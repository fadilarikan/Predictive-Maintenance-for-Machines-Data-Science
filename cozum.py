
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import confusion_matrix,classification_report

from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier



### Veri Yükleme
csv_measurements = os.path.join('measurements.csv')
df_measurements = pd.read_csv(csv_measurements, parse_dates=['measurement_time'])
df_measurements = df_measurements.sort_values(by=['measurement_time'], ascending=[True])

csv_failures = os.path.join('failures.csv')
df_failures = pd.read_csv(csv_failures, parse_dates=['failure_time'])
df_failures = df_failures.sort_values(by=['failure_time'], ascending=[True])

### Veri Birleştirme
df_combined = pd.merge_asof(
    df_measurements,
    df_failures,
    left_on='measurement_time',
    right_on='failure_time',
    by='gadget_id',
    direction='forward',
)

### Sütun Ekleme
df_combined['time_to_fail'] = df_combined['failure_time']-df_combined['measurement_time']
df_combined['fail_in_1h'] = np.where(df_combined['time_to_fail']<pd.Timedelta(hours=1), 1, 0)

### Ölçüleri Düzenleme
df_combined = df_combined.reset_index(drop=True)
df_combined = df_combined.sort_values(by=['gadget_id', 'measurement_time'], ascending=[True, True])

df_combined['temperature_6h_std'] = df_combined.groupby('gadget_id')['temperature'].rolling(6).std(ddof=0).reset_index(drop=True)
df_combined['pressure_6h_mean'] = df_combined.groupby('gadget_id')['pressure'].rolling(6).mean().reset_index(drop=True)

### Eğitim ve Test Kümesi Oluşturma
X = ['vibration_y', 'pressure_6h_mean', 'temperature_6h_std']
y = 'fail_in_1h'
cols = X + [y]

df_to_split = df_combined.copy()
df_to_split = df_to_split.dropna(subset=cols)
df_to_split = df_to_split.drop(['Unnamed: 10', 'Unnamed: 11'], axis=1)
df_to_split = df_to_split.reset_index(drop=True)

##### Binarye çevirme
binner = KBinsDiscretizer(n_bins=10, encode='onehot-dense', strategy='kmeans')
binner.fit(df_to_split[X])
arr_bins= binner.transform(df_to_split[X])
df_bins = pd.DataFrame(arr_bins)

X = list(df_bins.columns)
cols = X + [y]

df_to_split = pd.concat([df_to_split, df_bins], axis=1)

df_train = df_to_split[df_to_split['gadget_id'].isin([1,2,3,4])].reset_index(drop=True).copy()
df_test = df_to_split[df_to_split['gadget_id'].isin([5,6])].reset_index(drop=True).copy()

print(f"Training data: {df_train.shape}")
print(f"Test data: {df_test.shape}")

### Tahmin parametreleri
w0 = 1
w1 = 6
pos_label = 1
### Neural Network
nn = MLPClassifier( # sklearn.neural_network.MLPClassifier
    solver='lbfgs', 
    alpha=1e-6,
    max_iter=10000,
    activation='relu',
    hidden_layer_sizes=(32,32,32),
)
nn.fit(df_train[X], df_train[y])
df_test['nn'] = nn.predict(df_test[X])
cm = confusion_matrix(df_test['fail_in_1h'], df_test['nn'])

print("------------------------------")
print("Neural Network:")
print(classification_report(df_test['fail_in_1h'],df_test['nn']))
print("CM:\n ",cm)

### Random Forest 
random_forest = RandomForestClassifier(
    min_samples_leaf=7,
    random_state=45,
    n_estimators=50,
    class_weight={0:w0, 1:w1}
)
random_forest.fit(df_train[X], df_train[y])
df_test['random_forest'] = random_forest.predict(df_test[X])
cm = confusion_matrix(df_test['fail_in_1h'], df_test['random_forest'])

print("------------------------------")
print("Random Forest:")
print(classification_report(df_test['fail_in_1h'],df_test['random_forest']))
print("CM:\n ",cm)

### KNN
knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(df_train[X], df_train[y])
df_test['knn'] = knn.predict(df_test[X])
cm = confusion_matrix(df_test['fail_in_1h'], df_test['knn'])

print("------------------------------")
print("KNN:")
print(classification_report(df_test['fail_in_1h'],df_test['knn']))
print("CM:\n ",cm)

### SVM
svm = SVC()
svm.fit(df_train[X], df_train[y])
df_test['svm'] = svm.predict(df_test[X])
cm = confusion_matrix(df_test['fail_in_1h'], df_test['svm'])

print("------------------------------")
print("SVC:")
print(classification_report(df_test['fail_in_1h'],df_test['svm']))
print("CM:\n ",cm)

#print(svm.predict([[10,18.5,3.41657]]))

### NAIVE BAYES
bayes = GaussianNB()
bayes.fit(df_train[X], df_train[y])
df_test['bayes'] = bayes.predict(df_test[X])
cm = confusion_matrix(df_test['fail_in_1h'], df_test['bayes'])

print("------------------------------")
print("Naive Bayes:")
print(classification_report(df_test['fail_in_1h'],df_test['bayes']))
print("CM:\n ",cm)

### DECISION TREE
decision= DecisionTreeClassifier(criterion="entropy")
decision.fit(df_train[X], df_train[y])
df_test['decision'] = decision.predict(df_test[X])
cm = confusion_matrix(df_test['fail_in_1h'], df_test['decision'])

print("------------------------------")
print("Decision Tree:")
print(classification_report(df_test['fail_in_1h'],df_test['decision']))
print("CM:\n ",cm)



#XGBoost
xgboost = XGBClassifier()
xgboost.fit(df_train[X], df_train[y])
df_test['xgboost'] = xgboost.predict(df_test[X])
cm = confusion_matrix(df_test['fail_in_1h'], df_test['xgboost'])

print("------------------------------")
print("XGBoost:")
print(classification_report(df_test['fail_in_1h'],df_test['xgboost']))
print("CM:\n ",cm)


#Gradient Boosting
gb_model = GradientBoostingClassifier()
gb_model.fit(df_train[X], df_train[y])
df_test['gb_model'] = gb_model.predict(df_test[X])
cm = confusion_matrix(df_test['fail_in_1h'], df_test['gb_model'])

print("------------------------------")
print("Gradient Boosting:")
print(classification_report(df_test['fail_in_1h'],df_test['gb_model']))
print("CM:\n ",cm)

### Logistic Regression
logistic = LogisticRegression()
logistic.fit(df_train[X], df_train[y])
df_test['logistic'] = logistic.predict(df_test[X])
cm = confusion_matrix(df_test['fail_in_1h'], df_test['logistic'])

print("------------------------------")
print("Logistic Regression:")
print(classification_report(df_test['fail_in_1h'],df_test['logistic']))
print("CM:\n ",cm)


#print(logistic.predict([[5,2,1]]))
