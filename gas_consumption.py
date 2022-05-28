import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'C:\Users\aa092\OneDrive\桌面\machinelearning\petrol_consumption.csv')



#[ Column , row ]
X = dataset.iloc[:,0:4].values
y = dataset.iloc[:, 4].values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=3)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

regressor = RandomForestRegressor(n_estimators=40, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


print('平均絕對誤差:', metrics.mean_absolute_error(y_test, y_pred))
print('均方誤差:', metrics.mean_squared_error(y_test, y_pred))
print('均方根誤差:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))






