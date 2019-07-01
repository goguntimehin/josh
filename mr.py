import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.api as s
from sklearn.linear_model import LinearRegression

data = pd.read_csv(r'C:\Users\Josh\Downloads\NBA_train.csv')
print(data)
y = data['W']
X = data[['PTS', 'oppPTS']]

# This is to split our module into train and testing

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.7)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)
model  = linreg.fit(X_test,y_test)

# To calculate the  RMSE and R2

print ("R^2 is: \n", model.score(X_test, y_test))
predictions = model.predict(X_test)
from sklearn.metrics import mean_squared_error
print ('RMSE is: \n', mean_squared_error(y_test, predictions))