import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv(r'C:\Users\Josh\Downloads\NBA_train.csv')
print(data)
y = data['W']
X = data[['PTS', 'oppPTS']]

# To perform Naive Baye's

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.7)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

nb = GaussianNB()
nb.fit(X, y)
y_pred = nb.predict(X_test)
accuracy_nb= round(nb.score(X_test, y_pred) * 100,2)
print("NB accuracy is:",accuracy_nb)