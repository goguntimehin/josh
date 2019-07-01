import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

data = pd.read_csv(r'C:\Users\Josh\Downloads\NBA_train.csv')
print(data)
y = data['W']
X = data[['PTS', 'oppPTS']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.7)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# To perform Knn

##KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X, y)
y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_test, y_pred) * 100, 2)
print("KNN accuracy is:",acc_knn)