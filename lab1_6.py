import pandas as pd
from matplotlib.testing.jpl_units import km

from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_csv(r'C:\Users\Josh\Downloads\NBA_train.csv')
print(data)

from sklearn.preprocessing import LabelEncoder
data= data.apply(LabelEncoder().fit_transform)

X = data

from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
scaler.fit(X)
X_scaled_array = scaler.transform(X)
X_scaled = pd.DataFrame(X_scaled_array, columns = X.columns)

from sklearn.cluster import KMeans
nclusters = 3 # this is the k in kmeans
seed = 0
km = KMeans(n_clusters=nclusters, random_state=seed)
km.fit(X)

# predict the cluster for each data point
y_cluster_kmeans= km.predict(X)
from sklearn import metrics
score = metrics.silhouette_score(X, y_cluster_kmeans)
print(score)

# k means determine k
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])



# Plot the elbow
plt.plot(K, distortions, 'bx-');
plt.xlabel('k');
plt.ylabel('Distortion');
plt.title('The Elbow Method showing the optimal k');
plt.show()





