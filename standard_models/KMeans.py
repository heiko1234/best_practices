

#Imports
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


features = subdata

scaler = StandardScaler()
features_std = scaler.fit_transform(features)

features_std

cluster = KMeans(n_clusters = 9, random_state = 0)   #oder 10

model = cluster.fit(features_std)

model.labels_



