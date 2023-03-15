


#Imports
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


features = subdata

scaler = StandardScaler()
features_std = scaler.fit_transform(features)

features_std

cluster = DBSCAN(n_jobs=-1)

model = cluster.fit(features_std)

model.labels_


