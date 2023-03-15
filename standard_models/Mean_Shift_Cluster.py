


#Imports
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MeanShift

features = subdata

scaler = StandardScaler()
features_std = scaler.fit_transform(features)

cluster = MeanShift(n_jobs=-1)

model = cluster.fit(features_std)

model.labels_

