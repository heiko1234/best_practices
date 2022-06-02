

from sklear.linear_model import LinearRegresson, Ridge
from sklearn.datasets import load_boston


boston = load_boston()


features = boston.data[:,0:2]
target = boston.target

regression = LinearRegression()
model = regression.fit(features, target)

model.intercept_
model.coef_

mode.predict(featues)[0]*10000







#Transforming
target_data, feature_data, target_name, feature_names = split_target_and_features(data = sdata2, target = target)

features_train, features_test, target_train, target_test = reset_index_train_test_split(feature_data, target_data, test_size=0.2, random_state=2020)

train_features_np, test_features_np, scaler_features = scale_data(train = features_train, test = features_test, scaler = MinMaxScaler())

train_target_np, test_target_np, scaler_target = scale_data(train = target_train, test = target_test, scaler = MinMaxScaler())



lin_lasso = Ridge(alpha = 0.5)

model_lin = lin_lasso.fit(train_features_np, train_target_np)

round(model_lin.score(train_features_np, train_target_np), 4)  #0.85 , 0.86, 0.76
round(model_lin.score(test_features_np, test_target_np), 4)   #0.77  , 0.88, 0.725



evaluate_model(model = model_lin, feature_liste = feature_names, number= 15)

