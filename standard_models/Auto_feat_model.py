
from autofeat import FeatureSelector, AutoFeatRegressor


# remove nan rows
data_nan = sdata2

#data_nan = data


data_nan = data_nan.dropna(axis = 1)
data_nan = data_nan.reset_index(drop = True)
data_nan




#Transforming
target_data, feature_data, target_name, feature_names = split_target_and_features(data = sdata2, target = target)

features_train, features_test, target_train, target_test = reset_index_train_test_split(feature_data, target_data, test_size=0.2, random_state=2020)

train_features_np, test_features_np, scaler_features = scale_data(train = features_train, test = features_test, scaler = MinMaxScaler())

train_target_np, test_target_np, scaler_target = scale_data(train = target_train, test = target_test, scaler = MinMaxScaler())


X_all = scaler_features.transform(sdata2[features])
X_all
y_all = scaler_target.transform(sdata2[target].to_numpy().reshape(-1,1)).flatten()
y_all


features = [item for item in data_nan.columns 
                if target not in item 
                if "Feature_1" not in item 
                if "Feature_2" not in item 
                if "Feature_3" not in item 
                if "Feature_4" not in item
                if "Feature_5" not in item
                if "Feature_6" not in item 
                if "Feature_7" not in item
                if "Feature_8" not in item
                if "Feature_9" not in item
                if "diff" not in item
]

features = features


afreg = AutoFeatRegressor(verbose=1, feateng_steps=3)

# df = afreg.fit_transform(data_nan[features], data_nan[target])
afreg_model = afreg.fit(features_train, target_train)

# test on real targets
print("Final R^2: %.4f" % afreg.score(data_nan[features], data_nan[target]))   #0.87



round(afreg_model.score(features_train, target_train), 4)  #0.8778
round(afreg_model.score(features_test, target_test), 4)   #0.74
round(afreg_model.score(feature_data, target_data), 4)   #0.86


raw_data["predicted_target_afreg"] = afreg_model.predict(raw_data[features])

r2_score(raw_data[target], raw_data["predicted_target"])  #0.81


# Formular
# [AutoFeat] Final score: 0.8687





dd = afreg.predict(data_nan[features])
dd


data["predicted_target_afreg"] = afreg.predict(data[features])

raw_data["predicted_target_afreg"] = afreg.predict(raw_data[features])


# raw_data["predicted_target"] = 0

raw_data = raw_data[raw_data["predicted_target_afreg"] <= 200]
raw_data = raw_data[raw_data["predicted_target_afreg"] >= 0]


plot_multi(x = raw_data["Date/Time"], y = [raw_data[target], raw_data["predicted_target"], raw_data["predicted_target_afreg"]],
            color = ["royalblue", "red", "black"], title = "Color prediction", online = False)

r2_score(raw_data[target], raw_data["predicted_target_afreg"])  #0.71



