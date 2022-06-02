
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline

# # from sklearn.datasets import load_boston
# boston = load_boston()

# features = boston.data[:,0:2]
# target = boston.target


features_df = sdata2[features]
target_df = sdata2[target]



#Transforming
target_data, feature_data, target_name, feature_names = split_target_and_features(data = sdata2, target = target)

features_train, features_test, target_train, target_test = reset_index_train_test_split(feature_data, target_data, test_size=0.2, random_state=2020)

train_features_np, test_features_np, scaler_features = scale_data(train = features_train, test = features_test, scaler = MinMaxScaler())

train_target_np, test_target_np, scaler_target = scale_data(train = target_train, test = target_test, scaler = MinMaxScaler())


X_all = scaler_features.transform(sdata2[features])
X_all
y_all = scaler_target.transform(sdata2[target].to_numpy().reshape(-1,1)).flatten()
y_all





interaction = PolynomialFeatures(
    degree = 3, 
    include_bias = False,
    interaction_only = True
)




features_interaction = interaction.fit_transform(train_features_np)
features_interaction


regression = LinearRegression()

model = regression.fit(features_interaction, train_target_np)

model.intercept_
model.coef_

model.score(features_interaction, train_target_np) # 0.91


round(model.score(features_train, target_train), 4)  #0.8778
round(model.score(features_test, target_test), 4)   #0.74
round(model.score(feature_data, target_data), 4)   #0.86


#####################
#####################
#####################


# pipe = Pipeline([('interaction', PolynomialFeatures(degree = 3, 
#     include_bias = False,
#     interaction_only = True)), ('regression', LinearRegression())])

pipe = Pipeline([('interaction', PolynomialFeatures(degree = 3, 
    include_bias = False,
    interaction_only = True)), ('regression', Ridge(alpha=0.5))])



pipe.fit(train_features_np, train_target_np)


round(pipe.score(train_features_np, train_target_np), 4)  #0.79
round(pipe.score(test_features_np, test_target_np), 4)   #0.77
round(pipe.score(X_all, y_all), 4)   #0.78

raw_data["predicted_target"] = 0
raw_data["predicted_target"] = apply_model(data = raw_data, 
                                        feature_names = feature_names, 
                                        scaler_features = scaler_features, 
                                        scaler_target = scaler_target, 
                                        model= pipe)


r2_score(raw_data[target], raw_data["predicted_target"])  #0.81


plot_multi(x = raw_data["Date/Time"], y = [raw_data[target], raw_data["predicted_target"]],
            color = ["royalblue", "red", "black"], title = "Color prediction", online = False)



# Franzi 
def interpret_model(feat_cols, model):
    """
    Print feature importances for the given model

    Inputs:
        - feat_cols: list of features in the same order as the feature columns in X the model was trained on
        - model: trained model (either linear model with coef_ attribute or tree model with feature_importances_)
    """
    if hasattr(model, "intercept_"):
        print("intercept:", model.intercept_)
    try:
        fdict = dict(zip(feat_cols, model.feature_importances_))
    except:
        fdict = dict(zip(feat_cols, model.coef_))
    for f in sorted(fdict, key=fdict.get, reverse=True):
        if np.abs(fdict[f]) > 0.00001:
            print("%50s: %.5f" % (f, fdict[f]))
    return max(fdict, key=lambda x: np.abs(fdict[x]))


interpret_model(feat_cols= features, model = model)

