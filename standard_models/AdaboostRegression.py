




from standard_models.mlutils import plotly_gridsearch_heatmap



# from O19_Standard_Models.mlutils import plotly_gridsearch_heatmap


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold, KFold
import numpy as np



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




### AdaBoostRegressor

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor





y = train_target_np
X = train_features_np

# 

#Transforming
target_data, feature_data, target_name, feature_names = split_target_and_features(data = sdata2, target = target)

features_train, features_test, target_train, target_test = reset_index_train_test_split(feature_data, target_data, test_size=0.2, random_state=2020)

train_features_np, test_features_np, scaler_features = scale_data(train = features_train, test = features_test, scaler = MinMaxScaler())

train_target_np, test_target_np, scaler_target = scale_data(train = target_train, test = target_test, scaler = MinMaxScaler())



y = train_target_np
X = train_features_np


# 
# y = target_train
# X = features_train
# y = y.to_numpy()

y = y.flatten()
# y


X_all = scaler_features.transform(sdata2[features])
X_all
y_all = scaler_target.transforms(data2[target].to_numpy().reshape(-1,1)).flatten()
y_all

# X_all = data[features].to_numpy()
# y_all = data[target].to_numpy()


#
model_adaboost = AdaBoostRegressor(DecisionTreeRegressor(max_leaf_nodes=200, min_samples_split=50), n_estimators=130, learning_rate=0.05)
#model_adaboost.fit(X, y)

model_adaboost.fit(train_features_np, train_target_np.flatten())


cv = KFold(n_splits = 10)
single_scores = cross_val_score(model_adaboost, X_all, y_all, scoring = "r2", cv = cv, n_jobs= -1 )
single_scores = cross_val_score(model_adaboost, features_train, target_train, scoring = "r2", cv = cv, n_jobs= -1 )
single_scores.mean() #0.54


model_adaboost.score( train_features_np, train_target_np) # 0.938
model_adaboost.score(test_features_np, test_target_np) #0.826

model_adaboost.score(X_all, y_all)


r2_score(data[target], data["predicted_target"])  #0.776



raw_data["predicted_target"] = apply_model(data = raw_data, 
                                        feature_names = feature_names, 
                                        scaler_features = scaler_features, 
                                        scaler_target = scaler_target, 
                                        model= model_adaboost)


raw_data = raw_data[raw_data["predicted_target"].notna()]
raw_data = raw_data[raw_data[target].notna()]

r2_score(raw_data[target], raw_data["predicted_target"])  #0.75



plot_multi(x = raw_data["Date/Time"], y = [raw_data[target], raw_data["predicted_target"]],
            color = ["royalblue", "red", "black"], title = "Color prediction AdaBoost", online = False)





#grid search parameters
parameters = dict(n_estimators=range(100,200,10), learning_rate=[0.05, 0.1, 0.15, 0.2] )
# parameters = dict(max_leaf_nodes=range(100, 200, 10), min_samples_split=[10, 20, 30, 40] )  #geht nich
plotly_gridsearch_heatmap(model = model_adaboost, parameters = parameters, X = X, y = y, scoring=None, online=False)

