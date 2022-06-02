



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import plot_partial_dependence, permutation_importance



#Decision tree + feature importance

# load classification dataset
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# train decision tree
clf = DecisionTreeClassifier(max_leaf_nodes=8, min_samples_split=10, class_weight="balanced", ccp_alpha=0.01)
clf.fit(X_train, y_train)
print("Accuracy on test data: {:.2f}".format(clf.score(X_test, y_test)))



# plot tree itself
plt.figure(figsize=(12, 6))
plot_tree(clf, feature_names=data.feature_names, filled=True, class_names=data.target_names, proportion=True);

plt.show()

# plt.savefig("expl_dt.pdf", dpi=300, bbox_inches="tight")


# from IPython.display import Image
# from sklearn import tree
# import pydotplus


# dot_data = tree.export_graphviz(clf, #out_file= None, 
#                             feature_names = data.feature_names, 
#                             class_names = data.target_names)

# graph = pydotplus.graph_from_dot_data(dot_data)

# Image(graph.create_png())


# plot feature importances
sort_idx = clf.feature_importances_.argsort()[-5:]
plt.figure(figsize=(5, 3.5))
plt.barh(range(len(sort_idx)), clf.feature_importances_[sort_idx], color='#15317E', )
plt.yticks(range(len(sort_idx)), data.feature_names[sort_idx])
plt.xlabel("feature importance", fontsize=14)

plt.show()

# plt.savefig("expl_dt_featimp.pdf", dpi=300, bbox_inches="tight")


# Linear regression: feature effects
cal_housing = fetch_california_housing()
X = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
y = cal_housing.target
y -= y.mean()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)


# fit linear regression model
lm = LinearRegression(fit_intercept=False).fit(X_train, y_train)
print(f"Test R2 score: {lm.score(X_test, y_test):.2f}")


# global feature effects
feature_effects = np.array([lm.coef_[i]*X_test.to_numpy()[:, i] for i in range(X_test.shape[1])]).T

# effects for one instance
instance_effects = feature_effects[np.random.randint(X_test.shape[0])]

# plot
sorted_idx = feature_effects.mean(axis=0).argsort()
plt.figure(figsize=(10, 6))
plt.boxplot(feature_effects[:, sorted_idx], vert=False, labels=np.array(cal_housing.feature_names)[sorted_idx])
plt.plot(instance_effects[sorted_idx], range(1, 1+len(sorted_idx)), "x", color="#C70039", mew=2.)
plt.title("Feature Effects")

plt.show()

# plt.savefig("expl_lm_effects.pdf", dpi=300, bbox_inches="tight");


# Permutation Effects

# load regression dataset
cal_housing = fetch_california_housing()
X = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
y = cal_housing.target
y -= y.mean()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)


# fit random forest
est = RandomForestRegressor(max_leaf_nodes=250, min_samples_split=25, n_estimators=100)
est.fit(X_train, y_train)
print(f"Test R2 score: {est.score(X_test, y_test):.2f}")


result = permutation_importance(est, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
sorted_idx = result.importances_mean.argsort()
sorted_idx
np.array(cal_housing.feature_names)[sorted_idx]

plt.figure(figsize=(10, 6))
plt.boxplot(result.importances[sorted_idx].T, vert=False, labels=np.array(cal_housing.feature_names)[sorted_idx])
plt.title("Permutation Importance")

plt.show()


import plotly.express as px
df = px.data.tips()
df
fig = px.box(df, x = "sex", y="total_bill")
result.importances[sorted_idx].T
np.array(cal_housing.feature_names)[sorted_idx]
fig = px.box(result.importances[sorted_idx], x = np.array(cal_housing.feature_names)[sorted_idx], y="total_bill")
fig.show()
# plt.savefig("expl_perm_imp.pdf", dpi=300, bbox_inches="tight");


# Plotly
data = pd.DataFrame(data=result.importances[sorted_idx].T, columns =np.array(cal_housing.feature_names)[sorted_idx] )
data
fig = px.box(data, x = data.columns, title="Permutation Importance")
fig.show()



#ICE/PD Plot

# pdp
features_plot = ['MedInc', 'AveOccup', 'Latitude', 'Longitude']
plt.figure(figsize=(16, 4))
display = plot_partial_dependence(
    est, X_train, features_plot, kind="both", subsample=50, line_kw={"color": '#15317E', "label": None},
    n_cols=4, n_jobs=-1, grid_resolution=20, random_state=13, ax=plt.gca()
)
display.figure_.subplots_adjust(wspace=0.1, hspace=0.3)

plt.show()

# plt.savefig("expl_pdp.pdf", dpi=300, bbox_inches="tight")


######
######

from sklearn.inspection import plot_partial_dependence, permutation_importance
import matplotlib.pyplot as plt


sdata2 = data[[target] + features]

round(model_lin.score(train_features_np, train_target_np), 4)  #0.85 , 0.86
round(model_lin.score(test_features_np, test_target_np), 4)   #0.77  , 0.88


est = lin_lasso

est = afreg
est = model_rf
est = model_adaboost
X_test = test_features_np
y_test = test_target_np


result = permutation_importance(est, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
sorted_idx = result.importances_mean.argsort()
plt.figure(figsize=(10, 6))
plt.boxplot(result.importances[sorted_idx].T, vert=False, labels=np.array(data[features].columns)[sorted_idx])
plt.title("Permutation Importance")
plt.show()



###

X_train = train_features_np
X_train
est
features


plt.figure(figsize=(16, 8))
display = plot_partial_dependence(
    estimator = est,
    X = X_train,
    features = features, 
    feature_names= features, kind="both", subsample=50, line_kw={"color": '#15317E', "label": None},
    n_cols=4, n_jobs=-1, grid_resolution=20, random_state=13, ax=plt.gca()
)
display.figure_.subplots_adjust(wspace=0.1, hspace=0.3)

plt.show()




