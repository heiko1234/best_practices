
# # https://towardsdatascience.com/powerful-feature-selection-with-recursive-feature-elimination-rfe-of-sklearn-23efb2cdb54e


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


sdata2 = sdata2.dropna(axis = 1)
sdata2 = sdata2.reset_index(drop = True)


# Feature, target arrays
X, y = sdata2.iloc[:,1:], sdata2.iloc[:,0]


# Train/test set generation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Scale train and test sets with StandardScaler
# X_train
# X_test
X_train_std = StandardScaler().fit_transform(X_train)
X_test_std = StandardScaler().fit_transform(X_test)



# Fix the dimensions of the target array
y_train = y_train.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)
# y_test
# y_train


# Init, fit, test Lasso Regressor
forest = RandomForestRegressor()
model_fit = forest.fit(X_train_std, y_train)
forest.score(X_test_std, y_test)



feature_weight_df = pd.DataFrame(
                    zip(X_train.columns, abs(forest.feature_importances_)),
                    columns=["feature", "weight"],).sort_values("weight").reset_index(drop=True)

feature_weight_df



from sklearn.feature_selection import RFE


# Init the transformer
rfe = RFE(estimator=RandomForestRegressor(), n_features_to_select=10)


# Fit to the training data
rfe_model = rfe.fit(X_train_std, y_train)


X_train.loc[:, rfe_model.support_]


print(X_train.columns[rfe.support_])


print("Trainign R-sqaured:", rfe.score(X_train_std, y_train))
print("Testing R-squared:", rfe.score(X_test_std, y_test))




# # Init, fit, score
# forest = RandomForestRegressor()
# _ = forest.fit(rfe.transform(X_train_std), y_train)
# forest.score(rfe.transform(X_test_std), y_test)


