

# Ridge

from sklearn.linear_model import Ridge
from sklearn.datasets import load_boston
from sklearn.preporcessing import StandardScaler

boston = load_boston()
features = boston.data
target = boston.target

scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# Ridge
regression = Ridge(alpha = 0.5)

model = regression.fit(features_standadized, target)



## Lasso 

from sklearn.linear_model import Lasso

# data preprocessing analog

regression = Lasso(alpha = 0.5)

model = regression.fit(features_standardized, target)

model = regression.fit(features_standardized, target)
model = regression.fit(train_features_np, train_target_np)


model.coef_
# coef = 0: not used in Model









