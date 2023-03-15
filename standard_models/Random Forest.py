

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

from sklearn.datasets import load_boston


# Test data
# boston = load_boston()
# features = boston.data[:,0:2]
# target = boston.target


# 
#Transforming
target_data, feature_data, target_name, feature_names = split_target_and_features(data = sdata2, target = target)

features_train, features_test, target_train, target_test = reset_index_train_test_split(feature_data, target_data, test_size=0.2, random_state=2020)

train_features_np, test_features_np, scaler_features = scale_data(train = features_train, test = features_test, scaler = MinMaxScaler())

train_target_np, test_target_np, scaler_target = scale_data(train = target_train, test = target_test, scaler = MinMaxScaler())


X_all = scaler_features.transform(sdata2[features])
X_all
y_all = scaler_target.transform(sdata2[target].to_numpy().reshape(-1,1)).flatten()
y_all



randomforest = RandomForestRegressor(random_state=0, n_jobs=-1, n_estimators=150, max_features=1, bootstrap = True)

model_rf = randomforest.fit(train_features_np, train_target_np.flatten())
# yes, it needs to be flatten!


round(model_rf.score(train_features_np, train_target_np), 4)  #0.97
round(model_rf.score(test_features_np, test_target_np), 4)   #0.86
round(model_rf.score(X_all, y_all), 4)   #0.96


evaluate_model(model = model_rf, feature_liste = feature_names, number= 15)



importances = model_rf.feature_importances_
importances

importances_df = pd.DataFrame(zip(feature_names, abs(model_rf.feature_importances_)),
                    columns=["feature", "weight"],).sort_values("weight").reset_index(drop=True)

importances_df





def plotly_pareto(data,column_of_names, column_of_values, yname = None, title = None, online = False):
    """plotly pareto makes a combined bar and lineplot for absolut and percentage values

    Arguments:
        data {[type]} -- pandas DataFrame
        column_of_names {[type]} -- Name of column in data with categories for plot
        column_of_values {[type]} -- Name of column in data with values for plot

    Keyword Arguments:
        yname {[type]} -- to rename y axis, or to use name of column of values (default: {None})
    """
    #
    if yname == None:
        yname = column_of_values
    #
    data_sort = data.sort_values(by = column_of_values, ascending=False ).reset_index(drop = True)
    YY = data_sort.loc[:, column_of_values].tolist()
    XX = data_sort.loc[:, column_of_names].tolist()
    x_list = ["I." + str(i) for i in XX]
    x_list = np.asarray(x_list)
    y_per = [y /sum(YY) *100 for y in YY]
    #
    #x_list
    #y_per
    #
    output = []
    for i in range(1, (len(y_per)+1)):
        output.append(sum(y_per[0:i]) )
    #output
    # Create figure with secondary y-axis
    # fig = go.Figure()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
        name = "Barplot",
        x = x_list,
        y = YY,
        ),
        secondary_y = False
    )
    #
    fig.add_trace(
        go.Scatter(
            x = x_list,
            y = output,
            mode = "lines+markers",
            name = "percentage line",
            marker = dict(
                color = "red"
            )
        ),
        secondary_y = True
    )
    #
    if title == None:
        title = "Paretoplot"
    #
    fig.update_layout(
        title_text = title,
        xaxis = dict(categoryorder = "array", categoryarray = x_list)
    )
    fig.update_yaxes(title_text = "percentage",
                    range = (0, 101),
                    showgrid = True,
                    gridwidth = 1,
                    gridcolor = "white",
                    secondary_y = True)
    fig.update_yaxes(title_text = yname,
                    showgrid=True,
                    gridwidth = 1,
                    gridcolor = "black",
                    secondary_y = False)
    # finanly show plot
    if online == True:
        return(fig)
    if online != True:
        plotly.offline.plot(fig, filename = "pareto"+".html")


plotly_pareto(data = importances_df.iloc[-15:,:], 
                column_of_names = "feature", 
                column_of_values = "weight", 
                yname = None, title = None, online = False)



raw_data["predicted_target"] = apply_model(data = raw_data, 
                                        feature_names = feature_names, 
                                        scaler_features = scaler_features, 
                                        scaler_target = scaler_target, 
                                        model= model_rf)



r2_score(raw_data[target], raw_data["predicted_target"])  #0.81




plot_multi(x = raw_data["Date/Time"], y = [raw_data[target], raw_data["predicted_target"]],
            color = ["royalblue", "red", "black"], title = "Color prediction Random Forest Regression", online = False)


# # selector

# from sklear.features_selector import SelectFromModel

# randomforest = RandomForestClassifier(random_state=0, n_njobs=-1)

# selector = SelectFromModel(randomforest, threshold=0.3)

# features_important = selector.fit_transform(featrues, target)

# model = randomforest.fit(features_important, target)





