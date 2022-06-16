
#####

import numpy as np
import plotly
import plotly.graph_objs as go
# import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV


def plotly_distribution(data, target=None, plot=True):
    """
    Plot the distributions of all variables next to one another

    Inputs:
        - data: data frame with variables to plot
        - target: means any parameter in the data, can be None
        - plot False or True, True = plot
    """
    X_scaled = MinMaxScaler().fit_transform(data)

    if target is None:
        color = "royalblue"
    else:
        color = data[target]

    # FRANZI: list comprehension instead of for loop
    # FRANZI: x_output: numpy can generate arrays directly, much more efficient than a for loop!!!
    data_list = [go.Scatter(
                    x=i + 1 + 0.1*np.random.randn(data.shape[0]),
                    y=X_scaled[:, i],
                    mode='markers',
                    name=data.columns[i],
                    marker=dict(
                        size=10,
                        color=color
                    )
                ) for i in range(0, data.shape[1])]

    layout = {
        "xaxis": dict(
            range=(0, data.shape[1]+1),
            constrain="domain",
            ticktext=data.columns.to_list(),
            tickvals=list(range(1, data.shape[1]+1)),
        ),
        "yaxis": dict(
            range=(-0.05, 1.05),
            constrain="domain",
            title="scaled parameter",
        ),
        "title": dict(
            text="distribution of each parameter",
        ),
        "showlegend": False
    }

    fig = {
        "data": data_list,
        "layout": layout,
    }
    if plot:
        plotly.offline.plot(fig, filename="plotly_data_distribution.html")
        
    else:
        return plotly.graph_objs.Figure(fig)


def plotly_gridsearch_heatmap(model, parameters, X, y, scoring=None, plot=False):
    # FRANZI: at a few places you were using "parameter" instead of "parameters".
    # this didn't result in an error here, because instead of a main loop like below
    # you've defined your variables as global variables above in the script and they
    # were used instead. But this would throw an error later or have unintended effects.

    # select only on a training split to avoid overfitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)
    gs = GridSearchCV(model, parameters, scoring=scoring, n_jobs=-1, cv=5)
    gs.fit(X_train, y_train)
    print("Grid Search score:", gs.score(X_test, y_test))

    p1, p2 = sorted(parameters.keys())

    fig = go.Figure(data=go.Heatmap(
                        z=gs.cv_results_["mean_test_score"].reshape(len(parameters[p1]), -1),
                        x=list(parameters[p2]),
                        y=list(parameters[p1]),
                        hoverongaps=False
                    ))
    fig.update_layout(
            title = "heatmap of gridsearch",
            xaxis=dict(title=p2),
            yaxis=dict(title=p1),
        )
    # fig.update_xaxes(side="top")

    if plot:
        plotly.offline.plot(fig, filename="plotly_gridsearch_heatmap.html")
    else:
        return plotly.graph_objs.Figure(fig)





if __name__ == '__main__':
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor

    do_path = r"/home/heiko/Repos/data/ChemicalPlant/ChemicalManufacturingProcess.csv"
    do = pd.read_csv(do_path, sep=";")
    do.head()
    do.shape  # 176, 58
    # do = do.iloc[:, 1:]

    # do = pd.DataFrame()

    # select a few indexes

    data = do.iloc[:, [0, 1, 15, 16, 17, 18, 19, 20]]
    data
    cd = data.dropna()
    cd

    plotly_distribution(data=data, target=data.columns[0], plot=True)

    plotly_distribution(data=data, plot=True)

    ####

    y = cd.iloc[:, 0].to_numpy()
    y = y.flatten()
    X = cd.iloc[:, 1:].to_numpy()
    model = RandomForestRegressor(n_estimators=10)
    parameter = {"min_samples_leaf": range(1, 11), "max_depth": range(2, 21)}
    scoring = None

    plotly_gridsearch_heatmap(model=model, parameters=parameter, X=X, y=y, scoring=None, plot=True)
