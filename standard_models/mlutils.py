import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


import plotly
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split, GridSearchCV



def plot_distributions(df, vis_cols=None, fname=None):
    """
    Plot the distributions of all variables next to one another

    Inputs:
        - df: data frame with variables to plot
        - vis_cols: which variables to plot (default: all) - first variable will be used to color the dots,
                    i.e. you might want to put your target variable as the first here
        - fname: if specified, file name where the plot will be saved
    """
    if vis_cols is None:
        vis_cols = df.columns
    X_s = MinMaxScaler().fit_transform(df[vis_cols])
    # plot individual features
    plt.figure(figsize=(15, 7))
    for i, f in enumerate(vis_cols):
        plt.scatter(i + 0.1*np.random.randn(len(X_s[:, i])), X_s[:, i], s=10, c=df[vis_cols[0]], alpha=0.6, cmap="PiYG")
    plt.xticks(list(range(len(vis_cols))), vis_cols, rotation=90)
    plt.xlim(-1, len(vis_cols))
    plt.colorbar()
    if fname is not None:
        plt.savefig("%s.pdf" % fname, dpi=300, bbox_inches="tight")


def plot_heatmap(matrix, ticks_bottom=None, ticks_left=None, title=None, figsize=(12, 10)):
    """
    Plot a matrix as a heatmap incl. colorbar and text annotations.

    Inputs:
        - matrix: n x m matrix to be plotted
        - ticks_bottom: m ticks displayed on the x axis
        - ticks_left: n ticks displayed on the y axis
        - title: title of the figure
        - figsize: size of the figure (default: (12, 10), good for square matrices);
                   if None: don't create a new figure
    """
    assert matrix.shape[0] == len(ticks_left), "ticks_left has to contain %i entries, not %i" % (matrix.shape[0], len(ticks_left))
    assert matrix.shape[1] == len(ticks_bottom), "ticks_bottom has to contain %i entries, not %i" % (matrix.shape[1], len(ticks_bottom))
    if figsize is not None:
        plt.figure(figsize=figsize)
    plt.imshow(matrix)
    if ticks_bottom is not None:
        plt.xticks(list(range(len(ticks_bottom))), ticks_bottom, rotation=90)
    if ticks_left is not None:
        plt.yticks(list(range(len(ticks_left))), ticks_left)
    if title is not None:
        plt.title(title)
    # Loop over data dimensions and create text annotations
    if max(matrix.shape) < 50:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                plt.text(j, i, round(matrix[i, j], 2), ha="center", va="center", color="k")
    plt.colorbar()


def plot_true_pred(y_true, y_pred, title="", c_var=None, label=None):
    """
    Create a scatter plot with true vs. predicted values for each data point.

    Inputs:
        - y_true: array with true target values for n data points
        - y_pred: same shape as y_true but with predicted values
        - title: e.g. model name to include in the title of the plot (optional)
        - c_var: array of the same shape as y_true to color the points (optional; e.g. use an interesting feature)
        - label: name of the variable used for coloring the plot (will appear in the legend; optional)
    """
    plt.figure(figsize=(6.5, 6.5))
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], c="k", alpha=0.7)
    plt.scatter(y_true, y_pred, s=10, alpha=0.7, c=c_var, label=label)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("%s $R^2$: %.4f" % (title, r2_score(y_true, y_pred)))
    plt.grid()
    plt.axis("scaled")
    if label is not None:
        plt.legend(loc=0, fontsize=14)


def select_hyperparams(model, params, X, y, scoring=None):
    """
    Use a grid search to evaluate different parameter combinations and plot the results

    Inputs:
        - model: initialized sklearn model instance for which to check the parameters
        - params: dict with {"param_name": [values to check]}, where the param_name is the same as the
                  parameters used when initializing the models and the values are all the values to be tested
        - X, y: data, i.e., feature matrix and corresponding targets
        - scoring: name of metric used to evaluate the parameter combinations (default: whatever is implemented in model.score())
    """
    # select only on a training split to avoid overfitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)
    gs = GridSearchCV(model, params, scoring=scoring, n_jobs=-1, cv=5)
    gs.fit(X_train, y_train)
    # best parameter values (careful - might be overfitting)
    print("best parameters:", gs.best_params_)
    # plot results --> pick the most restrictive setting that still gives good results
    p1, p2 = sorted(params.keys())
    plt.figure()
    plt.imshow(gs.cv_results_["mean_test_score"].reshape(len(params[p1]), -1))
    plt.xticks(range(len(params[p2])), params[p2])
    plt.yticks(range(len(params[p1])), params[p1])
    plt.xlabel(p2)
    plt.ylabel(p1)
    plt.colorbar()


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


def interpret_xval(feat_cols, feature_importances):
    """
    Print feature importances collected in a cross-validation loop (e.g. from predict_xval())

    Inputs:
        - feat_cols: list of d features corresponding to the entries in feature_importances
        - feature_importances: array with k_folds x d_features feature importance scores from each xval fold
    """
    # statistics based on the feature importances collected in the xval
    fdict = dict(zip(feat_cols, feature_importances.mean(axis=0)))
    fdict_std = dict(zip(feat_cols, feature_importances.std(axis=0)))
    for f in sorted(fdict, key=fdict.get, reverse=True):
        if np.abs(fdict[f]) > 0.00001:
            print("%50s: %.5f +/- %.5f" % (f, fdict[f], fdict_std[f]))
    return max(fdict, key=lambda x: np.abs(fdict[x]))


def predict_xval(model, X, y, scoring=None, cv=10):
    """
    Evaluate the model in a k-fold cross-validation. The return values can be used with interpret_xval() and plot_true_pred().

    Inputs:
        - model: initialized sklearn model instance
        - X, y: data, i.e., feature matrix and corresponding targets
        - scoring: name of metric used to evaluate the parameter combinations (default: whatever is implemented in model.score())
        - cv: number of folds for the xval (default: 10)
    Returns:
        - y_true: true labels (for convenience, in the same order as y_pred)
        - y_pred: array with predicted labels from the test split in each xval fold
        - test_indices: data point indices of y_true and y_pred
        - feature_importances: array with k_folds x d_features feature importance scores from each xval fold
                               (if the model has coef_ or feature_importances_ attribute after training)
    """
    # collect predicted test values across folds
    y_true, y_pred = [], []
    test_indices = []
    feature_importances = []
    kf = KFold(n_splits=cv, shuffle=True, random_state=27)
    for train_index, test_index in kf.split(X):
        test_indices.append(test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # save test y
        y_true.append(y_test)
        # make & save prediction
        model.fit(X_train, y_train)
        y_pred.append(model.predict(X_test))
        # save feature importances / weights of the model for later interpretation
        if hasattr(model, "coef_"):
            feature_importances.append(model.coef_)
        elif hasattr(model, "feature_importances_"):
            feature_importances.append(model.feature_importances_)
    # transform into proper numpy arrays
    if len(y.shape) > 1:
        # target matrix or vector?!
        y_true, y_pred = np.vstack(y_true), np.vstack(y_pred)
    else:
        y_true, y_pred = np.hstack(y_true), np.hstack(y_pred)
    test_indices, feature_importances = np.hstack(test_indices), np.vstack(feature_importances)
    # real cross-validated score (instead of computing it from the predicted values like when creating the plot)
    xval_scores = cross_val_score(model, X, y, scoring=scoring, cv=cv, n_jobs=-1)
    print("## cross-validated score: %.4f +/- %.4f" % (xval_scores.mean(), xval_scores.std()))
    return y_true, y_pred, test_indices, feature_importances


def eval_model(model, df, feat_cols, target_cols):
    # extract X and y from dataframe
    X = df[feat_cols].to_numpy()
    y = df[target_cols].to_numpy()
    if len(target_cols) == 1:
        y = y.flatten()
    # scale the data (important for the linear regression models, especially to interpret the coefficients)
    X = StandardScaler().fit_transform(X)
    print("### Model:", str(model))
    # check model fit on the whole dataset (!)
    model.fit(X, y)
    try:
        best_feat = interpret_model(feat_cols, model)
        print("# best feature: ", best_feat)
    except:
        best_feat = target_cols[0]
    plot_true_pred(y, model.predict(X), model.__repr__() + " train", df[best_feat], best_feat)
    # realistic performance with xval
    y_true, y_pred, test_indices, feature_importances = predict_xval(model, X, y)
    if len(feature_importances):
        best_feat = interpret_xval(feat_cols, feature_importances)
    else:
        best_feat = target_cols[0]
    plot_true_pred(y_true, y_pred, model.__repr__() + " xval", df[best_feat].to_numpy()[test_indices], best_feat)






#####


def plotly_distribution(data, target=None, online=False):
    """
    Plot the distributions of all variables next to one another

    Inputs:
        - data: data frame with variables to plot
        - target: means any parameter in the data, can be None
        - online False or True, False = plot
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
    if online:
        return plotly.graph_objs.Figure(fig)
    else:
        plotly.offline.plot(fig, filename="plotly_data_distribution.html")


def plotly_gridsearch_heatmap(model, parameters, X, y, scoring=None, online=False):
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

    if online:
        return plotly.graph_objs.Figure(fig)
    else:
        plotly.offline.plot(fig, filename="plotly_gridsearch_heatmap.html")

