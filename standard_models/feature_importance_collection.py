
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score

from sklearn.preprocessing import (
    StandardScaler
)


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
    # realistic performance with xval
    y_true, y_pred, test_indices, feature_importances = predict_xval(model, X, y)
    if len(feature_importances):
        best_feat = interpret_xval(feat_cols, feature_importances)
    else:
        best_feat = target_cols[0]
    return best_feat

