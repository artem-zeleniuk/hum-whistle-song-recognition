"""Models on fixed-length feature vectors: random forest, SVM, XGBoost and LightGBM."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MODELS = ("random_forest", "svm", "xgboost", "lightgbm")


def build(name, seed, n_jobs=1, sweep=False):
    """One untrained classifier.

    `sweep=True` gives the smaller XGBoost used to compare numbers of temporal parts (200 trees,
    learning rate 0.1), the default is the final setting (up to 500 trees, learning rate 0.05).
    """
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5,
                                      min_samples_leaf=2, random_state=seed, n_jobs=n_jobs)
    if name == "svm":
        # probability=True adds Platt scaling so that the SVM gives class probabilities like the others
        return SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=seed)
    if name == "xgboost":
        from xgboost import XGBClassifier
        if sweep:
            return XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, min_child_weight=5,
                                 subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=15,
                                 eval_metric="mlogloss", random_state=seed, n_jobs=n_jobs, verbosity=0)
        return XGBClassifier(n_estimators=500, max_depth=3, learning_rate=0.05, min_child_weight=5,
                             reg_alpha=0.1, reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8,
                             early_stopping_rounds=20, eval_metric="mlogloss", random_state=seed,
                             n_jobs=n_jobs, verbosity=0)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=500, max_depth=3, learning_rate=0.05, num_leaves=15,
                              min_child_samples=10, reg_alpha=0.1, reg_lambda=1.0, subsample=0.8,
                              colsample_bytree=0.8, random_state=seed, n_jobs=n_jobs, verbose=-1)
    raise ValueError(name)


def fit_predict(name, seed, x_train, y_train, x_val, y_val, x_test, n_jobs=1, sweep=False, scale=True):
    """Train one classifier and return its class probabilities.

    The features are standardised with the mean and standard deviation of the training set. The two
    boosting models stop adding trees when the validation loss has not improved for 20 rounds (15 in
    the sweep) and predict with the best number of trees.

    Returns a dict with `val` and `test` (class probabilities), `train_accuracy` and `rounds` (number of
    trees used, None for the models without early stopping).
    """
    if scale:
        scaler = StandardScaler().fit(x_train)
        x_train, x_val, x_test = (scaler.transform(x) for x in (x_train, x_val, x_test))
    model = build(name, seed, n_jobs=n_jobs, sweep=sweep)
    rounds = None
    if name == "xgboost":
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        rounds = int(model.best_iteration) + 1
    elif name == "lightgbm":
        from lightgbm import early_stopping, log_evaluation
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)],
                  callbacks=[early_stopping(20, verbose=False), log_evaluation(0)])
        rounds = int(model.best_iteration_)
    else:
        model.fit(x_train, y_train)
    train = model.predict_proba(x_train)
    return {
        "val": model.predict_proba(x_val).astype(np.float32),
        "test": model.predict_proba(x_test).astype(np.float32),
        "train_accuracy": float(np.mean(train.argmax(axis=1) == y_train)),
        "rounds": rounds,
    }


def tune_xgboost(x_train, y_train, groups, seed=0, n_iter=30, n_jobs=1):
    """Random search over XGBoost settings, scored by cross-validation inside the training set.

    Thirty settings are drawn. Each is scored by the mean accuracy of three folds of the training set
    that keep the clips of one participant together, so the search never sees the validation or the
    test fold. Returns the best settings and their cross-validation accuracy.
    """
    from scipy.stats import randint, uniform
    from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
    from xgboost import XGBClassifier
    space = {
        "max_depth": randint(2, 8),
        "learning_rate": uniform(0.01, 0.19),
        "n_estimators": randint(100, 500),
        "min_child_weight": randint(1, 10),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "reg_alpha": uniform(0, 1),
        "reg_lambda": uniform(0.5, 2),
    }
    folds = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        XGBClassifier(eval_metric="mlogloss", random_state=seed, n_jobs=1, verbosity=0), space, n_iter=n_iter,
        cv=folds, scoring="accuracy", random_state=seed, n_jobs=n_jobs, refit=False)
    search.fit(x_train, y_train, groups=groups)
    best = {key: (float(value) if isinstance(value, float) else int(value))
            for key, value in search.best_params_.items()}
    return best, float(search.best_score_)


def fit_predict_fixed(settings, seed, x_train, y_train, x_val, x_test, n_jobs=1):
    """Train XGBoost with given settings and a fixed number of trees (no early stopping)."""
    from xgboost import XGBClassifier
    model = XGBClassifier(eval_metric="mlogloss", random_state=seed, n_jobs=n_jobs, verbosity=0, **settings)
    model.fit(x_train, y_train)
    train = model.predict_proba(x_train)
    return {"val": model.predict_proba(x_val).astype(np.float32),
            "test": model.predict_proba(x_test).astype(np.float32),
            "train_accuracy": float(np.mean(train.argmax(axis=1) == y_train)), "rounds": int(settings["n_estimators"])}


def drop_correlated(x, threshold=0.85):
    """Indices of the columns to keep: a column goes if it correlates above `threshold` with an earlier one."""
    corr = np.abs(np.corrcoef(x, rowvar=False))
    upper = np.triu(corr, k=1)
    return np.flatnonzero(~(upper > threshold).any(axis=0))


def select_features(x_train, y_train, x_val, x_test, seed=0, n_jobs=1):
    """Three-step feature selection fitted on the training set only.

    1. Drop columns with variance below 0.01.
    2. Drop columns that correlate above 0.85 with an earlier column.
    3. Standardise, fit a random forest with 100 trees and keep the columns whose importance is above
       the mean importance.

    Returns the three reduced and standardised matrices and the numbers of columns left after each step.
    """
    variance = VarianceThreshold(threshold=0.01).fit(x_train)
    x_train, x_val, x_test = (variance.transform(x) for x in (x_train, x_val, x_test))
    n_variance = x_train.shape[1]
    keep = drop_correlated(x_train)
    x_train, x_val, x_test = (x[:, keep] for x in (x_train, x_val, x_test))
    n_correlation = x_train.shape[1]
    scaler = StandardScaler().fit(x_train)
    x_train, x_val, x_test = (scaler.transform(x) for x in (x_train, x_val, x_test))
    forest = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=n_jobs).fit(x_train, y_train)
    chosen = SelectFromModel(forest, threshold="mean", prefit=True)
    x_train, x_val, x_test = (chosen.transform(x) for x in (x_train, x_val, x_test))
    counts = {"after_variance": int(n_variance), "after_correlation": int(n_correlation),
              "selected": int(x_train.shape[1])}
    return x_train, x_val, x_test, counts
