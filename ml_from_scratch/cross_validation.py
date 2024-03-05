import numpy as np


from copy import deepcopy
from ._resampling import KFold
from .metrics import __all__


def cross_validation_score(X,
                           y,
                           estimator,
                           scoring,
                           cv=5):
    """
    """

    X = np.array(X).copy()
    y = np.array(y).copy()

    # Split data
    k_fold = KFold(n_folds=cv)

    scoring = __all__[scoring]
    score_train_list = []
    score_valid_list = []
    for i, (idx_train, idx_valid) in enumerate(k_fold.split(X)):
        X_train, y_train = X[idx_train], y[idx_train]
        X_valid, y_valid = X[idx_valid], y[idx_valid]

        # Fit the model
        model = deepcopy(estimator)
        model.fit(X_train, y_train)

        # Predict the model
        y_pred_train = model.predict(X_train)
        y_pred_valid = model.predict(X_valid)

        # score
        score_train = scoring(y_train, y_pred_train)
        score_valid = scoring(y_valid, y_pred_valid)

        #
        score_train_list.append(score_train)
        score_valid_list.append(score_valid)

    return score_train_list, score_valid_list