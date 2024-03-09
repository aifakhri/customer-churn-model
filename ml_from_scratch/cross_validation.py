import itertools
import numpy as np
import pandas as pd


from copy import deepcopy
from ._resampling import KFold
from .metrics import accuracy_score, recall_score, precision_score



SCORES = {
    "accuracy": accuracy_score,
    "recall": recall_score,
    "precision": precision_score
}



class GridSearchCV:
    def __init__(
        self,
        estimator,
        param_grid,
        scoring="accuracy",
        cv=5
    ):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv

    def _cross_validation_score(self,
                                X,
                                y,
                                estimator,
                                cv=5):
        """Currently the cross validation score can be used for Classification Metrics only
        """

        X = np.array(X).copy()
        y = np.array(y).copy()

        # Split data
        k_fold = KFold(n_folds=cv)

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

            # 
            score_train = SCORES[self.scoring](y_train, y_pred_train)
            score_valid = SCORES[self.scoring](y_valid, y_pred_valid)

            score_train_list.append(score_train)
            score_valid_list.append(score_valid)

        avg_score_train = np.mean(score_train_list)
        avg_score_valid = np.mean(score_valid_list)

        return avg_score_train, avg_score_valid


    def _parameters_combinations(self):
        """
        """

        # Extract key and values
        param_values = [ val for val in self.param_grid.values() ]
        param_keys = [ key for key in self.param_grid.keys() ]

        # Get the parameter combination
        param_combinations = [
            dict(zip(param_keys, comb)) for comb in itertools.product(*param_values)
        ]

        return param_combinations

    def _initiate_hyper_parameters(self, estimator, grid_parameters):
        """
        """

        for parameter in grid_parameters:
            # Check the grid parameter is available as the estimator attribute
            is_param_available = estimator.__dict__.get(parameter, False)

            # Check whether the estimator attribute as a sub-estimator i.e Random Forest or Bagging
            if not is_param_available:
                is_param_available = estimator.__dict__["estimator"].__dict__.get(parameter, False)
                if not is_param_available:
                    pass
                else:
                    estimator.__dict__["estimator"].__dict__[parameter] = grid_parameters[parameter]
            else:
                # attribute is available
                # assign grid parameter to the estimator's attribute
                estimator.__dict__[parameter] = grid_parameters[parameter]


    def fit(self, X, y):
        """
        """

        # Get all parameter combinations
        parameters = self._parameters_combinations()

        # itreate thorough the combinations
        self._parameters = []
        self._cv_score_train = []
        self._cv_score_valid = []

        self.cv_results = []
        for i, parameter in enumerate(parameters):
            # instantiate the estimator
            estimator = self.estimator()

            # change estimators value based on the grid parameters
            self._initiate_hyper_parameters(estimator=estimator,
                                            grid_parameters=parameter)

            cv_score_train, cv_score_valid = self._cross_validation_score(X=X,
                                                                          y=y,
                                                                          cv=self.cv,
                                                                          estimator=estimator)

            self._parameters.append(parameter)
            results = {
                "parameter": parameter,
                "score_method": self.scoring,
                "cv_score_train": cv_score_train,
                "cv_score_valid": cv_score_valid
            }

            self.cv_results.append(deepcopy(results))

        results_table = pd.DataFrame(self.cv_results)

        return results_table