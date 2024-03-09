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
    """Class to run GridSearchCV
    """
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
        """Function to run cross validation
        Currently the cross validation score can be used for Classification Metrics only
        
        Parameters
        ----------
        X : array
            The train data predictors
        y : array
            The train data target variable
        estimator : object
            The model or estimator that would be used
        cv : int
            The number of folds

        Returns
        -------
        avg_train_score : int
            The average training score for each fold
        avg_valid_score : int
            The average validation score for each fold
        """

        # Change the predictors and target to array
        X = np.array(X).copy()
        y = np.array(y).copy()

        # Instantiate the KFold
        k_fold = KFold(n_folds=cv)

        # Prepare empty list to store the train and validation score
        score_train_list = []
        score_valid_list = []
        for i, (idx_train, idx_valid) in enumerate(k_fold.split(X)):
            # Get the index of training and validatin data from the K fold
            X_train, y_train = X[idx_train], y[idx_train]
            X_valid, y_valid = X[idx_valid], y[idx_valid]

            # Fit the model
            model = deepcopy(estimator)
            model.fit(X_train, y_train)

            # Predict the model
            y_pred_train = model.predict(X_train)
            y_pred_valid = model.predict(X_valid)

            # Get the train and valid score
            score_train = SCORES[self.scoring](y_train, y_pred_train)
            score_valid = SCORES[self.scoring](y_valid, y_pred_valid)

            # Store the score
            score_train_list.append(score_train)
            score_valid_list.append(score_valid)

        # Get the average of each train and validatin score
        avg_score_train = np.mean(score_train_list)
        avg_score_valid = np.mean(score_valid_list)

        return avg_score_train, avg_score_valid


    def _parameters_combinations(self):
        """Function to generate hyperparameter combination

        Returns:
        -------
        param_combination : list
            The list of combination of hyperparameters
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
        """Procedure to change the estimator attributes to the desired hyperparameters
    
        Parameters
        ----------
        estimator : object
            The instantiated estimator
        grid_parameters : dict
            The combination of the hyperparameters
        """

        for parameter in grid_parameters:
            # Check the grid parameter is available as the estimator attribute
            is_param_available = estimator.__dict__.get(parameter, False)

            # Check whether the estimator attribute as a sub-estimator i.e Random Forest or Bagging
            if not is_param_available:
                is_param_available = estimator.__dict__["estimator"].__dict__.get(parameter, False)
                if not is_param_available:
                    # This for future development
                    pass
                else:
                    # assign grid parameter to the estimator's attribute
                    estimator.__dict__["estimator"].__dict__[parameter] = grid_parameters[parameter]
            else:
                # attribute is available
                # assign grid parameter to the estimator's attribute
                estimator.__dict__[parameter] = grid_parameters[parameter]

    def fit(self, X, y):
        """Fitting the train data and test data
    
        Parameters
        ----------
        X : array
            The predictors of the training data
        y : array
            The target of the training data

        Returns
        -------
        resulst_table : pd.DataFrame
            The table to show the results of GridSearchCV
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

            # Get the cross validation scores
            cv_score_train, cv_score_valid = self._cross_validation_score(X=X,
                                                                          y=y,
                                                                          cv=self.cv,
                                                                          estimator=estimator)
            # store the parameters
            self._parameters.append(parameter)
            
            # Group the parameters into dictionary
            results = {
                "parameter": parameter,
                "score_method": self.scoring,
                "cv_score_train": cv_score_train,
                "cv_score_valid": cv_score_valid
            }

            self.cv_results.append(deepcopy(results))

        # Change the results int a table
        results_table = pd.DataFrame(self.cv_results)

        return results_table