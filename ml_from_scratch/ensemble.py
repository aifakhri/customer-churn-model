import numpy as np

from ._base_ensemble import BaseEnsemble

from sklearn.tree import DecisionTreeClassifier


class RandomForestClassifier(BaseEnsemble):
    """
    """

    def __init__(
        self,
        n_estimators=100,
        max_features="sqrt",
        random_state=123,
        criterion="gini",
        max_depth=None,
        min_sample_leaf=2,
        min_impurity_decrease=0.0,
    ):
        self.estimator = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_leaf=min_sample_leaf,
            min_impurity_decrease=min_impurity_decrease
        )
        super().__init__(
            estimator=self.estimator,
            n_estimators=n_estimators,
            max_features=max_features,
            random_state=random_state
        )