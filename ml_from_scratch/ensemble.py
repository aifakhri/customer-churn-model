import numpy as np

from ._base_ensemble import BaseEnsemble

from sklearn.tree import DecisionTreeClassifier



class RandomForestClassifier(BaseEnsemble):
    """Random Forest Classifier

    Parameters
    ----------
    n_estimators: int
        The number of estimators to be used
    max_features: int
        The maximum features for Random Forest. There are only 2 options
            - sqrt
            - log2
    random_state: int
        The random state to maintain the reproducability
    criterion: str
        The Decision Tree criterion: gini, entropy, log_loss
    max_depth: int
        The maximum depth of the Decision Tree
    min_sample_leaf: int
        Minimum sample of the required to split
    min_impurity_decrease: float
        The number of impurity decrease that is desired to decide whether
        the node should be splitted again or not
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