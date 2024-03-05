import numpy as np

from ._base_tree import BaseDecisionTree
from ._tree_criterion import gini, entropy




CRITERION = {
    "gini": gini,
    "entropy": entropy
}



def _majority_vote(y):
    """
    """

    unique_val , n_unique = np.unique(y, return_counts=True)

    max_unique = np.argmax(n_unique)
    y_pred = unique_val[max_unique]

    return y_pred

class DecisionTreeClassifier(BaseDecisionTree):
    """The Decision Tree for the classification problem
    """

    def __init__(
        self,
        criterion="gini",
        max_depth=None,
        min_sample_split=2,
        min_sample_leaf=1
    ):
        super().__init__(
            criterion=criterion,
            max_depth=max_depth,
            min_sample_split=min_sample_split,
            min_sample_leaf=min_sample_leaf
        )

    def fit(self, X, y):
        """
        """
        self._impurity_method = CRITERION[self.criterion]
        self._split_technique = _majority_vote

        super(DecisionTreeClassifier, self).fit(X, y)