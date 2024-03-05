import numpy as np



class Tree:
    """This is the tree object for building a blueprint of the Decision Tree
    """
    def __init__(
        self,
        feature=None,
        threshold=None,
        value=None,
        impurity=None,
        children_left=None,
        children_right=None,
        is_leaf=False,
        n_samples=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.impurity = impurity
        self.children_left = children_left
        self.children_right = children_right
        self.is_leaf = is_leaf
        self.n_samples = n_samples


class BaseDecisionTree:
    """
    """

    def __init__(
        self,
        criterion,
        max_depth,
        min_sample_split,
        min_sample_leaf,
        min_impurity_decrease
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.min_sample_leaf = min_sample_leaf
        self.min_impurity_decrease = min_impurity_decrease

    def _best_split(self, X, y):
        """Function to find the feature that would be used 
        for splitting the tree
        
        """

        n_data = len(y)

        # Stopping criterion if sample is less than the minimum sample split
        if n_data < self.min_sample_split:
            return None, None

        parent_node = np.column_stack((X, y))
        information_gain = 0.0
        feature, threshold = None, None

        for feature_i in range(self.n_features):
            # Get all data for feature_i
            X_i = X[:, feature_i]
            
            threshold = self._possible_threshold(X=X_i)
            for i in range(len(threshold)):
                left_child, right_child = self._split_data(data=parent_node,
                                                           feature=feature_i,
                                                           threshold=threshold[i])
        
                left_y = left_child[:, self.n_features]
                right_y = right_child[:, self.n_features]

                # Check sample leaf
                cond_left = len(left_y) >= self.min_sample_leaf
                cond_right = len(right_y) >= self.min_sample_leaf

                if cond_left and cond_right:
                    current_gain = ""


    def _split_data(self, data, feature, threshold):
        """
        """

        split_left = data[:, feature]
        data_left = data[split_left]
        data_right = data[~split_left]
    
        return data_left, data_right

    def _possible_threshold(X):
        """
        """

        data = np.array(X).copy()

        unique_data = np.unique(data)
        unique_data.sort()

        unique_length = len(unique_data)

        threshold = np.empty(unique_length-1)

        for i in range(unique_length-1):
            value_1 = unique_length[i]
            value_2 = unique_length[i+1]

            threshold[i] = 0.5 * (value_1, value_2)

        return threshold

    def _grow_tree(self, X, y, depth):
        """Function to grow the Decision Tree
        """

        # Initialize the impurity algorithm and splitting technique
        node_impurity = self._impurity_method(y)
        node_value = self._splitting_technique(y)
        
        # Initialize the tree object as a blueprint of the tree
        node = Tree(
            impurity=node_impurity,
            value=node_value,
            is_leaf=True,
            n_samples=len(y)
        )

        # Initialize the stopping condition
        if self.max_depth is None:
            condition = True
        else:
            condition = depth < self.max_depth

        if condition:
            # 1. Find the Best split

            # 2. Grow Tree's child
            pass

    def _prune_tree(self, X, y):
        """Function to prune the Established Tree
        """
    
        pass


    def fit(self, X, y):
        """
        """

        # Copy data
        X = np.array(X).copy()
        y = np.array(y).copy()
    
        # Extract features and samples counts
        self.n_samples, self.n_features = X.shape
    
        # Grow the tree
        self.trees = self._grow_tree(X=X, y=y)
    
        # Prune tree
        self._prune_tree()
