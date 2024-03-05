import numpy as np

from copy import deepcopy



class BaseEnsemble:
    """
    """

    def __init__(
        self,
        estimator,
        n_estimators,
        max_features=None,
        random_state=None
    ):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state

    def _build_estimators(self, base_, n_estimators):
        """
        """
        estimators = []
        for _ in range(n_estimators):
            estimators.append(deepcopy(base_))

        return estimators

    def _produce_bootstrap_sample(self,
                                  n_estimators,
                                  n_population,
                                  n_samples,):
        """
        """

        # Generate Random Seed for reproducability
        np.random.seed(self.random_state)

        sample_size = (n_estimators, n_samples)
        bootstrap_samples = np.random.choice(n_population,
                                             size=sample_size,
                                             replace=True)

        return bootstrap_samples

    def _select_features(self,
                         n_estimators,
                         n_population,
                         n_features):
        """
        """

        # Generate Random Seed for reproducability
        np.random.seed(self.random_state)
    
        # Generate Features
        feature_size = (n_estimators, n_features)
        features_ = np.empty(feature_size)

        for i in range(n_estimators):
            features_[i] = np.random_choice(n_population,
                                            n_features,
                                            replace=False)

        return features_

    def fit(self, X, y):
        """
        """

        # Copy and transform data input (X) and (y) into array
        X = np.array(X).copy()
        y = np.array(y).copy()

        # Extract information from data: n_data, samples, and classes
        self.n_samples, self.n_features = X.shape
        self.classes = y.unique()


        # Group the estimators into a list's object
        self.estimators = self._build_estimators(base_=self.estimator,
                                                 n_estimators=self.n_estimators)

        # Generate the Bootstrap Sample based on the data index
        bootstrap_samples = self._produce_bootstrap_sample(
            n_estimators=self.n_estimators,
            n_population=self.n_samples,
            n_samples=self.n_samples)

        # Feature Selection for Random Forest: Sqrt or Log
        if self.max_features == "sqrt":
            max_features = int(np.sqrt(self.n_features))
        elif self.max_features == "log2":
            max_features = int(np.log2(self.n_features))
    
        # 2. Select the features randomly
        self.selected_features = self._select_features(
            n_estimators=self.n_estimators,
            n_population=self.n_samples,
            n_features=max_features
        )

        # Fit each estimators with the data and the features
        for b in range(self.n_estimators):
            # Select bootstrap features
            X_bootstrap = X[:, self.selected_features[b]]

            # Select boostrap sample
            X_bootstrap = X_bootstrap[bootstrap_samples[b], :]
            y_bootstrap = y[bootstrap_samples[b]]

            estimator = self.estimators[b]
            estimator.fit(X_bootstrap, y_bootstrap)