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
        features_ = np.empty(feature_size, dtype='int')

        for i in range(n_estimators):
            features_[i] = np.random.choice(n_population,
                                            n_features,
                                            replace=False)
            features_[i].sort()

        return features_

    def _predict_ensemble(self, X, estimators, selected_feature):
        """
        """

        X = np.array(X).copy()
        n_samples = X.shape[0]

        n_estimators = len(estimators)

        pred_size = (n_estimators, n_samples)
        y_preds = np.empty(pred_size)

        for i, estimator in enumerate(estimators):
            X_i = X[:, selected_feature[i]]
            y_preds[i] = estimator.predict(X_i)

        return y_preds


    def _aggregate_ensemble_prediction(self, y_ensembles):
        """
        """

        n_samples = y_ensembles.shape[1]

        y_pred = np.empty(n_samples)
        for i in range(n_samples):
            y = y_ensembles[:, i]
            y_pred[i] = self._majority_vote_calculation(y=y)

        return y_pred

    def _majority_vote_calculation(self, y):
        """
        """

        val, count = np.unique(y, return_counts=True)

        max_idx = np.argmax(count)
        y_pred = val[max_idx]

        return y_pred

    def fit(self, X, y):
        """
        """

        # Copy and transform data input (X) and (y) into array
        X = np.array(X).copy()
        y = np.array(y).copy()

        # Extract information from data: n_data, samples, and classes
        self.n_samples, self.n_features = X.shape


        # Group the estimators into a list's object
        self.estimators = self._build_estimators(base_= self.estimator,
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
            n_population=self.n_features,
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

    def predict(self, X):
        """
        """

        y_pred_ensembles = self._predict_ensemble(
            X=X, 
            estimators=self.estimators,
            selected_feature=self.selected_features
        )

        y_preds = self._aggregate_ensemble_prediction(y_ensembles=y_pred_ensembles)

        return y_preds