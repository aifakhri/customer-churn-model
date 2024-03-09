import numpy as np




class KFold:
    def __init__(
        self,
        n_folds=5,
        random_state=42
    ):
        self.n_folds = n_folds
        self.random_state = random_state


    def _iter_indices(self, X):
        """Iterate the data based on the n_folds
        """

        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        np.random.seed(self.random_state)
        np.random.shuffle(indices)


        # Define fold size and its length
        fold_length = n_samples // self.n_folds 
        fold_sizes = np.full(self.n_folds, fold_length, dtype="int")

        # Fold remainder for uneven splits
        fold_remainders = n_samples % self.n_folds

        # Assign the remainder to each folds
        fold_sizes[:fold_remainders] += 1
        # Assign data (index) to the fold
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current+fold_size
            yield indices[start:stop]
            current = stop

    def split(self, X):
        """Function to split data based on the n_folds
        """

        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        for valid_idx in self._iter_indices(X):
            train_index = np.array(
                [idx for idx in indices if idx not in valid_idx]
            )

            yield (train_index, valid_idx)
            