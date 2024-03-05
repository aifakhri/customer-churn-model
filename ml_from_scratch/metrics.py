import numpy as np





def accuracy_score(y_true, y_pred):
    n_true = np.sum(y_true == y_pred)
    n_total = len(y_true)

    return n_true/n_total