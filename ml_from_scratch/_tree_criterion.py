import numpy as np



def gini(y):
    """
    """

    # Extract class
    K, n_unique = np.unique(y, return_counts=True)
    unique_class = dict(zip(k, n_unique))

    n_target = len(y)
    proba = {}
    for k in K:
        proba[k] = unique_class[k] / n_target

    # Calculate impurity
    impurity = 0
    for k in K:
        impurity += proba[k] * (1-proba[k])

    return impurity


def entropy(y):
    """
    """

    K, n_unique = np.unique(y, return_counts=True)
    unique_class = dict(zip(k, n_unique))

    n_target = len(y)
    proba = {}
    for k in K:
        proba[k] = unique_class[k] / n_target

    # Calculate impurity
    impurity = 0
    for k in K:
        impurity += proba[k] * np.log(proba[k])

    return impurity