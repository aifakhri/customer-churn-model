import numpy as np





def classification_report(y_true, y_pred, score):
    """The function to generate classification metrics

    Currently only accuracy, precision, and recall that can be generated

    Parameters
    ----------
    y_true : array
        The actual prediction based on the dataset
    y_pred : array
        The prediction results from the model
    score : str
        The type of the score. Currently only these following score can be generated
            - Accruacy Score
            - Precisiion Score
            - Recall Score

    Returns
    -------
    report : int
        The specific metrics based on the score
    """

    # Generate True Positive, True Negative, False Positive, and False Negative
    tp = np.array((y_pred == 1) & (y_true == 1)).sum()
    tn = np.array((y_pred == 0) & (y_true == 0)).sum()
    fp = np.array((y_pred == 1) & (y_true == 0)).sum()
    fn = np.array((y_pred == 0) & (y_true == 1)).sum()

    # Calculate the accuracy, recall, and precision
    accuracy_score = (tp + tn) / (tp + tn + fp + fn)
    recall_score = tp / (tp + fn)
    precision_score = tp / (tp + fp)

    # Group the result into dictionary
    report = {
        "accuracy_score": accuracy_score,
        "recall_score": recall_score,
        "precision_score": precision_score
    }

    return report[score]

def accuracy_score(y_true, y_pred):
    """Function to generate accuracy score
   
     Parameters
    ----------
    y_true : array
        The actual prediction based on the dataset
    y_pred : array
        The prediction results from the model
   
    Return
    ------
    accuracy : int
        The accuracy metric
    """

    # Get accuracy from classification report
    accuracy = classification_report(y_true=y_true,
                                     y_pred=y_pred,
                                     score="accuracy_score")

    return accuracy

def recall_score(y_true, y_pred):
    """Function to generate recall score
    
    Parameters
    ----------
    y_true : array
        The actual prediction based on the dataset
    y_pred : array
        The prediction results from the model
    
    Return
    ------
    accuracy : int
        The accuracy metric
    """

    # Get recall score from classification report
    recall = classification_report(y_true=y_true,
                                     y_pred=y_pred,
                                     score="recall_score")

    return recall

def precision_score(y_true, y_pred):
    """Function to generate precision score

    Parameters
    ----------
    y_true : array
        The actual prediction based on the dataset
    y_pred : array
        The prediction results from the model
    
    Return
    ------
    accuracy : int
        The accuracy metric
    """

    # Get precision score from classification report
    precision = classification_report(y_true=y_true, 
                                     y_pred=y_pred,
                                     score="precision_score")

    return precision