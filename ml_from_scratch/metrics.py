import numpy as np





def classification_report(y_true, y_pred, score=""):
    """
    """

    tp = np.array((y_pred == 1) & (y_true == 1)).sum()
    tn = np.array((y_pred == 0) & (y_true == 0)).sum()
    fp = np.array((y_pred == 1) & (y_true == 0)).sum()
    fn = np.array((y_pred == 0) & (y_true == 1)).sum()

    if score == "":
        confusion_matrix = np.array([[tp, fp], [fn, tn]])
        return confusion_matrix
    else:

        accuracy_score = (tp + tn) / (tp + tn + fp + fn)
        recall_score = tp / (tp + fn)
        precision_score = tp / (tp + fp)

        report = {
            "accuracy_score": accuracy_score,
            "recall_score": recall_score,
            "precision_score": precision_score
        }

        return report[score]

def accuracy_score(y_true, y_pred):
    accuracy = classification_report(y_true=y_true,
                                     y_pred=y_pred,
                                     score="accuracy_score")

    return accuracy

def recall_score(y_true, y_pred):
    recall = classification_report(y_true=y_true,
                                     y_pred=y_pred,
                                     score="recall_score")

    return recall

def precision_score(y_true, y_pred):
    precision = classification_report(y_true=y_true, 
                                     y_pred=y_pred,
                                     score="precision_score")

    return precision