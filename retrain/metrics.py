from sklearn import metrics
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


def accuracy(y_true, y_pred, normalize=True, sample_weight=None):
    """ Accuracy classification score.
    Args:
        y_true: 1d array-like, or label indicator array / sparse matrix
        y_pred: 1d array-like, or label indicator array / sparse matrix
        normalize: bool, optional (default=True)
        sample_weight: array-like of shape (n_samples,), default=None
    Return:
        accuracy score: float
    """
    return accuracy_score(y_true, y_pred, normalize=normalize, sample_weight=sample_weight)


def f1(y_true, y_pred, pos_label=0, average='binary'):
    """ F1 = 2 * (precision * recall) / (precision + recall).
    Args:
        y_true: 1d array-like, or label indicator array / sparse matrix
        y_pred: 1d array-like, or label indicator array / sparse matrix
        pos_label: str or int, 1 by default. setting labels=[pos_label] and average != 'binary' will report scores for that label only.
        average: string, [None, ‘binary’ (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’].
    Return:
        f1 score: float
    """
    return f1_score(y_true, y_pred, average='weighted') 


def precision(y_true, y_pred, pos_label=0, average='binary'):
    """ the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives.
    Args:
        y_true: 1d array-like, or label indicator array / sparse matrix
        y_pred: 1d array-like, or label indicator array / sparse matrix
        pos_label: str or int, 1 by default. setting labels=[pos_label] and average != 'binary' will report scores for that label only.
        average: string, [None, ‘binary’ (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’].
    Return:
        precision score: float
    """
    return precision_score(y_true, y_pred, average='weighted') 


def recall(y_true, y_pred, pos_label=0, average='binary'):
    """ the ratio tp / (tp + fn) where tp is the number of true positives and fp the number of false negatives.
    Args:
        y_true: 1d array-like, or label indicator array / sparse matrix
        y_pred: 1d array-like, or label indicator array / sparse matrix
        pos_label: str or int, 1 by default. setting labels=[pos_label] and average != 'binary' will report scores for that label only.
        average: string, [None, ‘binary’ (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’].
    Return:
        recall score: float
    """
    return recall_score(y_true, y_pred, average='weighted')


def cfm(y_true, y_pred):
    """ Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
        y_true: 1d array-like, or label indicator array / sparse matrix
        y_pred: 1d array-like, or label indicator array / sparse matrix
    Return:
        confusion matrix: ndarray of shape (n_classes, n_classes)
    """
    return metrics.multilabel_confusion_matrix(y_true, y_pred)


def classification_report(y_true, y_pred, target_names=['barking', 'howling', 'crying', 'others']):
    """ Build a text report showing the main classification metrics.
    Args:
        y_true: 1d array-like, or label indicator array / sparse matrix
        y_pred: 1d array-like, or label indicator array / sparse matrix
        target_names: list of strings. display names matching the labels (same order).
    Return:
        report: string
    """
    return metrics.classification_report(y_true, y_pred, target_names=target_names, digits=len(target_names))

def evaluate(y_t, y_p):
    ap = metrics.average_precision_score(
            y_t, y_p, average=None)
    try:
        auc = metrics.roc_auc_score(y_t, y_p, average=None)
    except ValueError:
        auc = 0
    return ap, auc