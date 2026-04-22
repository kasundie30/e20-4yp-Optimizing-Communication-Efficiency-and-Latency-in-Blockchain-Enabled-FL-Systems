"""
Model Quality Metrics for Evaluation

This module provides functions to calculate model quality metrics such as:
- Precision
- Recall
- F1 Score
"""

from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_precision(y_true, y_pred, average='binary'):
    """
    Calculate precision score.

    Parameters:
    - y_true (array-like): True labels.
    - y_pred (array-like): Predicted labels.
    - average (str): Averaging method ('macro', 'micro', 'weighted', etc.).

    Returns:
    - float: Precision score.
    """
    return precision_score(y_true, y_pred, average=average)

def calculate_recall(y_true, y_pred, average='binary'):
    """
    Calculate recall score.

    Parameters:
    - y_true (array-like): True labels.
    - y_pred (array-like): Predicted labels.
    - average (str): Averaging method.

    Returns:
    - float: Recall score.
    """
    return recall_score(y_true, y_pred, average=average)

def calculate_f1_score(y_true, y_pred, average='binary'):
    """
    Calculate F1 score.

    Parameters:
    - y_true (array-like): True labels.
    - y_pred (array-like): Predicted labels.
    - average (str): Averaging method.

    Returns:
    - float: F1 score.
    """
    return f1_score(y_true, y_pred, average=average)