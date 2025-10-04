"""Matplotlib-based visualization helpers for model evaluation.

This module contains small, reusable plotting helpers used by the training
script to produce diagnostic figures. Each plotting function returns the
matplotlib Axes instance so callers can further customize the figure or save
it to disk (as done in `Modelo/main.py`). The helpers are intentionally
lightweight and accept either a caller-provided Axes or create their own.

Functions:
    - plot_confusion_matrix(cm, labels, ax=None)
    - plot_cv_scores(scores, ax=None)
    - plot_truepositives_vs_others(y_true, y_pred, ax=None)
    - plot_train_test_accuracy(train_acc, test_acc, ax=None)
    - compute_cv_scores(estimator, X, y, cv=10)

Notes on inputs:
    - `cm` should be a 2D numpy array of shape (n_classes, n_classes)
    - `labels` should be an ordered sequence of strings corresponding to the
        rows/columns of the confusion matrix
    - `scores` may be any iterable of numeric CV scores (will be converted to
        a 1D numpy array)
"""

from typing import Optional, Sequence, Mapping

import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import numpy as np
from sklearn.model_selection import cross_val_score


def plot_confusion_matrix(cm: np.ndarray, labels: Sequence[str], ax: Optional[plt.Axes] = None):
    """Plot a confusion matrix (2D array) with labels.

    Args:
        cm: confusion matrix array (shape [n_classes, n_classes])
        labels: sequence of class labels (length n_classes)
        ax: optional matplotlib Axes to draw on
    Returns:
        ax: the Axes with the plot
    """
    # Create new Axes if caller did not provide one
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    # Normalize inputs and prepare ticks
    labels = list(labels)
    ticks = np.arange(len(labels))

    # Draw heatmap of the confusion matrix
    cmap = plt.get_cmap('Blues')
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # Set tick labels and axis labels
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_title('Confusion matrix')

    # Annotate each cell with the integer count. Use a threshold based on the
    # maximum cell value to decide text color for readability. Protect against
    # empty matrices by checking cm.size.
    fmt = 'd'
    thresh = cm.max() / 2. if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(int(cm[i, j]), fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    return ax


def plot_cv_scores(scores, ax: Optional[plt.Axes] = None):
    """Plot N-fold CV scores as a boxplot with points.

    Args:
        scores: sequence of CV scores
        ax: optional Axes
    Returns:
        ax
    """
    # Ensure scores are a 1-D numpy array for plotting. Accepts lists, tuples,
    # pandas Series, or numpy arrays.
    scores = np.asarray(list(scores))

    # Create Axes if necessary
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    # Draw boxplot + scatter of individual fold scores to show distribution
    ax.boxplot(scores, vert=True, notch=True, patch_artist=True)
    ax.scatter(np.ones_like(scores), scores, color='black', alpha=0.6)
    ax.set_ylabel('Score')
    ax.set_xticks([1])
    ax.set_xticklabels(['CV folds'])
    ax.set_title('Cross-validation scores')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    return ax


def plot_truepositives_vs_others(y_true, y_pred, ax: Optional[plt.Axes] = None):
    """Compare counts of true positives vs other outcomes per-class.

    Args:
        y_true: true labels
        y_pred: predicted labels
        ax: optional Axes
    Returns:
        ax
    """
    # Convert inputs to numpy arrays for vectorized operations
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Determine the set of labels present in either true or predicted arrays
    labels = np.unique(np.concatenate([y_true, y_pred]))

    # Count true positives (correct predictions) and other outcomes (where
    # the true label was the class but the prediction was different)
    tp_counts = []
    other_counts = []
    for lab in labels:
        tp = np.sum((y_true == lab) & (y_pred == lab))
        others = np.sum((y_true == lab) & (y_pred != lab))
        tp_counts.append(int(tp))
        other_counts.append(int(others))

    x = np.arange(len(labels))
    width = 0.35

    # Create Axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    # Two side-by-side bars per class: true positives and others
    ax.bar(x - width / 2, tp_counts, width, label='True Positives', color='tab:green')
    ax.bar(x + width / 2, other_counts, width, label='Others (false/misclassified)', color='tab:orange')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel('Count')
    ax.set_title('True positives vs others (per class)')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    return ax


def plot_train_test_accuracy(train_acc: float, test_acc: float, ax: Optional[plt.Axes] = None):
    """Simple bar chart comparing train and test accuracy.
    """
    # Create Axes if caller didn't provide one
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    labels = ['Train', 'Test']
    values = [train_acc, test_acc]
    colors = ['tab:blue', 'tab:purple']

    # Draw bars and format the y-axis to [0, 1] for accuracy
    ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Accuracy')
    ax.set_title('Train vs Test Accuracy')

    # Annotate numeric values above each bar for readability
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.3f}", ha='center')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    return ax


def compute_cv_scores(estimator, X, y, cv: int = 10, scoring: str = 'accuracy'):
    """Compute cross-validation scores using sklearn.cross_val_score and return the array of scores.

    Args:
        estimator: an sklearn estimator implementing fit/predict
        X: feature matrix
        y: labels
        cv: number of folds
        scoring: scoring string passed to cross_val_score
    Returns:
        scores: numpy array of CV scores
    """
    # Use all cores for faster CV when available; callers may choose to pass an
    # unfitted estimator or one configured with specific hyperparameters.
    scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return scores


def plot_binary_confusion(y_true, y_pred, labels: Optional[Sequence[str]] = None, ax: Optional[plt.Axes] = None, normalize: bool = False, cmap: str = 'Blues'):
    """Plot a 2x2 binary confusion matrix showing TN, FP, FN, TP.

    This helper is for binary classification only. It accepts raw label arrays
    and computes the four counts automatically. If `labels` is provided it
    should be a sequence of two strings used for tick labels (order: negative,
    positive). If `normalize` is True, values are shown as proportions per true
    class (row-normalized).

    Returns the Axes instance.
    """
    # Convert inputs to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    # Determine unique labels (must be exactly two)
    unique = np.unique(np.concatenate([y_true, y_pred]))
    if unique.size != 2:
        raise ValueError("plot_binary_confusion requires exactly two distinct labels in y_true/y_pred")

    neg_label, pos_label = unique[0], unique[1]

    # Compute counts
    TN = int(np.sum((y_true == neg_label) & (y_pred == neg_label)))
    FP = int(np.sum((y_true == neg_label) & (y_pred == pos_label)))
    FN = int(np.sum((y_true == pos_label) & (y_pred == neg_label)))
    TP = int(np.sum((y_true == pos_label) & (y_pred == pos_label)))

    cm = np.array([[TN, FP], [FN, TP]])

    # Optionally normalize per true-class (row-wise)
    display_cm = cm.astype(float)
    if normalize:
        row_sums = display_cm.sum(axis=1, keepdims=True)
        # avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            display_cm = np.divide(display_cm, row_sums, where=row_sums != 0)

    # Prepare axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    im = ax.imshow(display_cm, interpolation='nearest', cmap=plt.get_cmap(cmap))
    ax.figure.colorbar(im, ax=ax)

    # Tick labels: use provided labels if given, otherwise stringified unique labels
    if labels is not None:
        tick_labels = list(labels)
        if len(tick_labels) != 2:
            raise ValueError("labels must be a sequence of two strings for binary confusion")
    else:
        tick_labels = [str(neg_label), str(pos_label)]

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax.set_yticklabels(tick_labels)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_title('Binary confusion matrix')

    # Annotate cells with TN/FP/FN/TP and values. Use count and, if normalized,
    # show proportion on the next line.
    labels_map = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            if normalize:
                frac = display_cm[i, j]
                txt = f"{labels_map[i][j]}\n{val:d}\n({frac:.2f})"
            else:
                txt = f"{labels_map[i][j]}\n{val:d}"
            ax.text(j, i, txt, ha='center', va='center', color='white' if display_cm[i, j] > display_cm.max() / 2. else 'black')

    plt.tight_layout()
    return ax

def plotkfold_results(results: Mapping[str, np.ndarray], ax: Optional[plt.Axes] = None):
    """Plot per-fold k-fold results produced by `ten_fold_cross_validation`.

    Args:
        results: mapping with keys 'accuracy', 'precision', 'recall', 'f1'
                 each value should be a 1-D array-like of per-fold scores.
        ax: (ignored) kept for API compatibility with other helpers. The
            function creates a 2x2 set of subplots and returns the figure
            and axes.

    Returns:
        fig, axes: matplotlib Figure and a 2x2 array of Axes objects.

    Behavior/notes:
        - Missing metrics will be skipped but the function will attempt to
          plot any of the four supported metrics that are present in `results`.
        - Each subplot shows the per-fold points, a boxplot, and the mean/std
          annotated in the title.
    """
    # Convert results to a dict-like mapping and validate inputs
    if results is None:
        raise ValueError("results must be a mapping produced by ten_fold_cross_validation")

    metrics = ["accuracy", "precision", "recall", "f1"]
    present = [m for m in metrics if m in results]
    if not present:
        raise ValueError(f"No supported metrics found in results. Expected one of {metrics}")

    # Create a 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.reshape(-1)

    for idx, metric in enumerate(metrics):
        ax_i = axes[idx]
        if metric not in results:
            # Clear unused subplot
            ax_i.axis('off')
            continue

        vals = np.asarray(results[metric])
        # Defensive: flatten and ensure 1-D
        vals = vals.ravel()

        # Boxplot + scatter of fold scores
        ax_i.boxplot(vals, vert=True, notch=True, patch_artist=True)
        ax_i.scatter(np.arange(1, len(vals) + 1), vals, color='black', alpha=0.7)
        ax_i.set_xticks(range(1, len(vals) + 1))
        ax_i.set_xticklabels([str(i) for i in range(1, len(vals) + 1)])
        ax_i.set_ylabel(metric.capitalize())
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        ax_i.set_title(f"{metric.capitalize()} per fold (mean={mean:.3f}, std={std:.3f})")
        ax_i.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    return fig, axes