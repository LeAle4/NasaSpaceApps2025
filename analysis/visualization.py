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
from scipy import sparse as sp
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

class Graph:
    """Lightweight wrapper around a matplotlib Figure and Axes.

    The Graph class standardizes how plotting functions hand back drawable
    objects. It wraps a Figure and its primary Axes and provides helpers to
    save the figure, apply tight layout, and (optionally) create a Qt
    FigureCanvas for embedding in PyQt/PySide applications.

    Usage:
        g = Graph()                      # new figure + ax
        ax = g.ax                         # use as normal matplotlib Axes
        g.tight_layout()
        g.save('out.png')

        # or wrap an existing Axes
        g = Graph.from_axes(ax)
        canvas = g.to_qt_widget()        # returns FigureCanvasQTAgg if PyQt5/PySide2 present
    """

    def __init__(self, fig: Optional[plt.Figure] = None, ax: Optional[plt.Axes] = None, figsize=(6, 4)):
        if fig is not None and ax is not None:
            self.fig = fig
            self.ax = ax
        elif fig is not None and ax is None:
            self.fig = fig
            # take first axes or create one
            axes = fig.get_axes()
            self.ax = axes[0] if axes else fig.add_subplot(1, 1, 1)
        elif ax is not None:
            self.ax = ax
            self.fig = ax.figure
        else:
            self.fig, self.ax = plt.subplots(figsize=figsize)

    @classmethod
    def from_axes(cls, ax: plt.Axes):
        """Create a Graph wrapper from an existing Axes."""
        return cls(ax=ax)

    @classmethod
    def from_figure(cls, fig: plt.Figure):
        """Create a Graph wrapper from an existing Figure (uses first axes)."""
        return cls(fig=fig)

    def get_axes(self) -> plt.Axes:
        return self.ax

    def get_figure(self) -> plt.Figure:
        return self.fig

    def tight_layout(self, **kwargs):
        try:
            self.fig.tight_layout(**kwargs)
        except Exception:
            # fallback to plt.tight_layout if fig.tight_layout fails
            plt.tight_layout(**kwargs)

    def save(self, path: str, dpi: int = 150, bbox_inches: str = 'tight') -> None:
        """Save the figure to disk."""
        try:
            self.fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
        except Exception:
            plt.savefig(path, dpi=dpi, bbox_inches=bbox_inches)

    def clear(self):
        """Clear the underlying axes (useful before replotting)."""
        self.ax.cla()

    def to_qt_widget(self):
        """Return a Qt widget (FigureCanvas) for embedding in PyQt/PySide apps.

        This attempts to import PyQt5 first, then PySide2. If neither is
        available an ImportError is raised. The returned object is a
        `matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg` instance which
        can be inserted into Qt layouts directly.
        """
        # Lazy import to avoid hard dependency at module import time
        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        except Exception:
            try:
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            except Exception:
                raise ImportError("PyQt/PySide or the Qt backend for matplotlib is not available")

        canvas = FigureCanvas(self.fig)
        return canvas

    def __repr__(self) -> str:
        return f"<Graph fig={hex(id(self.fig))} ax={hex(id(self.ax))}>"

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
    cmap = mcm.get_cmap('Blues')
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
    return Graph.from_axes(ax)


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
    return Graph.from_axes(ax)


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
    return Graph.from_axes(ax)


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
    return Graph.from_axes(ax)


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


def plot_roc_auc(y_true, y_score, ax: Optional[plt.Axes] = None, pos_label=1, savepath: Optional[str] = None, show: bool = False):
    """Plot ROC curve and annotate AUC.

    This function is robust to being passed either continuous scores (recommended)
    or discrete predicted labels. It supports binary and multiclass inputs.

    Behavior:
    - Binary: plots single ROC curve and AUC.
    - Multiclass: plots per-class ROC curves and a micro-average curve; overall
      AUC is computed with `multi_class='ovr'` and `average='macro'`.

    Args:
        y_true: true labels (array-like)
        y_score: either continuous scores/probabilities for the positive class
                 (preferred) or discrete predicted labels. For multiclass, a
                 2D array of shape (n_samples, n_classes) with scores is also accepted.
        ax: optional Axes
        pos_label: positive label for binary case (default=1)
        savepath: optional PNG path to save the created figure
        show: whether to call plt.show()

    Returns:
        ax: matplotlib Axes with the ROC plot
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    classes = np.unique(y_true)
    n_classes = len(classes)

    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(figsize=(7, 6))

    # Binary case
    if n_classes == 2:
        # If y_score is 2D (probabilities), pick the column for pos_label when available
        if y_score.ndim == 2 and y_score.shape[1] >= 2:
            # try to find index of pos_label in classes
            try:
                idx = list(classes).index(pos_label)
            except ValueError:
                idx = 1
            score = y_score[:, idx]
        else:
            # If discrete predicted labels given (1D), convert to indicator for positive class
            if y_score.ndim == 1:
                score = (y_score == pos_label).astype(float)
            else:
                score = y_score

        # Compute ROC and AUC (roc_auc_score accepts discrete scores though
        # continuous probabilities are preferred)
        fpr, tpr, _ = roc_curve(y_true, score, pos_label=pos_label)
        auc_score = float(roc_auc_score(y_true, score))
        ax.plot(fpr, tpr, color='tab:blue', lw=2, label=f'ROC (AUC = {auc_score:.3f})')

    else:
        # Multiclass: need binarized form for curves
        y_true_bin = label_binarize(y_true, classes=classes)
        # Ensure dense numpy arrays for later indexing and ravel operations
        toarr = getattr(y_true_bin, 'toarray', None)
        if callable(toarr):
            y_true_bin = toarr()
        y_true_bin = np.asarray(y_true_bin)

        # Convert y_score to an (n_samples, n_classes) array if it's not already
        if y_score.ndim == 1:
            # discrete predicted labels -> binarize
            y_score_bin = label_binarize(y_score, classes=classes)
            toarr = getattr(y_score_bin, 'toarray', None)
            if callable(toarr):
                y_score_bin = toarr()
            y_score_bin = np.asarray(y_score_bin)
        elif y_score.ndim == 2 and y_score.shape[1] == n_classes:
            y_score_bin = y_score
        else:
            # fallback: attempt to binarize along classes
            try:
                y_score_bin = label_binarize(y_score, classes=classes)
                toarr = getattr(y_score_bin, 'toarray', None)
                if callable(toarr):
                    y_score_bin = toarr()
                y_score_bin = np.asarray(y_score_bin)
            except Exception:
                raise ValueError("y_score shape is incompatible with multiclass y_true")

        # Compute per-class ROC and plot
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score_bin[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            ax.plot(fpr[i], tpr[i], lw=1, alpha=0.6, label=f'Class {classes[i]} (AUC={roc_auc[i]:.3f})')

        # Micro-average ROC
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_score_bin.ravel())
        auc_micro = auc(fpr_micro, tpr_micro)
        ax.plot(fpr_micro, tpr_micro, color='black', lw=2, label=f'micro-average (AUC={auc_micro:.3f})')

        auc_score = float(roc_auc_score(y_true_bin, y_score_bin, multi_class='ovr', average='macro'))

    # Diagonal reference
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Chance')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC)')
    ax.legend(loc='lower right')
    ax.grid(linestyle='--', alpha=0.3)

    plt.tight_layout()

    if savepath is not None and created_fig is not None:
        try:
            created_fig.savefig(savepath, dpi=150, bbox_inches='tight')
        except Exception:
            plt.savefig(savepath, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return Graph.from_axes(ax)


def plot_pr_auc(y_true, y_score, ax: Optional[plt.Axes] = None, pos_label=1, savepath: Optional[str] = None, show: bool = False):
    """Plot Precision-Recall curve and annotate Average Precision (AP) for binary classification.

    Args:
        y_true: true binary labels (array-like)
        y_score: target scores; probabilities or decision function values for the
            positive class (array-like, same length as y_true)
        ax: optional matplotlib Axes to draw on. If None, a new figure is created.
        pos_label: label considered positive when building PR curve (default=1)
        savepath: optional path to save the figure as PNG
        show: whether to call plt.show() after plotting

    Returns:
        ax: the Axes containing the Precision-Recall plot
    """
    # Normalize inputs
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # If y_true looks already binarized/multilabel (2D), treat accordingly
    if y_true.ndim == 2:
        y_true_bin = y_true
        n_classes = y_true.shape[1]
        classes = list(range(n_classes))
    else:
        classes = np.unique(y_true)
        n_classes = len(classes)

    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(figsize=(6, 5))

    # Binary case
    if n_classes == 2:
        # Ensure y_true is 1-D labels (0/1 or original labels)
        if y_true.ndim == 2:
            # collapse to single label vector if necessary
            if y_true.shape[1] == 2:
                y_true = np.argmax(y_true, axis=1)
            else:
                raise ValueError("Unsupported binarized y_true shape for binary PR curve")

        # Extract probability-like scores for positive class when possible
        if y_score.ndim == 2 and y_score.shape[1] >= 2:
            try:
                idx = list(classes).index(pos_label)
            except ValueError:
                idx = 1
            score = y_score[:, idx]
        else:
            # If discrete predicted labels given as y_score, convert to indicator
            if y_score.ndim == 1 and set(np.unique(y_score)).issubset(set(classes)):
                score = (y_score == pos_label).astype(float)
            else:
                score = y_score

        precision, recall, _ = precision_recall_curve(y_true, score, pos_label=pos_label)
        ap = float(average_precision_score(y_true, score))
        ax.plot(recall, precision, color='tab:purple', lw=2, label=f'PR (AP = {ap:.3f})')

    else:
        # Multiclass: ensure y_true_bin and y_score_bin are (n_samples, n_classes)
        if y_true.ndim == 1:
            y_true_bin = label_binarize(y_true, classes=classes)
            toarr = getattr(y_true_bin, 'toarray', None)
            if callable(toarr):
                y_true_bin = toarr()
            y_true_bin = np.asarray(y_true_bin)
        else:
            y_true_bin = np.asarray(y_true)

        # Prepare y_score_bin
        if y_score.ndim == 1:
            # discrete predicted labels -> one-hot
            y_score_bin = label_binarize(y_score, classes=classes)
            if sp is not None and sp.issparse(y_score_bin):
                y_score_bin = y_score_bin.toarray()
            y_score_bin = np.asarray(y_score_bin)
        elif y_score.ndim == 2 and y_score.shape[1] == n_classes:
            y_score_bin = y_score
        else:
            # Try to coerce
            try:
                y_score_bin = label_binarize(y_score, classes=classes)
                if sp is not None and sp.issparse(y_score_bin):
                    y_score_bin = y_score_bin.toarray()
                y_score_bin = np.asarray(y_score_bin)
            except Exception:
                raise ValueError("y_score shape is incompatible with multiclass y_true for PR curve")

        # Per-class PR curves and AP
        ap_per_class = []
        for i in range(n_classes):
            prec, rec, _ = precision_recall_curve(y_true_bin[:, i], y_score_bin[:, i])
            ap_i = float(average_precision_score(y_true_bin[:, i], y_score_bin[:, i]))
            ap_per_class.append(ap_i)
            ax.plot(rec, prec, lw=1, alpha=0.6, label=f'Class {classes[i]} (AP={ap_i:.3f})')

        # Micro-average across classes
        prec_micro, rec_micro, _ = precision_recall_curve(y_true_bin.ravel(), y_score_bin.ravel())
        ap_micro = float(average_precision_score(y_true_bin, y_score_bin, average='micro'))
        ax.plot(rec_micro, prec_micro, color='black', lw=2, label=f'micro-average (AP={ap_micro:.3f})')

        ap = float(np.mean(ap_per_class))

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='lower left')
    ax.grid(linestyle='--', alpha=0.3)

    plt.tight_layout()

    if savepath is not None and created_fig is not None:
        try:
            created_fig.savefig(savepath, dpi=150, bbox_inches='tight')
        except Exception:
            plt.savefig(savepath, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return Graph.from_axes(ax)


def compute_brier_score(y_true, y_prob, pos_label=1):
    """Compute the Brier score for binary or multiclass probability predictions.

    For binary classification this wraps sklearn.metrics.brier_score_loss.
    For multiclass, the score is computed as the mean squared error between
    the one-hot encoded true labels and the predicted probability matrix.

    Args:
        y_true: true labels (array-like)
        y_prob: predicted probabilities. For binary either a 1-D array with
                probabilities for the positive class or a 2-D array of shape
                (n_samples, n_classes) for multiclass.
        pos_label: label considered positive in binary case (default=1)

    Returns:
        float: Brier score (lower is better)
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    classes = np.unique(y_true)
    n_classes = len(classes)

    # Binary case
    if n_classes == 2:
        # If y_prob is 2D, select the column corresponding to pos_label
        if y_prob.ndim == 2 and y_prob.shape[1] >= 2:
            try:
                idx = list(classes).index(pos_label)
            except ValueError:
                idx = 1
            prob = y_prob[:, idx]
        else:
            prob = y_prob
        return float(brier_score_loss(y_true, prob, pos_label=pos_label))

    # Multiclass: expect y_prob to be shape (n_samples, n_classes)
    if y_prob.ndim != 2 or y_prob.shape[1] != n_classes:
        raise ValueError("For multiclass Brier score, y_prob must be shape (n_samples, n_classes)")

    # One-hot encode true labels in the class order
    y_true_bin = label_binarize(y_true, classes=classes)
    if hasattr(y_true_bin, 'toarray'):
        y_true_bin = y_true_bin.toarray()
    y_true_bin = np.asarray(y_true_bin)

    # Mean squared error across all classes and samples
    mse = np.mean((y_true_bin - y_prob) ** 2)
    return float(mse)

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
    return Graph.from_figure(fig), axes