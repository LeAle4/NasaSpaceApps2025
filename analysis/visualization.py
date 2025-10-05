"""Condensed plotting helpers for model evaluation.

This file provides a compact, opinionated set of plotting helpers that return
Graph wrappers (figure+axes). The functions are simpler than the original
version: they assume reasonably-formed inputs and avoid many fallbacks. This
keeps the code readable and easy to maintain.

Each plotting function returns a `Graph` instance. Use `Graph.save(path)` to
write output to disk or `visualization.save_graphs(graphs, out_dir)` to save a
mapping of graphs to a folder.
"""

from typing import Optional, Sequence, Mapping

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

# Try seaborn styles if available, otherwise fall back to matplotlib default
try:
    plt.style.use('seaborn-darkgrid')
except Exception:
    try:
        plt.style.use('seaborn')
    except Exception:
        # final fallback to matplotlib's default style
        plt.style.use('default')
_cycle = plt.rcParams.get('axes.prop_cycle')
if _cycle is not None and hasattr(_cycle, 'by_key'):
    DEFAULT_PALETTE = _cycle.by_key().get('color', ['C0', 'C1', 'C2'])
else:
    DEFAULT_PALETTE = ['C0', 'C1', 'C2']


class Graph:
    """Small wrapper around a matplotlib Figure and Axes.

    Keeps a consistent API: `Graph.save(path)`, `Graph.tight_layout()` and
    `Graph.from_axes(ax)` for convenience.
    """

    def __init__(self, fig=None, ax=None, figsize=(6, 4)):
        if ax is not None:
            self.ax = ax
            self.fig = ax.figure
        elif fig is not None:
            axes = fig.get_axes()
            self.fig = fig
            self.ax = axes[0] if axes else fig.add_subplot(1, 1, 1)
        else:
            self.fig, self.ax = plt.subplots(figsize=figsize)

    @classmethod
    def from_axes(cls, ax):
        return cls(ax=ax)

    @classmethod
    def from_figure(cls, fig):
        return cls(fig=fig)

    def tight_layout(self, **kwargs):
        self.fig.tight_layout(**kwargs)

    def save(self, path: str, dpi: int = 150):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        self.fig.savefig(path, dpi=dpi, bbox_inches='tight')


def plot_confusion_matrix(cm: np.ndarray, labels: Sequence[str], ax: Optional[plt.Axes] = None):
    """Heatmap-style confusion matrix with annotations."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=cm.sum()))
    ticks = np.arange(len(labels))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')

    # annotate with count and percentage
    total = cm.sum() if cm.size else 1
    thresh = cm.max() / 2. if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = int(cm[i, j])
            pct = count / total if total else 0.0
            txt = f"{count}\n{pct:.1%}"
            ax.text(j, i, txt, ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black', fontsize=9)

    ax.set_title('Confusion matrix (counts and % of total)')

    return Graph.from_axes(ax)


def plot_cv_scores(scores, ax: Optional[plt.Axes] = None):
    """Boxplot of cross-validation scores."""
    scores = np.asarray(list(scores))
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    # Boxplot with individual points jittered and mean annotation
    bp = ax.boxplot(scores, notch=False, patch_artist=True, widths=0.4)
    # Use matplotlib defaults for box styling; keep jittered points for detail
    # jittered points
    jitter = (np.random.rand(len(scores)) - 0.5) * 0.12
    ax.scatter(np.ones_like(scores) + jitter, scores, color='black', alpha=0.75, s=20)
    mean = float(np.mean(scores))
    std = float(np.std(scores))
    ax.plot([0.6, 1.4], [mean, mean], color='C3', linestyle='--', linewidth=1.5, label=f'mean={mean:.3f}')
    ax.set_xticks([])
    ax.set_title('Cross-validation scores')
    ax.legend()
    ax.set_ylabel('Score')
    ax.grid(axis='y', alpha=0.3)
    # annotate mean/std
    ax.text(1.02, mean, f"μ={mean:.3f}\nσ={std:.3f}", va='center', transform=ax.get_yaxis_transform(), fontsize=9)
    return Graph.from_axes(ax)


def plot_permutation_importance(importances, feature_names, top_n: int = 30, ax: Optional[plt.Axes] = None):
    """Plot permutation importances as a horizontal bar chart and return Graph.

    Args:
        importances: 1-D array-like of importance values (same order as feature_names)
        feature_names: sequence of feature names
        top_n: number of top features to show
        ax: optional matplotlib Axes

    Returns:
        Graph wrapping the created Axes
    """
    arr = np.asarray(importances)
    # Build a Series for easy sorting and selection
    s = pd.Series(arr, index=list(feature_names))
    s = s.sort_values(ascending=False)
    to_plot = s.head(top_n).sort_values()

    if ax is None:
        # height proportional to number of bars
        fig, ax = plt.subplots(figsize=(8, max(3, len(to_plot) * 0.28)))

    idx = np.asarray(list(to_plot.index))
    vals = np.asarray(to_plot.values)
    bars = ax.barh(idx, vals, color=DEFAULT_PALETTE[1], alpha=0.85)
    ax.set_xlabel('Permutation importance (higher = more important)')
    ax.set_title(f'Permutation importances (top {min(top_n, len(s))})')
    ax.grid(axis='x', alpha=0.2)
    # annotate numeric values at the end of each bar
    for bar in bars:
        w = bar.get_width()
        ax.text(w + max(1e-6, 0.01 * w), bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va='center', fontsize=8)
    return Graph.from_axes(ax)


def plot_truepositives_vs_others(y_true, y_pred, ax: Optional[plt.Axes] = None):
    """Group-wise bar chart comparing true positives vs misclassified counts."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    # ensure correct boolean precedence
    tp = [int(((y_true == lab) & (y_pred == lab)).sum()) for lab in labels]
    others = [int(((y_true == lab) & (y_pred != lab)).sum()) for lab in labels]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, tp, width, label='True Positives', color='tab:green')
    ax.bar(x + width / 2, others, width, label='Others', color='tab:orange')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.legend()
    ax.set_ylabel('Count')
    # annotate counts above bars
    for i in range(len(labels)):
        ax.text(x[i] - width / 2, tp[i] + max(1, int(0.01 * sum(tp + others))), f"{tp[i]}", ha='center', va='bottom', fontsize=8)
        ax.text(x[i] + width / 2, others[i] + max(1, int(0.01 * sum(tp + others))), f"{others[i]}", ha='center', va='bottom', fontsize=8)
    return Graph.from_axes(ax)


def plot_train_test_accuracy(train_acc: float, test_acc: float, ax: Optional[plt.Axes] = None):
    """Simple train vs test accuracy bar chart."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(['Train', 'Test'], [train_acc, test_acc], color=['tab:blue', 'tab:purple'])
    ax.set_ylim(0, 1)
    # numeric labels
    for i, v in enumerate([train_acc, test_acc]):
        ax.text(i, v + 0.02, f"{v:.3f}", ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Accuracy')
    return Graph.from_axes(ax)


def compute_cv_scores(estimator, X, y, cv: int = 10, scoring: str = 'accuracy'):
    """Compute cross-validation scores using sklearn.cross_val_score.

    Parameters
    ----------
    estimator : estimator object
        Fitted or unfitted estimator compatible with sklearn API.
    X, y : array-like
        Data used for cross-validation.
    cv : int
        Number of folds.
    scoring : str
        Scoring metric to pass to cross_val_score.

    Returns
    -------
    numpy.ndarray
        1-D array of cross-validation scores.
    """
    return cross_val_score(estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1)


def plot_roc_auc(y_true, y_score, ax: Optional[plt.Axes] = None, pos_label=1):
    """Plot ROC curve. For multiclass, expects y_score shape (n, n_classes)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    classes = np.unique(y_true)
    n_classes = len(classes)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    if n_classes == 2:
        if y_score.ndim == 2:
            idx = list(classes).index(pos_label) if pos_label in classes else 1
            score = y_score[:, idx]
        else:
            score = y_score
        fpr, tpr, _ = roc_curve(y_true, score, pos_label=pos_label)
        auc_val = roc_auc_score(y_true, score)
        ax.plot(fpr, tpr, lw=2, color=DEFAULT_PALETTE[0], label=f'ROC (AUC={auc_val:.3f})')
        ax.set_title(f'ROC curve (AUC={auc_val:.3f})')
    else:
        y_true_bin = np.asarray(label_binarize(y_true, classes=classes))
        if y_score.ndim == 1:
            y_score_bin = np.asarray(label_binarize(y_score, classes=classes))
        else:
            y_score_bin = np.asarray(y_score)
        aucs = []
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score_bin[:, i])
            auc_i = auc(fpr, tpr)
            aucs.append(auc_i)
            ax.plot(fpr, tpr, lw=1.5, alpha=0.8, label=f'Class {classes[i]} (AUC={auc_i:.3f})')
        mean_auc = float(np.mean(aucs)) if aucs else None
        if mean_auc is not None:
            ax.set_title(f'Multiclass ROC (mean AUC={mean_auc:.3f})')
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    return Graph.from_axes(ax)


def plot_pr_auc(y_true, y_score, ax: Optional[plt.Axes] = None, pos_label=1):
    """Plot precision-recall curve. For multiclass expects y_score to be (n, n_classes)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    classes = np.unique(y_true)
    n_classes = len(classes)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    if n_classes == 2:
        if y_score.ndim == 2:
            idx = list(classes).index(pos_label) if pos_label in classes else 1
            score = y_score[:, idx]
        else:
            score = y_score
        precision, recall, _ = precision_recall_curve(y_true, score, pos_label=pos_label)
        ap = average_precision_score(y_true, score)
        ax.plot(recall, precision, lw=2, color=DEFAULT_PALETTE[1], label=f'PR (AP={ap:.3f})')
        ax.set_title(f'Precision-Recall (AP={ap:.3f})')
    else:
        y_true_bin = np.asarray(label_binarize(y_true, classes=classes))
        if y_score.ndim == 1:
            y_score_bin = np.asarray(label_binarize(y_score, classes=classes))
        else:
            y_score_bin = np.asarray(y_score)
        aps = []
        for i in range(n_classes):
            prec, rec, _ = precision_recall_curve(y_true_bin[:, i], y_score_bin[:, i])
            ap_i = average_precision_score(y_true_bin[:, i], y_score_bin[:, i])
            aps.append(ap_i)
            ax.plot(rec, prec, lw=1.2, alpha=0.8, label=f'Class {classes[i]} (AP={ap_i:.3f})')
        if aps:
            ax.set_title(f'Multiclass PR (mean AP={float(np.mean(aps)):.3f})')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower left')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    return Graph.from_axes(ax)


def compute_brier_score(y_true, y_prob, pos_label=1):
    """Simple Brier score: binary or multiclass (MSE vs one-hot).

    This is a compact replacement for the more defensive original.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    classes = np.unique(y_true)
    if len(classes) == 2:
        if y_prob.ndim == 2:
            idx = list(classes).index(pos_label) if pos_label in classes else 1
            prob = y_prob[:, idx]
        else:
            prob = y_prob
        from sklearn.metrics import brier_score_loss
        return float(brier_score_loss(y_true, prob, pos_label=pos_label))
    # multiclass mean squared error against one-hot
    y_true_bin = label_binarize(y_true, classes=classes)
    return float(np.mean((y_true_bin - y_prob) ** 2))


def plotkfold_results(results: Mapping[str, np.ndarray]):
    """2x2 plot of per-fold metrics returned by ten_fold_cross_validation.

    Parameters
    ----------
    results : Mapping[str, numpy.ndarray]
        Mapping expected to contain arrays for keys 'accuracy', 'precision',
        'recall', and 'f1'. Each array should have length equal to number of folds.

    Returns
    -------
    (Graph, numpy.ndarray)
        A Graph wrapping the figure and the flattened axes array.
    """
    metrics = ["accuracy", "precision", "recall", "f1"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    for i, metric in enumerate(metrics):
        ax = axes[i]
        vals = np.asarray(results.get(metric, []))
        if vals.size == 0:
            ax.axis('off')
            continue
        # Plot per-fold points and a mean ± std band.
        x = np.arange(1, len(vals) + 1)
        ax.scatter(x, vals, color=DEFAULT_PALETTE[0], edgecolor='k', zorder=3, s=40)
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        ax.fill_between([0.5, len(vals) + 0.5], mean - std, mean + std, color=DEFAULT_PALETTE[0], alpha=0.12)
        ax.hlines(mean, xmin=0.5, xmax=len(vals) + 0.5, colors=DEFAULT_PALETTE[0], linestyles='--', label=f'mean={mean:.3f}')
        ax.set_xlim(0.5, len(vals) + 0.5)
        ax.set_xticks(x)
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"{metric.capitalize()} per fold (mean={mean:.3f}, std={std:.3f})")
        ax.grid(axis='y', alpha=0.25)
        # Slight padding so markers don't touch the axis bounds
        y_min, y_max = ax.get_ylim()
        pad = max(0.02, 0.05 * (y_max - y_min))
        ax.set_ylim(y_min, 1)
    return Graph.from_figure(fig), axes


def save_graphs(graphs: Mapping[str, Graph], out_dir: str):
    """Save all Graph objects from the `graphs` mapping into `out_dir`.

    Each key is used as the filename (appending .png). Returns list of saved paths.
    """
    if not graphs:
        return []
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for name, g in graphs.items():
        if not isinstance(g, Graph):
            continue
        path = os.path.join(out_dir, f"{name}.png")
        g.tight_layout()
        g.save(path)
        saved.append(path)
    return saved
 