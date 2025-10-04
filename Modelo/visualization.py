"""Visualization helpers for model evaluation.

Provides functions to build evaluation plots (PR, ROC, calibration,
feature importances) and return either Matplotlib figures or base64-encoded
PNG strings. Designed to be imported and used by `ensemble.train_stack`.
"""
from typing import Optional, Sequence, Tuple, List, Any
import io
import base64
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def fig_to_base64(fig) -> Optional[str]:
    """Convert a Matplotlib figure to a base64-encoded PNG string.

    Returns None if Matplotlib isn't available.
    """
    if plt is None:
        return None
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def plot_pr_curve(precision: Optional[Sequence[float]], recall: Optional[Sequence[float]], ap: Optional[float] = None):
    """Create a Precision-Recall plot and return base64 PNG string.

    precision and recall are sequences (np arrays or lists) from
    sklearn.metrics.precision_recall_curve. ap is optional and shown in legend.
    """
    if plt is None or precision is None or recall is None:
        return None
    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=f"AP={ap:.3f}" if ap is not None else None)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall (confirmed)")
    if ap is not None:
        ax.legend()
    return fig_to_base64(fig)


def plot_roc_curve(fpr: Optional[Sequence[float]], tpr: Optional[Sequence[float]], auc: Optional[float] = None):
    """Create an ROC plot and return base64 PNG string."""
    if plt is None or fpr is None or tpr is None:
        return None
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC={auc:.3f}" if auc is not None else None)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (confirmed)")
    if auc is not None:
        ax.legend()
    return fig_to_base64(fig)


def plot_calibration_curve(prob_true: Optional[Sequence[float]], prob_pred: Optional[Sequence[float]]):
    """Create a calibration curve plot and return base64 PNG string."""
    if plt is None or prob_true is None or prob_pred is None:
        return None
    fig, ax = plt.subplots()
    ax.plot(prob_pred, prob_true, marker="o")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curve (confirmed)")
    return fig_to_base64(fig)


def plot_feature_importances(importances: Optional[List[Tuple[str, Any]]], title: str = "Feature importances (top)", top_n: int = 20):
    """Plot horizontal bar chart for feature importances and return base64 PNG string.

    `importances` is a list of (name, value) pairs sorted descending by value.
    """
    if plt is None or not importances:
        return None
    names, vals = zip(*importances[:top_n])
    fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.25)))
    ax.barh(range(len(names)), vals[::-1])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(list(names)[::-1])
    ax.set_title(title)
    return fig_to_base64(fig)
