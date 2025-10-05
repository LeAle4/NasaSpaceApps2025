"""Tools to visualize a trained RandomForest model and feature relationships.

Functions:
- load_model(path): loads joblib model
- plot_gini_feature_importance(clf, feature_names, top_n=20, savepath=None)
- plot_permutation_importance(clf, X, y, top_n=20, n_repeats=10, savepath=None)
- plot_feature_correlations(X, y, feature_names, top_n=10, savepath=None)

These helpers save PNGs into `viz_out/` when savepath is provided and return
matplotlib Axes/objects for further customization.
"""
from typing import Optional, Sequence
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.inspection import permutation_importance

# Use existing visualization helpers where sensible
try:
    from Modelo import visualization
except Exception:
    visualization = None


def load_model(path: str):
    """Load a joblib model from disk."""
    return joblib.load(path)


def plot_gini_feature_importance(clf, feature_names: Sequence[str], top_n: int = 20, savepath: Optional[str] = None):
    """Plot Gini feature importances from a RandomForest-like estimator.

    Args:
        clf: fitted estimator with attribute `feature_importances_`.
        feature_names: list of feature names matching training columns.
        top_n: how many top features to show.
        savepath: optional path to save PNG.
    Returns:
        ax: matplotlib Axes with the barplot.
    """
    if not hasattr(clf, 'feature_importances_'):
        raise ValueError('Estimator has no attribute feature_importances_')

    fi = np.asarray(clf.feature_importances_)
    idx = np.argsort(fi)[::-1][:top_n]
    names = [feature_names[i] for i in idx]
    values = fi[idx]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(names))))
    ax.barh(range(len(names))[::-1], values[::-1], color='tab:blue')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1])
    ax.set_xlabel('Gini importance')
    ax.set_title('Top feature importances (Gini)')
    plt.tight_layout()

    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
    return ax


def plot_permutation_importance(clf, X, y, feature_names: Sequence[str], top_n: int = 20, n_repeats: int = 10, savepath: Optional[str] = None):
    """Compute and plot permutation importances on X,y.

    Returns the permutation importance result object and the Axes.
    """
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    pi = permutation_importance(clf, X_arr, y_arr, n_repeats=n_repeats, random_state=0, n_jobs=-1)

    means = pi.importances_mean
    idx = np.argsort(means)[::-1][:top_n]
    names = [feature_names[i] for i in idx]
    values = means[idx]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(names))))
    ax.barh(range(len(names))[::-1], values[::-1], color='tab:green')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1])
    ax.set_xlabel('Permutation importance (mean decrease in score)')
    ax.set_title('Top features by permutation importance')
    plt.tight_layout()

    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
    return pi, ax


def plot_feature_correlations(X, y, feature_names: Sequence[str], top_n: int = 10, savepath: Optional[str] = None):
    """Plot simple scatter/density plots of top_n features vs target.

    For classification, this will show violin or box plots per class for each
    top feature. X may be a DataFrame or 2D array; y is the target labels.
    """
    # Convert to pandas DataFrame for convenience
    if isinstance(X, pd.DataFrame):
        dfX = X.copy()
    else:
        dfX = pd.DataFrame(np.asarray(X), columns=list(feature_names))

    y_ser = pd.Series(y, name='target')

    # Compute correlation (Pearson) between each numeric feature and numeric target
    # For categorical target, compute ANOVA-like effect via group means difference magnitude
    numeric = dfX.select_dtypes(include=[np.number])
    corrs = {}
    try:
        # attempt numeric correlation; if target is categorical, coerce to codes
        y_num = pd.to_numeric(y_ser, errors='coerce')
        for col in numeric.columns:
            corrs[col] = abs(numeric[col].corr(y_num))
    except Exception:
        # fallback: use group mean differences
        for col in numeric.columns:
            try:
                grp = dfX[col].groupby(y_ser).mean()
                corrs[col] = float((grp.max() - grp.min()))
            except Exception:
                corrs[col] = 0.0

    # Get top_n features by correlation measure
    sorted_feats = sorted(corrs.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_features = [f for f, _ in sorted_feats]

    # Create subplots
    n = len(top_features)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)

    for i, feat in enumerate(top_features):
        ax = axes[i]
        try:
            if y_ser.nunique() <= 10:
                # Categorical target: boxplot per class
                data = [dfX.loc[y_ser == lab, feat].dropna() for lab in sorted(y_ser.unique())]
                ax.boxplot(data, labels=[str(l) for l in sorted(y_ser.unique())])
                ax.set_title(f"{feat} by class")
                ax.set_ylabel(feat)
            else:
                # Numeric target: scatter
                ax.scatter(dfX[feat], y_ser, alpha=0.6)
                ax.set_xlabel(feat)
                ax.set_ylabel('target')
                ax.set_title(f"{feat} vs target")
        except Exception as exc:
            ax.text(0.5, 0.5, f"Error plotting {feat}: {exc}", ha='center')

    # Turn off unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
    return fig, top_features


if __name__ == '__main__':
    # Quick demo: load GOAT.joblib and show Gini importances if feature names available
    model_path = os.path.join(os.path.dirname(__file__), 'GOAT.joblib')
    if os.path.exists(model_path):
        clf = load_model(model_path)
        print('Loaded model; feature_importances_ present:', hasattr(clf, 'feature_importances_'))
    else:
        print('No GOAT.joblib found in module directory.')
