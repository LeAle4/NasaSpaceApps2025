"""Compute and save feature importances for the project's trained model.

This script provides utilities to compute:
 - Gini importances (if model exposes feature_importances_)
 - Permutation importances (sklearn)
 - SHAP values (optional; only used if `shap` is installed)

Outputs are saved under `Modelo/viz_out/` and a CSV with ranked features is
written as `feature_importances.csv`.
"""
from typing import Optional, Sequence
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance


DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'GOAT.joblib')
DEFAULT_FEATURES_PATH = os.path.join(os.path.dirname(__file__), 'data', 'non_candidates_processed_features.csv')
DEFAULT_LABELS_PATH = os.path.join(os.path.dirname(__file__), 'data', 'non_candidates_processed_labels.csv')
VIZ_OUT = os.path.join(os.path.dirname(__file__), 'viz_out')


def ensure_outdir():
    os.makedirs(VIZ_OUT, exist_ok=True)


def load_model(path: str):
    return joblib.load(path)


def load_data(features_path: str, labels_path: str):
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path)
    if hasattr(y, 'shape') and y.shape[1] == 1:
        y = y.iloc[:, 0]
    return X, y


def gini_importances(clf, feature_names: Sequence[str]):
    if not hasattr(clf, 'feature_importances_'):
        return None
    fi = np.asarray(clf.feature_importances_)
    df = pd.DataFrame({'feature': list(feature_names), 'gini_importance': fi})
    df = df.sort_values('gini_importance', ascending=False).reset_index(drop=True)
    return df


def permutation_importances(clf, X, y, feature_names: Sequence[str], n_repeats: int = 30):
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    res = permutation_importance(clf, X_arr, y_arr, n_repeats=n_repeats, random_state=0, n_jobs=-1)
    df = pd.DataFrame({'feature': list(feature_names), 'perm_mean': res.importances_mean, 'perm_std': res.importances_std})
    df = df.sort_values('perm_mean', ascending=False).reset_index(drop=True)
    return res, df


def try_shap(clf, X, feature_names: Sequence[str]):
    try:
        import shap
    except Exception:
        return None

    # Use TreeExplainer for tree-based models for speed, otherwise KernelExplainer
    try:
        explainer = shap.TreeExplainer(clf)
    except Exception:
        try:
            explainer = shap.Explainer(clf, X)
        except Exception:
            return None

    # Compute SHAP values for a subset if dataset is large
    sample = X if len(X) <= 1000 else X.sample(1000, random_state=0)
    shap_values = explainer(sample)
    # shap_values may be an Explanation object; convert to mean absolute per feature
    try:
        arr = np.abs(shap_values.values)
    except Exception:
        arr = np.abs(np.asarray(shap_values))

    mean_abs = np.mean(arr, axis=0)
    df = pd.DataFrame({'feature': list(feature_names), 'shap_mean_abs': mean_abs})
    df = df.sort_values('shap_mean_abs', ascending=False).reset_index(drop=True)
    return shap_values, df


def plot_bar_from_df(df, value_col: str, title: str, fname: str, top_n: int = 20, color: str = 'tab:blue'):
    ensure_outdir()
    n = min(len(df), top_n)
    sub = df.iloc[:n]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * n)))
    ax.barh(range(n)[::-1], sub[value_col].values[::-1], color=color)
    ax.set_yticks(range(n))
    ax.set_yticklabels(sub['feature'].values[::-1])
    ax.set_xlabel(value_col)
    ax.set_title(title)
    plt.tight_layout()
    outpath = os.path.join(VIZ_OUT, fname)
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    return outpath


def main(model_path: Optional[str] = None, features_path: Optional[str] = None, labels_path: Optional[str] = None):
    model_path = model_path or DEFAULT_MODEL_PATH
    features_path = features_path or DEFAULT_FEATURES_PATH
    labels_path = labels_path or DEFAULT_LABELS_PATH

    ensure_outdir()

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    clf = load_model(model_path)
    X, y = load_data(features_path, labels_path)
    feature_names = list(X.columns)

    results = {}

    # Gini
    gini_df = gini_importances(clf, feature_names)
    if gini_df is not None:
        gini_csv = os.path.join(VIZ_OUT, 'gini_importances.csv')
        gini_df.to_csv(gini_csv, index=False)
        out = plot_bar_from_df(gini_df, 'gini_importance', 'Gini Feature Importances', 'gini_importances.png')
        results['gini_csv'] = gini_csv
        results['gini_png'] = out

    # Permutation
    try:
        perm_res, perm_df = permutation_importances(clf, X, y, feature_names)
        perm_csv = os.path.join(VIZ_OUT, 'permutation_importances.csv')
        perm_df.to_csv(perm_csv, index=False)
        out = plot_bar_from_df(perm_df, 'perm_mean', 'Permutation Importances (mean)', 'permutation_importances.png', color='tab:green')
        results['perm_csv'] = perm_csv
        results['perm_png'] = out
    except Exception as exc:
        results['perm_error'] = str(exc)

    # SHAP (optional)
    shap_result = try_shap(clf, X, feature_names)
    if shap_result is not None:
        shap_vals, shap_df = shap_result
        shap_csv = os.path.join(VIZ_OUT, 'shap_importances.csv')
        shap_df.to_csv(shap_csv, index=False)
        try:
            # summary_plot requires matplotlib figure; save as png
            import shap
            fig = plt.figure(figsize=(8, 6))
            shap.summary_plot(shap_vals, features=X if len(X) <= 1000 else X.sample(1000, random_state=0), show=False)
            outpath = os.path.join(VIZ_OUT, 'shap_summary.png')
            fig.savefig(outpath, dpi=150, bbox_inches='tight')
            results['shap_png'] = outpath
        except Exception as exc:
            results['shap_plot_error'] = str(exc)
        results['shap_csv'] = shap_csv
    else:
        results['shap_available'] = False

    # Combined CSV: merge available importances
    comb = pd.DataFrame({'feature': feature_names})
    if gini_df is not None:
        comb = comb.merge(gini_df[['feature', 'gini_importance']], on='feature', how='left')
    if 'perm_df' in locals():
        comb = comb.merge(perm_df[['feature', 'perm_mean', 'perm_std']], on='feature', how='left')
    if 'shap_df' in locals():
        comb = comb.merge(shap_df[['feature', 'shap_mean_abs']], on='feature', how='left')

    comb = comb.sort_values(by=[col for col in ['perm_mean', 'shap_mean_abs', 'gini_importance'] if col in comb.columns and col is not None], ascending=False, na_position='last')
    comb_csv = os.path.join(VIZ_OUT, 'feature_importances_combined.csv')
    comb.to_csv(comb_csv, index=False)
    results['combined_csv'] = comb_csv

    print('Feature importance artifacts written to', VIZ_OUT)
    print(results)
    return results


if __name__ == '__main__':
    main()
