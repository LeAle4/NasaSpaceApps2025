"""Condensed metrics helpers for classifier evaluation.

This module provides two small utilities:
- evaluate_model: compute standard metrics and (optionally) permutation
  importances and a learning curve.
- ten_fold_cross_validation: run stratified k-fold CV and return per-fold
  accuracy/precision/recall/f1 arrays.

The implementation below is intentionally compact and makes reasonable
assumptions about inputs (fitted estimators, aligned arrays). It focuses on
clarity rather than exhaustive defensive coding.
"""

from typing import Dict, Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    log_loss,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
    top_k_accuracy_score,
)

from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve, StratifiedKFold


def _compute_brier(y_true, y_proba):
    """Brier score: binary uses sklearn's brier_score_loss; multiclass uses MSE vs one-hot."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    classes = np.unique(y_true)
    if len(classes) == 2:
        # If proba matrix given, take positive class column
        if y_proba.ndim == 2:
            pos_col = 1 if y_proba.shape[1] > 1 else 0
            return float(brier_score_loss(y_true, y_proba[:, pos_col]))
        return float(brier_score_loss(y_true, y_proba))
    # multiclass: mean squared error against one-hot
    from sklearn.preprocessing import label_binarize

    y_true_bin = label_binarize(y_true, classes=classes)
    return float(np.mean((y_true_bin - y_proba) ** 2))


def evaluate_model(
    estimator,
    X_train,
    X_test,
    y_train,
    y_test,
    *,
    top_k: int = 3,
    compute_permutation: bool = False,
    n_repeats: int = 10,
    random_state: int = 42,
    compute_learning_curve: bool = False,
    cv_for_learning: int = 5,
) -> Dict[str, Any]:
    """Compute common evaluation metrics for a fitted classifier.

    Returns a dict with scalar metrics and optional objects under keys
    'permutation_importance' and 'learning_curve' when requested.
    """
    results: Dict[str, Any] = {}

    # Basic predictions
    y_pred = estimator.predict(X_test)
    results["accuracy"] = float(accuracy_score(y_test, y_pred))
    # Convert confusion matrix to plain lists so the result is JSON-serializable
    results["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
    results["classification_report"] = classification_report(y_test, y_pred, output_dict=True)

    # Additional scalar metrics
    results["balanced_accuracy"] = float(balanced_accuracy_score(y_test, y_pred))
    results["cohen_kappa"] = float(cohen_kappa_score(y_test, y_pred))
    results["mcc"] = float(matthews_corrcoef(y_test, y_pred))

    # Probabilistic metrics if available
    y_proba = estimator.predict_proba(X_test) if hasattr(estimator, "predict_proba") else None
    if y_proba is not None:
        # Probabilistic metrics: try to compute but skip keys on failure
        try:
            results["log_loss"] = float(log_loss(y_test, y_proba))
        except Exception:
            # omit key if not computable
            results.pop("log_loss", None)
        try:
            results["brier_score"] = _compute_brier(y_test, y_proba)
        except Exception:
            results.pop("brier_score", None)
        # top-k accuracy for multiclass: only add when computable
        try:
            results[f"top_{top_k}_accuracy"] = float(top_k_accuracy_score(y_test, y_proba, k=top_k))
        except Exception:
            # do not include top_k key if not available
            results.pop(f"top_{top_k}_accuracy", None)
    else:
        results.update({"log_loss": None, "brier_score": None, f"top_{top_k}_accuracy": None})

    # Optional permutation importance (may be slow)
    if compute_permutation:
        # compute permutation importances and keep only JSON-friendly arrays
        pi = permutation_importance(estimator, X_test, y_test, n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
        imp_mean = getattr(pi, 'importances_mean', None)
        imp_std = getattr(pi, 'importances_std', None)
        results["permutation_importance"] = {
            "importances_mean": imp_mean.tolist() if imp_mean is not None else None,
            "importances_std": imp_std.tolist() if imp_std is not None else None,
        }

    # Optional learning curve
    if compute_learning_curve:
        lc = learning_curve(
            estimator,
            np.vstack([X_train, X_test]),
            np.concatenate([y_train, y_test]),
            cv=cv_for_learning,
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 5),
        )
        # Convert arrays to lists for JSON compatibility
        results["learning_curve"] = {
            "train_sizes": lc[0].tolist(),
            "train_scores": [s.tolist() for s in lc[1]],
            "test_scores": [s.tolist() for s in lc[2]],
        }

    # OOB score if present (RandomForest)
    # Only include oob_score when present to avoid nulls in JSON
    oob_val = getattr(estimator, "oob_score_", None)
    if oob_val is not None:
        results["oob_score"] = float(oob_val)

    return results


def ten_fold_cross_validation(estimator, X, y, n_splits: int = 10, random_state: int = 42):
    """Stratified k-fold cross-validation returning per-fold metrics.

    Returns a dict with numpy arrays for 'accuracy','precision','recall','f1'.
    """
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    accs, precisions, recalls, f1s = [], [], [], []
    for train_idx, test_idx in skf.split(X_arr, y_arr):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]
        # fit a fresh clone of the estimator when possible
        try:
            from sklearn.base import clone
            est = clone(estimator)
        except Exception:
            est = estimator
        est.fit(X_train, y_train)
        y_pred = est.predict(X_test)
        accs.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, average='macro', zero_division=0))
        recalls.append(recall_score(y_test, y_pred, average='macro', zero_division=0))
        f1s.append(f1_score(y_test, y_pred, average='macro', zero_division=0))

    return {
        'accuracy': np.array(accs),
        'precision': np.array(precisions),
        'recall': np.array(recalls),
        'f1': np.array(f1s),
    }