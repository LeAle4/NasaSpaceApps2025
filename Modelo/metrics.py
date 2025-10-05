"""Model evaluation helpers: compute a broad set of metrics for classifiers.

This module provides a single high-level helper `evaluate_model` which
computes a wide range of model diagnostics useful for reporting and
debugging. The function is intentionally conservative: expensive analyses
(permutation importance, learning curves) are optional and disabled by
default.

Returned structure
------------------
The function returns a dict with scalar metrics (floats) and optional keys
for more complex objects:

- "confusion_matrix": 2D array
- "classification_report": dict
- "balanced_accuracy", "cohen_kappa", "mcc", "accuracy": floats
- "log_loss": float or None (requires predict_proba)
- "brier_score": float or None
- "top_{k}_accuracy": float or None (requires predict_proba)
- "oob_score": float or None (if estimator exposes oob_score_)
- "permutation_importance": dict with mean/std and the raw result object
- "learning_curve": dict with train_sizes, train_scores, test_scores

Usage example
-------------
from Modelo.metrics import evaluate_model

# estimator should be a fitted classifier (e.g., RandomForestClassifier)
metrics = evaluate_model(estimator, X_train, X_test, y_train, y_test,
                         compute_permutation=False, compute_learning_curve=False)

print(metrics['accuracy'])
print(metrics['classification_report'])

"""
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    log_loss,
    cohen_kappa_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    classification_report,
    top_k_accuracy_score,
    accuracy_score,
)
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance

# Brier score util: use existing visualization.compute_brier_score if available
try:
    from Modelo.visualization import compute_brier_score
except Exception:
    from sklearn.metrics import brier_score_loss


def evaluate_model(
    estimator,
    X_train,
    X_test,
    y_train,
    y_test,
    *,
    top_k: int = 3,
    compute_permutation: bool = True,
    n_repeats: int = 10,
    random_state: int = 42,
    compute_learning_curve: bool = False,
    cv_for_learning: int = 5,
) -> Dict[str, Any]:
    """Compute a suite of evaluation metrics for a fitted classifier.

    Returns a dictionary with scalar metrics and optional objects under keys
    'permutation_importance' and 'learning_curve' when requested.
    """
    results: Dict[str, Any] = {}

    # Predictions
    y_pred = estimator.predict(X_test)

    # Try to get probabilities or decision function for continuous scores
    y_proba = None
    y_score = None
    if hasattr(estimator, "predict_proba"):
        try:
            y_proba = estimator.predict_proba(X_test)
            y_score = y_proba
        except Exception:
            y_proba = None
    if y_proba is None and hasattr(estimator, "decision_function"):
        try:
            y_score = estimator.decision_function(X_test)
        except Exception:
            y_score = None

    # Basic counts and accuracy
    results["accuracy"] = float(accuracy_score(y_test, y_pred))
    results["confusion_matrix"] = confusion_matrix(y_test, y_pred)

    # Classification report (string and parsed dict)
    try:
        rep = classification_report(y_test, y_pred, output_dict=True)
        results["classification_report"] = rep
    except Exception:
        results["classification_report"] = classification_report(y_test, y_pred)

    # Balanced accuracy, Kappa, MCC
    try:
        results["balanced_accuracy"] = float(balanced_accuracy_score(y_test, y_pred))
    except Exception:
        results["balanced_accuracy"] = None
    try:
        results["cohen_kappa"] = float(cohen_kappa_score(y_test, y_pred))
    except Exception:
        results["cohen_kappa"] = None
    try:
        results["mcc"] = float(matthews_corrcoef(y_test, y_pred))
    except Exception:
        results["mcc"] = None

    # Log loss (requires probabilities)
    if y_proba is not None:
        try:
            results["log_loss"] = float(log_loss(y_test, y_proba))
        except Exception:
            results["log_loss"] = None
    else:
        results["log_loss"] = None

    # Brier score
    try:
        if y_proba is not None:
            # Use provided prob matrix when possible
            results["brier_score"] = float(compute_brier_score(y_test, y_proba))
        else:
            # Fallback: try compute_brier_score with discrete predictions
            results["brier_score"] = float(compute_brier_score(y_test, y_pred))
    except Exception:
        results["brier_score"] = None

    # Top-k accuracy (requires probabilities)
    if y_proba is not None:
        try:
            results[f"top_{top_k}_accuracy"] = float(top_k_accuracy_score(y_test, y_proba, k=top_k))
        except Exception:
            results[f"top_{top_k}_accuracy"] = None
    else:
        results[f"top_{top_k}_accuracy"] = None

    # OOB score if available (RandomForest)
    try:
        oob = getattr(estimator, "oob_score_", None)
        results["oob_score"] = float(oob) if oob is not None else None
    except Exception:
        results["oob_score"] = None

    # Permutation importance (optional)
    if compute_permutation:
        try:
            pi = permutation_importance(estimator, X_test, y_test, n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
            # Provide a small summary and the full object. Access attributes defensively
            importances_mean = getattr(pi, 'importances_mean', None)
            importances_std = getattr(pi, 'importances_std', None)
            results["permutation_importance"] = {
                "importances_mean": importances_mean,
                "importances_std": importances_std,
                "result_obj": pi,
            }
        except Exception:
            results["permutation_importance"] = None

    # Learning curve (optional) - compute train/test scores vs training set sizes
    if compute_learning_curve:
        try:
            from sklearn.model_selection import learning_curve

            lc = learning_curve(
                estimator,
                np.vstack([X_train, X_test]) if hasattr(X_train, "shape") else np.concatenate([X_train, X_test]),
                np.concatenate([y_train, y_test]),
                cv=cv_for_learning,
                n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 5),
            )
            # learning_curve returns (train_sizes, train_scores, test_scores)
            train_sizes, train_scores, test_scores = lc[0], lc[1], lc[2]
            results["learning_curve"] = {
                "train_sizes": train_sizes,
                "train_scores": train_scores,
                "test_scores": test_scores,
            }
        except Exception:
            results["learning_curve"] = None

    return results
