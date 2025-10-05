"""Machine-learning helpers for batch prediction.

This module centralizes model loading, feature preparation and prediction
so the app can call a single function from `backend.backend` while keeping
the implementation easy to test and modify.
"""
from typing import Optional, Dict, Any, Tuple
import os
import pandas as pd

from .analysis import model as model_wrapper


def _find_model_path(candidates: Optional[Tuple[str, ...]] = None) -> Optional[str]:
    candidates = candidates or ("noncand37.joblib", "noncand25.joblib", "modeltest.joblib")
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def predict_batch(
    batch_df: pd.DataFrame,
    data_headers: list,
    model_path: Optional[str] = None,
    fillna_value: Any = 0,
    positive_class: Any = 1,
) -> Dict[str, Any]:
    """Run prediction on a batch DataFrame.

    Args:
        batch_df: original DataFrame containing raw rows (must include data_headers)
        data_headers: list of column names used as features
        model_path: optional explicit path to a .joblib model file. If None the
            function will try common candidate names in the current working dir.
        fillna_value: value used to fill NaNs after coercion to numeric
        positive_class: the label considered as "confirmed" (default 1)

    Returns a dict with keys:
        model_path, predictions, probabilities, results_df, confirmed_df, rejected_df

    Raises exceptions from underlying libraries to let callers decide how to
    handle them (backend.backend will catch and convert to messages).
    """
    if batch_df is None:
        raise ValueError("batch_df is None")

    # extract features
    features = batch_df[data_headers]

    # coerce numeric and fill missing
    X = features.copy()
    X = X.apply(pd.to_numeric, errors='coerce')
    if X.isnull().any().any():
        X = X.fillna(fillna_value)

    # resolve model path
    resolved_model = model_path if model_path else _find_model_path()
    if resolved_model is None:
        # allow model_wrapper.load to raise with clearer message if needed
        resolved_model = model_path or "noncand37.joblib"

    # load model wrapper (may raise)
    clf_wrapper = model_wrapper.Model.load(resolved_model)
    estimator = getattr(clf_wrapper, 'model', clf_wrapper)

    # run predictions
    preds = estimator.predict(X)
    probs = None
    if hasattr(estimator, 'predict_proba'):
        try:
            probs = estimator.predict_proba(X)
        except Exception:
            probs = None

    # construct results DataFrame
    results_df = batch_df.copy()
    results_df['prediction'] = preds
    if probs is not None:
        try:
            if probs.shape[1] >= 2:
                results_df['prediction_proba_pos'] = probs[:, 1]
            else:
                results_df['prediction_proba'] = probs[:, 0]
        except Exception:
            pass

    # split confirmed/rejected
    try:
        mask_confirmed = results_df['prediction'] == positive_class
    except Exception:
        mask_confirmed = results_df['prediction'].astype(bool)

    confirmed_df = results_df[mask_confirmed].reset_index(drop=True)
    rejected_df = results_df[~mask_confirmed].reset_index(drop=True)

    return {
        'model_path': resolved_model,
        'predictions': preds,
        'probabilities': probs,
        'results_df': results_df,
        'confirmed_df': confirmed_df,
        'rejected_df': rejected_df,
    }
