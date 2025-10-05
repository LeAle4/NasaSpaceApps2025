"""Machine-learning helpers for batch prediction.

This module centralizes model loading, feature preparation and prediction
so the app can call a single function from `backend.backend` while keeping
the implementation easy to test and modify.
"""
from typing import Optional, Dict, Any, Tuple
import os
import pandas as pd

from .analysis import model as model_wrapper
from .analysis import visualization as visualization
import os


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

    # verify that all requested feature columns exist and preserve order
    missing = [c for c in data_headers if c not in batch_df.columns]
    if missing:
        raise KeyError(f"Missing feature columns in batch_df: {missing}")

    # extract features in the exact order provided by data_headers
    features = batch_df.loc[:, data_headers]

    # coerce numeric and fill missing (preserve column order)
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

    # Ensure feature order matches what the estimator expects (some sklearn
    # estimators require the same column order used during fit and expose
    # `feature_names_in_`). If present, reorder X to match. If not present,
    # we continue with the order derived from `data_headers`.
    try:
        expected_features = getattr(estimator, 'feature_names_in_', None)
        if expected_features is not None:
            # feature_names_in_ may be numpy array - ensure same type and values
            expected = list(expected_features)
            # check for missing features
            missing_from_expected = [f for f in expected if f not in X.columns]
            if missing_from_expected:
                raise KeyError(f"Model expects features not present in batch_df: {missing_from_expected}")
            # reorder X to match expected order
            X = X.loc[:, expected]
    except Exception:
        # Any issue inspecting/reordering should not crash here; raise a
        # clearer KeyError up the stack so callers can show a friendly message.
        raise

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


def train_from_database(
    confirmed_csv: str,
    rejected_csv: str,
    data_headers: list,
    out_dir: Optional[str] = None,
    params: Optional[dict] = None,
    random_state: Optional[int] = None,
    test_size: float = 0.4,
):
    """Train a model using two CSV files (confirmed/rejected) and produce visualizations.

    Returns a dict with keys: status, message, saved_paths (list), metrics (dict)
    Raises exceptions on unexpected IO/ML errors so callers (backend) can present them.
    """
    if not os.path.exists(confirmed_csv):
        raise FileNotFoundError(f"Confirmed CSV not found: {confirmed_csv}")
    if not os.path.exists(rejected_csv):
        raise FileNotFoundError(f"Rejected CSV not found: {rejected_csv}")

    # Load CSVs
    df_conf = pd.read_csv(confirmed_csv)
    df_rej = pd.read_csv(rejected_csv)

    # Label and combine: use 1 for confirmed, -1 for false positives (rejected)
    df_conf = df_conf.copy()
    df_rej = df_rej.copy()
    df_conf['__label__'] = 1
    df_rej['__label__'] = -1
    df = pd.concat([df_conf, df_rej], ignore_index=True)

    # Verify features present
    missing = [c for c in data_headers if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns in combined database: {missing}")

    X = df.loc[:, data_headers].copy()
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    y = df['__label__']

    # Delegate training to analysis.model.train_save_model
    results = model_wrapper.train_save_model(X, y, params=params, random_state=random_state, test_size=test_size)

    # Create visualizations (Graph objects)
    graphs = model_wrapper.create_visualizations(results, random_state=random_state)

    # Save graphs to out_dir (default App/viz_out)
    if out_dir is None:
        out_dir = os.path.join(os.getcwd(), 'App', 'viz_out')
    saved_paths = visualization.save_graphs(graphs, out_dir)

    metrics = {
        'train_acc': float(results.get('train_acc')) if results.get('train_acc') is not None else None,
        'test_acc': float(results.get('test_acc')) if results.get('test_acc') is not None else None,
    }

    return {
        'status': 'success',
        'message': f"Training complete. Saved {len(saved_paths)} visualization(s)",
        'saved_paths': saved_paths,
        'metrics': metrics,
        # include the trained wrapper so callers (UI/backend) can offer to persist it
        'model_wrapper': results.get('wrapper'),
    }
