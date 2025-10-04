"""Ensemble models for exoplanet identification.

This module provides a ready-to-adapt ensemble pipeline tailored to the
Kepler/KOI dataset shape used in this project. It focuses on recall for the
"confirmed" exoplanet class and includes:

- Builders for base learners: RandomForest, AdaBoost (wrapping RF),
    GradientBoosting, and an MLP pipeline.
- A `train_stack` orchestration that fits base learners, handles sample/class
    weighting, trains a StackingClassifier, and returns an evaluation report.
- A 10-fold stratified CV helper to capture per-fold recall and average
    precision for the confirmed class.

The code favors readability and conservative compatibility (it attempts
workarounds for scikit-learn API differences and missing sample_weight
support in older MLP implementations).
"""
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import (
    recall_score,
    precision_recall_curve,
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import compute_sample_weight
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ------------------ Hyperparameter placeholders ------------------

# Random forest base params
RF_PARAMS: Dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": None,
    "random_state": 42,
    # class_weight left empty here; compute and pass during training where available
}

# AdaBoost params (wraps a base estimator)
ADB_PARAMS: Dict[str, Any] = {
    "n_estimators": 100,
    "learning_rate": 1.0,
    "random_state": 42,
}

# MLP params
MLP_PARAMS: Dict[str, Any] = {
    "hidden_layer_sizes": (100, ),
    "activation": "relu",
    "solver": "adam",
    "alpha": 1e-4,
    "max_iter": 300,
    "random_state": 42,
}

# Stacking params
STACK_PARAMS: Dict[str, Any] = {
    "final_estimator": LogisticRegression(max_iter=1000),
    "cv": 5,
    "stack_method": "predict_proba",
}


# Gradient Boosting params and builder
GB_PARAMS: Dict[str, Any] = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 3,
    "random_state": 42,
}

CLASS_WEIGHTS: Dict[int,float] = {
    -1:-1.0,
    0:0.0,
    1:10.0
} 

# ------------------ Model builders ------------------

def build_random_forest(class_weight: Optional[Dict[int, float]] = None) -> RandomForestClassifier:
    """Build a RandomForestClassifier with default params.

    If `class_weight` is provided it is passed to the estimator so RF will
    internally weight classes during training.
    """
    params = RF_PARAMS.copy()
    if class_weight is not None:
        params["class_weight"] = class_weight
    return RandomForestClassifier(**params)


def build_adaboost(base_estimator: Optional[Any] = None) -> AdaBoostClassifier:
    """Build an AdaBoostClassifier that wraps a base estimator."""
    
    # prefer explicit base estimator (default: a RandomForest)
    if base_estimator is None:
        base_estimator = build_random_forest()
    
    return AdaBoostClassifier(estimator=base_estimator, **ADB_PARAMS)


def build_mlp(sample_weighted: bool = True) -> Pipeline:
    """Return a small pipeline wrapping StandardScaler and MLPClassifier.

    Historically some sklearn MLP implementations didn't accept ``sample_weight``
    in fit(). The ``sample_weighted`` boolean is informational — callers that
    need to weight samples can either pass sample_weight to .fit (if supported)
    or use the provided resampling workaround in ``train_stack``.
    """
    scaler = StandardScaler()
    mlp = MLPClassifier(**MLP_PARAMS)
    return Pipeline([("scaler", scaler), ("mlp", mlp)])

def build_gradient_boost() -> GradientBoostingClassifier:
    """Return a GradientBoostingClassifier.

    Note: sklearn's GradientBoostingClassifier doesn't accept class_weight directly.
    If class weighting is required, consider using sample_weight during fit or
    use HistGradientBoostingClassifier (which supports class_weight in newer sklearn).
    """
    # GradientBoostingClassifier does not accept class_weight; leave out.
    return GradientBoostingClassifier(**GB_PARAMS)

# ------------------ Training / evaluation utils ------------------

def compute_class_weights_from_y(y: np.ndarray, upweight_confirmed: float = CLASS_WEIGHTS[1]) -> Dict[int, float]:
    """Compute class weights from labels and optionally upweight confirmed class.

    Returns a mapping suitable for passing as ``class_weight`` to estimators
    that accept it (for example RandomForest). The baseline uses
    sklearn.utils.class_weight.compute_class_weight("balanced", ...)
    and multiplies the weight for the CONFIRMED class (label 2) by
    ``upweight_confirmed`` so false-negatives for confirmed planets are
    penalized more heavily.
    """
    classes = np.unique(y)
    # compute balanced class weights
    balanced = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    cw = {int(c): float(w) for c, w in zip(classes, balanced)}
    cw[1] = cw[1]*upweight_confirmed
    return cw

def train_stack(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    upweight_confirmed: float = CLASS_WEIGHTS[1],
) -> Tuple[StackingClassifier, Dict[str, Any]]:
    """Orchestrate training of base learners and a stacking classifier.

    Steps performed (high level):
      1. Split data into train/test using stratified sampling.
      2. Compute class weights (balanced -> upweight confirmed class).
      3. Build base estimators (RF, AdaBoost, GradientBoost, MLP pipeline).
      4. Fit RF, AdaBoost and GradientBoost on training data (passing class
         weights where supported). For MLP, if ``sample_weight`` cannot be
         passed at fit-time, a probabilistic resampling approach is used.
      5. Fit a StackingClassifier on the training data (by default this will
         refit the base estimators internally). The stacking meta-estimator is
         LogisticRegression and the stack uses predict_proba to create meta-features.
      6. Evaluate on the held-out test set: classification_report, confusion
         matrix, recall for the CONFIRMED class, and average-precision (AP).
      7. Optionally compute stratified 10-fold CV metrics across the full
         dataset for recall and AP (see `cv10` in the returned report).

    Returns:
      (fitted_stack, report_dict)

    The returned report emphasizes recall/AP for the CONFIRMED class (label 2).
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_size, random_state=random_state)

    # class weights
    cw = compute_class_weights_from_y(y_train, upweight_confirmed=upweight_confirmed)

    rf = build_random_forest(class_weight=cw)
    adb = build_adaboost(base_estimator=build_random_forest(class_weight=cw))
    gb = build_gradient_boost()
    mlp_pipe = build_mlp(sample_weighted=True)

    estimators = [("rf", rf), ("adb", adb), ("gb", gb), ("mlp", mlp_pipe)]

    stack = StackingClassifier(estimators=estimators, **STACK_PARAMS)

    # compute sample weights for mlp if desired
    sample_w = compute_sample_weight(class_weight="balanced", y=y_train)
    # upweight confirmed samples further if desired
    sample_w = np.where(y_train == 2, sample_w * upweight_confirmed, sample_w)

    # Fit base estimators separately when they accept sample_weight in fit
    # RandomForest and AdaBoost accept class_weight/sample_weight internally when constructed
    rf.fit(X_train, y_train)
    adb.fit(X_train, y_train)
    # GradientBoostingClassifier doesn't accept class_weight; use sample weights if desired
    try:
        gb.fit(X_train, y_train)
    except TypeError:
        # try fitting with sample weights if signature requires it
        try:
            gb.fit(X_train, y_train, sample_weight=compute_sample_weight(class_weight="balanced", y=y_train))
        except Exception:
            # fallback: fit without sample weights
            gb.fit(X_train, y_train)

    # For older sklearn MLP implementations that do not accept sample_weight
    # in MLP.fit, we fall back to resampling the training set with replacement.
    # The resampling probabilities are proportional to a computed sample weight
    # vector so higher-weighted examples appear more often in the MLP training set.
    def _resample_by_weight(Xa, ya, weights, n_samples=None, random_state=42):
        if n_samples is None:
            n_samples = len(ya)
        rng = np.random.RandomState(random_state)
        probs = weights.astype(float) / float(np.sum(weights))
        idx = rng.choice(len(ya), size=n_samples, replace=True, p=probs)
        return Xa.iloc[idx, :].values if hasattr(Xa, 'iloc') else Xa[idx], ya[idx]

    X_mlp, y_mlp = _resample_by_weight(X_train, y_train, sample_w, n_samples=len(y_train), random_state=42)
    mlp_pipe.fit(X_mlp, y_mlp)

    # Now fit stacking (it will re-fit underlying estimators by default). To avoid refitting,
    # set passthrough or re-use fitted estimators. For simplicity we allow refit here but pass
    # `n_jobs=1` to avoid nested parallelism issues.
    stack.set_params(n_jobs=1)
    stack.fit(X_train, y_train)

    # Evaluation
    y_pred = stack.predict(X_test)
    y_proba = stack.predict_proba(X_test)

    # Identify index of confirmed class (2) in classes_
    classes = stack.classes_
    try:
        idx_confirmed = list(classes).index(2)
    except ValueError:
        idx_confirmed = None

    recall = recall_score(y_test, y_pred, labels=[2], average="macro") if idx_confirmed is not None else None

    # Precision-recall for confirmed class
    if idx_confirmed is not None:
        # ensure y_proba is a numpy array
        y_proba_arr = np.asarray(y_proba)
        if y_proba_arr.ndim == 1:
            # binary predict_proba sometimes returns shape (n_samples,) for single-column; guard it
            proba_confirmed = y_proba_arr
        else:
            proba_confirmed = y_proba_arr[:, idx_confirmed]
        precision, recall_vals, thresholds = precision_recall_curve((y_test == 2).astype(int), proba_confirmed)
        ap = average_precision_score((y_test == 2).astype(int), proba_confirmed)
    else:
        precision, recall_vals, thresholds, ap = None, None, None, None

    report = {
        "recall_confirmed": recall,
        "average_precision_confirmed": ap,
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    # Compute stratified 10-fold CV metrics for the confirmed class as an additional important metric
    def cv_metrics_10fold(X_all, y_all, estimators, n_splits=10, random_state=42):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fold_results = []
        for train_idx, test_idx in skf.split(X_all, y_all):
            Xa_train, Xa_test = X_all.iloc[train_idx], X_all.iloc[test_idx]
            ya_train, ya_test = y_all[train_idx], y_all[test_idx]

            # For each fold build a fresh stacking estimator to avoid state carryover
            try:
                estimators_for_fold = [(name, clone(est)) for name, est in estimators]
            except Exception:
                # fallback if clone not possible (e.g., pipeline with external state)
                estimators_for_fold = estimators

            clf = StackingClassifier(estimators=estimators_for_fold, **STACK_PARAMS)
            clf.set_params(n_jobs=1)
            try:
                clf.fit(Xa_train, ya_train)
            except Exception:
                # If fitting the full stacking classifier fails (resource/compatibility), skip this fold
                continue

            y_pred_fold = clf.predict(Xa_test)
            # Get proba for confirmed class if available
            try:
                y_proba_fold = np.asarray(clf.predict_proba(Xa_test))
                if y_proba_fold.ndim == 1:
                    proba_conf = y_proba_fold
                else:
                    classes_fold = clf.classes_
                    if 2 in classes_fold:
                        idx_conf = list(classes_fold).index(2)
                        proba_conf = y_proba_fold[:, idx_conf]
                    else:
                        proba_conf = None
            except Exception:
                proba_conf = None

            recall_f = None
            ap_f = None
            try:
                if 2 in np.unique(ya_test):
                    recall_f = recall_score(ya_test, y_pred_fold, labels=[2], average="macro")
                if proba_conf is not None:
                    ap_f = average_precision_score((ya_test == 2).astype(int), proba_conf)
            except Exception:
                pass

            fold_results.append({"recall": recall_f, "average_precision": ap_f})

        # Aggregate
        recalls = [f["recall"] for f in fold_results if f["recall"] is not None]
        aps = [f["average_precision"] for f in fold_results if f["average_precision"] is not None]
        return {"n_folds": len(fold_results), "recall_mean": np.mean(recalls) if recalls else None,
                "recall_std": np.std(recalls) if recalls else None,
                "ap_mean": np.mean(aps) if aps else None, "ap_std": np.std(aps) if aps else None,
                "per_fold": fold_results}

    try:
        report["cv10"] = cv_metrics_10fold(X.fillna(0), y, estimators)
    except Exception as e:
        report["cv10"] = {"error": str(e)}

    return stack, report


# ------------------ Grid search helpers ------------------

def grid_search_rf(X, y, param_grid=None, scoring: str = "recall", cv: int = 5):
    """Grid search for RandomForest with scoring focused on recall for confirmed class.

    Note: `scoring='recall'` uses macro recall by default; for class-specific recall use a custom scorer.
    """
    if param_grid is None:
        param_grid = {"n_estimators": [100, 200], "max_depth": [None, 10, 20]}

    rf = RandomForestClassifier(random_state=42)

    # custom scorer example (recall for label 2)
    from sklearn.metrics import make_scorer
    from sklearn.metrics import recall_score

    def recall_confirmed(y_true, y_pred):
        return recall_score(y_true, y_pred, labels=[2], average="macro")

    scorer = make_scorer(recall_confirmed)

    gs = GridSearchCV(rf, param_grid, scoring=scorer, cv=cv, n_jobs=1)
    gs.fit(X, y)
    return gs


# ------------------ Small CLI/demo example ------------------
if __name__ == "__main__":
    # Minimal demo: load KOI table if available and run a quick train
    try:
        df = pd.read_csv("koi_exoplanets.csv")
    except Exception as e:
        print("Could not load koi_exoplanets.csv in current directory:", e)
        raise SystemExit(1)

    # Use the data hooks defined in this module to prepare features and labels
    X = modify_features(df)
    y = modify_labels(df, label_col="koi_disposition")
    print("Features shape:", X.shape)
    print("Label distribution:", pd.Series(y).value_counts().to_dict())

    stack, report = train_stack(X.fillna(0), y)
    print("Report:")
    print(report)