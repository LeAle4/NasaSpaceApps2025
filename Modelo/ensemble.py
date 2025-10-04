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
import time

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
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import compute_sample_weight
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import visualization as viz

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# ------------------ Module-level defaults / constants ------------------
# Centralize common defaults so they are easy to find and modify for
# debugging and experimentation.
RANDOM_STATE: int = 42
TEST_SIZE_DEFAULT: float = 0.2
UPWEIGHT_CONFIRMED_DEFAULT: float = 10.0
STACK_CV_DEFAULT: int = 5
CV10_SPLITS_DEFAULT: int = 10
SAMPLE_WEIGHT_METHOD: str = "balanced"
N_JOBS_DEFAULT: int = -1

# Label constants (useful for readability in the rest of the module)
LABEL_CONFIRMED: int = 2
LABEL_CANDIDATE: int = 1
LABEL_FALSE_POSITIVE: int = 0

# A small mapping used by some callers/tests; kept here for convenience
CLASS_WEIGHTS: Dict[int, float] = {
    -1: -1.0,
    0: 1.0,
    1: UPWEIGHT_CONFIRMED_DEFAULT,
}

# ------------------ Hyperparameter placeholders ------------------

# Random forest base params
RF_PARAMS: Dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": None,
    "random_state": RANDOM_STATE,
    "n_jobs": N_JOBS_DEFAULT,
    # class_weight left empty here; compute and pass during training where available
}

# AdaBoost params (wraps a base estimator)
ADB_PARAMS: Dict[str, Any] = {
    "n_estimators": 100,
    "learning_rate": 1.0,
    "random_state": RANDOM_STATE,
}

# MLP params
MLP_PARAMS: Dict[str, Any] = {
    "hidden_layer_sizes": (100, ),
    "activation": "relu",
    "solver": "adam",
    "alpha": 1e-4,
    "max_iter": 300,
    "random_state": RANDOM_STATE,
}

# Stacking params
STACK_PARAMS: Dict[str, Any] = {
    "final_estimator": LogisticRegression(max_iter=1000),
    "cv": STACK_CV_DEFAULT,
    "stack_method": "predict_proba",
}


# Gradient Boosting params and builder
GB_PARAMS: Dict[str, Any] = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 3,
    "random_state": RANDOM_STATE,
}

# (CLASS_WEIGHTS already defined above to use the centralized UPWEIGHT_CONFIRMED_DEFAULT)

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


def build_mlp(sample_weighted: bool = True, verbose: bool = False) -> Pipeline:
    """Return a small pipeline wrapping StandardScaler and MLPClassifier.

    If ``verbose`` is True the underlying MLPClassifier will be created with
    verbose=True so that it prints iteration losses to stdout during training.

    Historically some sklearn MLP implementations didn't accept ``sample_weight``
    in fit(). The ``sample_weighted`` boolean is informational — callers that
    need to weight samples can either pass sample_weight to .fit (if supported)
    or use the provided resampling workaround in ``train_stack``.
    """
    scaler = StandardScaler()
    params = MLP_PARAMS.copy()
    if verbose:
        # allow the demo/user to see MLP iteration losses
        params["verbose"] = True
    mlp = MLPClassifier(**params)
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

def compute_class_weights_from_y(y: np.ndarray, upweight_confirmed: float = UPWEIGHT_CONFIRMED_DEFAULT) -> Dict[int, float]:
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
    balanced = compute_class_weight(class_weight=SAMPLE_WEIGHT_METHOD, classes=classes, y=y)
    cw = {int(c): float(w) for c, w in zip(classes, balanced)}
    cw[1] = cw[1]*upweight_confirmed
    return cw

def train_stack(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = TEST_SIZE_DEFAULT,
    random_state: int = RANDOM_STATE,
    upweight_confirmed: float = UPWEIGHT_CONFIRMED_DEFAULT,
    verbose: bool = False,
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
    # Split features/labels into train/test (stratified to preserve label ratios)
    train_features, test_features, train_labels, test_labels = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )

    # class weights (use balanced weights then upweight the confirmed class)
    class_weights = compute_class_weights_from_y(train_labels, upweight_confirmed=upweight_confirmed)

    # Build base learners using descriptive names
    if verbose:
        print("[train_stack] Building base learners...")
    rf_clf = build_random_forest(class_weight=class_weights)
    adb_clf = build_adaboost(base_estimator=build_random_forest(class_weight=class_weights))
    gb_clf = build_gradient_boost()
    # pass verbosity down to the MLP so it prints its internal training progress
    mlp_pipeline = build_mlp(sample_weighted=True, verbose=verbose)

    estimators = [("rf", rf_clf), ("adb", adb_clf), ("gb", gb_clf), ("mlp", mlp_pipeline)]

    stack = StackingClassifier(estimators=estimators, **STACK_PARAMS)

    # compute sample weights once and reuse (used for MLP and GB fallback)
    base_sample_weights = compute_sample_weight(class_weight=SAMPLE_WEIGHT_METHOD, y=train_labels)
    # upweight confirmed samples further if requested
    sample_weights = np.where(train_labels == LABEL_CANDIDATE, base_sample_weights * upweight_confirmed, base_sample_weights)

    # Fit base estimators separately when they accept class_weight/sample_weight
    if verbose:
        t0 = time.time()
        print(f"[train_stack] Fitting RandomForest (n_samples={len(train_labels)})...")
    rf_clf.fit(train_features, train_labels)
    if verbose:
        print(f"[train_stack] RandomForest fit complete in {time.time()-t0:.2f}s")
        t0 = time.time()
        print(f"[train_stack] Fitting AdaBoost (wraps RF)...")
    adb_clf.fit(train_features, train_labels)
    if verbose:
        print(f"[train_stack] AdaBoost fit complete in {time.time()-t0:.2f}s")
    # GradientBoostingClassifier doesn't accept class_weight; attempt sample_weight if signature allows
    if verbose:
        print("[train_stack] Fitting GradientBoostingClassifier...")
        t0 = time.time()
    try:
        gb_clf.fit(train_features, train_labels)
    except TypeError:
        try:
            gb_clf.fit(train_features, train_labels, sample_weight=base_sample_weights)
        except Exception:
            gb_clf.fit(train_features, train_labels)
    if verbose:
        print(f"[train_stack] GradientBoost fit complete in {time.time()-t0:.2f}s")

    # Fit the MLP pipeline. Modern scikit-learn pipelines accept sample_weight
    # by passing the keyword for the final estimator using the pipeline step
    # name ("mlp__sample_weight") to Pipeline.fit. We attempt to pass
    # sample_weight; if the installed sklearn doesn't support it we fall back
    # to fitting without weights.
    if verbose:
        print("[train_stack] Fitting MLP pipeline (attempting to pass sample_weight)...")
        t0 = time.time()
    
    # When using Pipeline, pass sample weights for the final estimator using <step>__sample_weight
    mlp_pipeline.fit(train_features, train_labels, mlp__sample_weight=sample_weights)
    
    if verbose:
        print(f"[train_stack] MLP pipeline fit complete in {time.time()-t0:.2f}s")

    # Now fit the stacking classifier on the training set. We set `n_jobs=1` to
    # avoid nested parallelism issues that can arise when base learners run in parallel.
    if verbose:
        print("[train_stack] Fitting StackingClassifier (this may take a while)...")
        t0 = time.time()
    stack.set_params(n_jobs=1)
    stack.fit(train_features, train_labels)
    if verbose:
        print(f"[train_stack] StackingClassifier fit complete in {time.time()-t0:.2f}s")

    # Evaluation
    y_pred = stack.predict(test_features)
    y_proba = stack.predict_proba(test_features)

    # Identify index of confirmed class (2) in classes_
    classes = stack.classes_
    try:
        idx_confirmed = list(classes).index(LABEL_CONFIRMED)
    except ValueError:
        idx_confirmed = None

    recall = recall_score(test_labels, y_pred, labels=[LABEL_CONFIRMED], average="macro") if idx_confirmed is not None else None

    # Precision-recall for confirmed class - cache boolean masks for reuse
    confirmed_mask_test = (test_labels == LABEL_CONFIRMED)
    confirmed_mask_train = (train_labels == LABEL_CONFIRMED)
    if idx_confirmed is not None:
        # ensure y_proba is a numpy array
        y_proba_arr = np.asarray(y_proba)
        if y_proba_arr.ndim == 1:
            # binary predict_proba sometimes returns shape (n_samples,) for single-column; guard it
            proba_confirmed = y_proba_arr
        else:
            proba_confirmed = y_proba_arr[:, idx_confirmed]

        precision, recall_vals, thresholds = precision_recall_curve(confirmed_mask_test.astype(int), proba_confirmed)
        ap = average_precision_score(confirmed_mask_test.astype(int), proba_confirmed)

        # Additional diagnostics: ROC/AUC, calibration curve, feature importances and plots
        try:
            fpr, tpr, roc_thresh = roc_curve(confirmed_mask_test.astype(int), proba_confirmed)
            roc_auc = roc_auc_score(confirmed_mask_test.astype(int), proba_confirmed)
        except Exception:
            fpr = tpr = roc_thresh = roc_auc = None

        try:
            prob_true, prob_pred = calibration_curve(confirmed_mask_test.astype(int), proba_confirmed, n_bins=10)
        except Exception:
            prob_true = prob_pred = None

        # Feature importances from tree-based base learners (if available)
        feature_names = None
        try:
            if hasattr(train_features, "columns"):
                feature_names = list(train_features.columns)
            else:
                feature_names = [f"f{i}" for i in range(train_features.shape[1])]
        except Exception:
            feature_names = None

        rf_importances = None
        gb_importances = None
        try:
            if hasattr(rf_clf, "feature_importances_") and feature_names is not None:
                vals = list(rf_clf.feature_importances_)
                rf_importances = sorted(zip(feature_names, vals), key=lambda x: x[1], reverse=True)
        except Exception:
            rf_importances = None
        try:
            if hasattr(gb_clf, "feature_importances_") and feature_names is not None:
                vals = list(gb_clf.feature_importances_)
                gb_importances = sorted(zip(feature_names, vals), key=lambda x: x[1], reverse=True)
        except Exception:
            gb_importances = None

        plots = {}
        # Use visualization helpers to produce base64 PNGs
        try:
            plots["pr_curve"] = viz.plot_pr_curve(precision, recall_vals, ap)
        except Exception:
            plots["pr_curve"] = None
        try:
            plots["roc_curve"] = viz.plot_roc_curve(fpr, tpr, roc_auc)
        except Exception:
            plots["roc_curve"] = None
        try:
            plots["calibration_curve"] = viz.plot_calibration_curve(prob_true, prob_pred)
        except Exception:
            plots["calibration_curve"] = None
        try:
            plots["rf_feature_importances"] = viz.plot_feature_importances(rf_importances, title="RandomForest feature importances (top)")
        except Exception:
            plots["rf_feature_importances"] = None
        try:
            plots["gb_feature_importances"] = viz.plot_feature_importances(gb_importances, title="GradientBoost feature importances (top)")
        except Exception:
            plots["gb_feature_importances"] = None
    else:
        precision = recall_vals = thresholds = ap = None
        fpr = tpr = roc_thresh = roc_auc = None
        prob_true = prob_pred = None
        rf_importances = gb_importances = None
        plots = {}

    report = {
        "recall_confirmed": recall,
        "average_precision_confirmed": ap,
        "classification_report": classification_report(test_labels, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(test_labels, y_pred).tolist(),
    }

    # Attach diagnostic arrays and plots to the report for downstream analysis
    report.update({
        "pr_curve": {"precision": None if precision is None else precision.tolist(),
                      "recall": None if recall_vals is None else recall_vals.tolist(),
                      "thresholds": None if thresholds is None else thresholds.tolist()},
        "roc": {"fpr": None if fpr is None else fpr.tolist(),
                "tpr": None if tpr is None else tpr.tolist(),
                "thresholds": None if roc_thresh is None else roc_thresh.tolist(),
                "auc": roc_auc},
        "calibration": {"prob_true": None if prob_true is None else prob_true.tolist(),
                        "prob_pred": None if prob_pred is None else prob_pred.tolist()},
        "feature_importances": {"rf": rf_importances, "gb": gb_importances},
        "plots": plots,
    })

    # Compute stratified 10-fold CV metrics for the confirmed class as an additional important metric
    def cv_metrics_10fold(X_all, y_all, estimators, n_splits=CV10_SPLITS_DEFAULT, random_state=RANDOM_STATE):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fold_results = []
        fold_no = 0
        for train_idx, test_idx in skf.split(X_all, y_all):
            fold_no += 1
            if verbose:
                print(f"[cv_metrics_10fold] Starting fold {fold_no}/{n_splits}...")
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
                if verbose:
                    print(f"[cv_metrics_10fold] Fold {fold_no} fit failed, skipping")
                continue

            y_pred_fold = clf.predict(Xa_test)
            # Get proba for confirmed class if available
            try:
                y_proba_fold = np.asarray(clf.predict_proba(Xa_test))
                if y_proba_fold.ndim == 1:
                    proba_conf = y_proba_fold
                else:
                    classes_fold = clf.classes_
                    if LABEL_CONFIRMED in classes_fold:
                        idx_conf = list(classes_fold).index(LABEL_CONFIRMED)
                        proba_conf = y_proba_fold[:, idx_conf]
                    else:
                        proba_conf = None
            except Exception:
                proba_conf = None

            recall_f = None
            ap_f = None
            try:
                if LABEL_CONFIRMED in np.unique(ya_test):
                    recall_f = recall_score(ya_test, y_pred_fold, labels=[LABEL_CONFIRMED], average="macro")
                if proba_conf is not None:
                    ap_f = average_precision_score((ya_test == LABEL_CONFIRMED).astype(int), proba_conf)
            except Exception:
                pass

            fold_results.append({"recall": recall_f, "average_precision": ap_f})
            if verbose:
                print(f"[cv_metrics_10fold] Fold {fold_no} results: recall={recall_f}, ap={ap_f}")

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

def grid_search_rf(X, y, param_grid=None, scoring: str = "recall", cv: int = STACK_CV_DEFAULT):
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
    # Use the data hooks defined in this module to prepare features and labels
    X = pd.read_csv("parameters.csv", )
    print("Loaded features from parameters.csv")
    y = pd.read_csv("labels.csv")
    print("Loaded labels from labels.csv")
    y = np.asarray(y).ravel()
    print("Features shape:", X.shape)
    print("Label distribution:", pd.Series(y).value_counts().to_dict())
    # Train a stacking classifier and print the report. Enable verbose to show
    # progress for each major training step and, for the MLP, internal losses.
    print("Starting training (verbose=True). You will see step-by-step progress...")
    print(X, y)

    stack_clf, report = train_stack(X, y, test_size=0.2, random_state=42, verbose=True)

    print("Training complete. Summary:")
    print("  - recall_confirmed:", report.get("recall_confirmed"))
    print("  - average_precision_confirmed:", report.get("average_precision_confirmed"))
    if "cv10" in report:
        cv10 = report["cv10"]
        if isinstance(cv10, dict) and "n_folds" in cv10:
            print(f"  - cv10: {cv10['n_folds']} folds, recall_mean={cv10.get('recall_mean')}, ap_mean={cv10.get('ap_mean')}")
        else:
            print("  - cv10: ", cv10)