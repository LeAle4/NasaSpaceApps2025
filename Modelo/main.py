from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import pandas as pd
import visualization as viz
import matplotlib.pyplot as plt
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
from pathlib import Path
import base64


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
LABEL_CONFIRMED: int = 1
LABEL_CANDIDATE: int = 0
LABEL_FALSE_POSITIVE: int = -1

# A small mapping used by some callers/tests; kept here for convenience
CLASS_WEIGHTS: Dict[int, float] = {
    LABEL_FALSE_POSITIVE: -1.0,
    LABEL_CANDIDATE: 1.0,
    LABEL_CONFIRMED: UPWEIGHT_CONFIRMED_DEFAULT,
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


# ------------------ Simplified pipeline ------------------
def simple_train_pipeline(X: pd.DataFrame, y: np.ndarray, test_size: float = TEST_SIZE_DEFAULT,
                          random_state: int = RANDOM_STATE) -> Dict[str, Any]:
    """A simplified training pipeline that fits a few base learners, a stacking
    classifier, computes a small evaluation report and returns visualization
    images (base64) using the `visualization` module.
    """
    # ensure X is a DataFrame
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )

    # sample weights: balanced, upweight confirmed
    base_sw = compute_sample_weight(class_weight=SAMPLE_WEIGHT_METHOD, y=y_train)
    sample_weights = np.where(y_train == LABEL_CONFIRMED, base_sw * UPWEIGHT_CONFIRMED_DEFAULT, base_sw)

    # build learners
    rf = build_random_forest(class_weight=None)
    gb = build_gradient_boost()
    mlp = build_mlp(sample_weighted=False, verbose=False)

    # fit base learners (simple approach)
    rf.fit(X_train, y_train)
    try:
        gb.fit(X_train, y_train, sample_weight=sample_weights)
    except TypeError:
        gb.fit(X_train, y_train)
    # fit MLP without sample_weight to keep compatibility
    mlp.fit(X_train, y_train)

    # stacking
    stack = StackingClassifier(estimators=[('rf', rf), ('gb', gb), ('mlp', mlp)],
                               final_estimator=LogisticRegression(max_iter=1000), cv=STACK_CV_DEFAULT,
                               stack_method='predict_proba', n_jobs=1)
    stack.fit(X_train, y_train)

    # predict / proba
    y_pred = stack.predict(X_test)
    y_proba = stack.predict_proba(X_test)

    # find index of confirmed class
    idx_conf = None
    try:
        idx_conf = int(np.where(stack.classes_ == LABEL_CONFIRMED)[0][0])
    except Exception:
        idx_conf = None

    recall_conf = None
    ap_conf = None
    precision = recall_vals = thresholds = None
    fpr = tpr = roc_thresh = None
    roc_auc = None
    prob_true = prob_pred = None

    if idx_conf is not None:
        recall_conf = recall_score(y_test, y_pred, labels=[LABEL_CONFIRMED], average='macro')
        # ensure y_proba is numpy array and slice the confirmed column
        y_proba = np.asarray(y_proba)
        y_score = y_proba[:, int(idx_conf)]
        ap_conf = average_precision_score((y_test == LABEL_CONFIRMED).astype(int), y_score)

        # precision-recall
        precision, recall_vals, thresholds = precision_recall_curve((y_test == LABEL_CONFIRMED).astype(int), y_score)

        # ROC and AUC (may fail for degenerate labels)
        try:
            fpr, tpr, roc_thresh = roc_curve((y_test == LABEL_CONFIRMED).astype(int), y_score)
            roc_auc = roc_auc_score((y_test == LABEL_CONFIRMED).astype(int), y_score)
        except Exception:
            fpr = tpr = roc_thresh = roc_auc = None

        # calibration curve
        try:
            prob_true, prob_pred = calibration_curve((y_test == LABEL_CONFIRMED).astype(int), y_score, n_bins=10)
        except Exception:
            prob_true = prob_pred = None
    else:
        precision = recall_vals = thresholds = None
        fpr = tpr = roc_thresh = None
        roc_auc = None
        prob_true = prob_pred = None

    # feature importances
    feature_names = list(X.columns)
    rf_importances = None
    gb_importances = None
    try:
        rf_importances = list(zip(feature_names, rf.feature_importances_))
        rf_importances = sorted(rf_importances, key=lambda x: x[1], reverse=True)
    except Exception:
        rf_importances = None
    try:
        gb_importances = list(zip(feature_names, gb.feature_importances_))
        gb_importances = sorted(gb_importances, key=lambda x: x[1], reverse=True)
    except Exception:
        gb_importances = None

    # visualizations via visualization.py (imported as viz at top)
    # prepare arrays/lists for visualization (convert numpy arrays to lists)
    def as_list(x):
        if x is None:
            return None
        try:
            return x.tolist()
        except Exception:
            return list(x)

    plots = {
        'pr_curve': viz.plot_pr_curve(as_list(precision), as_list(recall_vals), float(ap_conf) if ap_conf is not None else None),
        'roc_curve': viz.plot_roc_curve(as_list(fpr), as_list(tpr), float(roc_auc) if roc_auc is not None else None),
        'calibration': viz.plot_calibration_curve(as_list(prob_true), as_list(prob_pred)),
        'rf_importances': viz.plot_feature_importances(rf_importances, title='RF importances (top)'),
        'gb_importances': viz.plot_feature_importances(gb_importances, title='GB importances (top)'),
    }

    report = {
        'recall_confirmed': recall_conf,
        'average_precision_confirmed': ap_conf,
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'plots': plots,
    }

    return report


if __name__ == '__main__':
    # Runner: load local parameters.csv and labels.csv from this module folder
    base = Path(__file__).parent
    X_path = base / 'parameters.csv'
    y_path = base / 'labels.csv'

    if not X_path.exists() or not y_path.exists():
        print(f"Missing data files at {X_path} or {y_path}. Place 'parameters.csv' and 'labels.csv' in {base}.")
    else:
        X = pd.read_csv(X_path)
        y = pd.read_csv(y_path).squeeze().values
        print('Loaded data shapes:', X.shape, y.shape)
        report = simple_train_pipeline(X, y)
        print('Recall (confirmed):', report.get('recall_confirmed'))
        print('Average precision (confirmed):', report.get('average_precision_confirmed'))
        # write out images to files for quick inspection if available
        out_dir = base / 'viz_out'
        out_dir.mkdir(exist_ok=True)
        for name, b64 in report['plots'].items():
            if b64:
                out_file = out_dir / (name + '.png')
                with open(out_file, 'wb') as fh:
                        fh.write(base64.b64decode(b64))
        print('Wrote visualization images to', out_dir)

