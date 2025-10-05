"""Train a Random Forest classifier on precomputed feature and label CSVs.

This module provides a small, self-contained training script used by the
project to fit a RandomForestClassifier on the features stored in
`Modelo/data/parameters.csv` and the labels in `Modelo/data/labels.csv`.

It exposes three main functions:
- `load_data` - read features and labels from CSV files and normalize shapes
- `train_save_model` - fit a RandomForest, compute basic metrics, save model
- `main` - entrypoint that wires constants and starts training

Notes:
- This file focuses on clarity and small-scale reproducible experiments; it
    intentionally keeps defaults simple and stores artifacts under the `Modelo`
    folder (model file and visualizations).
"""
from typing import Optional

import pandas as pd
import numpy as np
import visualization
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score

# Progress helpers: print timestamped stage/step messages
import progress

FEATURE_PATH = "data/parameters.csv"
LABEL_PATH = "data/labels.csv"
MODELOUT = "model.joblib"
VIZOUT = "viz_out"
TEST_SIZE = 0.4
RANDOM_STATE = 69
KFOLDDIV = 10  # number of folds for cross-validation

RF_ESTIMATORS = 100
RF_N_JOBS = -1

def visualize_model():
    """Placeholder for future visualization helpers.

    Currently visualization functions are implemented in the sibling
    `visualization` module and are invoked from `train_save_model`. This
    function exists as a clear extension point if additional model-level
    visualizations or interactive outputs are required later.
    """
    # Intentionally left blank; visualization handled in `visualization.py`.
    return None

def load_data(features_path: str, labels_path: str):
    """Load feature matrix X and label vector y from CSV files.

    Args:
        features_path: path to CSV containing feature columns (one row per
                       example).
        labels_path: path to CSV containing labels. The file may be a single
                     column (common) or have multiple columns.

    Returns:
        X: pandas.DataFrame with features
        y: pandas.Series or 1-D array-like with labels

    Behavior:
        If the labels CSV contains exactly one column, the column is
        converted to a Series to simplify downstream scikit-learn usage.
    """
    progress.stage("data_load", f"Loading features from {features_path} and labels from {labels_path}")
    progress.step("Reading features CSV")
    X = pd.read_csv(features_path)
    progress.step("Reading labels CSV")
    y = pd.read_csv(labels_path)

    # If the labels file only has one column (common case), convert to a
    # pandas Series so scikit-learn receives a 1-dimensional target vector.
    if hasattr(y, "shape") and y.shape[1] == 1:
        y = y.iloc[:, 0]

    return X, y

def train_save_model(X, y, output_path: str, params: Optional[dict] = None):
    """Train a RandomForestClassifier, produce visualizations, and save model.

    Args:
        X: feature DataFrame
        y: label Series/array
        output_path: path where the trained model will be saved (joblib)
        params: optional dict of parameters to pass to RandomForestClassifier

    Actions and outputs:
        - Splits the data into train/test using TEST_SIZE and RANDOM_STATE
        - Trains a RandomForestClassifier with provided params
        - Computes confusion matrix, train/test accuracy
        - Generates several plots using the `visualization` module and
          saves them under the `VIZOUT` directory
        - Serializes the trained model to `output_path` using joblib
    """
    params = params or {}

    progress.stage("training", "Starting model training")
    # Create classifier with user-provided or default parameters
    progress.step(f"RandomForest params: {params}")
    clf = RandomForestClassifier(**params)

    # Stratify split ensures label proportions are preserved in train/test
    progress.step("Splitting data into train and test sets")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    progress.step("Fitting RandomForest on training data")
    # Fit the model on the training partition
    clf.fit(X_train, y_train)

    progress.step("Predicting on test set and computing metrics")
    # Predict on the test set and compute metrics
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    test_acc = accuracy_score(y_test, y_pred)
    train_acc = accuracy_score(y_train, clf.predict(X_train))

    # Prepare labels for confusion matrix plotting: include any label that
    # appears in either ground truth or predictions and convert to strings
    labels = [str(x) for x in np.unique(np.concatenate([y_test, y_pred]))]

    progress.stage("visualization", "Creating and saving visualizations")
    # Save confusion matrix figure
    ax = visualization.plot_confusion_matrix(cm, labels=labels)
    ax.figure.savefig(str(VIZOUT + "/confusion_matrix.png"))

    # Compute cross-validation scores for additional diagnostics and save plot
    progress.step("Computing cross-validation scores")
    scores = visualization.compute_cv_scores(clf, X, y, cv=10)
    ax = visualization.plot_cv_scores(scores)
    ax.figure.savefig(str(VIZOUT + "/cv_scores.png"))

    # Plot ROC AUC: prefer probability scores or decision function values
    progress.step("Preparing scores for ROC/PR plotting")
    progress.step(f"X_test shape={getattr(X_test, 'shape', None)}, y_test shape={getattr(y_test, 'shape', None)}")
    if hasattr(clf, "predict_proba"):
        try:
            y_score = clf.predict_proba(X_test)
        except Exception:
            y_score = y_pred
    elif hasattr(clf, "decision_function"):
        try:
            y_score = clf.decision_function(X_test)
        except Exception:
            y_score = y_pred
    else:
        y_score = y_pred

    progress.step(f"y_score preview: shape={np.asarray(y_score).shape}, dtype={np.asarray(y_score).dtype}")
    # Show a small sample for quick inspection
    try:
        sample_vals = np.asarray(y_score)[:8]
    except Exception:
        sample_vals = str(type(y_score))
    progress.step(f"y_score sample={sample_vals}")

    ax = visualization.plot_roc_auc(y_test, y_score)
    ax.figure.savefig(str(VIZOUT + "/roc_auc.png"))

    # PR curve (use same y_score selection as above)
    ax = visualization.plot_pr_auc(y_test, y_score)
    ax.figure.savefig(str(VIZOUT + "/pr_auc.png"))

    # Plot true positives vs other predictions and save
    ax = visualization.plot_truepositives_vs_others(y_test, y_pred)
    ax.figure.savefig(str(VIZOUT + "/tp_vs_others.png"))

    progress.step("Saving train/test accuracy plot")
    # Visualize train vs test accuracy and save
    ax = visualization.plot_train_test_accuracy(float(train_acc), float(test_acc))
    ax.figure.savefig(str(VIZOUT + "/train_test_accuracy.png"))

    progress.stage("persistence", "Saving trained model to disk")
    # Persist trained model to disk and inform the user
    joblib.dump(clf, output_path)
    progress.step(f"Saved model to {output_path}")

    # Run aggregated evaluation and save metrics
    try:
        from Modelo.metrics import evaluate_model
        progress.stage("evaluation", "Computing aggregated evaluation metrics")
        metrics = evaluate_model(clf, X_train, X_test, y_train, y_test, compute_permutation=False)
        import json
        metrics_path = str(VIZOUT + "/metrics.json")
        with open(metrics_path, 'w', encoding='utf-8') as fh:
            json.dump(metrics, fh, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o), indent=2)
        progress.step(f"Saved aggregated metrics to {metrics_path}")

        # Print a concise human-readable summary
        summary_keys = ["accuracy", "balanced_accuracy", "log_loss", "brier_score", "cohen_kappa", "mcc"]
        summary = {k: metrics.get(k) for k in summary_keys}
        progress.step(f"Metrics summary: {summary}")
    except Exception as ex:
        progress.step(f"Aggregated evaluation failed: {ex}")


def ten_fold_cross_validation(estimator, X, y, random_state: int = RANDOM_STATE):
    """Run a stratified 10-fold cross-validation and return per-fold metrics.

    Args:
        estimator: an sklearn estimator (unfitted)
        X: features (DataFrame or array)
        y: labels (Series or array)
        random_state: seed for the StratifiedKFold splitter

    Returns:
        results: dict with keys 'accuracy', 'precision', 'recall', 'f1'
                 each maps to a numpy array of length 10 (per-fold scores)
    """
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    skf = StratifiedKFold(n_splits=KFOLDDIV, shuffle=True, random_state=random_state)

    accs = []
    precisions = []
    recalls = []
    f1s = []

    progress.stage("cross_validation", "Running stratified 10-fold cross-validation")
    fold = 0
    for train_idx, test_idx in skf.split(X_arr, y_arr):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]

        fold += 1
        progress.step(f"Starting fold {fold}/{KFOLDDIV}")

        est = estimator
        # If estimator is a class instance that was already fitted, clone is safer,
        # but to avoid adding sklearn.base dependency we create a fresh instance when possible.
        try:
            from sklearn.base import clone
            est = clone(estimator)
        except Exception:
            # fallback: use the provided estimator and refit (may overwrite state)
            est = estimator

        est.fit(X_train, y_train)
        y_pred = est.predict(X_test)

        accs.append(accuracy_score(y_test, y_pred))
        # average='macro' to treat classes equally regardless of support
        precisions.append(precision_score(y_test, y_pred, average='macro', zero_division=0))
        recalls.append(recall_score(y_test, y_pred, average='macro', zero_division=0))
        f1s.append(f1_score(y_test, y_pred, average='macro', zero_division=0))

    results = {
        'accuracy': np.array(accs),
        'precision': np.array(precisions),
        'recall': np.array(recalls),
        'f1': np.array(f1s),
    }
    return results

def ten_fold_test_saved_model(model_path: str, X, y):
    """Load a saved model and run a single 10-fold evaluation using the loaded estimator.

    This is a convenience wrapper that loads the joblib model and calls
    `ten_fold_cross_validation` with it.

    Returns the same dict structure as `ten_fold_cross_validation`.
    """
    progress.stage("load_model", f"Loading model from {model_path}")
    clf = joblib.load(model_path)
    progress.step("Model loaded; starting 10-fold evaluation")
    return ten_fold_cross_validation(clf, X, y)

def main():

    features_path = FEATURE_PATH
    labels_path = LABEL_PATH
    model_out = MODELOUT

    X, y = load_data(features_path, labels_path)

    progress.stage("pre_training", "Data loaded and ready; beginning training pipeline")

    # Basic parameter mapping: try to read from data/parameters.csv header 'rf_n_estimators' etc if present
    # Otherwise use a small default for quick runs
    rf_params = {
        "n_estimators": RF_ESTIMATORS,
        "random_state": RANDOM_STATE,
        "n_jobs": RF_N_JOBS,
    }

    train_save_model(X, y, model_out, rf_params)
    # Ensure visualization output directory exists
    briescore = visualization.compute_brier_score(y, np.asarray(y))  # Dummy example; replace with real scores if available
    print(f"Brier score: {briescore}")

    progress.stage("kfold_evaluation", "Running k-fold evaluation and saving results")
    results = ten_fold_test_saved_model(model_out, X, y)
    fig, axes = visualization.plotkfold_results(results)
    fig.savefig(str(VIZOUT + "/kfold_results.png"))

    progress.step(f"Saved k-fold results to {VIZOUT + '/kfold_results.png'}")

if __name__ == "__main__":
    main()
