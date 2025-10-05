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
from sklearn.inspection import permutation_importance as _perm
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score

# Progress helpers: print timestamped stage/step messages
import progress

FEATURE_PATH = "data/non_candidates_processed_features.csv"
LABEL_PATH = "data/non_candidates_processed_labels.csv"
MODELOUT = "model.joblib"
VIZOUT = "viz_out"
TEST_SIZE = 0.4
RANDOM_STATE = 420
KFOLDDIV = 10  # number of folds for cross-validation

RF_ESTIMATORS = 100
RF_N_JOBS = -1

class Model:
    """Light wrapper around a scikit-learn RandomForestClassifier.

    This class encapsulates a RandomForestClassifier instance and exposes a
    small API that will make later refactors and testing easier. It intentionally
    mirrors the common sklearn estimator methods so it can be dropped into
    existing code that expects an estimator.

    Usage examples:
        m = Model(params={"n_estimators": 10, "random_state": 0})
        m.fit(X, y)
        preds = m.predict(X_test)
        m.save("model.joblib")
        m2 = Model.load("model.joblib")
    """

    def __init__(self, params: Optional[dict] = None, estimator=None, random_state: Optional[int] = None):
        """Create a Model wrapper.

        Args:
            params: dict of parameters to pass to RandomForestClassifier if
                    no `estimator` instance is provided.
            estimator: an existing sklearn-like estimator instance (optional)
        """
        self.params = params or {}
        self.random_state = random_state
        if estimator is not None:
            # Allow wrapping a pre-built estimator (e.g., loaded from disk)
            self.model = estimator
        else:
            # ensure reproducible RandomForest when random_state provided
            if self.random_state is not None and 'random_state' not in self.params:
                self.params['random_state'] = self.random_state
            self.model = RandomForestClassifier(**self.params)

    # Estimator façade -------------------------------------------------
    def fit(self, X, y, **fit_kwargs):
        """Fit the internal RandomForest model and return self.

        Accepts additional keyword arguments forwarded to the estimator.fit
        method for flexibility.
        """
        self.model.fit(X, y, **fit_kwargs)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise AttributeError("Underlying estimator has no 'predict_proba' method")

    def decision_function(self, X):
        if hasattr(self.model, "decision_function"):
            return self.model.decision_function(X)
        raise AttributeError("Underlying estimator has no 'decision_function' method")

    def score(self, X, y):
        return self.model.score(X, y)

    # Persistence -----------------------------------------------------
    def save(self, path: str):
        """Persist the wrapped estimator to disk using joblib."""
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str):
        """Load a joblib-serialized estimator and return a Model wrapper.

        The loaded object is wrapped so callers can rely on the same API.
        """
        estimator = joblib.load(path)
        return cls(estimator=estimator)

    # Parameters & helpers -------------------------------------------
    def set_params(self, **params):
        """Set parameters on the underlying estimator and update local copy."""
        self.params.update(params)
        try:
            self.model.set_params(**params)
        except Exception:
            # Some wrapped objects may not support set_params; ignore gracefully
            pass

    def get_params(self, deep=True):
        try:
            return self.model.get_params(deep=deep)
        except Exception:
            return dict(self.params)

    def feature_importances_(self):
        """Return model.feature_importances_ if available, else None."""
        return getattr(self.model, "feature_importances_", None)

    def permutation_importance(self, X, y, scoring=None, n_repeats: int = 30, random_state: Optional[int] = None):
        """Convenience wrapper around sklearn.inspection.permutation_importance.

        Returns the same object sklearn returns (Bunch-like) so callers can
        examine importances_mean, importances_std, etc.
        """

        return _perm(self.model, X, y, scoring=scoring, n_repeats=n_repeats, random_state=random_state)

    def clone(self):
        """Return a fresh clone of this wrapper (with a cloned estimator) if possible."""
        try:
            from sklearn.base import clone as _clone
            return Model(params=self.params.copy(), estimator=_clone(self.model))
        except Exception:
            # Fall back to serializing and loading to produce an independent copy
            import tempfile
            fp = tempfile.NamedTemporaryFile(delete=False)
            joblib.dump(self.model, fp.name)
            loaded = joblib.load(fp.name)
            return Model(estimator=loaded)


def visualize_model():
    """Placeholder for future visualization helpers.

    Currently visualization functions are implemented in the sibling
    `visualization` module and are invoked from `train_save_model`. This
    function exists as a clear extension point if additional model-level
    visualizations or interactive outputs are required later.
    """
    # Intentionally left blank; visualization handled in `visualization.py`.
    return None

def train_save_model(X, y, params: Optional[dict] = None, random_state: Optional[int] = None):
    """Train a RandomForest model and return a results dict.

    This function only performs the data split and fitting. It returns a
    dictionary with the trained wrapper, train/test splits, predictions and
    basic metrics so callers can separately visualize or persist artifacts.

    Returns:
        dict containing keys: wrapper, clf, X_train, X_test, y_train, y_test,
        y_pred, cm, train_acc, test_acc, labels, y_score
    """
    params = params or {}

    progress.stage("training", "Starting model training")
    progress.step(f"RandomForest params: {params}")

    # Create Model wrapper and perform stratified train/test split
    wrapper = Model(params=params, random_state=random_state)

    progress.step("Splitting data into train and test sets")
    # Respect caller-provided random_state; fall back to module-level constant
    rs = RANDOM_STATE if random_state is None else random_state
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=rs, stratify=y
    )

    progress.step("Fitting RandomForest on training data")
    wrapper.fit(X_train, y_train)
    clf = wrapper.model

    progress.step("Predicting on test set and computing metrics")
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    test_acc = accuracy_score(y_test, y_pred)
    train_acc = accuracy_score(y_train, clf.predict(X_train))

    # Labels for plotting
    labels = [str(x) for x in np.unique(np.concatenate([y_test, y_pred]))]

    # Prepare y_score (prefer probabilities or decision function if available)
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

    results = {
        'wrapper': wrapper,
        'clf': clf,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'cm': cm,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'labels': labels,
        'y_score': y_score,
    }

    progress.step("Training complete; returning results")
    return results

def create_visualizations(results: dict, output_path: str | None = None, viz_out: str = VIZOUT):
    """Given training results, create visualization Graph objects and return them.

    This function builds the standard diagnostic plots using the
    `visualization` module and returns a dict of Graph wrappers keyed by id.
    """
    wrapper = results['wrapper']
    clf = results['clf']
    X = pd.concat([results['X_train'], results['X_test']], axis=0)
    y = pd.concat([results['y_train'], results['y_test']], axis=0)
    y_test = results['y_test']
    y_pred = results['y_pred']
    cm = results['cm']
    train_acc = results['train_acc']
    test_acc = results['test_acc']
    labels = results['labels']
    y_score = results['y_score']

    progress.stage("visualization", "Creating visualization Graph objects")

    graphs = {}

    # Confusion matrix (visualization functions already return Graph)
    g = visualization.plot_confusion_matrix(cm, labels=labels)
    graphs['confusion_matrix'] = g

    # Compute cross-validation scores for diagnostics
    progress.step("Computing cross-validation scores")
    scores = visualization.compute_cv_scores(clf, X, y, cv=10)
    g = visualization.plot_cv_scores(scores)
    graphs['cv_scores'] = g

    progress.step(f"y_score preview: shape={np.asarray(y_score).shape}, dtype={np.asarray(y_score).dtype}")
    try:
        sample_vals = np.asarray(y_score)[:8]
    except Exception:
        sample_vals = str(type(y_score))
    progress.step(f"y_score sample={sample_vals}")

    # ROC AUC
    g = visualization.plot_roc_auc(y_test, y_score)
    graphs['roc_auc'] = g

    # PR curve
    g = visualization.plot_pr_auc(y_test, y_score)
    graphs['pr_auc'] = g

    # True positives vs others
    g = visualization.plot_truepositives_vs_others(y_test, y_pred)
    graphs['tp_vs_others'] = g

    # Train vs test accuracy
    progress.step("Creating train/test accuracy plot")
    g = visualization.plot_train_test_accuracy(float(train_acc), float(test_acc))
    graphs['train_test_accuracy'] = g

    return graphs


def compute_and_save_metrics(clf, X_train, X_test, y_train, y_test, viz_out: str = VIZOUT, compute_permutation: bool = False, random_state: Optional[int] = None):
    """Compute a comprehensive set of evaluation metrics and save them to disk.

    This helper prefers the project's `metrics.evaluate_model` function when
    available (it may compute additional aggregates). If that module isn't
    importable, it falls back to computing a reasonable set of metrics using
    scikit-learn. The returned object is always a plain dict suitable for
    JSON serialization (numpy arrays are converted where necessary).

    Args:
        clf: trained classifier (has predict / predict_proba as applicable)
        X_train, X_test, y_train, y_test: data splits
        viz_out: directory to write `metrics.json`
        compute_permutation: whether to compute permutation importances

    Returns:
        dict of metrics
    """
    progress.stage("evaluation", "Computing aggregated evaluation metrics")

    # Try to use project's evaluate_model if present
    try:
        from metrics import evaluate_model
        metrics = evaluate_model(clf, X_train, X_test, y_train, y_test, compute_permutation=compute_permutation)
    except Exception:
        # Fallback: compute a set of common metrics using sklearn
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, brier_score_loss
        from sklearn.metrics import cohen_kappa_score
        try:
            from sklearn.metrics import matthews_corrcoef as mcc
        except Exception:
            def mcc(y_true, y_pred):
                try:
                    from sklearn.metrics import matthews_corrcoef as _mcc
                    return _mcc(y_true, y_pred)
                except Exception:
                    return None

        y_pred_test = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)

        metrics = {}
        try:
            metrics['accuracy'] = float(accuracy_score(y_test, y_pred_test))
            metrics['train_accuracy'] = float(accuracy_score(y_train, y_pred_train))
        except Exception:
            pass

        try:
            metrics['balanced_accuracy'] = float(balanced_accuracy_score(y_test, y_pred_test))
        except Exception:
            metrics['balanced_accuracy'] = None

        # Probabilistic metrics if available
        try:
            if hasattr(clf, 'predict_proba'):
                proba = clf.predict_proba(X_test)
                # if binary or multiclass, compute log_loss where sensible
                metrics['log_loss'] = float(log_loss(y_test, proba))
                # brier score only for binary; attempt and fallback
                try:
                    metrics['brier_score'] = float(brier_score_loss(y_test, proba[:, 1]))
                except Exception:
                    metrics['brier_score'] = None
            else:
                metrics['log_loss'] = None
                metrics['brier_score'] = None
        except Exception:
            metrics['log_loss'] = None
            metrics['brier_score'] = None

        try:
            metrics['cohen_kappa'] = float(cohen_kappa_score(y_test, y_pred_test))
        except Exception:
            metrics['cohen_kappa'] = None

        try:
            metrics['mcc'] = float(mcc(y_test, y_pred_test))
        except Exception:
            metrics['mcc'] = None

        # Add confusion matrix and classification report for completeness
        try:
            from sklearn.metrics import confusion_matrix, classification_report
            metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred_test).tolist()
            metrics['classification_report'] = classification_report(y_test, y_pred_test, output_dict=True)
        except Exception:
            pass

        # Optionally compute permutation importances (may be slow)
        if compute_permutation:
            try:
                from sklearn.inspection import permutation_importance as _perm
                rs = RANDOM_STATE if random_state is None else random_state
                perm = _perm(clf, X_test, y_test, n_repeats=30, random_state=rs)
                metrics['permutation_importances'] = {
                    'importances_mean': perm.importances_mean.tolist(),
                    'importances_std': perm.importances_std.tolist(),
                }
            except Exception:
                metrics['permutation_importances'] = None

    return metrics