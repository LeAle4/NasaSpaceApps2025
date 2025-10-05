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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Progress helpers: print timestamped stage/step messages
import progress

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
        """Return predicted class labels for X.

        Parameters
        ----------
        X : array-like
            Samples to predict.

        Returns
        -------
        array-like
            Predicted labels produced by the underlying estimator.
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise AttributeError("Underlying estimator has no 'predict_proba' method")

    def decision_function(self, X):
        """Call the underlying estimator's decision_function if available.

        Many scikit-learn estimators provide `decision_function` for scoring
        (e.g., SVMs). If the wrapped estimator doesn't implement it this
        method raises AttributeError to make the absence explicit.
        """
        df = getattr(self.model, "decision_function", None)
        if callable(df):
            return df(X)
        raise AttributeError("Underlying estimator has no 'decision_function' method")

    def score(self, X, y):
        """Return the default estimator score on the given test data and labels.

        This delegates to the underlying estimator's `score` method which
        commonly returns accuracy for classifiers.
        """
        return self.model.score(X, y)

    def get_scores(self, X):
        """Return the best available score representation for plotting:
        prefer predict_proba -> decision_function -> raw predictions.
        This centralizes the logic used multiple times in the module.
        """
        if hasattr(self.model, "predict_proba"):
            try:
                return self.model.predict_proba(X)
            except Exception:
                pass
        df = getattr(self.model, "decision_function", None)
        if callable(df):
            try:
                return df(X)
            except Exception:
                pass
        # Fallback: return hard labels
        return self.model.predict(X)

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

def train_save_model(X, y, params: Optional[dict] = None, random_state: Optional[int] = None, test_size: float = 0.4):
    """Train a RandomForest model and return a results dict.

    This function only performs the data split and fitting. It returns a
    dictionary with the trained wrapper, train/test splits, predictions and
    basic metrics so callers can separately visualize or persist artifacts.

    Returns:
        dict containing keys: wrapper, clf, X_train, X_test, y_train, y_test,
        y_pred, cm, train_acc, test_acc, labels, y_score
    """
    # Parameters (brief):
    # - X, y: feature matrix and label vector (pandas or numpy)
    # - params: parameters forwarded to RandomForestClassifier (e.g. n_estimators)
    # - random_state: integer seed used for splitting and model initialization
    # - test_size: fraction of data to reserve for testing
    params = params or {}

    progress.stage("training", "Starting model training")
    # Show params at higher verbosity as structured details
    progress.step("RandomForest initialization", verbosity=2, details={"params": params})

    # Create Model wrapper and perform stratified train/test split
    wrapper = Model(params=params, random_state=random_state)

    # Resolve random state used for splitting and logging
    rs = 42 if random_state is None else random_state
    progress.step("Splitting data into train and test sets", verbosity=1, details={"test_size": test_size, "random_state": rs})
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=rs, stratify=y
    )
    # Report resulting split shapes and label distribution at verbose level
    try:
        y_train_dist = dict(zip(*np.unique(y_train, return_counts=True)))
    except Exception:
        y_train_dist = None
    progress.step("Train/test split complete", verbosity=2, details={"X_train_shape": getattr(X_train, 'shape', None), "X_test_shape": getattr(X_test, 'shape', None), "y_train_distribution": y_train_dist})

    progress.step("Fitting RandomForest on training data", verbosity=1)
    wrapper.fit(X_train, y_train)
    clf = wrapper.model
    # After training, report model summary and OOB (if present) at verbose level
    try:
        model_oob = getattr(clf, 'oob_score_', None)
    except Exception:
        model_oob = None
    progress.step("Model fitted", verbosity=2, details={"oob_score": model_oob, "n_estimators": wrapper.get_params().get('n_estimators')})

    progress.step("Predicting on test set and computing metrics", verbosity=1)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    test_acc = accuracy_score(y_test, y_pred)
    train_acc = accuracy_score(y_train, clf.predict(X_train))

    # Labels for plotting: unique values present in true/pred
    labels = [str(x) for x in np.unique(np.concatenate([y_test, y_pred]))]

    # Score object used by ROC/PR plotting (probabilities or decision scores)
    y_score = wrapper.get_scores(X_test)
    # Provide a brief preview of the score object at verbose level
    try:
        score_shape = np.asarray(y_score).shape
        score_dtype = str(np.asarray(y_score).dtype)
    except Exception:
        score_shape = None
        score_dtype = None
    progress.step("Prepared score object for ROC/PR plotting", verbosity=2, details={"score_shape": score_shape, "score_dtype": score_dtype})

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

def create_visualizations(results: dict, random_state: Optional[int] = None):
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
    progress.step("Preparing combined data for visualizations", verbosity=2, details={"X_shape": getattr(X, 'shape', None), "y_shape": getattr(y, 'shape', None)})

    graphs = {}

    # Confusion matrix
    graphs['confusion_matrix'] = visualization.plot_confusion_matrix(cm, labels=labels)

    # Cross-validation scores (simple diagnostic)
    progress.step("Computing cross-validation scores", verbosity=1, details={"cv": 10})
    scores = visualization.compute_cv_scores(clf, X, y, cv=10)
    graphs['cv_scores'] = visualization.plot_cv_scores(scores)

    # K-fold detailed per-metric results and plot
    try:
        progress.step("Computing per-fold metrics (k-fold)", verbosity=1)
        from metrics import ten_fold_cross_validation
        rs_k = 42 if random_state is None else random_state
        kfold_results = ten_fold_cross_validation(clf, X, y, n_splits=10, random_state=rs_k)
        kf_graph, _ = visualization.plotkfold_results(kfold_results)
        graphs['kfold_results'] = kf_graph
        # Summarize k-fold means at verbose level
        try:
            kf_summary = {k: float(np.mean(v)) for k, v in kfold_results.items()}
        except Exception:
            kf_summary = None
        progress.step("K-fold metrics computed", verbosity=2, details={"kfold_summary": kf_summary})
    except Exception as e:
        progress.step(f"Skipping k-fold detailed plot: {e}", verbosity=1)

    # ROC and PR curves
    progress.step("Creating ROC AUC plot", verbosity=1)
    graphs['roc_auc'] = visualization.plot_roc_auc(y_test, y_score)
    progress.step("Creating Precision-Recall plot", verbosity=1)
    graphs['pr_auc'] = visualization.plot_pr_auc(y_test, y_score)

    # True positives vs others and train/test accuracy
    progress.step("Creating true-positives vs others plot", verbosity=1)
    graphs['tp_vs_others'] = visualization.plot_truepositives_vs_others(y_test, y_pred)
    progress.step("Creating train/test accuracy plot", verbosity=1, details={"train_acc": float(train_acc), "test_acc": float(test_acc)})
    graphs['train_test_accuracy'] = visualization.plot_train_test_accuracy(float(train_acc), float(test_acc))

    # Permutation importances (quick mode) - keep optional and non-fatal
    try:
        progress.step("Computing permutation importances (quick)", verbosity=1, details={"n_repeats": 10, "n_features": getattr(results['X_test'], 'shape', (None, None))[1] if hasattr(results['X_test'], 'shape') else None})
        perm_res = wrapper.permutation_importance(results['X_test'], results['y_test'], n_repeats=10, random_state=random_state)
        perm_mean = getattr(perm_res, 'importances_mean', None)
        if perm_mean is not None:
            graphs['permutation_importance'] = visualization.plot_permutation_importance(perm_mean, results['X_test'].columns, top_n=30)
    except Exception as e:
        progress.step(f"Skipping permutation importance plot: {e}", verbosity=1)

    return graphs


def compute_metrics(clf, X_train, X_test, y_train, y_test, compute_permutation: bool = False, random_state: Optional[int] = None):
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
    progress.step("Starting metrics aggregation", verbosity=1)

    # Prefer project-level evaluate_model if available (keeps output consistent)
    try:
        from metrics import evaluate_model
        progress.step("Using project evaluate_model implementation", verbosity=2)
        return evaluate_model(clf, X_train, X_test, y_train, y_test, compute_permutation=compute_permutation)
    except Exception:
        progress.step("Project evaluate_model not available; using fallback sklearn-based metrics", verbosity=1)
        pass

    # Fallback: core metrics computed with sklearn (kept concise)
    from sklearn.metrics import balanced_accuracy_score, log_loss, brier_score_loss, cohen_kappa_score
    try:
        from sklearn.metrics import matthews_corrcoef as _mcc
    except Exception:
        _mcc = None

    y_pred_test = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)

    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred_test)),
        'train_accuracy': float(accuracy_score(y_train, y_pred_train)),
        'balanced_accuracy': float(balanced_accuracy_score(y_test, y_pred_test)) if len(np.unique(y_test)) > 1 else None,
    }

    # Probabilistic metrics when available
    if hasattr(clf, 'predict_proba'):
        try:
            proba = clf.predict_proba(X_test)
            metrics['log_loss'] = float(log_loss(y_test, proba))
            # brier for binary problems only
            try:
                metrics['brier_score'] = float(brier_score_loss(y_test, proba[:, 1]))
            except Exception:
                metrics['brier_score'] = None
        except Exception:
            metrics['log_loss'] = None
            metrics['brier_score'] = None
    else:
        metrics['log_loss'] = None
        metrics['brier_score'] = None

    metrics['cohen_kappa'] = float(cohen_kappa_score(y_test, y_pred_test)) if len(np.unique(y_test)) > 1 else None
    if _mcc is not None:
        try:
            metrics['mcc'] = float(_mcc(y_test, y_pred_test))
        except Exception:
            metrics['mcc'] = None
    else:
        metrics['mcc'] = None

    # Confusion matrix and classification report
    try:
        from sklearn.metrics import confusion_matrix, classification_report
        metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred_test).tolist()
        metrics['classification_report'] = classification_report(y_test, y_pred_test, output_dict=True)
    except Exception:
        pass

    # Optional permutation importances (may be slow) - convert numpy -> list for JSON
    if compute_permutation:
        try:
            rs = 42 if random_state is None else random_state
            progress.step("Computing permutation importances for metrics", verbosity=2, details={"n_repeats": 30})
            perm = _perm(clf, X_test, y_test, n_repeats=30, random_state=rs)
            imp_mean = getattr(perm, 'importances_mean', None)
            imp_std = getattr(perm, 'importances_std', None)
            metrics['permutation_importances'] = {
                'importances_mean': imp_mean.tolist() if imp_mean is not None else None,
                'importances_std': imp_std.tolist() if imp_std is not None else None,
            }
        except Exception:
            metrics['permutation_importances'] = None

    return metrics