Developer documentation — NasaSpaceApps2025
===========================================

This file documents the recent changes made to the `Modelo/` package (visualization
helpers, metrics/evaluation helpers) and how to use them. It is aimed at
developers who want to run, extend, or debug the training/evaluation pipeline.

Summary of recent changes
-------------------------
- Added/extended plotting helpers in `Modelo/visualization.py`:
  - `plot_confusion_matrix(cm, labels, ax=None)`
  - `plot_cv_scores(scores, ax=None)`
  - `plot_truepositives_vs_others(y_true, y_pred, ax=None)`
  - `plot_train_test_accuracy(train_acc, test_acc, ax=None)`
  - `plot_roc_auc(y_true, y_score, ax=None, pos_label=1, savepath=None, show=False)`
    - Supports binary and multiclass. Accepts probability matrices (preferred) or
      will attempt sensible fallbacks for discrete predictions.
  - `plot_pr_auc(y_true, y_score, ax=None, pos_label=1, savepath=None, show=False)`
    - Supports binary and multiclass PR curves; computes per-class and micro-average AP.
  - `compute_brier_score(y_true, y_prob, pos_label=1)`
    - Binary: wraps sklearn's Brier score when probabilities provided.
    - Multiclass: computes mean squared error between one-hot labels and probability matrix.
  - `plotkfold_results(results, ax=None)` — 2x2 set of per-fold metric plots.

- Added `Modelo/metrics.py` with a high-level `evaluate_model(...)` function that
  computes many numeric diagnostics (accuracy, balanced accuracy, Cohen's kappa,
  MCC, classification report, log-loss, Brier score, top-k accuracy, OOB score if
  present) and optionally computes permutation importance and a learning curve.

- `Modelo/main.py` now:
  - Adds intermediate progress messages at key stages (data load, split, train, predict, visualize, save).
  - Prefers `clf.predict_proba(X_test)` (or `decision_function`) for ROC/PR/log-loss/Brier calculations.
  - Falls back to safe one-hot conversions when only discrete predictions are available.
  - Calls `evaluate_model(...)` and writes `viz_out/metrics.json` containing aggregated metrics.

Why these changes
-----------------
- PR/ROC plotting and metrics like log-loss / Brier score require probability-like
  scores. Real-world pipelines sometimes only have discrete predictions; the code
  now handles both cases but recommends using `predict_proba` when available.
- `evaluate_model` centralizes many commonly used diagnostics so you can produce
  a compact metrics report for experiments or CI.

Files modified/added
--------------------
- Modified: `Modelo/visualization.py` — added ROC/PR/Brier utilities and resilient input handling.
- Added: `Modelo/metrics.py` — evaluate_model and docstring with usage examples.
- Modified: `Modelo/main.py` — added progress checkpoints, probability handling, calls to visualization helpers, and call to `evaluate_model` (saves `viz_out/metrics.json`).
- Added: `README.md` and this `DOCUMENTATION.md` (developer-facing notes).

API reference (short)
---------------------
- visualization.plot_roc_auc(y_true, y_score, ax=None, pos_label=1, savepath=None, show=False)
  - y_score: prefer `predict_proba(X)` (n_samples, n_classes) or a 1D array of positive-class scores.
  - Returns: matplotlib Axes with ROC plot. For multiclass, plots each class + micro-average.

- visualization.plot_pr_auc(y_true, y_score, ax=None, pos_label=1, savepath=None, show=False)
  - Accepts same score shapes as ROC helper. Plots per-class PR curves and a micro-average curve.

- visualization.compute_brier_score(y_true, y_prob, pos_label=1)
  - Binary: pass 1-D positive-class probabilities or a 2-D probability matrix.
  - Multiclass: pass (n_samples, n_classes) probability matrix.
  - Returns float (lower is better).

- metrics.evaluate_model(estimator, X_train, X_test, y_train, y_test, *, top_k=3, compute_permutation=True, n_repeats=10, random_state=42, compute_learning_curve=False, cv_for_learning=5)
  - estimator must be a fitted classifier (or at least implement predict/predict_proba).
  - Returns dict with keys: accuracy, confusion_matrix, classification_report (dict), balanced_accuracy, cohen_kappa, mcc, log_loss, brier_score, top_k_accuracy, oob_score (if available), permutation_importance (if requested), learning_curve (if requested).

Examples
--------
1) Training & evaluation via `main.py` (recommended):

```powershell
python -m Modelo.main
```

This will create `viz_out/` and write visualization PNGs and `viz_out/metrics.json`.

2) Quick programmatic evaluation (from Python):

```python
from Modelo.metrics import evaluate_model
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10, random_state=0)
clf.fit(X_train, y_train)
metrics = evaluate_model(clf, X_train, X_test, y_train, y_test, compute_permutation=False)
print(metrics['accuracy'])
```

3) Plot ROC/PR with probabilities:

```python
from Modelo import visualization
probs = clf.predict_proba(X_test)
visualization.plot_roc_auc(y_test, probs, show=True)
visualization.plot_pr_auc(y_test, probs, show=True)
```

Troubleshooting (common issues)
------------------------------
- Error: "Expected 2D array, got 1D array instead" when calling `average_precision_score` or `precision_recall_curve`.
  - Cause: discrete labels passed where probabilities were expected.
  - Fix: pass `clf.predict_proba(X_test)` or convert discrete predictions to a 2-D one-hot matrix using `sklearn.preprocessing.label_binarize`.

- AUC/PR curves look odd (step functions or flat): likely due to using discrete predictions instead of continuous probabilities. Use probabilistic outputs when possible.

- Permutation importance is slow on large feature sets — reduce `n_repeats` or compute on a smaller validation subset.

Next steps / suggestions
-----------------------
- Run the linter / type checker to resolve remaining Pylance type warnings (some helper functions use dynamic shapes which can trigger static analyzer warnings).
- Add unit tests for: shape handling in plotting functions, evaluate_model outputs on synthetic datasets, and basic CLI invocation of `main.py`.
- Consider an example Jupyter notebook (`examples/run_training.ipynb`) that runs a quick end-to-end pipeline and displays figures inline.

If you want I can:
- Run static checks and fix any trivial lint/type issues now (up to 3 quick fixes),
- Add the example notebook, or
- Add unit tests for the new helpers.


