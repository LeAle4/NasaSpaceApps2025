"""Lightweight pipeline runner for training and evaluating the RandomForest model.

This script wires together the Model wrapper, training helpers and metrics
utilities already present in `analysis/model.py` and `analysis/metrics.py`.

It is intentionally minimal: it loads feature and label CSVs from the module
default locations, trains a model, saves the trained estimator to disk, runs
evaluation, writes `metrics.json` into the visualization output directory, and
prints a short summary to stdout.

Usage: run this file as a script from the repository root, e.g.:
	python -m analysis.test

The script avoids heavy plotting and long-running permutation importance by
default; these can be toggled with the `compute_permutation` flag below.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd

import visualization
from model import train_save_model, create_visualizations, compute_and_save_metrics
from metrics import evaluate_model

FEATURE_PATH = "data/non_candidates_processed_features.csv"
LABEL_PATH = "data/non_candidates_processed_labels.csv"
MODELOUT = "modeltest.joblib"
VIZOUT = "viz_out"
TEST_SIZE = 0.4
RANDOM_STATE = 420
KFOLDDIV = 10  # number of folds for cross-validation

RF_ESTIMATORS = 100
RF_N_JOBS = -1

def _resolve(path: str) -> str:
	base = Path(__file__).parent
	return str((base / path).resolve())


def run_pipeline(
	feature_csv: str = FEATURE_PATH,
	label_csv: str = LABEL_PATH,
	model_out: str = MODELOUT,
	viz_out: str = VIZOUT,
	compute_permutation: bool = False,
	random_state: Optional[int] = RANDOM_STATE,
):
	# Prepare paths
	feat = _resolve(feature_csv)
	lab = _resolve(label_csv)
	out_model = _resolve(model_out)
	out_viz = _resolve(viz_out)
	os.makedirs(out_viz, exist_ok=True)

	print(f"Loading features from: {feat}")
	X = pd.read_csv(feat)
	print(f"Loading labels from: {lab}")
	y_df = pd.read_csv(lab)
	if y_df.shape[1] == 1:
		y = y_df.iloc[:, 0].squeeze()
	else:
		if 'koi_disposition' in y_df.columns:
			y = y_df['koi_disposition'].squeeze()
		else:
			y = y_df.iloc[:, 0].squeeze()

	print(f"Starting training on {len(X)} samples and {X.shape[1]} features")
	results = train_save_model(X, y, params = {"n_estimators": RF_ESTIMATORS, "n_jobs": RF_N_JOBS}, random_state=random_state)

	# Save the model
	print(f"Saving trained model to: {out_model}")
	wrapper = results['wrapper']

	wrapper.save(out_model)


	# Compute metrics (use project's evaluate_model if available)
	print("Computing metrics (this may compute permutation importances if enabled)...")
	try:
		metrics = evaluate_model(results['clf'], results['X_train'], results['X_test'], results['y_train'], results['y_test'], compute_permutation=compute_permutation)
	except Exception:
		# fallback to compute_and_save_metrics from model module if present
		metrics = compute_and_save_metrics(results['clf'], results['X_train'], results['X_test'], results['y_train'], results['y_test'], viz_out=out_viz, compute_permutation=compute_permutation, random_state=random_state)

	metrics_path = os.path.join(out_viz, "metrics.json")
	with open(metrics_path, "w", encoding="utf-8") as fh:
		json.dump(metrics, fh, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o), indent=2)
	print(f"Saved metrics to: {metrics_path}")

	# Create visualizations (the function returns Graph objects; many implementations
	# write files themselves or return matplotlib objects)
	try:
		print("Creating visualizations...")
		graphs = create_visualizations(results)
		visualization.save_graphs(graphs, viz_out)
		print(f"Created {len(graphs)} visualization graph objects (if any).")
	except Exception as e:
		print(f"Visualization generation failed: {e}")

	print("Pipeline complete. Summary:")
	print(f"  model: {out_model}")
	print(f"  metrics: {metrics_path}")


if __name__ == '__main__':
	# Lightweight smoke-run: don't compute expensive permutation by default
	run_pipeline(compute_permutation=True)

