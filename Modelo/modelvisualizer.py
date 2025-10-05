import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import train_test_split

FEATURE_PATH = "data/non_candidates_processed_features.csv"
LABEL_PATH = "data/non_candidates_processed_labels.csv"
MODELIN = "GOAT.joblib"
VIZOUT = "viz_out"
TEST_SIZE = 0.4
RANDOM_STATE = 420


def _resolve_path(rel_path: str) -> str:
	"""Resolve a path relative to this file's directory."""
	base = os.path.dirname(__file__)
	return os.path.join(base, rel_path)


def compute_and_save_importances(model_path: str = MODELIN,
								 feature_csv: str = FEATURE_PATH,
								 label_csv: str = LABEL_PATH,
								 out_dir: str = VIZOUT,
								 test_size: float = TEST_SIZE,
								 random_state: int = RANDOM_STATE):
	"""Load model and data, compute feature importances (model and permutation),
	save CSVs and plots to `out_dir`, and print top features.

	Outputs:
	- feature_importances_model.csv / .png (if model exposes feature_importances_)
	- feature_importances_permutation.csv / .png
	"""
	model_abspath = _resolve_path(model_path)
	features_abspath = _resolve_path(feature_csv)
	labels_abspath = _resolve_path(label_csv)
	out_dir_abspath = _resolve_path(out_dir)
	os.makedirs(out_dir_abspath, exist_ok=True)

	# Load data (assume rows correspond; files don't include a shared index)
	if not os.path.exists(features_abspath) or not os.path.exists(labels_abspath):
		raise FileNotFoundError(f"Feature or label file not found: {features_abspath}, {labels_abspath}")

	X = pd.read_csv(features_abspath)  # features as columns, rows are samples
	y_df = pd.read_csv(labels_abspath)
	# If labels file has a single column, take it as the Series
	if y_df.shape[1] == 1:
		y = y_df.iloc[:, 0].squeeze()
	else:
		# If labels file contains an index column plus label, prefer the 'koi_disposition' column if present
		if 'koi_disposition' in y_df.columns:
			y = y_df['koi_disposition'].squeeze()
		else:
			# fallback: take the first column
			y = y_df.iloc[:, 0].squeeze()

	# Ensure X and y have the same number of rows (samples)
	if len(X) != len(y):
		raise ValueError(f"Feature matrix and label vector have different lengths: {len(X)} vs {len(y)}.\n"
						 "Make sure both CSVs are row-aligned or provide an index to join on.")

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

	# Load model
	if not os.path.exists(model_abspath):
		raise FileNotFoundError(f"Model file not found: {model_abspath}")
	model = joblib.load(model_abspath)

	# 1) Model-based importances (if available)
	model_importances = None
	if hasattr(model, "feature_importances_"):
		model_importances = pd.Series(model.feature_importances_, index=X.columns)
		model_importances = model_importances.sort_values(ascending=False)
		# Save CSV and plot
		csv_path = os.path.join(out_dir_abspath, "feature_importances_model.csv")
		model_importances.to_csv(csv_path, header=["importance"]) 

		plt.figure(figsize=(8, max(4, len(model_importances) * 0.2)))
		model_importances.head(30).sort_values().plot(kind='barh')
		plt.title('Model feature importances')
		plt.xlabel('Importance')
		plt.tight_layout()
		png_path = os.path.join(out_dir_abspath, "feature_importances_model.png")
		plt.savefig(png_path)
		plt.close()
		print(f"Saved model feature importances to: {csv_path} and {png_path}")
	else:
		print("Model has no attribute 'feature_importances_'. Skipping model-based importances.")

	# 2) Permutation importance (overall, using accuracy) -- more model-agnostic
	perm_result = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=random_state)
	perm_importances = pd.Series(perm_result.importances_mean, index=X.columns)
	perm_importances = perm_importances.sort_values(ascending=False)
	csv_path = os.path.join(out_dir_abspath, "feature_importances_permutation.csv")
	perm_importances.to_csv(csv_path, header=["importance"])

	plt.figure(figsize=(8, max(4, len(perm_importances) * 0.2)))
	perm_importances.head(30).sort_values().plot(kind='barh', color='C1')
	plt.title('Permutation feature importances (test set)')
	plt.xlabel('Decrease in score (mean over repeats)')
	plt.tight_layout()
	png_path = os.path.join(out_dir_abspath, "feature_importances_permutation.png")
	plt.savefig(png_path)
	plt.close()
	print(f"Saved permutation importances to: {csv_path} and {png_path}")

	# 3) Example: class-specific importance for class '2' (if present)
	unique_labels = set(y_test.unique())
	if 2 in unique_labels:
		scorer = make_scorer(recall_score, labels=[2], average='macro')
		class2_result = permutation_importance(model, X_test, y_test, scoring=scorer, n_repeats=30, random_state=random_state)
		class2_importances = pd.Series(class2_result.importances_mean, index=X.columns).sort_values(ascending=False)
		csv_path = os.path.join(out_dir_abspath, "feature_importances_class2_permutation.csv")
		class2_importances.to_csv(csv_path, header=["importance"])
		plt.figure(figsize=(8, max(4, len(class2_importances) * 0.2)))
		class2_importances.head(30).sort_values().plot(kind='barh', color='C2')
		plt.title('Permutation importances for recall of class 2')
		plt.xlabel('Decrease in recall (mean over repeats)')
		plt.tight_layout()
		png_path = os.path.join(out_dir_abspath, "feature_importances_class2_permutation.png")
		plt.savefig(png_path)
		plt.close()
		print(f"Saved class-2 permutation importances to: {csv_path} and {png_path}")

	# Print top features to console
	print('\nTop features by permutation importance:')
	print(perm_importances.head(20))
	if model_importances is not None:
		print('\nTop features by model.feature_importances_:')
		print(model_importances.head(20))


if __name__ == '__main__':
	compute_and_save_importances()
