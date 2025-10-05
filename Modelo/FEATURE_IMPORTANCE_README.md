Feature importance helper
=========================

This helper computes feature importances for the trained RandomForest model and
saves CSV and PNG artifacts to the `viz_out/` directory inside `Modelo`.

What it computes
- Gini importance (from RandomForest.feature_importances_) — saved as `gini_importances.csv` and `gini_importances.png`.
- Permutation importance (sklearn) — saved as `permutation_importances.csv` and `permutation_importances.png`.
- Optional SHAP values if the `shap` package is installed — saved as `shap_importances.csv` and `shap_summary.png`.
- A combined summary CSV: `feature_importances_combined.csv`.

How to run

Open a terminal in the repository root and run:

```powershell
python Modelo\feature_importance.py
```

Notes
- The script expects the trained model at `Modelo/GOAT.joblib` and the feature/label
  CSVs at `Modelo/data/non_candidates_processed_features.csv` and
  `Modelo/data/non_candidates_processed_labels.csv`. You can pass different
  paths by editing the script or importing the `main()` function.
- Permutation importance may be slow for large datasets; adjust `n_repeats` in
  `feature_importance.py` to speed it up.
- If you want SHAP explanations, install `shap` in your environment (`pip install shap`).
