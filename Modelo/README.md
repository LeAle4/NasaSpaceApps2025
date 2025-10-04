Model training utilities

To train a quick Random Forest using the provided data, from the repository root run:

python -m Modelo.train_random_forest

This will read `Modelo/data/parameters.csv` and `Modelo/data/labels.csv`, train a RandomForest, and save the model to `Modelo/model.joblib`.

Requirements are listed in `requirements.txt` at the repo root.
