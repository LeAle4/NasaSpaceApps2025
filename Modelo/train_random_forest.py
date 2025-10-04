"""Simple Random Forest training script.

Loads features from Modelo/data/parameters.csv and labels from Modelo/data/labels.csv,
trains a RandomForestClassifier using parameters (if provided) and saves the model to Modelo/model.joblib.
Prints basic classification metrics.
"""
from pathlib import Path
from typing import Optional

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_data(features_path: Path, labels_path: Path):
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path)

    # If labels file has a single column header, extract series
    if y.shape[1] == 1:
        y = y.iloc[:, 0]

    return X, y


def train_save_model(X, y, output_path: Path, params: Optional[dict] = None):
    params = params or {}
    clf = RandomForestClassifier(**params)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print("Classification report:\n", classification_report(y_test, y_pred, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    joblib.dump(clf, output_path)
    print(f"Saved model to {output_path}")


def main():
    root = Path(__file__).parent
    features_path = root / "data" / "parameters.csv"
    labels_path = root / "data" / "labels.csv"
    model_out = root / "model.joblib"

    X, y = load_data(features_path, labels_path)

    # Basic parameter mapping: try to read from data/parameters.csv header 'rf_n_estimators' etc if present
    # Otherwise use a small default for quick runs
    rf_params = {
        "n_estimators": 100,
        "random_state": 42,
        "n_jobs": -1,
    }

    train_save_model(X, y, model_out, rf_params)


if __name__ == "__main__":
    main()
