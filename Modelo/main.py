"""Simple Random Forest training script.

Loads features from Modelo/data/parameters.csv and labels from Modelo/data/labels.csv,
trains a RandomForestClassifier using parameters (if provided) and saves the model to Modelo/model.joblib.
Prints basic classification metrics.
"""
from typing import Optional

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

FEATURE_PATH = "data/parameters.csv"
LABEL_PATH = "data/labels.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42

RF_ESTIMATORS = 100
RF_N_JOBS = -1

def load_data(features_path: str, labels_path: str):
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path)

    # If labels file has a single column header, extract series
    if y.shape[1] == 1:
        y = y.iloc[:, 0]

    return X, y

def train_save_model(X, y, output_path: str, params: Optional[dict] = None):
    params = params or {}
    clf = RandomForestClassifier(**params)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print("Classification report:\n", classification_report(y_test, y_pred, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    joblib.dump(clf, output_path)
    print(f"Saved model to {output_path}")

def main():

    features_path = FEATURE_PATH
    labels_path = LABEL_PATH
    model_out = "model.joblib"

    X, y = load_data(features_path, labels_path)

    # Basic parameter mapping: try to read from data/parameters.csv header 'rf_n_estimators' etc if present
    # Otherwise use a small default for quick runs
    rf_params = {
        "n_estimators": RF_ESTIMATORS,
        "random_state": RANDOM_STATE,
        "n_jobs": RF_N_JOBS,
    }

    train_save_model(X, y, model_out, rf_params)

if __name__ == "__main__":
    main()
