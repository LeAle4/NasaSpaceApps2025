"""Simplified ensemble utilities.

This module provides a compact alternative to `ensemble.py` with a smaller API:
 - `modify_features`, `modify_labels` (simple defaults)
 - `build_simple_stack()` returns a StackingClassifier with RF, GB, MLP
 - `train_and_evaluate(X, y)` fits the stack and returns a minimal report

Designed for quick experiments and easier reading; advanced features remain in `ensemble.py`.
"""
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, recall_score, average_precision_score

def build_simple_stack() -> StackingClassifier:
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=50, random_state=42)
    mlp = Pipeline([("scaler", StandardScaler()), ("mlp", MLPClassifier(hidden_layer_sizes=(50,), max_iter=200, random_state=42))])
    estimators = [("rf", rf), ("gb", gb), ("mlp", mlp)]
    stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=500), cv=5, stack_method="predict_proba")
    return stack


def train_and_evaluate(X: pd.DataFrame, y: np.ndarray, cv10: bool = True) -> Tuple[StackingClassifier, Dict[str, Any]]:
    stack = build_simple_stack()
    # Fit to full data
    stack.fit(X, y)
    report: Dict[str, Any] = {}
    # Optionally run stratified 10-fold CV for recall (confirmed class) and average precision
    if cv10:
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        scorer_recall = make_scorer(lambda yt, yp: recall_score(yt, yp, labels=[2], average="macro"))
        # average_precision_score requires probability - use cross_validate with 'predict_proba' via custom cv loop
        recalls = []
        aps = []
        for train_idx, test_idx in skf.split(X, y):
            Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
            ytr, yte = y[train_idx], y[test_idx]
            clf = build_simple_stack()
            clf.fit(Xtr, ytr)
            ypred = clf.predict(Xte)
            recalls.append(recall_score(yte, ypred, labels=[2], average="macro") if 2 in yte else None)
            try:
                proba = np.asarray(clf.predict_proba(Xte))
                if proba.ndim > 1:
                    classes = clf.classes_
                    if 2 in classes:
                        idx = list(classes).index(2)
                        aps.append(average_precision_score((yte == 2).astype(int), proba[:, idx]))
                    else:
                        aps.append(None)
                else:
                    aps.append(None)
            except Exception:
                aps.append(None)
        report["cv10"] = {"recall_per_fold": recalls, "ap_per_fold": aps, "recall_mean": np.nanmean([r for r in recalls if r is not None]) if any(r is not None for r in recalls) else None, "ap_mean": np.nanmean([a for a in aps if a is not None]) if any(a is not None for a in aps) else None}
    return stack, report
