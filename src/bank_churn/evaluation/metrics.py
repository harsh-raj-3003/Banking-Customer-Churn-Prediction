"""Metric helpers shared by the evaluation scripts.

Same definitions as `notebooks/tuning.ipynb`, so numbers stay comparable.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

BETA = 2.0


def fbeta_from_pr(precision: float, recall: float, beta: float = BETA) -> float:
    """F-beta from a precision/recall pair."""
    denominator = beta**2 * precision + recall
    if denominator == 0:
        return 0.0
    return (1 + beta**2) * precision * recall / denominator


def best_fbeta(y_true, proba, beta: float = BETA) -> tuple[float, float]:
    """Highest F-beta over all cut-offs, and the cut-off that reaches it."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    scores = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)
    index = int(np.argmax(np.nan_to_num(scores)))
    threshold = float(thresholds[min(index, len(thresholds) - 1)])
    return float(scores[index]), threshold


def confusion_counts(y_true, proba, threshold: float) -> dict:
    """TN / FP / FN / TP at one cut-off."""
    prediction = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(np.asarray(y_true), prediction, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def cost_of(y_true, proba, threshold: float, miss_cost: float = 4.0, alarm_cost: float = 1.0) -> float:
    """Business cost of one operating point: missed churners weigh more than false alarms."""
    counts = confusion_counts(y_true, proba, threshold)
    return miss_cost * counts["fn"] + alarm_cost * counts["fp"]


def metrics_at(y_true, proba, threshold: float, beta: float = BETA) -> dict:
    """Recall, precision, F-beta and the counts for one cut-off."""
    prediction = (np.asarray(proba) >= threshold).astype(int)
    recall = recall_score(y_true, prediction, zero_division=0)
    precision = precision_score(y_true, prediction, zero_division=0)
    return {
        "threshold": round(float(threshold), 4),
        "flagged_pct": round(float(prediction.mean()) * 100, 2),
        "recall": round(float(recall), 4),
        "precision": round(float(precision), 4),
        "f2": round(fbeta_from_pr(precision, recall, beta), 4),
        **confusion_counts(y_true, proba, threshold),
    }


def ranking_metrics(y_true, proba) -> dict:
    """Ranking quality: ROC-AUC and PR-AUC (independent of the cut-off)."""
    return {
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
    }
