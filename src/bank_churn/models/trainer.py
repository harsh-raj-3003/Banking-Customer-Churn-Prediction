"""Train, evaluate and export the final churn model.

Model choice (from FINDINGS.md)
-------------------------------
Study 3, trial 28: the best CV F2 in the reduced space (0.7111) and test F2 0.7000 at its
`n_estimators` of 400. The ten-trial comparison showed that 500 trees made both F2 and recall
worse, so 400 is pinned here.

Pipeline
--------
1. stratified 80/20 split (seed 42) - used only to measure the model honestly;
2. out-of-fold F2 on the training rows gives the cut-off;
3. test rows score the model once, for the metadata;
4. the exported model is refit on all 10,000 rows, which is what deployment needs.

Usage:
    PYTHONPATH=src .venv/Scripts/python.exe -m bank_churn.models.trainer
    PYTHONPATH=src .venv/Scripts/python.exe -m bank_churn.models.trainer --n-estimators 400
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from ..config.settings import (
    DECISION_THRESHOLD,
    FEATURE_ORDER,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    PROJECT_ROOT,
    SEED,
    TARGET,
    TEST_SIZE,
)
from ..data.loader import load_processed
from ..evaluation.metrics import best_fbeta, cost_of, metrics_at, ranking_metrics

# study 3, trial 28 (8 tuned parameters + the 5 the study fixed)
FINAL_PARAMS = {
    "learning_rate": 0.0133,
    "colsample_bytree": 0.7106,
    "scale_pos_weight": 1.8081,
    "max_depth": 5,
    "min_child_weight": 9,
    "reg_alpha": 0.0015,
    "subsample": 0.8356,
    "gamma": 0.1675,
    "colsample_bylevel": 0.8854,
    "colsample_bynode": 0.8725,
    "reg_lambda": 0.0085,
    "max_delta_step": 7.7363,
}

N_ESTIMATORS = 400
N_SPLITS = 5


def make_model(params: dict, n_estimators: int = N_ESTIMATORS) -> XGBClassifier:
    """XGBoost configured for training (all cores)."""
    return XGBClassifier(
        **{**params, "n_estimators": n_estimators},
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
    )


def out_of_fold_proba(model_params: dict, X, y, n_estimators: int) -> np.ndarray:
    """Out-of-fold churn probabilities for the training rows."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for tr_idx, va_idx in cv.split(X, y):
        fold_model = make_model(model_params, n_estimators)
        fold_model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof[va_idx] = fold_model.predict_proba(X.iloc[va_idx])[:, 1]
    return oof


def train_final_model(
    model_params: dict | None = None,
    n_estimators: int = N_ESTIMATORS,
    threshold: float = DECISION_THRESHOLD,
    verbose: bool = True,
) -> tuple[XGBClassifier, dict]:
    """Fit, measure and return the deployment model plus its metadata."""
    params = FINAL_PARAMS if model_params is None else model_params
    frame = load_processed()
    X, y = frame[FEATURE_ORDER], frame[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    oof = out_of_fold_proba(params, X_train, y_train, n_estimators)
    f2_oof, threshold_f2 = best_fbeta(y_train, oof)

    evaluated = make_model(params, n_estimators)
    evaluated.fit(X_train, y_train)
    test_proba = evaluated.predict_proba(X_test)[:, 1]

    test_metrics = metrics_at(y_test, test_proba, threshold_f2)
    metadata = {
        "model_version": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "source": "study 3 trial 28 (reduced space), n_estimators=400",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_estimators": n_estimators,
        "params": {**params, "n_estimators": n_estimators},
        "feature_order": FEATURE_ORDER,
        "n_features": len(FEATURE_ORDER),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "refit_on": "all rows (10,000)",
        "seed": SEED,
        "threshold": float(threshold),
        "threshold_f2": round(float(threshold_f2), 4),
        "oof_f2": round(float(f2_oof), 4),
        "test": {**test_metrics, **ranking_metrics(y_test, test_proba),
                 "cost": cost_of(y_test, test_proba, threshold_f2)},
        "operating_point": metrics_at(y_test, test_proba, threshold),
    }

    if verbose:
        print(f"params      : {len(params)} + n_estimators={n_estimators}")
        print(f"train/test  : {len(X_train):,} / {len(X_test):,} rows")
        print(f"OOF F2      : {f2_oof:.4f} at cut-off {threshold_f2:.4f}")
        print(f"test        : ROC-AUC {metadata['test']['roc_auc']:.4f} | "
              f"PR-AUC {metadata['test']['pr_auc']:.4f}")
        print(f"test @{threshold_f2:.3f}: recall {test_metrics['recall']:.4f} | "
              f"precision {test_metrics['precision']:.4f} | "
              f"TP {test_metrics['tp']} / FN {test_metrics['fn']} / FP {test_metrics['fp']}")
        print(f"test @{threshold:.2f} (operating): recall "
              f"{metadata['operating_point']['recall']:.4f} | precision "
              f"{metadata['operating_point']['precision']:.4f}")

    deployment_model = make_model(params, n_estimators)
    deployment_model.fit(X, y)
    if verbose:
        print(f"refit       : all {len(X):,} rows (deployment model)")
    return deployment_model, metadata


def save_model(model: XGBClassifier, metadata: dict) -> tuple[Path, Path]:
    """Write the model and its metadata; returns both paths."""
    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return MODEL_PATH, MODEL_METADATA_PATH


def main() -> None:
    """Command-line entry point: train, evaluate, export."""
    parser = argparse.ArgumentParser(description="Train and export the final churn model")
    parser.add_argument("--n-estimators", type=int, default=N_ESTIMATORS)
    parser.add_argument("--threshold", type=float, default=DECISION_THRESHOLD,
                        help="operating cut-off stored in the metadata")
    args = parser.parse_args()

    model, metadata = train_final_model(
        n_estimators=args.n_estimators, threshold=args.threshold
    )
    model_path, metadata_path = save_model(model, metadata)
    print(f"saved model    : {model_path.relative_to(PROJECT_ROOT)}")
    print(f"saved metadata : {metadata_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
