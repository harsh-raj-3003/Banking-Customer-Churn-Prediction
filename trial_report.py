"""Confusion report for the ten best trials of studies 2 and 3.

Run it with:
    PYTHONPATH=src .venv/Scripts/python.exe -m bank_churn.evaluation.trial_report
    PYTHONPATH=src .venv/Scripts/python.exe -m bank_churn.evaluation.trial_report --n-estimators 700

What it does
------------
1. Takes the top 5 trials of study 2 (13 tuned parameters) and the top 5 of study 3
   (8 tuned + 5 fixed), with the parameters recorded in FINDINGS.md.
2. Sets `n_estimators = 500` for all ten, as requested.
3. Scores each configuration with 5-fold stratified CV on the training rows, takes the
   cut-off from the out-of-fold F2, then scores the test rows once.
4. Reports recall, precision, F2, ROC-AUC, PR-AUC, the confusion matrix and the business cost
   (4 * FN + FP) for every configuration.
5. Adds three changes that should improve on the single best trial:
   - **isotonic calibration** on the out-of-fold scores, so a cut-off means the same thing
     across configurations (the studies drifted between 0.19 and 0.24);
   - an **ensemble** of the ten configurations (mean probability), because the top of the
     search space is flat and averaging usually beats picking one;
   - a **cost column** plus metrics at a fixed 0.30 cut-off, since the business decision is a
     cost, not an F2 value.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Temporary comparison script: run it from the project root with
#     .venv/Scripts/python.exe trial_report.py
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import argparse
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from bank_churn.config.settings import (
    DECISION_THRESHOLD,
    PROCESSED_DATA_PATH,
    PROJECT_ROOT,
    SEED,
    TARGET,
    TEST_SIZE,
)
from bank_churn.evaluation.metrics import best_fbeta, cost_of, metrics_at, ranking_metrics

N_ESTIMATORS = 500
N_SPLITS = 5

# study 3 fixed five parameters; they are merged into every study-3 configuration so all ten
# configurations carry the full 13 parameters.
STUDY3_FIXED = {
    "colsample_bylevel": 0.8854,
    "colsample_bynode": 0.8725,
    "reg_lambda": 0.0085,
    "max_delta_step": 7.7363,
}


@dataclass
class TrialConfig:
    """One set of hyperparameters taken from a study."""

    name: str
    study: str
    cv_f2: float
    params: dict = field(default_factory=dict)


TOP_TRIALS = [
    # ---- study 2, full 13-parameter space
    TrialConfig("s2-t19", "study 2", 0.7098, {
        "learning_rate": 0.0403, "max_depth": 6, "min_child_weight": 3, "gamma": 3.8734,
        "subsample": 0.9433, "colsample_bytree": 0.9196, "colsample_bylevel": 0.8854,
        "colsample_bynode": 0.8725, "reg_alpha": 0.4127, "reg_lambda": 0.0085,
        "scale_pos_weight": 1.4485, "max_delta_step": 7.7363,
    }),
    TrialConfig("s2-t25", "study 2", 0.7096, {
        "learning_rate": 0.0437, "max_depth": 6, "min_child_weight": 3, "gamma": 4.0971,
        "subsample": 0.8524, "colsample_bytree": 0.8950, "colsample_bylevel": 0.9473,
        "colsample_bynode": 0.8062, "reg_alpha": 0.2724, "reg_lambda": 0.0077,
        "scale_pos_weight": 1.0394, "max_delta_step": 8.0381,
    }),
    TrialConfig("s2-t12", "study 2", 0.7095, {
        "learning_rate": 0.0437, "max_depth": 6, "min_child_weight": 4, "gamma": 3.2864,
        "subsample": 0.8059, "colsample_bytree": 0.9169, "colsample_bylevel": 0.8684,
        "colsample_bynode": 0.9034, "reg_alpha": 0.7750, "reg_lambda": 0.0359,
        "scale_pos_weight": 1.3668, "max_delta_step": 8.5740,
    }),
    TrialConfig("s2-t14", "study 2", 0.7094, {
        "learning_rate": 0.0321, "max_depth": 5, "min_child_weight": 6, "gamma": 3.4791,
        "subsample": 0.6582, "colsample_bytree": 0.8124, "colsample_bylevel": 0.7807,
        "colsample_bynode": 0.7409, "reg_alpha": 1.9555, "reg_lambda": 0.0083,
        "scale_pos_weight": 2.5949, "max_delta_step": 9.1371,
    }),
    TrialConfig("s2-t24", "study 2", 0.7091, {
        "learning_rate": 0.0412, "max_depth": 7, "min_child_weight": 4, "gamma": 3.8079,
        "subsample": 0.8663, "colsample_bytree": 0.7900, "colsample_bylevel": 0.9739,
        "colsample_bynode": 0.9229, "reg_alpha": 2.0770, "reg_lambda": 0.0282,
        "scale_pos_weight": 1.8943, "max_delta_step": 6.6788,
    }),
    # ---- study 3, reduced space
    TrialConfig("s3-t28", "study 3", 0.7111, {
        **STUDY3_FIXED, "learning_rate": 0.0133, "max_depth": 5, "min_child_weight": 9,
        "gamma": 0.1675, "subsample": 0.8356, "colsample_bytree": 0.7106,
        "reg_alpha": 0.0015, "scale_pos_weight": 1.8081,
    }),
    TrialConfig("s3-t21", "study 3", 0.7097, {
        **STUDY3_FIXED, "learning_rate": 0.0133, "max_depth": 5, "min_child_weight": 11,
        "gamma": 1.2208, "subsample": 0.9539, "colsample_bytree": 0.6966,
        "reg_alpha": 1.3655, "scale_pos_weight": 2.0998,
    }),
    TrialConfig("s3-t25", "study 3", 0.7096, {
        **STUDY3_FIXED, "learning_rate": 0.0131, "max_depth": 6, "min_child_weight": 13,
        "gamma": 0.2691, "subsample": 0.9293, "colsample_bytree": 0.6868,
        "reg_alpha": 0.0200, "scale_pos_weight": 2.9099,
    }),
    TrialConfig("s3-t2", "study 3", 0.7089, {
        **STUDY3_FIXED, "learning_rate": 0.0281, "max_depth": 4, "min_child_weight": 13,
        "gamma": 1.8318, "subsample": 0.6461, "colsample_bytree": 0.7149,
        "reg_alpha": 0.0036, "scale_pos_weight": 4.8875,
    }),
    TrialConfig("s3-t6", "study 3", 0.7089, {
        **STUDY3_FIXED, "learning_rate": 0.0642, "max_depth": 8, "min_child_weight": 19,
        "gamma": 4.6094, "subsample": 0.7989, "colsample_bytree": 0.5109,
        "reg_alpha": 3.7959, "scale_pos_weight": 9.7263,
    }),
]


def load_split():
    """Processed data, split exactly as in the notebooks."""
    frame = pd.read_csv(PROCESSED_DATA_PATH)
    X = frame.drop(columns=[TARGET])
    y = frame[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)


def make_model(params: dict, n_estimators: int | None = None) -> XGBClassifier:
    """XGBoost with the trial parameters and the requested number of trees."""
    trees = N_ESTIMATORS if n_estimators is None else n_estimators
    return XGBClassifier(
        **{**params, "n_estimators": trees},
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
    )


def out_of_fold_proba(params: dict, X, y, n_splits: int | None = None) -> np.ndarray:
    """Out-of-fold churn probabilities for the training rows."""
    splits = N_SPLITS if n_splits is None else n_splits
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for tr_idx, va_idx in cv.split(X, y):
        model = make_model(params)
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
    return oof


def evaluate(config: TrialConfig, X_train, X_test, y_train, y_test, calibrate: bool = True):
    """Score one configuration and return its report rows plus the probabilities."""
    oof = out_of_fold_proba(config.params, X_train, y_train)
    model = make_model(config.params)
    model.fit(X_train, y_train)
    test_proba = model.predict_proba(X_test)[:, 1]

    f2_oof, threshold = best_fbeta(y_train, oof)
    rows = [{
        "config": config.name, "study": config.study, "variant": "raw",
        "thr": round(threshold, 4), "oof_f2": round(f2_oof, 4),
        **metrics_at(y_test, test_proba, threshold),
        **ranking_metrics(y_test, test_proba),
        "cost": cost_of(y_test, test_proba, threshold),
        "cost_at_0.30": cost_of(y_test, test_proba, DECISION_THRESHOLD),
    }]
    probabilities = {"raw": (oof, test_proba, threshold)}

    if calibrate:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(oof, y_train)
        oof_cal, test_cal = iso.predict(oof), iso.predict(test_proba)
        f2_oof_cal, threshold_cal = best_fbeta(y_train, oof_cal)
        rows.append({
            "config": config.name, "study": config.study, "variant": "calibrated",
            "thr": round(threshold_cal, 4), "oof_f2": round(f2_oof_cal, 4),
            **metrics_at(y_test, test_cal, threshold_cal),
            **ranking_metrics(y_test, test_proba),  # calibration preserves the ranking
            "cost": cost_of(y_test, test_cal, threshold_cal),
            "cost_at_0.30": cost_of(y_test, test_cal, DECISION_THRESHOLD),
        })
        probabilities["calibrated"] = (oof_cal, test_cal, threshold_cal)

    return rows, probabilities


def main() -> None:
    global N_ESTIMATORS, N_SPLITS

    parser = argparse.ArgumentParser(description="Compare the top trials of studies 2 and 3")
    parser.add_argument("--n-estimators", type=int, default=N_ESTIMATORS)
    parser.add_argument("--folds", type=int, default=N_SPLITS)
    parser.add_argument("--no-calibration", action="store_true", help="report raw scores only")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "artifacts" / "trial_report.csv"))
    args = parser.parse_args()

    N_ESTIMATORS = args.n_estimators
    N_SPLITS = args.folds
    calibrate = not args.no_calibration

    X_train, X_test, y_train, y_test = load_split()
    print(f"train {X_train.shape} | test {X_test.shape} | n_estimators = {N_ESTIMATORS} | "
          f"folds = {N_SPLITS}")
    print(f"churn rate: train {y_train.mean():.4f} | test {y_test.mean():.4f}")

    rows, probs = [], {}
    for config in TOP_TRIALS:
        config_rows, probabilities = evaluate(
            config, X_train, X_test, y_train, y_test, calibrate=calibrate
        )
        rows += config_rows
        probs.update({(config.name, variant): value for variant, value in probabilities.items()})
        print(f"  {config.name} | " + " | ".join(
            f"{row['variant']} thr {row['thr']:.3f} F2 {row['f2']:.4f}" for row in config_rows
        ))

    # ---- ensemble of all ten configurations (mean probability)
    for variant in (["calibrated", "raw"] if calibrate else ["raw"]):
        oof_mean = np.mean([probs[(c.name, variant)][0] for c in TOP_TRIALS], axis=0)
        test_mean = np.mean([probs[(c.name, variant)][1] for c in TOP_TRIALS], axis=0)
        f2_oof, threshold = best_fbeta(y_train, oof_mean)
        rows.append({
            "config": "ensemble-10", "study": "all", "variant": variant,
            "thr": round(threshold, 4), "oof_f2": round(f2_oof, 4),
            **metrics_at(y_test, test_mean, threshold),
            **ranking_metrics(y_test, test_mean),
            "cost": cost_of(y_test, test_mean, threshold),
            "cost_at_0.30": cost_of(y_test, test_mean, DECISION_THRESHOLD),
        })
        probs[("ensemble-10", variant)] = (oof_mean, test_mean, threshold)

    report = pd.DataFrame(rows)
    columns = ["config", "study", "variant", "thr", "oof_f2", "recall", "precision", "f2",
               "roc_auc", "pr_auc", "tn", "fp", "fn", "tp", "cost", "cost_at_0.30"]
    report = report[columns].sort_values(["variant", "f2"], ascending=[True, False])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out, index=False)
    pd.set_option("display.width", 200)
    print("\n=== confusion report (test rows, cut-off from out-of-fold F2) ===")
    print(report.to_string(index=False))

    best = report.sort_values("f2", ascending=False).iloc[0]
    cheapest = report.sort_values("cost").iloc[0]
    print(f"\nbest F2      : {best['config']} ({best['variant']}) | cut-off {best['thr']:.3f} | "
          f"recall {best['recall']:.4f} | precision {best['precision']:.4f} | "
          f"F2 {best['f2']:.4f} | cost {int(best['cost'])}")
    print(f"lowest cost  : {cheapest['config']} ({cheapest['variant']}) | "
          f"recall {cheapest['recall']:.4f} | precision {cheapest['precision']:.4f} | "
          f"cost {int(cheapest['cost'])}")

    _, best_test, best_thr = probs[(best["config"], best["variant"])]
    print(f"\nconfusion matrix of {best['config']} ({best['variant']}) at {best_thr:.3f}:")
    print(pd.DataFrame(
        confusion_matrix(y_test, (best_test >= best_thr).astype(int), labels=[0, 1]),
        index=["actual: stayed", "actual: churned"],
        columns=["pred: stayed", "pred: churned"],
    ))
    print(classification_report(y_test, (best_test >= best_thr).astype(int), digits=3,
                                target_names=["stayed (0)", "churned (1)"]))

    print("average over configurations by variant:")
    print(report.groupby("variant")[["f2", "recall", "precision", "cost"]].mean().round(4).to_string())
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
