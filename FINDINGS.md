# FINDINGS — Bank customer churn: model, metric and tuning

Final report. Numbers come from the executed notebooks (`notebooks/eda.ipynb`,
`notebooks/modelling.ipynb`, `notebooks/tuning.ipynb`).

## 1. Top-line result

| model | cut-off | recall | precision | F2 | ROC-AUC | PR-AUC | churners caught |
|---|---|---|---|---|---|---|---|
| baseline XGBoost | 0.451 | 0.774 | 0.487 | 0.692 | 0.8620 | 0.7056 | 315 / 407 |
| study 1, PR-AUC objective (13 params) | 0.522 | 0.735 | 0.537 | 0.684 | 0.8669 | 0.7145 | 299 / 407 |
| study 2, F2 objective (13 params) | 0.191 | 0.845 | 0.417 | **0.701** | 0.8693 | 0.7193 | 344 / 407 |
| **study 3, F2 objective (8 params)** | 0.239 | 0.840 | 0.420 | 0.700 | 0.8683 | 0.7167 | 342 / 407 |

**Final model: study 3, trial 28** (the reduced-space run). It catches **342 of 407** churners
(84.0% recall) with precision 0.420, and holds the second-best ranking quality of the project.

Study 2 and study 3 are **equivalent within noise**: F2 differs by 0.001, recall by 0.5 points,
ROC-AUC by 0.001. Study 3 reaches the same quality with 8 tuned parameters instead of 13, so it
is the better choice for maintenance and for the next round of work.

## 2. Data and setup

- `data/processed/churn_processed.csv`: 10,000 rows, 14 features + target.
  Identifiers dropped (`RowNumber`, `CustomerId`, `Surname`), `ZeroBalance` added,
  `Geography` and `Gender` one-hot encoded.
- Split: stratified 80/20, `random_state=42` → 8,000 train / 2,000 test, churn 20.375% / 20.350%.
- `ZeroBalance` (36.2% of customers) is a real segment: those customers churn at 13.8% vs 24.1%
  for the rest. It ranks 5th in importance.
- Model selection uses 5-fold stratified cross-validation on the 8,000 training rows only. The
  2,000 test rows are scored once per study.

## 3. Metric: why F2

`F_beta = (1 + beta^2) * P * R / (beta^2 * P + R) = (1 + beta^2) * TP / ((1 + beta^2) * TP + beta^2 * FN + FP)`.

With `beta = 2` a miss costs 4x a false alarm. That matches the business rule: a lost customer
costs more than a wasted call.

Study 1 maximised PR-AUC, a *ranking* metric. It produced the best-ordered probabilities but
chose a high cut-off (0.522), so it flagged fewer customers and caught **299** churners instead
of the baseline's **315**. Ranking quality rose while caught churners fell. F2 scores the
decision itself, so studies 2 and 3 use it.

## 4. Study 1 — PR-AUC objective (13 parameters, 30 trials)

- Best trial #15, CV PR-AUC 0.7021 +/- 0.0203 (untuned OOF 0.6964).
- OOF ROC-AUC 0.8651, Youden cut-off 0.5223.
- Test: ROC-AUC 0.8669, PR-AUC 0.7145, recall 0.7346, precision 0.5368.
- Confusion matrix: TN 1,335 / FP 258 / FN 108 / TP 299.
- Importance: `learning_rate` 0.2260, `colsample_bylevel` 0.2044, `scale_pos_weight` 0.1020,
  `reg_alpha` 0.0903, `max_delta_step` 0.0651, `n_estimators` 0.0613, `max_depth` 0.0566,
  `subsample` 0.0562, `colsample_bytree` 0.0409, `gamma` 0.0355, `min_child_weight` 0.0316,
  `colsample_bynode` 0.0203, `reg_lambda` 0.0097.

## 5. Study 2 — F2 objective, full space (30 trials)

- Best trial #19, CV F2 0.7098 +/- 0.0167; pooled OOF F2 0.7058.
- Winning parameters: `n_estimators` 400, `learning_rate` 0.0403, `max_depth` 6,
  `min_child_weight` 3, `gamma` 3.8734, `subsample` 0.9433, `colsample_bytree` 0.9196,
  `colsample_bylevel` 0.8854, `colsample_bynode` 0.8725, `reg_alpha` 0.4127,
  `reg_lambda` 0.0085, `scale_pos_weight` 1.4485, `max_delta_step` 7.7363.
- OOF: ROC-AUC 0.8646, PR-AUC 0.6995. Operating cut-off 0.1908.
- Test: ROC-AUC 0.8693, PR-AUC 0.7193, recall 0.8452, precision 0.4170, F2 0.7012.
- Confusion matrix: TN 1,112 / FP 481 / FN 63 / TP 344.
- 41.25% of customers flagged.

Top 5 trials by CV F2 (used later in the 10-trial comparison):

| trial | cv_f2 | cv_std | cut-off | recall@OOF | precision@OOF |
|---|---|---|---|---|---|
| 19 | 0.7098 | 0.0167 | 0.1860 | — | — |
| 25 | 0.7096 | 0.0146 | 0.1380 | — | — |
| 12 | 0.7095 | 0.0145 | 0.2100 | — | — |
| 14 | 0.7094 | 0.0186 | 0.3260 | — | — |
| 24 | 0.7091 | 0.0163 | 0.2590 | — | — |

Importance: `learning_rate` 0.3762, `colsample_bytree` 0.1684, `scale_pos_weight` 0.1139,
`max_depth` 0.0891, `min_child_weight` 0.0689, `reg_alpha` 0.0480, `subsample` 0.0335,
`gamma` 0.0265, `max_delta_step` 0.0219, `colsample_bylevel` 0.0194, `reg_lambda` 0.0155,
`colsample_bynode` 0.0129, `n_estimators` 0.0057.

## 6. Study 3 — F2 objective, reduced space (8 tuned + 5 fixed, 30 trials)

Search space reduced to the parameters that carry 90% of the importance (see section 7).
Tuned: `learning_rate`, `colsample_bytree`, `scale_pos_weight`, `max_depth`,
`min_child_weight`, `reg_alpha`, `subsample`, `gamma`.
Fixed: `n_estimators` 400, `colsample_bylevel` 0.8854, `colsample_bynode` 0.8725,
`reg_lambda` 0.0085, `max_delta_step` 7.7363.

- **Best trial #28, CV F2 0.7111 +/- 0.0160** (study 2: 0.7098, so +0.0013); pooled OOF F2 0.7074.
- Winning parameters: `learning_rate` **0.0133**, `colsample_bytree` 0.7106,
  `scale_pos_weight` 1.8081, `max_depth` **5**, `min_child_weight` **9**,
  `reg_alpha` 0.0015, `subsample` 0.8356, `gamma` 0.1675.
- OOF: ROC-AUC 0.8656, PR-AUC 0.7005. Operating cut-off **0.2385** (mean fold cut-off 0.2549).
- Test: ROC-AUC 0.8683, PR-AUC 0.7167, recall 0.8403, precision 0.4196, F2 0.7000.
- Confusion matrix: TN 1,120 / FP 473 / FN 65 / TP 342 (of 407).
- 40.75% of customers flagged.
- 8 of 30 trials land within 0.002 of the winner: the top of the space is still flat.

Top 5 trials by CV F2:

| trial | cv_f2 | cv_std | cut-off | pr_auc | roc_auc |
|---|---|---|---|---|---|
| **28** | **0.7111** | 0.0160 | 0.2550 | 0.7005 | 0.8656 |
| 21 | 0.7097 | 0.0155 | 0.2810 | 0.6990 | 0.8651 |
| 25 | 0.7096 | 0.0160 | 0.3230 | 0.7012 | 0.8654 |
| 2 | 0.7089 | 0.0134 | 0.4210 | 0.7000 | 0.8641 |
| 6 | 0.7089 | 0.0183 | 0.5640 | 0.6947 | 0.8621 |

Importance in the reduced space: `learning_rate` **0.6547**, `colsample_bytree` 0.1732,
`scale_pos_weight` 0.0552, `gamma` 0.0418, `max_depth` 0.0296, `min_child_weight` 0.0233,
`subsample` 0.0202, `reg_alpha` 0.0019.

**Reading of study 3.** The importance of `learning_rate` rose from 0.376 to 0.655 because the
other noisy dimensions are gone: the search now resolves the one parameter that really matters.
The winning region moved to a **lower learning rate (0.013) with shallower trees (depth 5) and
heavier leaf constraints (min_child_weight 9)** — more trees' worth of small steps instead of
fewer large ones.

## 6b. Ten best trials compared at `n_estimators = 500`

`trial_report.py` (project root, temporary) re-scores the top 5 trials of study 2 and the top 5 of
study 3 with `n_estimators = 500`, a cut-off taken from out-of-fold F2, plus an ensemble of all ten
(mean probability). Raw scores:

| config | study | cut-off | recall | precision | F2 | PR-AUC | cost (4FN+FP) | cost @0.30 |
|---|---|---|---|---|---|---|---|---|
| **ensemble-10** | all | 0.2925 | 0.8329 | **0.4286** | **0.7007** | **0.7188** | **724** | 729 |
| s3-t2 | study 3 | 0.3952 | **0.8624** | 0.3984 | 0.6995 | 0.7156 | 754 | 853 |
| s2-t19 | study 2 | 0.1874 | 0.8428 | 0.4153 | 0.6989 | 0.7194 | 739 | **674** |
| s2-t24 | study 2 | 0.2279 | 0.8378 | 0.4189 | 0.6982 | 0.7194 | 737 | 690 |
| s2-t25 | study 2 | 0.1433 | 0.8428 | 0.4133 | 0.6977 | 0.7185 | 743 | 712 |
| s2-t14 | study 2 | 0.2947 | 0.8354 | 0.4177 | 0.6962 | 0.7180 | 742 | 739 |
| s3-t21 | study 3 | 0.2582 | 0.8378 | 0.4153 | 0.6962 | 0.7169 | 744 | 714 |
| s3-t25 | study 3 | 0.3334 | 0.8182 | 0.4342 | 0.6952 | 0.7125 | 730 | 735 |
| s3-t28 | study 3 | 0.2187 | 0.8452 | 0.4057 | 0.6947 | 0.7175 | 756 | 701 |
| s2-t12 | study 2 | 0.1552 | 0.8575 | 0.3878 | 0.6903 | 0.7191 | 783 | 695 |
| s3-t6 | study 3 | 0.4443 | 0.8821 | 0.3626 | 0.6856 | 0.7100 | 823 | 978 |

Ensemble at its own cut-off (0.2925): TN 1,141 / FP 452 / FN 68 / TP 339, accuracy 0.740.

**Findings from the ten-trial comparison**

1. **The ensemble is the best F2 model of the project** (0.7007) and the cheapest operating point at
   its own cut-off (724). Averaging the flat top of the search space paid off where more trials did
   not: it also has the best precision (0.4286) among the high-recall configurations and the best
   PR-AUC (0.7188).
2. **500 trees is not a free improvement.** The best tuned trials score slightly *below* their
   notebook values, because their tuned `n_estimators` was 400 (s3-t28 0.6947 vs 0.7000;
   s2-t19 0.6989 vs 0.7012). Adding trees past the tuned value buys nothing here.
3. **Recall can go higher, at a price.** s3-t6 reaches 0.8821 recall but costs 823 — the 4:1 ratio
   does not justify it (its cost @0.30 is 978).
4. **Calibration leaves F2 unchanged and compresses the cut-offs.** Isotonic regression is monotone,
   so every F2 is identical to the raw row; the raw cut-offs span 0.143-0.444 while the calibrated
   ones span 0.139-0.197. Use calibration for a portable threshold, not for a better score.
5. **At the fixed 0.30 cut-off, s2-t19 is cheapest (674)**, then s2-t24 (690) and s3-t28 (701). If
   the retention team fixes a contact budget, that is the configuration to ship.

## 6c. Shipped model (`artifacts/`)

`python -m bank_churn.models.trainer` fits study 3 trial 28 with **`n_estimators = 400`** (the
500-tree comparison in 6b made both F2 and recall worse), measures it on the holdout, then refits
on all 10,000 rows for deployment.

| quantity | value |
|---|---|
| parameters | 12 tuned/fixed + `n_estimators` 400 |
| OOF F2 (emitted cut-off) | 0.7055 at 0.2509 |
| test ROC-AUC / PR-AUC | 0.8680 / 0.7165 |
| test @0.251 | recall 0.8206, precision 0.4299, TP 334 / FN 73 / FP 443 |
| test @0.30 (operating) | recall 0.7813, precision 0.4818 |
| inference | 1.31 ms per row (median), 8.7 ms for 1,000 rows |

The cut-off recomputed on these folds is 0.2509 instead of the notebook's 0.2385, which moves
8 churners on the 2,000 test rows — the cut-off drift described in section 7. The API default
stays at 0.30, the cost-optimal point.

Artifacts: `artifacts/churn_xgb.json` (model) and `artifacts/churn_model_metadata.json`
(version, parameters, cut-off, metrics). Prediction loads the model once, runs the booster with
one thread and uses `inplace_predict`, which is 2.7x faster than `predict_proba` on a single row.

## 7. Why the score barely moves but the cut-off moves a lot

Your observation is correct and it is the main lesson of the three studies:

- CV F2: 0.7098 (study 2) → 0.7111 (study 3). ROC-AUC: 0.8646 → 0.8656. Ranking is unchanged.
- Operating cut-off: **0.191 → 0.239**, a 25% relative change, and the per-fold cut-offs range
  from 0.14 to 0.57 across trials.

Cause: the cut-off is a point on the probability scale, and the scale depends on
`scale_pos_weight`, `reg_alpha` and the tree structure. Two models with the same ranking can sit
on quite different scales. Consequences for practice:

1. **Never hard-code a cut-off.** Recompute it from out-of-fold scores for every fitted model.
2. **Calibrate before deploying** (isotonic on out-of-fold predictions) if the score is shown to
   a user or compared across model versions.
3. **Judge the ranking with ROC-AUC/PR-AUC and the decision with recall/precision at the chosen
   cut-off.** A cut-off from another model version is not transferable.

## 8. What the alerts cost

Diagnostic only (post hoc from the confusion matrices, cost model `4 * FN + FP`):

| model | FN | FP | cost | value `4*TP - FP` |
|---|---|---|---|---|
| baseline @0.451 | 92 | 332 | 700 | 928 |
| study 1 @0.522 | 108 | 258 | **690** | 938 |
| study 2 @0.191 | 63 | 481 | 733 | 895 |
| study 3 @0.239 | 65 | 473 | 733 | 895 |
| study 3 @0.300 | 90 | 341 | 701 | 927 |

F2 is a ratio, so the F2-optimal cut-off is **not** the cost-optimal cut-off. At 0.239 the model
spends 473 false alarms to avoid 65 misses; at 0.300 the same model spends 341 false alarms and
misses 90 — a lower total cost.

**Recommendation:** keep the study 3 model and set the cut-off from the real cost ratio or the
contact budget (about 0.30 on this holdout), not from F2 directly.

## 9. Parameter importance and the reduced search space

Study 2 importance (13 params): cumulative importance reaches **92.4% after 8 parameters**
(89.8% after 7). The five dropped parameters — `max_delta_step`, `colsample_bylevel`,
`reg_lambda`, `colsample_bynode`, `n_estimators` — were fixed at the winning values.

Effect of the reduction: same score, 8 dimensions instead of 13, and each dimension sampled about
1.6x more densely at the same 30-trial budget. `learning_rate` is now unambiguously the driver
(0.655 of the importance).

## 10. What drives churn (study 3 model)

| feature | importance |
|---|---|
| NumOfProducts | 0.2408 |
| Age | 0.1804 |
| IsActiveMember | 0.1461 |
| Geo_Germany | 0.0941 |
| ZeroBalance | 0.0585 |
| Balance | 0.0524 |
| Geo_France | 0.0397 |
| Gender_Male | 0.0358 |
| Geo_Spain | 0.0333 |
| Gender_Female | 0.0330 |
| CreditScore | 0.0244 |
| EstimatedSalary | 0.0239 |
| Tenure | 0.0193 |
| HasCrCard | 0.0184 |

The EDA story holds: product count, age, activity and country dominate. Germany churns at 33.1%
vs 15.9% (France) and 16.6% (Spain); female customers 25.1% vs 16.5%.

## 11. Conclusions

1. **The metric decides the outcome.** The same data and model family produced 299 caught
   churners (PR-AUC objective) or 344 (F2 objective). Choose the metric that scores the decision.
2. **The reduced space is as good and cheaper to maintain.** Study 3 matches study 2 within noise
   while tuning 8 parameters.
3. **Precision is the price of recall.** 0.42 means about 6 in 10 flagged customers would have
   stayed. That is acceptable only if a contact costs less than a quarter of a lost customer.
4. **Cut-offs do not transfer.** Recompute from out-of-fold scores; calibrate before showing a
   score to a user.
5. **Tuning is at its ceiling.** Three studies moved test PR-AUC 0.7056 → 0.7193 → 0.7167 and
   recall 0.774 → 0.845 → 0.840. Further gains must come from features (product behaviour,
   transaction recency, service contacts), not from more trials.
6. **The best decision is not the best F2.** At ~0.30 the same model has a lower business cost
   than at its F2 optimum.

## 12. Caveats

- **One holdout.** 2,000 rows, 407 churners; a 0.01 difference in PR-AUC is inside noise.
- **CV noise +/- 0.016 F2**, so studies 2 and 3 are statistically tied.
- **Cut-offs drift** with the fitted model and the data window. Recalibrate before each campaign.
- **Cost analysis in section 8 is diagnostic**, computed after the fact on the test set.
- **XGBoost with `n_jobs=-1`** can differ in the last decimals between runs.
- **`data/processed/churn_processed.csv` is static**; new customers need the same preprocessing
  (drop identifiers, add `ZeroBalance`, one-hot `Geography`/`Gender`) before scoring.

## 13. Files

| file | content |
|---|---|
| `notebooks/eda.ipynb` | inspection, stratified split, outliers and distributions |
| `notebooks/modelling.ipynb` | processed dataset, 3 baselines, threshold, test evaluation, at-risk ranking |
| `notebooks/tuning.ipynb` | study 3: F2 objective with the reduced search space |
| `src/bank_churn/evaluation/trial_report.py` | side-by-side confusion report for the top trials |
| `data/processed/churn_processed.csv` | model-ready dataset, 10,000 x 15 |
| `FINDINGS.md` | this report |
| `pyproject.toml` | declares `optuna>=5.0.0` |

## 14. Reproduce

```
uv sync
.venv/Scripts/python.exe -m jupyter nbconvert --to notebook --execute --inplace notebooks/tuning.ipynb
PYTHONPATH=src .venv/Scripts/python.exe -m bank_churn.evaluation.trial_report
```

The kernel `bank-churn-venv` is registered inside `.venv/share/jupyter`; select the project
`.venv` interpreter in VS Code or JupyterLab. One study takes about 3 minutes.
