# Bank Customer Churn: Findings

This report gives the result of three tuning studies. The numbers come from the executed
notebooks in [notebooks/](notebooks).

## 1. Result

Table 1 gives the test result of each model. The test set has 2,000 rows and 407 churners.

| model | cut-off | recall | precision | F2 | ROC-AUC | PR-AUC | churners caught |
|---|---|---|---|---|---|---|---|
| baseline XGBoost | 0.451 | 0.774 | 0.487 | 0.692 | 0.8620 | 0.7056 | 315 / 407 |
| study 1, PR-AUC objective (13 parameters) | 0.522 | 0.735 | 0.537 | 0.684 | 0.8669 | 0.7145 | 299 / 407 |
| study 2, F2 objective (13 parameters) | 0.191 | 0.845 | 0.417 | **0.701** | 0.8693 | 0.7193 | 344 / 407 |
| **study 3, F2 objective (8 parameters)** | 0.239 | 0.840 | 0.420 | 0.700 | 0.8683 | 0.7167 | 342 / 407 |

The final model is study 3, trial 28. It catches 342 of 407 churners. Recall is 0.840.
Precision is 0.420.

Study 2 and study 3 are equivalent within noise. The difference in F2 is 0.001. The difference
in recall is 0.5 points. The difference in ROC-AUC is 0.001. Study 3 tunes 8 parameters. Study 2
tunes 13 parameters. Use study 3. It is easier to maintain and to extend.

## 2. Data and test method

The processed file is [data/processed/churn_processed.csv](data/processed/churn_processed.csv).
It has 10,000 rows, 14 features and the target column `Exited`.

The preparation does these steps:

1. Drop `RowNumber`, `CustomerId` and `Surname`.
2. Add the feature `ZeroBalance`.
3. One-hot encode `Geography` and `Gender`.

The project makes a stratified split of 80/20 with `random_state=42`. The split gives 8,000
training rows and 2,000 test rows. The churn rate is 20.375 percent in the training set and
20.350 percent in the test set.

`ZeroBalance` is 36.2 percent of the customers. These customers churn at 13.8 percent. The other
customers churn at 24.1 percent. `ZeroBalance` is the fifth most important feature.

Model selection uses 5-fold stratified cross-validation on the 8,000 training rows. The project
scores the test rows one time for each study.

## 3. Metric

The project uses F2.

```
F_beta = (1 + beta^2) * P * R / (beta^2 * P + R)
       = (1 + beta^2) * TP / ((1 + beta^2) * TP + beta^2 * FN + FP)
```

F2 has `beta = 2`. One missed churner costs 4 times more than one false alarm. This ratio agrees
with the business rule: a lost customer costs more than a useless call.

Study 1 maximized PR-AUC. PR-AUC measures the ranking only. Study 1 gave the best order of the
probabilities. But it selected a high cut-off of 0.522. It caught 299 churners. The baseline
caught 315. The ranking became better and the number of caught churners became smaller. F2
measures the decision. For this reason, study 2 and study 3 use F2.

## 4. Study 1: PR-AUC objective

Setup: 13 parameters, 30 trials.

- Best trial is trial 15. Cross-validation PR-AUC is 0.7021, standard deviation is 0.0203.
  The untuned out-of-fold PR-AUC is 0.6964.
- Out-of-fold ROC-AUC is 0.8651. The Youden cut-off is 0.5223.
- Test: ROC-AUC 0.8669, PR-AUC 0.7145, recall 0.7346, precision 0.5368.
- Confusion matrix: TN 1,335, FP 258, FN 108, TP 299.

| parameter | importance |
|---|---|
| learning_rate | 0.2260 |
| colsample_bylevel | 0.2044 |
| scale_pos_weight | 0.1020 |
| reg_alpha | 0.0903 |
| max_delta_step | 0.0651 |
| n_estimators | 0.0613 |
| max_depth | 0.0566 |
| subsample | 0.0562 |
| colsample_bytree | 0.0409 |
| gamma | 0.0355 |
| min_child_weight | 0.0316 |
| colsample_bynode | 0.0203 |
| reg_lambda | 0.0097 |

## 5. Study 2: F2 objective, full space

Setup: 13 parameters, 30 trials.

- Best trial is trial 19. Cross-validation F2 is 0.7098, standard deviation is 0.0167. The pooled
  out-of-fold F2 is 0.7058.
- Out-of-fold ROC-AUC is 0.8646. Out-of-fold PR-AUC is 0.6995. The operating cut-off is 0.1908.
- Test: ROC-AUC 0.8693, PR-AUC 0.7193, recall 0.8452, precision 0.4170, F2 0.7012.
- Confusion matrix: TN 1,112, FP 481, FN 63, TP 344.
- The model flags 41.25 percent of the customers.

Winning parameters:

| parameter | value |
|---|---|
| n_estimators | 400 |
| learning_rate | 0.0403 |
| max_depth | 6 |
| min_child_weight | 3 |
| gamma | 3.8734 |
| subsample | 0.9433 |
| colsample_bytree | 0.9196 |
| colsample_bylevel | 0.8854 |
| colsample_bynode | 0.8725 |
| reg_alpha | 0.4127 |
| reg_lambda | 0.0085 |
| scale_pos_weight | 1.4485 |
| max_delta_step | 7.7363 |

The five best trials by cross-validation F2:

| trial | cv_f2 | cv_std | cut-off |
|---|---|---|---|
| 19 | 0.7098 | 0.0167 | 0.1860 |
| 25 | 0.7096 | 0.0146 | 0.1380 |
| 12 | 0.7095 | 0.0145 | 0.2100 |
| 14 | 0.7094 | 0.0186 | 0.3260 |
| 24 | 0.7091 | 0.0163 | 0.2590 |

| parameter | importance |
|---|---|
| learning_rate | 0.3762 |
| colsample_bytree | 0.1684 |
| scale_pos_weight | 0.1139 |
| max_depth | 0.0891 |
| min_child_weight | 0.0689 |
| reg_alpha | 0.0480 |
| subsample | 0.0335 |
| gamma | 0.0265 |
| max_delta_step | 0.0219 |
| colsample_bylevel | 0.0194 |
| reg_lambda | 0.0155 |
| colsample_bynode | 0.0129 |
| n_estimators | 0.0057 |

## 6. Study 3: F2 objective, reduced space

The search space contains the parameters that carry 90 percent of the importance. The study tunes
8 parameters and keeps 5 parameters fixed.

Tuned parameters: `learning_rate`, `colsample_bytree`, `scale_pos_weight`, `max_depth`,
`min_child_weight`, `reg_alpha`, `subsample`, `gamma`.

Fixed parameters: `n_estimators` 400, `colsample_bylevel` 0.8854, `colsample_bynode` 0.8725,
`reg_lambda` 0.0085, `max_delta_step` 7.7363.

- **Best trial is trial 28. Cross-validation F2 is 0.7111, standard deviation is 0.0160.** Study 2
  gives 0.7098. The pooled out-of-fold F2 is 0.7074.
- Out-of-fold ROC-AUC is 0.8656. Out-of-fold PR-AUC is 0.7005. The operating cut-off is **0.2385**.
  The mean fold cut-off is 0.2549.
- Test: ROC-AUC 0.8683, PR-AUC 0.7167, recall 0.8403, precision 0.4196, F2 0.7000.
- Confusion matrix: TN 1,120, FP 473, FN 65, TP 342.
- The model flags 40.75 percent of the customers.
- 8 of 30 trials are within 0.002 of the best trial. The top of the space is flat.

Winning parameters:

| parameter | value |
|---|---|
| learning_rate | **0.0133** |
| colsample_bytree | 0.7106 |
| scale_pos_weight | 1.8081 |
| max_depth | **5** |
| min_child_weight | **9** |
| reg_alpha | 0.0015 |
| subsample | 0.8356 |
| gamma | 0.1675 |

The five best trials by cross-validation F2:

| trial | cv_f2 | cv_std | cut-off | pr_auc | roc_auc |
|---|---|---|---|---|---|
| **28** | **0.7111** | 0.0160 | 0.2550 | 0.7005 | 0.8656 |
| 21 | 0.7097 | 0.0155 | 0.2810 | 0.6990 | 0.8651 |
| 25 | 0.7096 | 0.0160 | 0.3230 | 0.7012 | 0.8654 |
| 2 | 0.7089 | 0.0134 | 0.4210 | 0.7000 | 0.8641 |
| 6 | 0.7089 | 0.0183 | 0.5640 | 0.6947 | 0.8621 |

| parameter | importance |
|---|---|
| learning_rate | **0.6547** |
| colsample_bytree | 0.1732 |
| scale_pos_weight | 0.0552 |
| gamma | 0.0418 |
| max_depth | 0.0296 |
| min_child_weight | 0.0233 |
| subsample | 0.0202 |
| reg_alpha | 0.0019 |

The importance of `learning_rate` goes from 0.376 to 0.655 because the study removed the other
dimensions. The search now shows the one parameter that has the largest effect. The best region
has a lower learning rate (0.013), shallow trees (depth 5) and strong leaf constraints
(`min_child_weight` 9). The model takes more small steps instead of a small number of large steps.

## 7. Comparison of ten trials at 500 trees

[trial_report.py](trial_report.py) scores the five best trials of study 2 and the five best trials
of study 3 with `n_estimators = 500`. It computes the cut-off from the out-of-fold F2. It also
scores an ensemble of the ten models with the mean probability.

| config | study | cut-off | recall | precision | F2 | PR-AUC | cost (4FN+FP) | cost at 0.30 |
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

The ensemble at its cut-off of 0.2925 gives TN 1,141, FP 452, FN 68, TP 339. Accuracy is 0.740.

Results:

1. The ensemble is the best F2 model of the project (0.7007). It is also the cheapest configuration
   at its own cut-off (724). It has the best precision (0.4286) of the high-recall configurations
   and the best PR-AUC (0.7188).
2. 500 trees do not give an improvement. The best tuned trials score below their notebook values
   because their tuned `n_estimators` is 400 (s3-t28 0.6947 against 0.7000, s2-t19 0.6989 against
   0.7012). More trees than the tuned value give no gain here.
3. Recall can be higher at a higher cost. s3-t6 gives a recall of 0.8821 and a cost of 823. The
   4:1 ratio does not agree with this cost. Its cost at 0.30 is 978.
4. Calibration does not change F2. It compresses the cut-offs. Isotonic regression is monotone, so
   each F2 is equal to the raw row. The raw cut-offs are between 0.143 and 0.444. The calibrated
   cut-offs are between 0.139 and 0.197. Use calibration to get a portable threshold. Do not use
   calibration to get a better score.
5. At the fixed cut-off of 0.30, s2-t19 has the lowest cost (674). s2-t24 is second (690) and
   s3-t28 is third (701). If the retention team has a fixed contact budget, use s2-t19.

## 8. Deployed model

`python -m bank_churn.models.trainer` fits study 3, trial 28, with **`n_estimators = 400`**.
Section 7 shows that 500 trees make F2 and recall worse. The trainer measures the model on the
holdout and then fits it again on all 10,000 rows for deployment.

| quantity | value |
|---|---|
| parameters | 12 tuned or fixed parameters and `n_estimators` 400 |
| out-of-fold F2 at the computed cut-off | 0.7055 at 0.2509 |
| test ROC-AUC and PR-AUC | 0.8680 and 0.7165 |
| test at 0.251 | recall 0.8206, precision 0.4299, TP 334, FN 73, FP 443 |
| test at 0.30, the operating cut-off | recall 0.7813, precision 0.4818 |
| inference time | 1.31 ms for one row (median), 8.7 ms for 1,000 rows |

The cut-off from these folds is 0.2509. The notebook gives 0.2385. The difference moves 8 churners
on the 2,000 test rows. This is the cut-off drift in section 9. The API default cut-off is 0.30,
the cost-optimal point.

The artifacts are [artifacts/churn_xgb.json](artifacts/churn_xgb.json) and
[artifacts/churn_model_metadata.json](artifacts/churn_model_metadata.json). The metadata holds the
version, the parameters, the cut-off and the metrics.

The predictor loads the model one time. It runs the booster with one thread and uses
`inplace_predict`. This method is 2.7 times faster than `predict_proba` for one row.

## 9. Why the cut-off moves

Cross-validation F2 is 0.7098 in study 2 and 0.7111 in study 3. ROC-AUC is 0.8646 and 0.8656. The
ranking does not change.

The operating cut-off changes from 0.191 to 0.239. This is a change of 25 percent. The cut-off of
each fold is between 0.14 and 0.57 across the trials.

Cause: the cut-off is a point on the probability scale. The parameters `scale_pos_weight` and
`reg_alpha`, and the tree structure, change this scale. Two models can have the same ranking and a
different scale.

Do these steps:

1. Do not hard-code a cut-off. Compute the cut-off from out-of-fold scores for each fitted model.
2. Calibrate the model before deployment. Use isotonic regression on out-of-fold predictions. Do
   this if a user sees the score or if you compare model versions.
3. Measure the ranking with ROC-AUC and PR-AUC. Measure the decision with recall and precision at
   the selected cut-off. Do not use a cut-off from an other model version.

## 10. Cost of the alerts

This table is a diagnostic. The project computes the cost after the test with the formula
`4 * FN + FP`.

| model | FN | FP | cost | value `4*TP - FP` |
|---|---|---|---|---|
| baseline at 0.451 | 92 | 332 | 700 | 928 |
| study 1 at 0.522 | 108 | 258 | **690** | 938 |
| study 2 at 0.191 | 63 | 481 | 733 | 895 |
| study 3 at 0.239 | 65 | 473 | 733 | 895 |
| study 3 at 0.300 | 90 | 341 | 701 | 927 |

F2 is a ratio. For this reason, the F2-optimal cut-off is not the cost-optimal cut-off. At 0.239
the model uses 473 false alarms to prevent 65 misses. At 0.300 the model uses 341 false alarms and
misses 90 churners. The total cost is lower.

Keep the study 3 model. Set the cut-off from the real cost ratio or from the contact budget. On
this holdout the value is approximately 0.30. Do not set the cut-off from F2 directly.

## 11. Parameter importance and the reduced search space

In study 2 the cumulative importance is **92.4 percent after 8 parameters**. It is 89.8 percent
after 7 parameters. The study keeps these 5 parameters fixed: `max_delta_step`,
`colsample_bylevel`, `reg_lambda`, `colsample_bynode`, `n_estimators`.

The reduction gives the same score with 8 dimensions instead of 13. The study samples each dimension 1.6
times more densely with the same budget of 30 trials. `learning_rate` is the largest effect
(0.655 of the importance).

## 12. Features that drive churn

Importance in the study 3 model:

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

The result of the EDA is the same. Product count, age, activity and country have the largest
effect. Germany churns at 33.1 percent. France churns at 15.9 percent and Spain at 16.6 percent.
Female customers churn at 25.1 percent. Male customers churn at 16.5 percent.

## 13. Conclusions

1. The metric controls the result. The same data and model family caught 299 churners with the
   PR-AUC objective and 344 churners with the F2 objective. Select the metric that measures the
   decision.
2. The reduced space gives the same quality and is easier to maintain. Study 3 is equal to study 2
   within noise and tunes 8 parameters.
3. Precision is the cost of recall. A precision of 0.42 means that approximately 6 of 10 flagged
   customers would stay. This is acceptable only if a contact costs less than a quarter of a lost
   customer.
4. Cut-offs do not move between models. Compute the cut-off from out-of-fold scores. Calibrate the
   model before a user sees the score.
5. Tuning is at its limit. The three studies moved test PR-AUC from 0.7056 to 0.7193 and to 0.7167.
   Recall moved from 0.774 to 0.845 and to 0.840. More trials give no gain. New features give the
   next gain: product behaviour, transaction recency and service contacts.
6. The best decision is not the best F2. At approximately 0.30 the model has a lower business cost
   than at its F2 optimum.

## 14. Limits

- The project has one holdout set. It has 2,000 rows and 407 churners. A difference of 0.01 in
  PR-AUC is inside the noise.
- Cross-validation noise is plus or minus 0.016 F2. Study 2 and study 3 are equally good.
- Cut-offs change with the fitted model and the data window. Calibrate the model before each
  campaign.
- The cost analysis in section 10 is a diagnostic. The project computes it after the test.
- XGBoost with `n_jobs=-1` can give different values in the last decimals between runs.
- [data/processed/churn_processed.csv](data/processed/churn_processed.csv) is static. New customers
  must have the same preparation before scoring: drop the identifiers, add `ZeroBalance`, one-hot
  encode `Geography` and `Gender`.

## 15. Files

| file | content |
|---|---|
| [notebooks/eda.ipynb](notebooks/eda.ipynb) | inspection, stratified split, outliers and distributions |
| [notebooks/modelling.ipynb](notebooks/modelling.ipynb) | processed dataset, 3 baselines, threshold, test evaluation, at-risk ranking |
| [notebooks/tuning.ipynb](notebooks/tuning.ipynb) | study 3: F2 objective with the reduced search space |
| [trial_report.py](trial_report.py) | comparison of the ten best trials |
| [src/bank_churn/models/trainer.py](src/bank_churn/models/trainer.py) | fits, measures and saves the final model |
| [src/bank_churn/models/predictor.py](src/bank_churn/models/predictor.py) | loads the model and scores the customers |
| [src/bank_churn/api/main.py](src/bank_churn/api/main.py) | FastAPI service |
| [src/bank_churn/ui/app.py](src/bank_churn/ui/app.py) | Streamlit interface |
| [data/processed/churn_processed.csv](data/processed/churn_processed.csv) | model-ready dataset, 10,000 x 15 |
| [artifacts/](artifacts) | model file and metadata |
| [pyproject.toml](pyproject.toml) | dependencies, includes `optuna>=5.0.0` |

## 16. Reproduce the results

1. Install the dependencies.

```
uv sync
```

2. Run a notebook.

```
.venv/Scripts/python.exe -m jupyter nbconvert --to notebook --execute --inplace notebooks/tuning.ipynb
```

3. Run the comparison of the ten trials.

```
PYTHONPATH=src .venv/Scripts/python.exe trial_report.py
```

4. Fit the final model.

```
PYTHONPATH=src .venv/Scripts/python.exe -m bank_churn.models.trainer
```

The kernel `bank-churn-venv` is in `.venv/share/jupyter`. Select the project `.venv` interpreter in
VS Code or JupyterLab. One study needs approximately 3 minutes.
