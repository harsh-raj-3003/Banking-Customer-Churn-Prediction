# Bank Customer Churn Prediction

This project predicts which bank customers will churn. It finds the customers with the highest
risk, so the retention team can contact them first. Recall is the primary metric: a lost customer
costs more than a false alarm.

Read [FINDINGS.md](FINDINGS.md) for the result of the tuning studies and for the decisions behind
the model.

## 1. Result

The deployed model is XGBoost with the parameters of study 3, trial 28, and 400 trees. Table 1
gives the measured values on the 2,000 test rows.

| quantity | value |
|---|---|
| recall at cut-off 0.30 | 0.781 |
| precision at cut-off 0.30 | 0.482 |
| recall at cut-off 0.251 | 0.821 |
| precision at cut-off 0.251 | 0.430 |
| ROC-AUC | 0.868 |
| PR-AUC | 0.717 |
| churners caught at 0.251 | 334 of 407 |
| time for one prediction | 1.31 ms (median) |
| time for 1,000 predictions | 8.7 ms |

## 2. Requirements

- Python 3.13 or later. The requirement is in [pyproject.toml](pyproject.toml).
- uv, the package manager. Read https://docs.astral.sh/uv/ for the installation.
- Windows, macOS or Linux. The commands in this file use the Windows path `.venv/Scripts/python.exe`.
  On macOS and Linux, use `.venv/bin/python`.

## 3. Install the dependencies

Run this command in the project root:

```
uv sync
```

The command reads [pyproject.toml](pyproject.toml) and [uv.lock](uv.lock). It creates the `.venv`
directory and installs the dependencies.

## 4. Deploy the service

Do these steps in this sequence.

### Step 1: Fit the model

```
PYTHONPATH=src .venv/Scripts/python.exe -m bank_churn.models.trainer
```

The trainer does these steps:

1. It makes a stratified split of 80/20 with seed 42.
2. It computes the out-of-fold scores and the cut-off.
3. It measures the model on the test rows.
4. It fits the model again on all 10,000 rows.
5. It writes the artifacts.

The command needs approximately 15 seconds. It writes
[artifacts/churn_xgb.json](artifacts/churn_xgb.json) and
[artifacts/churn_model_metadata.json](artifacts/churn_model_metadata.json).

The trainer has these options:

| option | default | effect |
|---|---|---|
| `--n-estimators` | 400 | number of trees |
| `--threshold` | 0.30 | stored operating cut-off |

Do not set `--n-estimators` to 500. Section 7 of FINDINGS.md shows that 500 trees make F2 and recall
worse.

### Step 2: Start the API

```
PYTHONPATH=src .venv/Scripts/python.exe -m uvicorn bank_churn.api.main:app --host 127.0.0.1 --port 8000
```

Keep the terminal open. The service loads the model one time at startup.

The service has these endpoints:

| method | path | effect |
|---|---|---|
| GET | `/health` | gives the status, the model version and the cut-off |
| GET | `/features` | gives the feature names and the permitted values |
| POST | `/predict` | gives the churn probability and the churn flag |

### Step 3: Examine the API

Open http://127.0.0.1:8000/docs in a browser. This page gives an interactive test for each
endpoint.

1. Send `GET /health`.
2. Make sure that the response shows `"model_ready": true`.
3. Send `POST /predict` with the body from section 6.

If `model_ready` is `false`, the model file is absent. Do step 1 again.

### Step 4: Start the interface

Open a second terminal. Run this command:

```
PYTHONPATH=src .venv/Scripts/python.exe -m streamlit run src/bank_churn/ui/app.py
```

Streamlit shows a local URL. Open the URL in a browser. The interface sends the customer data to
the API.

If the API is not at `http://127.0.0.1:8000`, set the variable `CHURN_API_URL` before you start
Streamlit:

```
set CHURN_API_URL=http://127.0.0.1:8001
```

On macOS and Linux, use `export CHURN_API_URL=http://127.0.0.1:8001` instead.

## 5. Stop the services

Press `Ctrl+C` in each terminal. Do this in the terminal of the API and in the terminal of the
interface.

## 6. Send a prediction request

Send this command in a terminal. The API must be active.

```
curl.exe -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"credit_score\": 619, \"geography\": \"Germany\", \"gender\": \"Female\", \"age\": 52, \"tenure\": 6, \"balance\": 130000, \"num_of_products\": 3, \"has_cr_card\": true, \"is_active_member\": false, \"estimated_salary\": 120000}"
```

The response has this form:

```
{"churn_probability": 0.978, "churn_flag": true, "threshold": 0.3, "model_version": "<build time>"}
```

`churn_flag` is `true` if `churn_probability` is equal to or larger than `threshold`.

The request has these fields:

| field | type | permitted values |
|---|---|---|
| `credit_score` | integer | 300 to 850 |
| `geography` | text | `France`, `Germany`, `Spain` |
| `gender` | text | `Female`, `Male` |
| `age` | integer | 18 to 100 |
| `tenure` | integer | 0 to 10 |
| `balance` | number | 0 or more |
| `num_of_products` | integer | 1 to 4 |
| `has_cr_card` | boolean | `true`, `false` |
| `is_active_member` | boolean | `true`, `false` |
| `estimated_salary` | number | 0 or more |

A field with an invalid value gives HTTP status 422. A request without a model gives HTTP status
503.

## 7. Project directories

| directory or file | content |
|---|---|
| [src/bank_churn/](src/bank_churn) | all source code |
| [src/bank_churn/config/](src/bank_churn/config) | paths, feature order, seed and cut-off |
| [src/bank_churn/data/](src/bank_churn/data) | loader for the raw and processed data |
| [src/bank_churn/features/](src/bank_churn/features) | preparation of the customer data |
| [src/bank_churn/models/](src/bank_churn/models) | trainer and predictor |
| [src/bank_churn/evaluation/](src/bank_churn/evaluation) | F2, recall, precision and cost metrics |
| [src/bank_churn/api/](src/bank_churn/api) | FastAPI service and request schemas |
| [src/bank_churn/ui/](src/bank_churn/ui) | Streamlit interface |
| [notebooks/](notebooks) | EDA, modelling and tuning notebooks |
| [data/raw/](data/raw) | the source file `Churn_Modelling.csv` |
| [data/processed/](data/processed) | the model-ready file `churn_processed.csv` |
| [artifacts/](artifacts) | model file, metadata and the trial report |
| [FINDINGS.md](FINDINGS.md) | the findings report |
| [trial_report.py](trial_report.py) | comparison of the ten best trials |
| [pyproject.toml](pyproject.toml) | dependencies and the Python version |

## 8. Limits

- The processed file is static. Apply the same preparation to new customers before you send them
  to the API.
- The cut-off changes with each fit. The API reads the cut-off from the model metadata. Compute the
  cut-off again when you fit a new model.
- The metrics come from one holdout set of 2,000 rows. A difference of 0.01 in PR-AUC is inside the
  noise.
- The model uses 14 features. Product behaviour, transaction recency and service contacts are not
  in the data. These features give the next gain.
