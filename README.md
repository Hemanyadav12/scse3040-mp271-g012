# G012 - MP271: Cab pick-up time (Surat Rides Cooperative)

Group G012 | Problem MP271

| Member | Enrolment |
|---|---|
| Heman | S24CSEU1551 |
| Prathamchaurasia | S24CSEU1568 |
| Akshej Gupta | S24CSEU1577 |
| Ananya Jain | S24CSEU1571 |

Predicts `pickup_eta_min` (regression, scored by MAE) and serves it with FastAPI.

## Run order (from the repo root, Python 3.12)

```bash
pip install -r requirements.txt            # + requirements-dev.txt for dvc
export PYTHONPATH=src                      # Windows PowerShell: $env:PYTHONPATH="src"

python -m mp271.run_data_checks            # 1. 7 detectors  -> reports/findings.yaml (data_issues)
python -m mp271.train                      # 2. v1: 5 MLflow runs, best registered as version 1 @champion
python -m mp271.monitor                    # 3. PSI per column + MAE per zone for v1 on later.csv
python -m mp271.retrain                    # 4. v2 on train + first 70% of later; promote if >=5% better
python -m mp271.predict_holdout            # 5. predictions/holdout_predictions.csv from @champion
pytest -q                                  # tests (no CSVs needed)

uvicorn app.main:app --port 8000           # serve locally
docker build -t mp271-api:1.0 .
kubectl apply -f k8s/
```

Re-running `train` on an existing `mlflow.db` adds more registry versions. To start again,
delete `mlflow.db` and `mlartifacts/` first.

## What the data checks found

| Issue | Present | Details |
|---|---|---|
| `leaking_column` | Yes | column=actual_pickup_after_min |
| `unit_mixup` | No | None |
| `sentinel_value` | Yes | column=distance_to_driver_km, value=999, count=67 |
| `duplicate_rows` | Yes | count=50 |
| `stuck_sensor` | Yes | column=hour, start_date=2026-03-26, end_date=2026-04-12 |
| `target_typos` | No | None |
| `spelling_variants` | Yes | column=car_type, true_levels=3 |

## Repository layout

```text
.
├── .github/workflows/ci.yml        # GitHub Actions CI running pytest on push/PR
├── app/
│   └── main.py                     # FastAPI service (/health, /predict) with JSON logging
├── data/raw/                       # DVC-tracked train.csv & later.csv + holdout_features.csv
├── k8s/
│   ├── deployment.yaml             # Kubernetes Deployment (4 replicas, readiness/liveness probes)
│   └── service.yaml                # Kubernetes Service mp271-svc (80 -> 8000)
├── models/
│   └── model.joblib                # Exported @champion model artifact
├── predictions/
│   └── holdout_predictions.csv     # September holdout predictions from @champion
├── reports/
│   └── findings.yaml               # Recorded data issues, MLflow tracking, drift, and v2 decision
├── src/mp271/
│   ├── checks.py                   # 7 data-problem detectors
│   ├── cleaning.py                 # Rule-driven feature and training data cleaning
│   ├── config.py                   # Parameter loader (params.yaml)
│   ├── data.py                     # Raw dataset loaders
│   ├── features.py                 # Scikit-learn preprocessing + regressor pipeline
│   ├── findings.py                 # Safe YAML writer for reports/findings.yaml
│   ├── logging_utils.py            # Structured JSON prediction logger
│   ├── monitor.py                  # PSI drift & per-zone MAE monitoring
│   ├── predict_holdout.py          # Holdout batch inference script
│   ├── registry.py                 # MLflow model registry loader (@champion / version)
│   ├── retrain.py                  # v2 retraining with >=5% MAE promotion rule
│   ├── run_data_checks.py          # CLI runner for data checks
│   └── train.py                    # MLflow experiment tracking & v1 champion registration
├── tests/                          # Unit and API tests (checks, cleaning, pipeline, monitor, API)
├── Dockerfile                      # Container image for mp271-api:1.0
├── params.yaml                     # Central configuration & thresholds
├── pytest.ini                      # Pytest path configuration
└── requirements.txt                # Runtime Python dependencies
```