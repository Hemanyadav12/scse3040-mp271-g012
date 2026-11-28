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

python -m cabeta.run_data_checks           # 1. 7 detectors  -> reports/findings.yaml (data_issues)
python -m cabeta.train                     # 2. v1: 5 MLflow runs, best registered as version 1 @champion
python -m cabeta.monitor                   # 3. PSI per column + MAE per zone for v1 on later.csv
python -m cabeta.retrain                   # 4. v2 on train + first 70% of later; promote if >=5% better
python -m cabeta.predict_holdout           # 5. predictions/holdout_predictions.csv from @champion
pytest -q                                  # tests (no CSVs needed)

uvicorn app.main:app --port 8000           # serve locally
docker build -t mp271-api:1.0 .
kubectl apply -f k8s/
```

Re-running `train` on an existing `mlflow.db` adds more registry versions. To start again,
delete `mlflow.db` and `mlartifacts/` first.