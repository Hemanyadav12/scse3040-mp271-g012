"""Loading models from the MLflow registry (by version or by @champion alias)."""
from pathlib import Path

import joblib
import mlflow

ROOT = Path(__file__).resolve().parents[2]


def _tracking_uri():
    return f"sqlite:///{ROOT / 'mlflow.db'}"


def load_registered(params, version=None, alias=None):
    mlflow.set_tracking_uri(_tracking_uri())
    name = params["mlflow"]["registered_model_name"]
    if version is not None:
        uri = f"models:/{name}/{version}"
    else:
        uri = f"models:/{name}@{alias or params['mlflow']['alias']}"
    return mlflow.sklearn.load_model(uri)


def load_champion(params):
    """Champion from the registry; falls back to the saved models/model.joblib."""
    try:
        return load_registered(params, alias=params["mlflow"]["alias"])
    except Exception as exc:  # e.g. artifacts missing on another machine
        print(f"registry load failed ({exc!r}); using {params['paths']['model']}")
        return joblib.load(ROOT / params["paths"]["model"])
