"""Reads params.yaml and gives paths relative to the repository root."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_params(path=None):
    path = Path(path) if path else ROOT / "params.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(rel_path):
    """Turn a path from params.yaml into an absolute path under the repo root."""
    return ROOT / rel_path


def tracking_uri():
    return f"sqlite:///{ROOT / 'mlflow.db'}"
