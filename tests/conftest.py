from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml


@pytest.fixture(scope="session")
def params():
    root = Path(__file__).resolve().parents[1]
    with open(root / "params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture()
def tiny_train(params):
    """Small made-up data (the real CSVs are never needed by tests)."""
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({
        "record_id": [f"R{i}" for i in range(n)],
        "date": pd.date_range("2026-01-01", periods=n).strftime("%Y-%m-%d"),
        "drivers_nearby": rng.integers(0, 40, n),
        "distance_to_driver_km": rng.uniform(0.1, 8, n),
        "hour": rng.integers(0, 24, n),
        "rain_mm": rng.uniform(0, 50, n),
        "surge": rng.uniform(1, 3, n),
        "road_closures": rng.integers(0, 7, n),
        "car_type": rng.choice(["mini", "sedan", "auto"], n),
        "zone": rng.choice(params["zones"], n),
    })
    df["pickup_eta_min"] = 5 + 2 * df["distance_to_driver_km"] + df["road_closures"]
    df["actual_pickup_after_min"] = df["pickup_eta_min"] + rng.normal(0, 0.1, n)
    return df
