import json

import pytest
from fastapi.testclient import TestClient

from app import main
from cabeta.features import build_pipeline

GOOD = {"drivers_nearby": 12, "distance_to_driver_km": 3.5, "hour": 9, "rain_mm": 10.0,
        "surge": 1.5, "road_closures": 4, "car_type": "sedan", "zone": "Airport"}


@pytest.fixture()
def client(tiny_train, params):
    pipe = build_pipeline("ridge", {"alpha": 1.0}, params)
    pipe.fit(tiny_train, tiny_train["pickup_eta_min"])
    main.model = pipe
    yield TestClient(main.app)
    main.model = None


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_predict_ok(client):
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    assert isinstance(r.json()["prediction"], float)