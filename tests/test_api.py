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

def test_road_closures_over_15_rejected(client):
    assert client.post("/predict", json={**GOOD, "road_closures": 16}).status_code == 422
    assert client.post("/predict", json={**GOOD, "road_closures": 15}).status_code == 200


def test_negative_drivers_rejected(client):
    assert client.post("/predict", json={**GOOD, "drivers_nearby": -1}).status_code == 422


def test_bad_category_and_missing_field_rejected(client):
    assert client.post("/predict", json={**GOOD, "zone": "Moon"}).status_code == 422
    bad = dict(GOOD); bad.pop("hour")
    assert client.post("/predict", json=bad).status_code == 422

def test_prediction_is_logged_as_json(client, capfd):
    client.post("/predict", json=GOOD)
    line = [l for l in capfd.readouterr().out.splitlines() if l.startswith("{")][-1]
    entry = json.loads(line)
    assert entry["event"] == "prediction" and "prediction" in entry and entry["inputs"]["zone"] == "Airport"