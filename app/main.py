"""FastAPI service: POST /predict, GET /health. Loads models/model.joblib (the champion)."""
import time
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from cabeta.cleaning import normalize_car_type
from cabeta.config import load_params, resolve
from cabeta.features import feature_columns
from cabeta.logging_utils import get_json_logger

PARAMS = load_params()
logger = get_json_logger("cabeta.api")
app = FastAPI(title="Surat Rides - pickup ETA", version="1.0")
model = None  # loaded lazily; tests can put a small model here

class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    drivers_nearby: int = Field(ge=0)                 # client rule: never negative
    distance_to_driver_km: float = Field(ge=0)
    hour: int = Field(ge=0, le=23)
    rain_mm: float = Field(ge=0)
    surge: float = Field(gt=0)
    road_closures: int = Field(ge=0, le=15)           # client rule: more than 15 is impossible
    car_type: str
    zone: str

    @field_validator("car_type")
    @classmethod
    def _car_type(cls, v):
        v = normalize_car_type(v)
        if v not in PARAMS["car_types"]:
            raise ValueError(f"car_type must be one of {PARAMS['car_types']}")
        return v

    @field_validator("zone")
    @classmethod
    def _zone(cls, v):
        v = v.strip()
        if v not in PARAMS["zones"]:
            raise ValueError(f"zone must be one of {PARAMS['zones']}")
        return v


def get_model():
    global model
    if model is None:
        path = resolve(PARAMS["paths"]["model"])
        if not path.exists():
            raise HTTPException(status_code=503, detail="model file not found")
        model = joblib.load(path)
    return model


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    t0 = time.perf_counter()
    row = pd.DataFrame([req.model_dump()])[feature_columns(PARAMS)]
    value = round(float(get_model().predict(row)[0]), 2)
    logger.info("prediction", extra={"fields": {
        "event": "prediction",
        "inputs": req.model_dump(),
        "prediction": value,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
    }})
    return {"prediction": value, "pickup_eta_min": value}
