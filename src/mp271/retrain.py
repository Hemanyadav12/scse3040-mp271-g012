"""Step 4: retrain v2 on train + first 70% of later, compare on the last 30%,
apply the client's promotion rule (v2 must be >= 5% better to become @champion)."""
import json

import joblib
import mlflow
from mlflow import MlflowClient

from .cleaning import clean_other, clean_train
from .config import load_params, resolve
from .data import load_raw, train_sha256
from .features import build_pipeline, mae
from .findings import update_findings
from .registry import load_registered
from .train import setup_experiment

import pandas as pd


def decide(mae_v1, mae_v2, min_improvement):
    """Pure rule: promote only if v2 is at least min_improvement (fraction) lower."""
    return "promote" if mae_v2 <= (1 - min_improvement) * mae_v1 else "keep"


def main():
    p = load_params()
    setup_experiment(p)
    client = MlflowClient()
    name = p["mlflow"]["registered_model_name"]

    raw = load_raw("train", p)
    train, report = clean_train(raw, set(load_raw("holdout", p).columns), p)
    later = clean_other(load_raw("later", p), report, p).sort_values(["date", "record_id"], kind="stable")

    cut = int(len(later) * p["monitoring"]["retrain_fraction"])
    later_fit, later_check = later.iloc[:cut], later.iloc[cut:]

    # same model family and settings as the v1 run (the champion's run)
    v1_version = client.get_model_version_by_alias(name, p["mlflow"]["alias"])
    v1_run = client.get_run(v1_version.run_id)
    model_name = v1_run.data.params["model_name"]
    model_params = json.loads(v1_run.data.params["model_params"])

    v1 = load_registered(p, version=int(v1_version.version))
    mae_v1 = mae(v1, later_check, p)

    train_plus = pd.concat([train, later_fit], ignore_index=True)
    v2 = build_pipeline(model_name, model_params, p)
    v2.fit(train_plus, train_plus[p["target"]])
    mae_v2 = mae(v2, later_check, p)
    decision = decide(mae_v1, mae_v2, p["monitoring"]["promote_min_improvement"])

    with mlflow.start_run(run_name="v2_retrain") as run:
        mlflow.log_param("data_sha256", train_sha256(p))
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_params", json.dumps(model_params))
        mlflow.log_param("n_train_rows", len(train_plus))
        mlflow.log_param("decision", decision)
        mlflow.log_metric("mae_check_v1", mae_v1)
        mlflow.log_metric("mae_check_v2", mae_v2)
        mlflow.sklearn.log_model(v2, artifact_path="model", serialization_format="cloudpickle")
        v2_run_id = run.info.run_id

    mv2 = mlflow.register_model(f"runs:/{v2_run_id}/model", name)
    if decision == "promote":
        client.set_registered_model_alias(name, p["mlflow"]["alias"], mv2.version)
        out = resolve(p["paths"]["model"])
        joblib.dump(v2, out)

    champion = client.get_model_version_by_alias(name, p["mlflow"]["alias"])
    from .findings import load_findings

    existing = load_findings()

    tracking = existing.get("tracking", {})
    tracking["champion_version"] = int(champion.version)
    update_findings("tracking", tracking)

    monitoring = existing.get("monitoring", {})
    monitoring.update({
        "mae_v1_on_check": round(mae_v1, 4),
        "mae_v2_on_check": round(mae_v2, 4),
        "decision": decision,
    })
    update_findings("monitoring", monitoring)
    print(f"v1 MAE on last 30% = {mae_v1:.3f} | v2 MAE = {mae_v2:.3f} | "
          f"improvement = {(1 - mae_v2 / mae_v1) * 100:.1f}% -> {decision.upper()}; "
          f"champion is now version {champion.version}")


if __name__ == "__main__":
    main()
