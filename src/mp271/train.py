"""Step 2: train v1 with MLflow tracking, register the best run as version 1."""

import json

import joblib
import mlflow
from mlflow import MlflowClient

from .cleaning import clean_other, clean_train
from .config import ROOT, load_params, resolve, tracking_uri
from .data import load_raw, train_sha256
from .features import build_pipeline, feature_columns, mae
from .findings import update_findings


def setup_experiment(p):
    mlflow.set_tracking_uri(tracking_uri())

    name = p["mlflow"]["experiment_name"]

    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(
            name,
            artifact_location=(ROOT / "mlartifacts").as_uri()
        )

    mlflow.set_experiment(name)


def main():
    p = load_params()
    setup_experiment(p)

    sha = train_sha256(p)

    raw = load_raw("train", p)

    holdout_cols = set(
        load_raw("holdout", p).columns
    )

    train, report = clean_train(
        raw,
        holdout_cols,
        p
    )

    later = clean_other(
        load_raw("later", p),
        report,
        p
    )

    update_findings(
    "data_issues",
    report
)

    cut = int(
        len(train) *
        (1 - p["mlflow"]["validation_fraction"])
    )

    fit_part = train.iloc[:cut]
    val_part = train.iloc[cut:]

    results = []

    for cand in p["mlflow"]["candidates"]:

        with mlflow.start_run(
            run_name=cand["run_name"]
        ) as run:

            mlflow.log_param(
                "data_sha256",
                sha
            )

            mlflow.log_param(
                "model_name",
                cand["model"]
            )

            mlflow.log_param(
                "model_params",
                json.dumps(cand["params"])
            )

            mlflow.log_param(
                "n_train_rows",
                len(train)
            )

            mlflow.log_param(
                "features",
                ",".join(feature_columns(p))
            )

            pipe = build_pipeline(
                cand["model"],
                cand["params"],
                p
            )

            pipe.fit(
                fit_part,
                fit_part[p["target"]]
            )

            mae_val = mae(
                pipe,
                val_part,
                p
            )

            # Refit on all cleaned training data
            pipe = build_pipeline(
                cand["model"],
                cand["params"],
                p
            )

            pipe.fit(
                train,
                train[p["target"]]
            )

            # Later MAE is informational only
            mae_later = mae(
                pipe,
                later,
                p
            )

            mlflow.log_metric(
                "mae_val",
                mae_val
            )

            mlflow.log_metric(
                "mae_later",
                mae_later
            )

            mlflow.sklearn.log_model(
                pipe,
                artifact_path="model",
                serialization_format="cloudpickle"
            )

            results.append(
                (
                    mae_val,
                    run.info.run_id,
                    cand["run_name"],
                    pipe
                )
            )

            print(
                f"{cand['run_name']:10s} "
                f"mae_val={mae_val:.3f} "
                f"mae_later={mae_later:.3f} "
                f"run={run.info.run_id}"
            )

    mae_val, best_id, best_name, best_pipe = min(
        results,
        key=lambda r: r[0]
    )

    name = p["mlflow"]["registered_model_name"]

    mv = mlflow.register_model(
        f"runs:/{best_id}/model",
        name
    )

    MlflowClient().set_registered_model_alias(
        name,
        p["mlflow"]["alias"],
        mv.version
    )

    out = resolve(p["paths"]["model"])

    out.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        best_pipe,
        out
    )

    update_findings(
        "tracking",
        {
            "experiment_name": p["mlflow"]["experiment_name"],
            "best_run_id": best_id,
            "registered_model_name": name,
            "champion_version": int(mv.version),
        }
    )

    print(
        f"\nv1 = {best_name} "
        f"(run {best_id}), "
        f"registered as '{name}' "
        f"version {mv.version} "
        f"with alias champion"
    )


if __name__ == "__main__":
    main()
