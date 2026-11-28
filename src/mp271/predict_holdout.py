"""Step 5: predict September (holdout_features.csv) with the @champion model."""
import pandas as pd

from .cleaning import clean_other
from .config import load_params, resolve
from .data import load_raw
from .features import feature_columns
from .findings import load_findings
from .registry import load_champion


def main():
    p = load_params()
    report = load_findings()["data_issues"]
    hold = load_raw("holdout", p)
    X = clean_other(hold, report, p)
    # keep the original row order of the holdout file
    X = X.set_index("record_id").loc[hold["record_id"]].reset_index()
    model = load_champion(p)
    pred = model.predict(X[feature_columns(p)])
    out = pd.DataFrame({"record_id": X["record_id"], "prediction": pred.round(3)})
    path = resolve(p["paths"]["predictions"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"wrote {len(out)} predictions to {path}")


if __name__ == "__main__":
    main()
