"""Run the seven data checks on raw train.csv and record data_issues in reports/findings.yaml."""
import json
from pathlib import Path
import pandas as pd
import yaml

from .checks import run_all_checks
from .findings import update_findings


def main():
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    train_path = Path(params.get("paths", {}).get("train", "data/raw/train.csv"))
    holdout_path = Path(params.get("paths", {}).get("holdout", "data/raw/holdout_features.csv"))

    train = pd.read_csv(train_path)
    holdout_cols = set(pd.read_csv(holdout_path).columns)

    issues = run_all_checks(train, holdout_cols, params)
    update_findings("data_issues", issues, "reports/findings.yaml")
    print(json.dumps(issues, indent=2))


if __name__ == "__main__":
    main()
