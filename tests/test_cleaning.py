import numpy as np
import pandas as pd

from mp271.checks import run_all_checks
from mp271.cleaning import clean_train


def test_clean_train_fixes_all_issues(tiny_train, params):
    df = tiny_train.copy()
    # Inject sentinel value
    df.loc[0:3, "rain_mm"] = -999.0
    # Inject stuck sensor window
    df.loc[10:19, "hour"] = 6
    # Inject spelling variants
    df.loc[0, "car_type"] = "MINI"
    df.loc[1, "car_type"] = "Mini "
    # Inject duplicate row
    df = pd.concat([df, df.iloc[[2]]], ignore_index=True)

    holdout_cols = set(tiny_train.columns) - {"pickup_eta_min", "actual_pickup_after_min"}
    issues = run_all_checks(df, holdout_cols, params)
    cleaned, report = clean_train(df, holdout_cols, params, report=issues)

    # Leaking column dropped
    assert "actual_pickup_after_min" not in cleaned.columns
    # Duplicates removed
    assert len(cleaned) == len(tiny_train)
    # Sentinel replaced with NaN
    assert cleaned["rain_mm"].isna().sum() >= 4
    # Stuck sensor window replaced with NaN
    assert cleaned.loc[10:19, "hour"].isna().all()
    # Spelling variants collapsed
    assert set(cleaned["car_type"].unique()).issubset({"mini", "sedan", "auto"})
