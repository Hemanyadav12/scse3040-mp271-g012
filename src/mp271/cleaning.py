"""Clean train.csv (all steps) or inference features (feature-side steps only).
Every rule is driven by the data_issues report so nothing is hardcoded."""
import re

import numpy as np
import pandas as pd


def _canonical_map(series):
    """Map messy spelling variants (e.g. 'Auto', 'AUTO', 'auto ') to clean lowercase."""
    mapping = {}
    for val in series.dropna().unique():
        clean = re.sub(r"[^a-z0-9]", "", str(val).lower())
        mapping[val] = clean
    return mapping


def clean_features(df, issues, params):
    """Clean feature columns (safe for both training data and inference rows)."""
    out = df.copy()

    # 1. Drop leaking column if present in dataframe
    leak = issues.get("leaking_column", {})
    if leak.get("present") and leak.get("column") in out.columns:
        out = out.drop(columns=[leak["column"]])

    # 2. Fix unit mix-up in specific group
    unit = issues.get("unit_mixup", {})
    if unit.get("present") and unit.get("column") in out.columns:
        group_col = params["group_column"]
        col = unit["column"]
        mask = out[group_col] == unit["group_value"]
        out.loc[mask, col] = out.loc[mask, col] / float(unit["factor"])

    # 3. Replace sentinel value with NaN
    sent = issues.get("sentinel_value", {})
    if sent.get("present") and sent.get("column") in out.columns:
        col = sent["column"]
        out.loc[out[col] == sent["value"], col] = np.nan

    # 4. Blank out stuck sensor window with NaN
    stuck = issues.get("stuck_sensor", {})
    if stuck.get("present") and stuck.get("column") in out.columns and "date" in out.columns:
        col = stuck["column"]
        mask = (out["date"] >= stuck["start_date"]) & (out["date"] <= stuck["end_date"])
        out.loc[mask, col] = np.nan

    # 5. Normalize spelling variants
    spell = issues.get("spelling_variants", {})
    if spell.get("present") and spell.get("column") in out.columns:
        col = spell["column"]
        out[col] = out[col].map(_canonical_map(out[col]))

    return out


def clean_train(train, issues, params):
    """Full cleaning pipeline for training data (deduplication + feature cleaning + target typos)."""
    out = train.copy()

    # 1. Drop exact duplicate rows
    dup = issues.get("duplicate_rows", {})
    if dup.get("present"):
        out = out.drop_duplicates().reset_index(drop=True)

    # 2. Clean features (leakage, units, sentinels, stuck sensor, spelling)
    out = clean_features(out, issues, params)

    # 3. Drop extreme target typos
    typo = issues.get("target_typos", {})
    target = params["target"]
    if typo.get("present") and target in out.columns:
        mult = params["data_checks"]["typo_median_multiple"]
        med = out[target].median()
        out = out[out[target] <= mult * med].reset_index(drop=True)

    return out
