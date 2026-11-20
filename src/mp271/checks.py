"""Detectors for the seven kinds of data problem. Every answer in
findings.yaml comes from these functions running on the real train.csv."""
import re

import numpy as np
import pandas as pd


def _num(v):
    """Make numpy numbers plain Python numbers (so YAML stays clean)."""
    v = float(v)
    return int(v) if v.is_integer() else round(v, 4)


def _feature_cols(df, exclude):
    return [c for c in df.select_dtypes(include="number").columns if c not in exclude]


def check_leaking_column(train, holdout_columns, target, threshold=0.95):
    best = None
    for c in train.columns:
        if c == target or c in holdout_columns:
            continue
        if pd.api.types.is_numeric_dtype(train[c]):
            r = abs(train[c].corr(train[target]))
            if r >= threshold and (best is None or r > best[1]):
                best = (c, r)
    return {"present": best is not None, "column": best[0] if best else None}


def check_sentinel_value(train, candidates, exclude, min_count=3):
    hits = []
    for c in _feature_cols(train, exclude):
        for v in candidates:
            n = int((train[c] == v).sum())
            if n >= min_count:
                hits.append((n, c, v))
    if not hits:
        return {"present": False, "column": None, "value": None, "count": None}
    n, c, v = max(hits)
    return {"present": True, "column": c, "value": _num(v), "count": n}


def check_duplicate_rows(train):
    n = int(train.duplicated().sum())
    return {"present": n > 0, "count": n if n > 0 else None}


def check_spelling_variants(train, skip_columns):
    """Find the text column whose raw levels collapse to fewer real levels."""
    best = None
    for c in train.select_dtypes(include=["object", "string"]).columns:
        if c in skip_columns:
            continue
        raw = train[c].dropna().unique()
        keys = {re.sub(r"[^a-z]", "", str(s).lower())[:3] for s in raw}
        if len(raw) > len(keys) and (best is None or len(raw) - len(keys) > best[2]):
            best = (c, len(keys), len(raw) - len(keys))
    if best is None:
        return {"present": False, "column": None, "true_levels": None}
    return {"present": True, "column": best[0], "true_levels": best[1]}


def check_target_typos(train, target, multiple=4):
    med = train[target].median()
    n = int((train[target] > multiple * med).sum())
    return {"present": n > 0, "count": n if n > 0 else None}


def check_stuck_sensor(train, exclude, min_days=7):
    """A column whose value is identical on every row of many consecutive days."""
    best = None
    for c in _feature_cols(train, exclude):
        per_day = train.groupby("date")[c].agg(["nunique", "first"]).sort_index()
        run_start, run_val, prev = None, None, None
        days = list(per_day.index)
        for i, d in enumerate(days):
            ok = per_day.loc[d, "nunique"] == 1
            val = per_day.loc[d, "first"]
            if ok and run_start is not None and val == run_val:
                prev = d
            else:
                if run_start is not None:
                    best = _keep(best, c, run_start, prev, min_days)
                run_start, run_val, prev = (d, val, d) if ok else (None, None, None)
        if run_start is not None:
            best = _keep(best, c, run_start, prev, min_days)
    if best is None:
        return {"present": False, "column": None, "start_date": None, "end_date": None}
    return {"present": True, "column": best[0], "start_date": best[1], "end_date": best[2]}


def _keep(best, col, start, end, min_days):
    span = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    if span >= min_days and (best is None or span > best[3]):
        return (col, str(start), str(end), span)
    return best


def check_unit_mixup(train, group_col, exclude, factors, ratio_threshold=5):
    """One group whose median for a column is ~10/12/60/100/1000 x the others."""
    best = None
    for c in _feature_cols(train, exclude):
        med = train.groupby(group_col)[c].median()
        if (med <= 0).any() or len(med) < 3:
            continue
        for g in med.index:
            others = med.drop(g).median()
            ratio = med[g] / others
            big = max(ratio, 1 / ratio)
            if big >= ratio_threshold:
                f = min(factors, key=lambda x: abs(x - big))
                if abs(f - big) / f < 0.3:
                    factor = f if ratio > 1 else 1 / f
                    if best is None or big > best[3]:
                        best = (c, g, _num(factor), big)
    if best is None:
        return {"present": False, "column": None, "group_value": None, "factor": None}
    return {"present": True, "column": best[0], "group_value": best[1], "factor": best[2]}


def run_all_checks(train, holdout_columns, params):
    """Returns the data_issues block of findings.yaml (plus nothing else)."""
    dc = params["data_checks"]
    target = params["target"]
    group_col = params["group_column"]
    ids = set(params["id_columns"])

    leak = check_leaking_column(train, holdout_columns, target, dc["leak_corr_threshold"])
    skip = ids | {target} | ({leak["column"]} if leak["present"] else set())

    dedup = train.drop_duplicates()
    sentinel = check_sentinel_value(train, dc["sentinel_candidates"], skip, dc["sentinel_min_count"])

    work = dedup.copy()
    if sentinel["present"]:
        work.loc[work[sentinel["column"]] == sentinel["value"], sentinel["column"]] = np.nan

    return {
        "leaking_column": leak,
        "unit_mixup": check_unit_mixup(work, group_col, skip, dc["unit_factors"], dc["unit_ratio_threshold"]),
        "sentinel_value": sentinel,
        "duplicate_rows": check_duplicate_rows(train),
        "stuck_sensor": check_stuck_sensor(work, skip, dc["stuck_min_days"]),
        "target_typos": check_target_typos(dedup, target, dc["typo_median_multiple"]),
        "spelling_variants": check_spelling_variants(train, ids),
    }


if __name__ == "__main__":
    import json

    from .config import load_params
    from .data import load_raw

    p = load_params()
    print(json.dumps(run_all_checks(load_raw("train", p), set(load_raw("holdout", p).columns), p), indent=2))
