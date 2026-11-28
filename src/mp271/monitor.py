"""Drift (PSI per numeric column) and v1's MAE per zone on later.csv."""
import numpy as np
import pandas as pd


def psi(expected, actual, bins=10, eps=1e-4):
    """Population Stability Index. Bin edges come from quantiles of `expected`."""
    expected = pd.Series(expected).dropna().to_numpy(dtype=float)
    actual = pd.Series(actual).dropna().to_numpy(dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, bins=edges)[0] / len(expected)
    a = np.histogram(actual, bins=edges)[0] / len(actual)
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_table(train, later, columns, bins=10):
    return {c: round(psi(train[c], later[c], bins), 4) for c in columns}


def mae_by_group(pipe, df, group_col, target, feature_cols):
    out = {}
    for g, part in df.groupby(group_col):
        errors = np.abs(part[target] - pipe.predict(part[feature_cols]))
        out[g] = round(float(np.mean(errors)), 4)
    return out


def main():
    # Imported here so this file loads even before teammates' modules exist.
    from .cleaning import clean_other, clean_train
    from .config import load_params
    from .data import load_raw
    from .features import feature_columns
    from .findings import update_findings
    from .registry import load_registered

    p = load_params()
    raw = load_raw("train", p)
    train, report = clean_train(raw, set(load_raw("holdout", p).columns), p)
    later = clean_other(load_raw("later", p), report, p)

    table = psi_table(train, later, p["features"]["numeric"], p["monitoring"]["psi_bins"])
    most = max(table, key=table.get)

    v1 = load_registered(p, version=1)
    by_zone = mae_by_group(v1, later, p["group_column"], p["target"], feature_columns(p))
    worst = max(by_zone, key=by_zone.get)

    update_findings("monitoring", {
        "psi": table, "most_drifted_feature": most,
        "mae_by_group": by_zone, "worst_served_group": worst,
    })
    print("PSI:", table, "\nmost drifted:", most)
    print("MAE by zone:", by_zone, "\nworst served:", worst)


if __name__ == "__main__":
    main()
