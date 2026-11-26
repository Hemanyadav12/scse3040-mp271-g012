"""Cleaning: applies a fix for every problem the checks found."""

import re

import numpy as np
import pandas as pd

from .checks import run_all_checks


_PREFIX = {
    "sed": "sedan",
    "min": "mini",
    "aut": "auto",
}


def normalize_car_type(value):
    """Normalize spelling variants of car types."""
    key = re.sub(r"[^a-z]", "", str(value).lower())
    return _PREFIX.get(key[:3], key)


def clean_train(raw, holdout_columns, params, report=None):
    """Return cleaned training dataframe and data-quality report."""

    report = report or run_all_checks(
        raw, holdout_columns, params
    )

    df = raw.drop_duplicates().copy()

    # Normalize car types
    s = report["spelling_variants"]

    if s["present"]:
        df[s["column"]] = df[s["column"]].map(
            normalize_car_type
        )

    # Fix unit mix-up
    u = report["unit_mixup"]

    if u["present"]:
        mask = (
            df[params["group_column"]]
            == u["group_value"]
        )

        df.loc[mask, u["column"]] = (
            df.loc[mask, u["column"]] / u["factor"]
        )

    # Replace sentinel values
    sv = report["sentinel_value"]

    if sv["present"]:
        df.loc[
            df[sv["column"]] == sv["value"],
            sv["column"]
        ] = np.nan

    # Handle stuck sensor
    st = report["stuck_sensor"]

    if st["present"]:
        mask = (
            (df["date"] >= st["start_date"])
            & (df["date"] <= st["end_date"])
        )

        df.loc[mask, st["column"]] = np.nan

    # Remove target typos
    if report["target_typos"]["present"]:

        med = df[params["target"]].median()

        df = df[
            df[params["target"]]
            <= params["data_checks"]["typo_median_multiple"] * med
        ]

    # Remove leaking column
    if report["leaking_column"]["present"]:

        df = df.drop(
            columns=[report["leaking_column"]["column"]]
        )

    return (
        df.sort_values(
            ["date", "record_id"],
            kind="stable"
        ).reset_index(drop=True),
        report,
    )


def clean_other(df, report, params):
    """Clean later data and holdout features."""

    df = df.copy()

    df["car_type"] = df["car_type"].map(
        normalize_car_type
    )

    if (
        report["leaking_column"]["present"]
        and report["leaking_column"]["column"] in df.columns
    ):
        df = df.drop(
            columns=[report["leaking_column"]["column"]]
        )

    if "date" in df.columns:
        df = df.sort_values(
            ["date", "record_id"],
            kind="stable"
        )

    return df.reset_index(drop=True)