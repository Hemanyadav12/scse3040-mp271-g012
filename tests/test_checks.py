import numpy as np

from mp271.checks import (check_leaking_column, check_sentinel_value,
                          check_stuck_sensor, check_target_typos, check_unit_mixup)


def test_leaking_column_found(tiny_train):
    holdout = set(tiny_train.columns) - {"pickup_eta_min", "actual_pickup_after_min"}
    r = check_leaking_column(tiny_train, holdout, "pickup_eta_min")
    assert r == {"present": True, "column": "actual_pickup_after_min"}


def test_sentinel_found(tiny_train):
    tiny_train.loc[:4, "rain_mm"] = -999.0
    r = check_sentinel_value(tiny_train, [-1, -999, 999], {"record_id", "pickup_eta_min"})
    assert r["column"] == "rain_mm" and r["value"] == -999 and r["count"] == 5


def test_stuck_sensor_window(tiny_train):
    tiny_train.loc[10:20, "hour"] = 6                              # 11 days, all 6
    r = check_stuck_sensor(tiny_train, {"pickup_eta_min", "actual_pickup_after_min"}, min_days=7)
    assert r["present"] and r["column"] == "hour"
    assert r["start_date"] == tiny_train.loc[10, "date"] and r["end_date"] == tiny_train.loc[20, "date"]


def test_unit_mixup_and_typos(tiny_train, params):
    tiny_train.loc[tiny_train.zone == "Airport", "distance_to_driver_km"] *= 100
    r = check_unit_mixup(tiny_train, "zone", {"record_id"}, [10, 12, 60, 100, 1000], 5)
    assert r["present"] and r["group_value"] == "Airport" and r["factor"] == 100
    tiny_train.loc[0, "pickup_eta_min"] = 500.0
    assert check_target_typos(tiny_train, "pickup_eta_min", 4)["count"] == 1
