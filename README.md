# SCSE3040 MLOps Mini Project — MP271 (Cab Pick-up Time, Surat)

- **Group ID:** G012
- **Problem ID:** MP271

## Team Members
- S24CSEU1551 — Heman.
- S24CSEU1568 — PRATHAMCHAURASIA
- S24CSEU1577 — AkshejGupta
- S24CSEU1571 — AnanyaJain

## What the data checks found

| Issue | Present | Details |
|---|---|---|
| `leaking_column` | Yes | column=actual_pickup_after_min |
| `unit_mixup` | No | None |
| `sentinel_value` | Yes | column=distance_to_driver_km, value=999, count=67 |
| `duplicate_rows` | Yes | count=50 |
| `stuck_sensor` | Yes | column=hour, start_date=2026-03-26, end_date=2026-04-12 |
| `target_typos` | No | None |
| `spelling_variants` | Yes | column=car_type, true_levels=3 |
