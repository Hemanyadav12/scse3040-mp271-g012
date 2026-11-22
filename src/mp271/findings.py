"""Helper to read and update sections of reports/findings.yaml safely."""
from pathlib import Path
import yaml


def load_findings(path="reports/findings.yaml"):
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def update_findings(section, data, path="reports/findings.yaml"):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = load_findings(p)
    existing[section] = data
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(existing, f, sort_keys=False)
