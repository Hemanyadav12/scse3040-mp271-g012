"""JSON logging to stdout: one JSON object per line."""
import json
import logging
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload)


class StdoutHandler(logging.StreamHandler):
    """Always writes to the *current* sys.stdout (so it also works under pytest/docker)."""

    def __init__(self):
        super().__init__(sys.stdout)

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, value):
        pass


def get_json_logger(name="mp271.api"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = StdoutHandler()  # screen only, never a file
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
