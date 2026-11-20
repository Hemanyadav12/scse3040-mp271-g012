"""Loading the CSV files and hashing the original training file."""
import hashlib

import pandas as pd

from .config import load_params, resolve


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_raw(name, params=None):
    """name is one of: train, later, holdout."""
    params = params or load_params()
    return pd.read_csv(resolve(params["paths"][name]))


def train_sha256(params=None):
    params = params or load_params()
    return sha256_file(resolve(params["paths"]["train"]))
