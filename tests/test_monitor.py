import numpy as np
import pandas as pd

from mp271.monitor import mae_by_group, psi, psi_table


class _ZeroModel:
    def predict(self, X):
        return np.zeros(len(X))


def test_psi_zero_for_same_data_and_large_for_shift():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 2000)
    assert psi(a, a) < 1e-6
    assert psi(a, a + 3) > 1


def test_psi_table_has_one_value_per_column():
    rng = np.random.default_rng(2)
    train = pd.DataFrame({"x": rng.normal(0, 1, 500), "y": rng.normal(5, 1, 500)})
    later = train.copy()
    table = psi_table(train, later, ["x", "y"])
    assert set(table) == {"x", "y"}
    assert all(v < 1e-6 for v in table.values())


def test_mae_by_group_is_computed_per_zone():
    df = pd.DataFrame({"zone": ["A", "A", "B"], "target": [2, 4, 10], "x": [0, 0, 0]})
    out = mae_by_group(_ZeroModel(), df, "zone", "target", ["x"])
    assert out == {"A": 3.0, "B": 10.0}
