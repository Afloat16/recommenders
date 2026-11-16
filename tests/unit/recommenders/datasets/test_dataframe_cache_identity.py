# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest

from recommenders.datasets.pandas_df_utils import PandasHash, lru_cache_df


def test_cached_dataframe_result_respects_row_labels():
    @lru_cache_df(maxsize=2)
    def labels(frame):
        return tuple(frame.index)

    first = pd.DataFrame({"rating": [1.0, 2.0]}, index=[10, 20])
    second = first.set_axis([30, 40])
    assert labels(first) == (10, 20)
    assert labels(second) == (30, 40)
    assert labels(first.copy()) == (10, 20)
    assert labels.cache_info().hits == 1
    assert labels.cache_info().misses == 2


def test_cached_dataframe_result_respects_dtypes_with_identical_bytes():
    @lru_cache_df(maxsize=2)
    def total(frame):
        return frame["rating"].sum()

    values = np.array([1.0, 2.0], dtype=np.float64)
    floating = pd.DataFrame({"rating": values})
    integral = pd.DataFrame({"rating": values.view(np.int64)})
    assert floating.values.tobytes() == integral.values.tobytes()

    assert total(floating) == 3.0
    assert total(integral) == integral["rating"].sum()
    assert total(floating.copy()) == 3.0
    assert total.cache_info().hits == 1
    assert total.cache_info().misses == 2


def test_cached_dataframe_result_checks_values_on_hash_collisions(monkeypatch):
    monkeypatch.setattr(PandasHash, "__hash__", lambda self: 7)

    @lru_cache_df(maxsize=2)
    def total(frame):
        return frame["rating"].sum()

    assert total(pd.DataFrame({"rating": [1.0]})) == 1.0
    assert total(pd.DataFrame({"rating": [2.0]})) == 2.0
    assert total(pd.DataFrame({"rating": [1.0]})) == 1.0
    assert total.cache_info().hits == 1
    assert total.cache_info().misses == 2


@pytest.mark.parametrize("other", [None, 1, object()])
def test_pandas_hash_equality_with_other_types(other):
    value = PandasHash(pd.DataFrame({"rating": [1.0]}))
    assert value != other
