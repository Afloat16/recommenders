# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest

from recommenders.models.sar.sar_singlenode import SARSingleNode
from recommenders.utils.python_utils import rescale


@pytest.mark.parametrize("dtype", [np.int8, np.int64, np.float32, np.float64])
@pytest.mark.parametrize("bounds", [(0, 1), (-2, 3)])
def test_rescale_constant_input_maps_to_lower_bound(dtype, bounds):
    values = np.full((2, 3), 4, dtype=dtype)
    original = values.copy()
    with np.errstate(divide="raise", invalid="raise"):
        actual = rescale(values, *bounds)
    np.testing.assert_array_equal(actual, np.full(values.shape, bounds[0]))
    np.testing.assert_array_equal(values, original)


def test_rescale_broadcast_bounds_handle_constant_rows():
    values = np.asarray([[4.0, 4.0, 4.0], [2.0, 6.0, 10.0], [0.0, 0.0, 0.0]])
    lower = np.asarray([[4.0], [2.0], [0.0]])
    upper = np.asarray([[4.0], [10.0], [0.0]])
    with np.errstate(divide="raise", invalid="raise"):
        actual = rescale(values, -1, 1, lower, upper)
    np.testing.assert_allclose(actual, [[-1, -1, -1], [-1, 0, 1], [-1, -1, -1]])


@pytest.mark.parametrize("rating", [0.0, 1.0, 5.0])
def test_sar_constant_rating_normalization_keeps_recommendations(rating):
    ratings = pd.DataFrame(
        {"userID": [0, 0, 1, 1], "itemID": [0, 1, 0, 1], "rating": [rating] * 4}
    )
    model = SARSingleNode(normalize=True)
    model.fit(ratings)
    query = pd.DataFrame({"userID": [0, 1]})
    with np.errstate(divide="raise", invalid="raise"):
        actual = model.score(query)
        recommendations = model.recommend_k_items(query, top_k=2)
    np.testing.assert_array_equal(actual, np.full((2, 2), rating))
    assert len(recommendations) == 4
    np.testing.assert_array_equal(recommendations.prediction, [rating] * 4)
