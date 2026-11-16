# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest

from recommenders.models.sar.sar_singlenode import SARSingleNode


@pytest.mark.parametrize("similarity", ["cooccurrence", "jaccard"])
@pytest.mark.parametrize(
    "rating_dtype, population",
    [
        (np.int8, 130),
        (np.uint8, 260),
        (np.int64, 130),
        (np.float32, 130),
        (np.float64, 130),
    ],
)
def test_sar_counts_are_not_limited_by_rating_dtype(
    similarity, rating_dtype, population
):
    shared = population // 2
    ratings = pd.DataFrame(
        {
            "userID": list(range(population)) + list(range(shared)),
            "itemID": ["a"] * population + ["b"] * shared,
            "rating": np.ones(population + shared, dtype=rating_dtype),
        }
    )
    original = ratings.copy(deep=True)
    model = SARSingleNode(similarity_type=similarity)
    model.fit(ratings)

    np.testing.assert_array_equal(model.item_frequencies, [population, shared])
    actual = model.item_similarity
    if similarity == "cooccurrence":
        expected = [[population, shared], [shared, shared]]
        np.testing.assert_array_equal(actual.toarray(), expected)
        if np.issubdtype(rating_dtype, np.floating):
            assert actual.dtype == np.dtype(rating_dtype)
        expected_scores = [[population + shared, 2 * shared], [population, shared]]
    else:
        expected = [[1.0, 0.5], [0.5, 1.0]]
        np.testing.assert_allclose(actual, expected)
        expected_scores = [[1.5, 1.5], [1.0, 0.5]]

    scores = model.score(pd.DataFrame({"userID": [0, population - 1]}))
    np.testing.assert_allclose(scores, expected_scores)
    pd.testing.assert_frame_equal(ratings, original)
