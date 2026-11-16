# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest

from recommenders.models.sar.sar_singlenode import SARSingleNode


@pytest.mark.parametrize("rating_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("users", [[0, 1], [1]])
@pytest.mark.parametrize("similarity", ["cooccurrence", "jaccard"])
def test_normalized_sar_scores_support_sparse_similarity(
    rating_dtype, users, similarity
):
    ratings = pd.DataFrame(
        {
            "userID": [0, 0, 1, 1],
            "itemID": [0, 1, 1, 2],
            "rating": np.asarray([1, 5, 4, 2], dtype=rating_dtype),
        }
    )
    original = ratings.copy(deep=True)
    model = SARSingleNode(similarity_type=similarity, normalize=True)
    model.fit(ratings)
    affinity = np.asarray([[1, 5, 0], [0, 4, 2]], dtype=float)[users]
    similarity_matrix = (
        np.asarray([[1, 1, 0], [1, 2, 1], [0, 1, 1]], dtype=float)
        if similarity == "cooccurrence"
        else np.asarray([[1, 0.5, 0], [0.5, 1, 0.5], [0, 0.5, 1]])
    )
    raw_scores = affinity @ similarity_matrix
    counts = (affinity != 0).astype(float) @ similarity_matrix
    lower = counts.min(axis=1, keepdims=True)
    upper = counts.max(axis=1, keepdims=True) * 5
    expected = (raw_scores - lower) / (upper - lower) * 4 + 1

    query = pd.DataFrame({"userID": users})
    actual = model.score(query)
    assert isinstance(actual, np.ndarray)
    np.testing.assert_allclose(actual, expected)
    assert np.isfinite(actual).all()

    recommendations = model.recommend_k_items(query, top_k=3)
    for row, user in enumerate(users):
        selected = recommendations[recommendations.userID == user]
        order = np.argsort(-expected[row])
        np.testing.assert_array_equal(selected.itemID, order)
        np.testing.assert_allclose(selected.prediction, expected[row, order])
    pd.testing.assert_frame_equal(ratings, original)
