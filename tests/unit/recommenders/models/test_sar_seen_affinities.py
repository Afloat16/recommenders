# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest

from recommenders.models.sar import SAR


@pytest.fixture(params=["jaccard", "cooccurrence"])
def affinity_model(request):
    return SAR(similarity_type=request.param)


@pytest.fixture(params=[np.float32, np.float64, np.int64])
def signed_training_data(request):
    return pd.DataFrame(
        {
            "userID": [
                "negative",
                "negative",
                "zero",
                "zero",
                "cancelled",
                "cancelled",
                "cancelled",
                "complete",
                "complete",
                "complete",
            ],
            "itemID": [10, 20, 20, 30, 10, 10, 30, 10, 20, 30],
            "rating": np.asarray(
                [-2, 2, 0, 1, -1, 1, 2, -3, 0, 2], dtype=request.param
            ),
        }
    )


def test_remove_seen_masks_observations_independently_of_affinity(
    affinity_model, signed_training_data
):
    model = affinity_model
    model.fit(signed_training_data)
    users = pd.DataFrame(
        {"userID": ["zero", "negative", "cancelled", "complete", "zero"]}
    )
    unmasked = np.asarray(model.score(users)).copy()
    affinity = model.user_affinity.copy()

    with np.errstate(invalid="raise"):
        masked = np.asarray(model.score(users, remove_seen=True))

    assert masked.shape == (4, 3)
    for row, user in enumerate(users.userID.unique()):
        seen = set(
            signed_training_data.loc[signed_training_data.userID == user, "itemID"]
        )
        for item, col in model.item2index.items():
            if item in seen:
                assert np.isneginf(masked[row, col])
            else:
                assert masked[row, col] == unmasked[row, col]
    np.testing.assert_array_equal(model.score(users), unmasked)
    np.testing.assert_array_equal(model.user_affinity.data, affinity.data)
    np.testing.assert_array_equal(model.user_affinity.indices, affinity.indices)
    np.testing.assert_array_equal(model.user_affinity.indptr, affinity.indptr)


def test_recommendations_exclude_negative_and_zero_rated_items(
    affinity_model, signed_training_data
):
    model = affinity_model
    model.fit(signed_training_data)
    users = pd.DataFrame({"userID": ["negative", "zero", "cancelled", "complete"]})
    recommendations = model.recommend_k_items(users, top_k=3, remove_seen=True)

    expected = {("negative", 30), ("zero", 10), ("cancelled", 20)}
    assert (
        set(recommendations[["userID", "itemID"]].itertuples(index=False, name=None))
        == expected
    )
    assert np.isfinite(recommendations.prediction).all()
