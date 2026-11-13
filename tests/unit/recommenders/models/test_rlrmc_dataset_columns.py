# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest

from recommenders.models.rlrmc.RLRMCdataset import RLRMCdataset


@pytest.mark.parametrize(
    "columns",
    [
        {},
        {"userID": "reader"},
        {"itemID": "article"},
        {"rating": "score"},
        {"userID": "reader", "itemID": "article", "rating": "score"},
    ],
)
@pytest.mark.parametrize("mean_center", [False, True])
@pytest.mark.parametrize("has_validation", [False, True])
def test_rlrmc_custom_columns_preserve_sparse_data(
    columns, mean_center, has_validation
):
    train = pd.DataFrame(
        {"userID": [20, 10, 20], "itemID": [9, 3, 3], "rating": [4.0, 2.0, 5.0]},
        index=[7, 4, 15],
    )
    validation = (
        pd.DataFrame(
            {"userID": [10, 30], "itemID": [9, 7], "rating": [3.0, 1.0]},
            index=[5, 2],
        )
        if has_validation
        else None
    )
    test = pd.DataFrame({"userID": [99], "itemID": [9], "rating": [6.0]})
    reference = RLRMCdataset(train, validation, test, mean_center=mean_center)
    renamed = [
        df.rename(columns=columns) if df is not None else None
        for df in (train, validation, test)
    ]
    original = [df.copy(deep=True) if df is not None else None for df in renamed]

    actual = RLRMCdataset(
        *renamed,
        mean_center=mean_center,
        col_user=columns.get("userID", "userID"),
        col_item=columns.get("itemID", "itemID"),
        col_rating=columns.get("rating", "rating"),
    )

    assert actual.model_param == reference.model_param
    assert actual.user2id == reference.user2id
    assert actual.item2id == reference.item2id
    assert actual.id2user == reference.id2user
    assert actual.id2item == reference.id2item
    assert actual.model_param["train_mean"] == (11.0 / 3 if mean_center else 0)
    for name in ("train", "validation"):
        expected_matrix, actual_matrix = getattr(reference, name), getattr(actual, name)
        if expected_matrix is None:
            assert actual_matrix is None
        else:
            assert actual_matrix.shape == expected_matrix.shape
            np.testing.assert_array_equal(actual_matrix.indptr, expected_matrix.indptr)
            np.testing.assert_array_equal(
                actual_matrix.indices, expected_matrix.indices
            )
            np.testing.assert_allclose(actual_matrix.data, expected_matrix.data)
    for current, before in zip(renamed, original):
        if current is not None:
            pd.testing.assert_frame_equal(current, before)
