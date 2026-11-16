# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import math

import numpy as np
import pandas as pd
import pytest

from recommenders.evaluation.python_evaluation import ndcg_at_k


@pytest.mark.parametrize(
    "rating_dtype",
    [
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.int64,
        np.float32,
        np.float64,
    ],
)
@pytest.mark.parametrize("k", [2, 3])
@pytest.mark.parametrize("discount", ["loge", "log2"])
def test_exponential_ndcg_does_not_overflow_rating_dtype(rating_dtype, k, discount):
    truth = pd.DataFrame(
        {
            "userID": [0, 0, 0],
            "itemID": [0, 1, 2],
            "rating": np.asarray([9, 8, 3], dtype=rating_dtype),
        }
    )
    original = truth.copy(deep=True)
    predictions = pd.DataFrame(
        {"userID": [0, 0, 0], "itemID": [1, 0, 2], "prediction": [3.0, 2.0, 1.0]}
    )
    logarithm = math.log if discount == "loge" else math.log2

    def dcg(grades):
        return sum(
            (2.0 ** float(grade) - 1.0) / logarithm(rank + 1)
            for rank, grade in enumerate(grades[:k], start=1)
        )

    expected = dcg([8, 9, 3]) / dcg([9, 8, 3])
    actual = ndcg_at_k(truth, predictions, k=k, score_type="exp", discfun_type=discount)
    assert actual == pytest.approx(expected, rel=1e-6)
    assert 0.0 <= actual <= 1.0
    pd.testing.assert_frame_equal(truth, original)
