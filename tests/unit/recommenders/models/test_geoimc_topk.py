# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from recommenders.models.geoimc.geoimc_data import DataPtr
from recommenders.models.geoimc.geoimc_predict import Inferer


@pytest.mark.parametrize("k", [0, 1, 3, 10])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_topk_selects_at_most_available_items(k, dtype):
    users = np.array([[1, 0], [0, 1]], dtype=dtype)
    items = np.array([[3, -4], [1, 7], [-2, 2]], dtype=dtype)
    before_users, before_items = users.copy(), items.copy()

    result = Inferer(k=k, transformation="topk").infer([users, items], W=None)

    expected_by_k = {
        0: [[0, 0, 0], [0, 0, 0]],
        1: [[1, 0, 0], [0, 1, 0]],
        3: [[1, 1, 1], [1, 1, 1]],
        10: [[1, 1, 1], [1, 1, 1]],
    }
    np.testing.assert_array_equal(result, expected_by_k[k])
    assert result.dtype == dtype
    np.testing.assert_array_equal(users, before_users)
    np.testing.assert_array_equal(items, before_items)


@pytest.mark.parametrize("shape", [(0, 3), (2, 0), (0, 0)])
@pytest.mark.parametrize("k", [0, 10])
def test_topk_empty_catalogs_preserve_output_shape(shape, k):
    users = np.empty((shape[0], 2))
    items = np.empty((shape[1], 2))

    result = Inferer(k=k, transformation="topk").infer([users, items], W=None)

    assert result.shape == shape
    assert result.size == 0


@pytest.mark.parametrize("k", [0, 1, 10])
def test_topk_with_real_data_pointer(k):
    users = np.array([[1.0, 0.0], [0.0, 1.0]])
    items = np.array([[3.0, -4.0], [1.0, 7.0], [-2.0, 2.0]])
    data = DataPtr(csr_matrix((2, 3)), [users, items])
    identity = np.eye(2)

    result = Inferer(k=k, transformation="topk").infer(
        data, W=[identity, identity, identity]
    )

    expected = (
        [[0, 0, 0], [0, 0, 0]]
        if k == 0
        else [[1, 0, 0], [0, 1, 0]]
        if k == 1
        else [[1, 1, 1], [1, 1, 1]]
    )
    np.testing.assert_array_equal(result, expected)


def test_topk_rejects_negative_count():
    with pytest.raises(ValueError, match="nonnegative"):
        Inferer(k=-1, transformation="topk").infer(
            [np.ones((2, 2)), np.ones((3, 2))], W=None
        )


@pytest.mark.parametrize("transformation", ["", "mean"])
def test_other_transformations_keep_existing_scores(transformation):
    users = np.eye(2)
    items = np.array([[3.0, -4.0], [1.0, 7.0], [-2.0, 2.0]])

    result = Inferer(transformation=transformation).infer([users, items], W=None)

    expected = (
        [[3.0, 1.0, -2.0], [-4.0, 7.0, 2.0]]
        if not transformation
        else [[1, 0, 0], [0, 1, 1]]
    )
    np.testing.assert_array_equal(result, expected)
