# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest

from recommenders.datasets.python_splitters import (
    numpy_stratified_split,
    python_stratified_split,
)


def _ratings():
    return pd.DataFrame(
        {
            "userID": [1] * 5 + [2] * 5,
            "itemID": list(range(5)) * 2,
            "rating": range(1, 11),
        }
    )


def _matrix():
    return np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 0.0], [0.0, 6.0, 7.0, 8.0, 9.0, 10.0]])


@pytest.mark.parametrize("seed", [0, 42, None])
@pytest.mark.parametrize("kind", ["pandas", "numpy"])
def test_stratified_split_preserves_global_random_state(kind, seed):
    original_state = np.random.get_state()
    try:
        np.random.seed(918)
        np.random.normal(size=3)
        expected_state = np.random.get_state()

        if kind == "pandas":
            python_stratified_split(_ratings(), ratio=0.6, seed=seed)
        else:
            numpy_stratified_split(_matrix(), ratio=0.6, seed=seed)

        actual_state = np.random.get_state()
        assert actual_state[0] == expected_state[0]
        np.testing.assert_array_equal(actual_state[1], expected_state[1])
        assert actual_state[2:] == expected_state[2:]
    finally:
        np.random.set_state(original_state)


@pytest.mark.parametrize(
    "seed, expected_indices",
    [
        (0, [[4, 3, 0, 9, 6, 5], [2, 1, 7, 8]]),
        (42, [[4, 0, 3, 6, 5, 8], [2, 1, 9, 7]]),
    ],
)
def test_python_stratified_split_preserves_seeded_partitions(seed, expected_indices):
    ratings = _ratings()
    original = ratings.copy(deep=True)
    first = python_stratified_split(ratings, ratio=0.6, seed=seed)
    second = python_stratified_split(ratings, ratio=0.6, seed=seed)

    for actual, repeated, indices in zip(first, second, expected_indices):
        pd.testing.assert_frame_equal(actual, original.loc[indices])
        pd.testing.assert_frame_equal(actual, repeated)
    pd.testing.assert_frame_equal(ratings, original)


@pytest.mark.parametrize(
    "seed, expected_test",
    [
        (0, [[1.0, 0.0, 3.0, 0.0, 0.0, 0.0], [0.0, 6.0, 0.0, 8.0, 0.0, 0.0]]),
        (42, [[0.0, 2.0, 0.0, 0.0, 5.0, 0.0], [0.0, 0.0, 7.0, 0.0, 9.0, 0.0]]),
    ],
)
def test_numpy_stratified_split_preserves_seeded_partitions(seed, expected_test):
    ratings = _matrix()
    original = ratings.copy()
    train, test = numpy_stratified_split(ratings, ratio=0.6, seed=seed)
    repeated_train, repeated_test = numpy_stratified_split(
        ratings, ratio=0.6, seed=seed
    )

    np.testing.assert_array_equal(test, expected_test)
    np.testing.assert_array_equal(train + test, original)
    np.testing.assert_array_equal(train * test, np.zeros_like(original))
    np.testing.assert_array_equal(train, repeated_train)
    np.testing.assert_array_equal(test, repeated_test)
    np.testing.assert_array_equal(ratings, original)
