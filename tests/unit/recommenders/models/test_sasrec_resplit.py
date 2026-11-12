# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import copy

import pytest

from recommenders.models.sasrec.util import SASRecDataSet


@pytest.mark.parametrize("with_time", [False, True])
@pytest.mark.parametrize("valid_size,test_size", [(1, 1), (2, 1)])
def test_sasrec_resplit_matches_fresh_dataset(
    tmp_path, with_time, valid_size, test_size
):
    path = tmp_path / "interactions.txt"
    interactions = [
        (1, 2),
        (1, 4),
        (2, 3),
        (1, 5),
        (2, 6),
        (1, 7),
        (2, 8),
        (2, 9),
        (1, 10),
    ]
    path.write_text(
        "".join(
            f"{user} {item}" + (f" {index}" if with_time else "") + "\n"
            for index, (user, item) in enumerate(interactions)
        ),
        encoding="utf-8",
    )
    reused = SASRecDataSet(filename=str(path))
    reused.split()
    previous = copy.deepcopy(reused.user_train)
    previous_reference = reused.user_train
    stats = reused.split(valid_size=valid_size, test_size=test_size, min_interactions=4)
    fresh = SASRecDataSet(filename=str(path))
    expected_stats = fresh.split(
        valid_size=valid_size, test_size=test_size, min_interactions=4
    )

    assert reused.User == fresh.User
    assert reused.user_train == fresh.user_train
    assert reused.user_valid == fresh.user_valid
    assert reused.user_test == fresh.user_test
    assert stats == expected_stats
    # Callers retaining the previous partition should keep the original result.
    assert previous_reference == previous


@pytest.mark.parametrize("with_time", [False, True])
def test_sasrec_reload_removes_deleted_users_and_items(tmp_path, with_time):
    path = tmp_path / "interactions.txt"

    def write(rows):
        path.write_text(
            "".join(
                f"{user} {item}" + (f" {i}" if with_time else "") + "\n"
                for i, (user, item) in enumerate(rows)
            ),
            encoding="utf-8",
        )

    write([(1, 1), (1, 2), (1, 3), (5, 8), (5, 9), (5, 10)])
    reused = SASRecDataSet(filename=str(path))
    reused.split()
    write([(1, 1), (1, 2), (1, 3)])
    actual = reused.split()
    fresh = SASRecDataSet(filename=str(path))
    expected = fresh.split()
    assert actual == expected
    assert reused.User == fresh.User
    assert reused.Items == fresh.Items
    assert reused.user_train == fresh.user_train
    assert reused.user_valid == fresh.user_valid
    assert reused.user_test == fresh.user_test
