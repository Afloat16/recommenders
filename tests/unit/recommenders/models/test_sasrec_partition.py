# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import pytest

from recommenders.models.sasrec.util import SASRecDataSet


@pytest.mark.parametrize("with_time", [False, True])
@pytest.mark.parametrize("valid_size,test_size", [(0, 0), (2, 0), (0, 2), (1, 1)])
def test_split_preserves_interactions_with_empty_holdouts(
    tmp_path, with_time, valid_size, test_size
):
    path = tmp_path / "interactions.txt"
    rows = [(1, item, item) for item in range(1, 6)] + [(2, 6, 6)]
    if with_time:
        rows.reverse()
    path.write_text(
        "\n".join(
            " ".join(str(v) for v in (row if with_time else row[:2])) for row in rows
        )
    )
    data = SASRecDataSet(filename=str(path))
    stats = data.split(valid_size=valid_size, test_size=test_size, min_interactions=3)
    train_end = 5 - valid_size - test_size
    valid_end = 5 - test_size
    assert data.user_train[1] == list(range(1, 6))[:train_end]
    assert data.user_valid[1] == list(range(1, 6))[train_end:valid_end]
    assert data.user_test[1] == list(range(1, 6))[valid_end:]
    assert data.user_train[1] + data.user_valid[1] + data.user_test[1] == list(
        range(1, 6)
    )
    assert data.user_train[2] == [6]
    assert data.user_valid[2] == data.user_test[2] == []
    assert stats["users_with_splits"] == int(valid_size + test_size > 0)
    assert stats["users_train_only"] == 2 - stats["users_with_splits"]
