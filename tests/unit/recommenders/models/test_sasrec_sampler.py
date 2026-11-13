# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import numpy as np
import pytest

from recommenders.models.sasrec.sampler import WarpSampler


@pytest.mark.parametrize(
    "histories, usernum, itemnum",
    [({}, 0, 3), ({1: [1]}, 1, 3), ({1: [1, 2, 3]}, 1, 3)],
)
def test_sampler_rejects_no_trainable_users(histories, usernum, itemnum):
    sampler = None
    try:
        with pytest.raises(ValueError, match="negative"):
            sampler = WarpSampler(histories, usernum, itemnum, batch_size=4)
    finally:
        if sampler is not None:
            sampler.close()


@pytest.mark.parametrize("workers", [1, 2])
def test_sampler_skips_untrainable_users_and_preserves_training_pairs(workers):
    # Users 1 and 2 cannot form a positive/negative training pair. Missing user
    # IDs should not be sampled, and repeated interactions are not new items.
    histories = {1: [1, 2, 3], 2: [1], 5: [2, 2, 1]}
    sampler = WarpSampler(
        histories, usernum=5, itemnum=3, batch_size=16, maxlen=4, n_workers=workers
    )
    try:
        for _ in range(3):
            users, sequences, positives, negatives = sampler.result_queue.get(timeout=5)
            assert set(users) == {5}
            for sequence, positive, negative in zip(sequences, positives, negatives):
                np.testing.assert_array_equal(sequence, [0, 0, 2, 2])
                np.testing.assert_array_equal(positive, [0, 0, 2, 1])
                np.testing.assert_array_equal(negative, [0, 0, 3, 3])
    finally:
        sampler.close()


def test_sampler_keeps_all_eligible_users():
    histories = {1: [1, 2], 2: [2, 3]}
    np.random.seed(17)
    sampler = WarpSampler(histories, usernum=2, itemnum=4, batch_size=32, maxlen=2)
    try:
        users, sequences, positives, negatives = sampler.result_queue.get(timeout=5)
        assert set(users) == {1, 2}
        for user, sequence, positive, negative in zip(
            users, sequences, positives, negatives
        ):
            assert sequence[-1] == histories[user][0]
            assert positive[-1] == histories[user][1]
            assert 1 <= negative[-1] <= 4
            assert negative[-1] not in histories[user]
    finally:
        sampler.close()
