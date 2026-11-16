# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest

from recommenders.models.deeprec.DataModel.ImplicitCF import ImplicitCF


def _dataset(include_filtered_user, include_test_user, columns):
    user, item, rating = columns
    records = [("good", "a", 1), ("other", "b", 2)]
    if include_filtered_user:
        records.insert(0, ("inactive", "a", 0))
        records.append(("inactive", "c", -1))
    train = pd.DataFrame(records, columns=columns)
    test = (
        pd.DataFrame([("test-only", "c", 1)], columns=columns)
        if include_test_user
        else None
    )
    return ImplicitCF(
        train, test, col_user=user, col_item=item, col_rating=rating, seed=1
    )


@pytest.mark.parametrize("filtered", [False, True])
@pytest.mark.parametrize("test_only", [False, True])
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize(
    "columns", [("userID", "itemID", "rating"), ("customer", "product", "value")]
)
def test_graph_sampling_preserves_actual_positive_user_ids(
    filtered, test_only, batch_size, columns
):
    data = _dataset(filtered, test_only, columns)
    user, item, _ = columns
    original = data.train.copy(deep=True)
    users, positives, negatives = data.train_loader(batch_size)

    assert len(users) == len(positives) == len(negatives) == batch_size
    observed = set(data.train[[user, item]].itertuples(index=False, name=None))
    for sampled_user, positive, negative in zip(users, positives, negatives):
        assert (sampled_user, positive) in observed
        assert (sampled_user, negative) not in observed
        assert 0 <= negative < data.n_items
    assert data.n_users_in_train == 2
    pd.testing.assert_frame_equal(data.train, original)


def test_graph_sampling_rejects_dataset_without_positive_users():
    data = ImplicitCF(
        pd.DataFrame({"userID": [1, 2], "itemID": [10, 20], "rating": [0, -1]})
    )

    with pytest.raises(ValueError, match="positive interactions"):
        data.train_loader(3)


def test_graph_sampling_keeps_full_catalog_negative_guard():
    data = ImplicitCF(
        pd.DataFrame({"userID": [1, 1], "itemID": [10, 20], "rating": [1, 1]})
    )

    with pytest.raises(ValueError, match="every item"):
        data.train_loader(1)


def test_lightgcn_trains_with_filtered_users_and_test_only_users():
    torch = pytest.importorskip("torch")
    from recommenders.models.deeprec.models.graphrec.lightgcn import LightGCN

    data = _dataset(True, True, ("userID", "itemID", "rating"))
    model = LightGCN(
        data.n_users,
        data.n_items,
        data.get_norm_adj_mat(),
        embed_size=4,
        n_layers=1,
        seed=4,
    )
    before = model.user_embedding.weight.detach().clone()

    model.fit(data, epochs=1, batch_size=3, eval_epoch=-1)

    assert not torch.equal(model.user_embedding.weight, before)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    assert np.isfinite(
        model.score(data.train[data.col_user].unique(), remove_seen=False)
    ).all()
