# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest

from recommenders.models.deeprec.DataModel.ImplicitCF import ImplicitCF

torch = pytest.importorskip("torch")
LightGCN = pytest.importorskip(
    "recommenders.models.deeprec.models.graphrec.lightgcn"
).LightGCN

DEFAULT_COLUMNS = ("userID", "itemID", "rating", "prediction")
CUSTOM_COLUMNS = ("customer id", "product-code", "observed rating", "model score")


def _model(columns):
    user, item, rating, prediction = columns
    train = pd.DataFrame([(10, 100, 1), (20, 200, 1)], columns=columns[:3])
    test = pd.DataFrame([(10, 300, 1), (20, 100, 1), (20, 300, 1)], columns=columns[:3])
    data = ImplicitCF(
        train,
        test,
        col_user=user,
        col_item=item,
        col_rating=rating,
        col_prediction=prediction,
        seed=1,
    )
    model = LightGCN(
        data.n_users,
        data.n_items,
        data.get_norm_adj_mat(),
        embed_size=2,
        n_layers=0,
        seed=2,
    )
    with torch.no_grad():
        model.user_embedding.weight.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=model.device)
        )
        model.item_embedding.weight.copy_(
            torch.tensor([[10.0, 3.0], [3.0, 10.0], [2.0, 2.0]], device=model.device)
        )
    model.data = data
    return model


@pytest.mark.parametrize(
    "columns",
    [DEFAULT_COLUMNS, CUSTOM_COLUMNS]
    + [
        tuple(
            CUSTOM_COLUMNS[i] if i == changed else DEFAULT_COLUMNS[i] for i in range(4)
        )
        for changed in range(4)
    ],
)
def test_lightgcn_evaluation_uses_dataset_columns(columns):
    model = _model(columns)
    before = model.data.test.copy(deep=True)

    actual = model.run_eval(top_k=2, metrics=["map", "recall", "precision", "ndcg"])

    # User 10 hits only rank 2; user 20 hits both ranks 1 and 2.
    expected = [0.75, 1.0, 0.75, (1.0 / np.log2(3.0) + 1.0) / 2.0]
    assert actual == pytest.approx(expected)
    pd.testing.assert_frame_equal(model.data.test, before)


@pytest.mark.parametrize("columns", [DEFAULT_COLUMNS, CUSTOM_COLUMNS])
def test_lightgcn_default_metric_order_with_dataset_columns(columns):
    actual = _model(columns).run_eval(top_k=2)
    assert actual == pytest.approx([1.0, (1.0 / np.log2(3.0) + 1.0) / 2.0, 0.75, 0.75])


@pytest.mark.parametrize("columns", [DEFAULT_COLUMNS, CUSTOM_COLUMNS])
def test_lightgcn_periodic_training_evaluation_with_dataset_columns(columns):
    model = _model(columns)
    before = model.user_embedding.weight.detach().clone()

    model.fit(model.data, epochs=1, batch_size=2, eval_epoch=1, top_k=2)

    assert not torch.equal(model.user_embedding.weight, before)
    assert np.isfinite(model.run_eval(top_k=2)).all()
