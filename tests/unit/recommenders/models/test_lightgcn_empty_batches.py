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


def _model(columns=("userID", "itemID", "rating", "prediction")):
    train = pd.DataFrame(
        [(10, 100, 1), (10, 200, 1), (20, 200, 1), (20, 300, 1)],
        columns=columns[:3],
    )
    user, item, rating, prediction = columns
    data = ImplicitCF(
        train,
        col_user=user,
        col_item=item,
        col_rating=rating,
        col_prediction=prediction,
        seed=2,
    )
    model = LightGCN(
        data.n_users,
        data.n_items,
        data.get_norm_adj_mat(),
        embed_size=3,
        n_layers=1,
        seed=3,
    )
    model.data = data
    return model


@pytest.mark.parametrize("input_dtype", [np.int64, np.float64, object])
@pytest.mark.parametrize("model_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("remove_seen", [False, True])
def test_lightgcn_scores_empty_user_batches(input_dtype, model_dtype, remove_seen):
    model = _model().to(dtype=model_dtype)
    users = np.array([], dtype=input_dtype)

    scores = model.score(users, remove_seen=remove_seen)

    assert scores.shape == (0, 3)
    assert scores.dtype == (np.float32 if model_dtype == torch.float32 else np.float64)
    assert users.size == 0


@pytest.mark.parametrize("use_id", [False, True])
@pytest.mark.parametrize("input_dtype", [np.int64, object])
@pytest.mark.parametrize(
    "columns",
    [
        ("userID", "itemID", "rating", "prediction"),
        ("customer", "product", "feedback", "score"),
    ],
)
def test_lightgcn_recommends_for_empty_user_frame(use_id, input_dtype, columns):
    model = _model(columns)
    frame = pd.DataFrame({columns[0]: pd.Series(dtype=input_dtype)})
    before = frame.copy(deep=True)

    result = model.recommend_k_items(frame, top_k=2, use_id=use_id)

    assert result.empty
    assert list(result.columns) == [columns[0], columns[1], columns[3]]
    pd.testing.assert_frame_equal(frame, before)


def test_lightgcn_nonempty_scoring_keeps_seen_item_mask():
    model = _model()
    scores = model.score(np.array([0, 1]))
    assert scores.shape == (2, 3)
    assert np.isneginf(scores[0, 0]) and np.isneginf(scores[0, 1])
    assert np.isneginf(scores[1, 1]) and np.isneginf(scores[1, 2])
    assert np.isfinite(scores[0, 2]) and np.isfinite(scores[1, 0])


def test_lightgcn_empty_scoring_still_requires_dataset():
    model = _model()
    model.data = None
    with pytest.raises(RuntimeError, match="requires a dataset"):
        model.score(np.array([], dtype=np.int64))


def test_lightgcn_nonempty_scoring_still_rejects_missing_users():
    with pytest.raises(ValueError, match="not in the training set"):
        _model().score(np.array([np.nan]))
