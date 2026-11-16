# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest

from recommenders.datasets.sparse import AffinityMatrix

torch = pytest.importorskip("torch")
StandardVAE = pytest.importorskip("recommenders.models.vae.standard_vae").StandardVAE
MultVAE = pytest.importorskip("recommenders.models.vae.multinomial_vae").MultVAE
RBM = pytest.importorskip("recommenders.models.rbm.rbm").RBM


def _model(model_class, n_items=3):
    if model_class is RBM:
        return RBM([1, 2, 3, 4, 5], n_items, hidden_units=3, seed=7)
    return model_class(
        n_users=2,
        original_dim=n_items,
        intermediate_dim=3,
        latent_dim=2,
        n_epochs=1,
        batch_size=2,
        drop_encoder=0,
        drop_decoder=0,
        seed=7,
        verbose=0,
    )


def _recommend(model, inputs, k, remove_seen=True):
    if isinstance(model, RBM):
        return model.recommend_k_items(inputs, top_k=k, remove_seen=remove_seen)
    return model.recommend_k_items(inputs, k=k, remove_seen=remove_seen)


@pytest.mark.parametrize("model_class", [StandardVAE, MultVAE, RBM])
@pytest.mark.parametrize("k", [1, 3, 10])
@pytest.mark.parametrize("remove_seen", [False, True])
def test_model_topk_matches_full_sort(model_class, k, remove_seen):
    model = _model(model_class)
    inputs = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    before = inputs.copy()
    scores = model.predict(inputs) if isinstance(model, RBM) else model._score(inputs)
    if remove_seen:
        scores[inputs != 0] = 0
    indices = np.argsort(-scores, axis=1)[:, : min(k, scores.shape[1])]
    expected = np.zeros_like(scores)
    np.put_along_axis(
        expected, indices, np.take_along_axis(scores, indices, axis=1), axis=1
    )

    actual = _recommend(model, inputs, k=k, remove_seen=remove_seen)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(inputs, before)


@pytest.mark.parametrize("model_class", [StandardVAE, MultVAE, RBM])
@pytest.mark.parametrize("k", [0, -1])
def test_model_topk_rejects_nonpositive_counts(model_class, k):
    with pytest.raises(ValueError, match="at least 1"):
        _recommend(_model(model_class), np.zeros((1, 3), dtype=np.float32), k=k)


@pytest.mark.parametrize("model_class", [StandardVAE, MultVAE])
@pytest.mark.parametrize("k", [1, 10])
def test_vae_topk_empty_users_keep_catalog_width(model_class, k):
    result = _model(model_class).recommend_k_items(
        np.empty((0, 3), dtype=np.float32), k=k
    )
    assert result.shape == (0, 3)
    assert result.dtype == np.float32


@pytest.mark.parametrize("model_class", [StandardVAE, MultVAE, RBM])
def test_model_topk_ties_keep_valid_scores_and_exclude_seen(model_class, monkeypatch):
    model = _model(model_class, n_items=5)
    if isinstance(model, RBM):
        monkeypatch.setattr(
            model,
            "_infer",
            lambda x, **kwargs: (np.full(x.shape, 0.2, dtype=np.float32), None),
        )
    else:
        monkeypatch.setattr(
            model, "_score", lambda x: np.full(x.shape, 0.2, dtype=np.float32)
        )
    inputs = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=np.float32)
    result = _recommend(model, inputs, k=2)
    assert np.all(np.count_nonzero(result, axis=1) == 2)
    assert np.all(result[inputs != 0] == 0)
    np.testing.assert_array_equal(
        result[result != 0], np.full(4, 0.2, dtype=np.float32)
    )


@pytest.mark.parametrize("model_class", [StandardVAE, MultVAE])
def test_vae_default_validation_k_handles_small_catalog(model_class):
    model = _model(model_class)
    mapper = AffinityMatrix(
        pd.DataFrame({"userID": [99], "itemID": [102], "rating": [1]}),
        items_list=[100, 101, 102],
    )
    validation, _, _ = mapper.gen_affinity_matrix()
    model.fit(
        np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
        validation,
        np.zeros((1, 3), dtype=np.float32),
        validation,
        mapper,
    )
    assert len(model.ndcg_per_epoch) == 1
    assert np.isfinite(model.ndcg_per_epoch).all()
