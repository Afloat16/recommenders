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


def _data():
    return ImplicitCF(
        pd.DataFrame(
            {
                "userID": [10, 10, 20, 20],
                "itemID": [100, 101, 101, 102],
                "rating": [1, 1, 1, 1],
            }
        ),
        seed=7,
    )


def _model(layers=2):
    data = _data()
    model = LightGCN(
        data.n_users,
        data.n_items,
        data.get_norm_adj_mat(),
        embed_size=3,
        n_layers=layers,
        seed=3,
    )
    model.data = data
    return model


def _dense_reference(model):
    weights = (
        torch.cat([model.user_embedding.weight, model.item_embedding.weight])
        .detach()
        .requires_grad_(True)
    )
    adjacency = torch.as_tensor(
        model.norm_adj.toarray().astype(np.float32),
        dtype=weights.dtype,
        device=weights.device,
    )
    layers = [weights]
    for _ in range(model.n_layers):
        layers.append(adjacency @ layers[-1])
    return torch.stack(layers).mean(dim=0), weights


@pytest.mark.parametrize("layers", [0, 1, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_lightgcn_dtype_move_preserves_propagation_and_gradients(layers, dtype):
    model = _model(layers).to(device="cpu", dtype=dtype)
    expected, weights = _dense_reference(model)
    actual = torch.cat(model._propagate())
    torch.testing.assert_close(actual, expected)

    coefficients = torch.linspace(0.2, 1.3, actual.numel(), dtype=dtype).reshape_as(
        actual
    )
    (actual * coefficients).sum().backward()
    (expected * coefficients).sum().backward()
    torch.testing.assert_close(model.user_embedding.weight.grad, weights.grad[:2])
    torch.testing.assert_close(model.item_embedding.weight.grad, weights.grad[2:])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_lightgcn_respects_default_embedding_dtype(dtype):
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        model = _model().cpu()
    finally:
        torch.set_default_dtype(original)

    expected, _ = _dense_reference(model)
    torch.testing.assert_close(torch.cat(model._propagate()), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_lightgcn_trains_and_scores_after_dtype_move(dtype):
    model = _model().to(device="cpu", dtype=dtype)
    data = model.data
    before = model.user_embedding.weight.detach().clone()
    model.fit(data, epochs=1, batch_size=2, eval_epoch=-1)

    assert not torch.equal(model.user_embedding.weight, before)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    expected, _ = _dense_reference(model)
    scores = model.score(np.array([0, 1]), remove_seen=False)
    torch.testing.assert_close(
        torch.from_numpy(scores), (expected[:2] @ expected[2:].T).detach()
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_lightgcn_loads_embedding_only_checkpoint_after_move(tmp_path, dtype):
    source = _model().cpu()
    legacy_state = {
        "user_embedding.weight": source.user_embedding.weight.detach().clone(),
        "item_embedding.weight": source.item_embedding.weight.detach().clone(),
    }
    checkpoint = tmp_path / "model.pt"
    torch.save(legacy_state, checkpoint)
    model = _model().to(device="cpu", dtype=dtype)
    with torch.no_grad():
        model.user_embedding.weight.zero_()
    model.load(str(checkpoint))

    assert set(model.state_dict()) == set(legacy_state)
    expected, _ = _dense_reference(model)
    torch.testing.assert_close(torch.cat(model._propagate()), expected)
    torch.testing.assert_close(
        model.user_embedding.weight, legacy_state["user_embedding.weight"].to(dtype)
    )


def test_lightgcn_device_follows_module_movement():
    model = _model().to("meta")
    assert model.device == model.user_embedding.weight.device == torch.device("meta")
    assert model.A_hat.device == torch.device("meta")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_lightgcn_cuda_cpu_round_trip_trains_and_scores():
    model = _model().cuda()
    cuda_scores = model.score(np.array([0, 1]), remove_seen=False)
    model.cpu()
    assert model.device == torch.device("cpu")
    np.testing.assert_allclose(
        model.score(np.array([0, 1]), remove_seen=False),
        cuda_scores,
        rtol=1e-5,
        atol=1e-6,
    )
    model.fit(model.data, epochs=1, batch_size=2, eval_epoch=-1)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
