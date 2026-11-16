# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest

from recommenders.datasets.sparse import AffinityMatrix

torch = pytest.importorskip("torch")
StandardVAE = pytest.importorskip("recommenders.models.vae.standard_vae").StandardVAE
MultVAE = pytest.importorskip("recommenders.models.vae.multinomial_vae").MultVAE


def _model(model_class, checkpoint):
    model = model_class(
        n_users=2,
        original_dim=3,
        intermediate_dim=3,
        latent_dim=2,
        n_epochs=1,
        batch_size=2,
        k=1,
        drop_encoder=0,
        drop_decoder=0,
        seed=1,
        verbose=0,
        save_path=checkpoint,
    )
    with torch.no_grad():
        for parameter in model.model.parameters():
            parameter.zero_()
        model.model.decoder_out.bias.copy_(
            torch.tensor([5.0, 0.0, -5.0], device=model.device)
        )
    return model


def _fit(model, relevant_item):
    mapper = AffinityMatrix(
        pd.DataFrame({"userID": [99], "itemID": [100 + relevant_item], "rating": [1]}),
        items_list=[100, 101, 102],
    )
    validation, _, _ = mapper.gen_affinity_matrix()
    validation_input = np.zeros((1, 3), dtype=np.float32)
    model.fit(
        np.array([[1, 0, 0], [1, 0, 0]], dtype=np.float32),
        validation,
        validation_input,
        validation,
        mapper,
    )
    return validation_input


@pytest.mark.parametrize("model_class", [StandardVAE, MultVAE])
@pytest.mark.parametrize("existing_checkpoint", [False, True])
@pytest.mark.parametrize("relevant_item", [0, 2])
def test_vae_saves_a_checkpoint_even_when_validation_ndcg_is_zero(
    model_class, existing_checkpoint, relevant_item, tmp_path
):
    checkpoint = tmp_path / "best.pt"
    model = _model(model_class, str(checkpoint))
    if existing_checkpoint:
        stale_state = {
            name: torch.zeros_like(value)
            for name, value in model.model.state_dict().items()
        }
        stale_state["decoder_out.bias"] = torch.tensor([-5.0, 0.0, 5.0])
        torch.save(stale_state, checkpoint)

    validation_input = _fit(model, relevant_item)
    assert model.ndcg_per_epoch == [1.0 if relevant_item == 0 else 0.0]
    trained_state = {
        name: value.detach().cpu().clone()
        for name, value in model.model.state_dict().items()
    }

    recommendations = model.recommend_k_items(validation_input, k=1)

    assert np.argmax(recommendations[0]) == 0
    assert np.isfinite(recommendations).all()
    saved_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for name, value in trained_state.items():
        torch.testing.assert_close(saved_state[name], value)


@pytest.mark.parametrize("model_class", [StandardVAE, MultVAE])
def test_vae_zero_score_without_checkpoint_keeps_in_memory_prediction(model_class):
    model = _model(model_class, None)
    validation_input = _fit(model, relevant_item=2)
    assert model.ndcg_per_epoch == [0.0]
    assert np.argmax(model.recommend_k_items(validation_input, k=1)[0]) == 0
