# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import pytest
import torch

from recommenders.models.embdotbias.model import EmbeddingDotBias


@pytest.mark.gpu
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("is_item", [False, True])
@pytest.mark.parametrize("method", ["weight", "bias"])
def test_embedding_inspection_preserves_model_state(device, training, is_item, method):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    classes = {"users": ["u0", "u1", "u2"], "items": ["i0", "i1", "i2"]}
    model = EmbeddingDotBias.from_classes(2, classes).to(device).double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(
                torch.arange(parameter.numel(), device=device).reshape_as(parameter)
                / 10
            )
    model.train(training)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    inputs = torch.tensor([[0, 1], [1, 2]], device=device)
    model(inputs).sum().backward()
    optimizer.step()
    optimizer.zero_grad()

    states = {name: module.training for name, module in model.named_modules()}
    parameters = {
        name: (parameter, parameter.device)
        for name, parameter in model.named_parameters()
    }
    before = model(inputs).detach().clone()
    layer = getattr(model, ("i_" if is_item else "u_") + method)
    expected = layer.weight.detach()[[2, 0]].cpu()
    if method == "bias":
        expected = expected.squeeze()
    ids = ["i2", "i0"] if is_item else ["u2", "u0"]

    actual = getattr(model, method)(ids, is_item=is_item)

    torch.testing.assert_close(actual, expected)
    assert actual.device.type == "cpu"
    assert not actual.requires_grad
    assert {name: module.training for name, module in model.named_modules()} == states
    for name, parameter in model.named_parameters():
        original, original_device = parameters[name]
        assert parameter is original
        assert parameter.device == original_device
    torch.testing.assert_close(model(inputs), before)
    model(inputs).sum().backward()
    optimizer.step()
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
