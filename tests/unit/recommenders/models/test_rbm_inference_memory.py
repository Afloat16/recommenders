# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import tracemalloc

import numpy as np
import pytest

torch = pytest.importorskip("torch")
RBM = pytest.importorskip("recommenders.models.rbm.rbm").RBM


@pytest.mark.parametrize("batch_size", [1, 4, 10])
@pytest.mark.parametrize("return_distributions", [False, True])
def test_rbm_chunked_outputs_match_direct_inference(batch_size, return_distributions):
    model = RBM([1, 2, 3, 4, 5], visible_units=6, hidden_units=3, seed=2)
    inputs = np.random.default_rng(2).integers(0, 6, size=(7, 6)).astype(np.float32)
    with torch.no_grad():
        expected_ratings, expected_probabilities = model.eval_out(
            torch.as_tensor(inputs, device=model.device)
        )

    if return_distributions:
        ratings, probabilities = model._infer(inputs, batch_size=batch_size)
        np.testing.assert_allclose(
            probabilities, expected_probabilities.cpu().numpy(), rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0, atol=1e-6)
    else:
        ratings, probabilities = model._infer(
            inputs, batch_size=batch_size, return_distributions=False
        )
        assert probabilities is None
    np.testing.assert_allclose(
        ratings, expected_ratings.cpu().numpy(), rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("users", [257, 1200])
@pytest.mark.parametrize("method", ["predict", "recommend"])
def test_rbm_public_inference_does_not_materialize_unused_distribution(users, method):
    model = RBM(np.arange(0.5, 5.5, 0.5), visible_units=64, hidden_units=4, seed=3)
    inputs = (
        np.random.default_rng(3).integers(0, 6, size=(users, 64)).astype(np.float32)
    )
    before = inputs.copy()

    def infer(values):
        if method == "predict":
            return model.predict(values)
        return model.recommend_k_items(values, top_k=3)

    infer(inputs[:1])
    tracemalloc.start()
    try:
        result = infer(inputs)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert result.shape == inputs.shape
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(inputs, before)
    # Allow the score-selection workspace, but not the unused ten-class output.
    # NumPy allocations are traced here; this is not a process-RSS assertion.
    assert peak < 6 * result.nbytes + 100_000
