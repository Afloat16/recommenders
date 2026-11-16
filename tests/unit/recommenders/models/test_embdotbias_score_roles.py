# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest
import torch

from recommenders.models.embdotbias.model import EmbeddingDotBias
from recommenders.models.embdotbias.utils import score


@pytest.mark.gpu
@pytest.mark.parametrize("catalog", ["normal", "reversed", "extra"])
@pytest.mark.parametrize(
    "columns", [("userID", "itemID", "prediction"), ("customer", "product", "estimate")]
)
@pytest.mark.parametrize("top_k", [None, 1])
def test_score_respects_explicit_model_entity_roles(catalog, columns, top_k):
    entries = [("users", ["u1", "u2"]), ("items", ["i1", "i2"])]
    if catalog == "reversed":
        entries.reverse()
    elif catalog == "extra":
        entries.insert(0, ("regions", ["north"]))
    model = EmbeddingDotBias.from_classes(
        2, dict(entries), user="users", item="items"
    ).double()
    with torch.no_grad():
        model.u_weight.weight.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float64)
        )
        model.i_weight.weight.copy_(
            torch.tensor([[2.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
        )
        model.u_bias.weight.copy_(torch.tensor([[0.1], [0.2]], dtype=torch.float64))
        model.i_bias.weight.copy_(torch.tensor([[0.3], [0.4]], dtype=torch.float64))
    user_col, item_col, prediction_col = columns
    query = pd.DataFrame(
        {user_col: ["u2", "u1", "u1", "u2"], item_col: ["i1", "i2", "i1", "i2"]},
        index=[4, 9, 2, 7],
    )
    original = query.copy(deep=True)

    actual = score(model, query, user_col, item_col, prediction_col, top_k)

    expected = pd.DataFrame(
        {
            user_col: ["u1", "u1", "u2", "u2"],
            item_col: ["i1", "i2", "i2", "i1"],
            prediction_col: np.asarray([2.4, 1.5, 6.6, 2.5]),
        }
    )
    if top_k == 1:
        expected = expected.iloc[[0, 2]]
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True), expected.reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(query, original)
