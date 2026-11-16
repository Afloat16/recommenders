# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

from collections import namedtuple
from types import SimpleNamespace

import pandas as pd
import pytest

from recommenders.models.cornac.cornac_utils import predict as cornac_predict
from recommenders.models.surprise.surprise_utils import predict as surprise_predict

Prediction = namedtuple("Prediction", "uid iid r_ui est details")
USER_MAP = {"u1": 0, "u2": 1}
ITEM_MAP = {"i1": 0, "i2": 1}


class CornacModel:
    train_set = SimpleNamespace(uid_map=USER_MAP, iid_map=ITEM_MAP)

    def rate(self, user_idx, item_idx):
        return float(10 * user_idx + item_idx)


class SurpriseModel:
    def predict(self, user, item):
        score = 10 * USER_MAP.get(user, 2) + ITEM_MAP.get(item, 2)
        return Prediction(user, item, None, float(score), {})


@pytest.mark.parametrize(
    "predict,model",
    [(cornac_predict, CornacModel()), (surprise_predict, SurpriseModel())],
    ids=["cornac", "surprise"],
)
@pytest.mark.parametrize(
    "columns",
    [
        ("userID", "itemID", "prediction"),
        ("user id", "item-id", "score value"),
        ("_user", "_item", "_score"),
        ("Index", "item", "score"),
        ("details", "r_ui", "est"),
    ],
)
@pytest.mark.parametrize("empty", [False, True])
def test_prediction_adapters_honor_dataframe_columns(predict, model, columns, empty):
    usercol, itemcol, predcol = columns
    data = pd.DataFrame(
        {
            "unused": [0.5, 1.5, 2.5],
            itemcol: ["i2", "i1", "missing"],
            usercol: ["u1", "u2", "missing"],
        },
        index=[9, 14, 27],
    )
    if empty:
        data = data.iloc[:0]
    original = data.copy(deep=True)

    result = predict(model, data, usercol=usercol, itemcol=itemcol, predcol=predcol)

    assert list(result.columns) == [usercol, itemcol, predcol]
    if empty:
        assert result.empty
    else:
        expected = pd.DataFrame(
            {
                usercol: ["u1", "u2", "missing"],
                itemcol: ["i2", "i1", "missing"],
                predcol: [1.0, 10.0, 22.0],
            }
        )
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)
        if predict is surprise_predict:
            assert list(result.index) == ["0", "1", "2"]
    pd.testing.assert_frame_equal(data, original)
