# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from recommenders.models.embdotbias.data_loader import RecoDataLoader
from recommenders.models.embdotbias.model import EmbeddingDotBias
from recommenders.models.embdotbias.training_utils import Trainer


class ConstantModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs):
        return self.value.expand(inputs.shape[0])


def ratings_loader(batch_size):
    return DataLoader(
        TensorDataset(
            torch.zeros(5, 2, dtype=torch.long),
            torch.tensor([[1.0], [2.0], [3.0], [4.0], [10.0]]),
        ),
        batch_size=batch_size,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("method", ["train_epoch", "validate"])
@pytest.mark.parametrize("batch_size", [2, 3, 5])
def test_epoch_loss_is_mean_over_ratings(method, batch_size):
    trainer = Trainer(ConstantModel(), learning_rate=0, weight_decay=0)
    actual = getattr(trainer, method)(ratings_loader(batch_size))
    assert actual == pytest.approx((0 + 1 + 4 + 9 + 81) / 5)


@pytest.mark.gpu
@pytest.mark.parametrize("method", ["train_epoch", "validate"])
def test_epoch_loss_counts_observed_batches_without_len(method):
    trainer = Trainer(ConstantModel(), learning_rate=0, weight_decay=0)
    batches = (batch for batch in ratings_loader(2))
    assert getattr(trainer, method)(batches) == pytest.approx(19.0)


@pytest.mark.gpu
@pytest.mark.parametrize("valid_pct", [0, 0.1])
def test_fit_accepts_no_validation_split(valid_pct):
    ratings = pd.DataFrame(
        {"userID": [0, 0, 1, 1], "itemID": [0, 1, 0, 1], "rating": [1.0, 2.0, 3.0, 4.0]}
    )
    data = RecoDataLoader.from_df(ratings, valid_pct=valid_pct, batch_size=3)
    assert data.valid is None
    model = EmbeddingDotBias.from_classes(2, data.classes)
    trainer = Trainer(model)
    before = [p.detach().clone() for p in model.parameters()]
    trainer.fit(data.train, data.valid, n_epochs=2)
    assert trainer.validate(None) is None
    assert any(
        not torch.equal(old, new) for old, new in zip(before, model.parameters())
    )


@pytest.mark.gpu
def test_validate_empty_loader_returns_none():
    trainer = Trainer(ConstantModel())
    empty = DataLoader(
        TensorDataset(torch.empty(0, 2, dtype=torch.long), torch.empty(0, 1))
    )
    assert trainer.validate(empty) is None


@pytest.mark.gpu
def test_validate_does_not_hide_model_division_errors():
    class FailingModel(ConstantModel):
        def forward(self, inputs):
            raise ZeroDivisionError("model failure")

    trainer = Trainer(FailingModel())
    with pytest.raises(ZeroDivisionError, match="model failure"):
        trainer.validate(ratings_loader(2))
