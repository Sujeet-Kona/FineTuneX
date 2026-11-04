"""Tests for token-weighted training and validation loops."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from finetunex.training.engine import evaluate, train_one_epoch


class ToyCausalModel(nn.Module):
    """Small differentiable model with a controllable scalar loss."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.received_keys = []

    def forward(self, input_ids, attention_mask, labels):
        self.received_keys.append({"input_ids", "attention_mask", "labels"})
        return SimpleNamespace(loss=self.weight.square())


def make_batch(supervised_tokens: int):
    """Create one padded batch with the requested number of target tokens."""
    if supervised_tokens < 1:
        raise ValueError("supervised_tokens must be positive")

    sequence_length = supervised_tokens + 1

    return {
        "input_ids": torch.ones((1, sequence_length), dtype=torch.long),
        "attention_mask": torch.ones((1, sequence_length), dtype=torch.long),
        "labels": torch.tensor(
            [[-100] + [1] * supervised_tokens],
            dtype=torch.long,
        ),
        "metadata": [{"record_id": "toy-record"}],
    }


def test_training_uses_token_weighted_accumulation():
    model = ToyCausalModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    result = train_one_epoch(
        model,
        [make_batch(1), make_batch(3)],
        optimizer,
        [model.weight],
        device="cpu",
        gradient_accumulation_steps=2,
        max_grad_norm=100.0,
    )

    # Both microbatches have mean loss 1.0, so weighted epoch loss is 1.0.
    assert result.mean_loss == pytest.approx(1.0)
    assert result.supervised_tokens == 4
    assert result.microbatches == 2
    assert result.optimizer_steps == 1
    assert result.mean_gradient_norm == pytest.approx(2.0)
    assert model.weight.item() == pytest.approx(0.8)


def test_final_partial_accumulation_window_is_processed():
    model = ToyCausalModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    result = train_one_epoch(
        model,
        [make_batch(1), make_batch(2), make_batch(3)],
        optimizer,
        [model.weight],
        device="cpu",
        gradient_accumulation_steps=2,
        max_grad_norm=100.0,
    )

    assert result.microbatches == 3
    assert result.optimizer_steps == 2
    assert result.supervised_tokens == 6


def test_training_excludes_metadata_from_model_inputs():
    model = ToyCausalModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    train_one_epoch(
        model,
        [make_batch(2)],
        optimizer,
        [model.weight],
        device="cpu",
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
    )

    assert model.received_keys == [{"input_ids", "attention_mask", "labels"}]


def test_validation_uses_token_weighted_loss_and_restores_training_mode():
    model = ToyCausalModel()
    model.train()

    result = evaluate(
        model,
        [make_batch(1), make_batch(3)],
        device="cpu",
    )

    assert result.mean_loss == pytest.approx(1.0)
    assert result.supervised_tokens == 4
    assert result.microbatches == 2
    assert result.optimizer_steps == 0
    assert model.training is True


def test_validation_does_not_change_parameters():
    model = ToyCausalModel()
    before = model.weight.detach().clone()

    evaluate(
        model,
        [make_batch(2)],
        device="cpu",
    )

    assert torch.equal(model.weight.detach(), before)


def test_validation_restores_eval_mode_if_model_started_in_eval():
    model = ToyCausalModel()
    model.eval()

    evaluate(
        model,
        [make_batch(2)],
        device="cpu",
    )

    assert model.training is False


def test_rejects_empty_training_dataloader():
    model = ToyCausalModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    with pytest.raises(ValueError, match="no batches"):
        train_one_epoch(
            model,
            [],
            optimizer,
            [model.weight],
            device="cpu",
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
        )


def test_rejects_empty_validation_dataloader():
    model = ToyCausalModel()

    with pytest.raises(ValueError, match="no batches"):
        evaluate(model, [], device="cpu")


def test_rejects_non_finite_training_loss():
    class NonFiniteModel(ToyCausalModel):
        def forward(self, input_ids, attention_mask, labels):
            return SimpleNamespace(loss=self.weight * torch.tensor(float("nan")))

    model = NonFiniteModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    with pytest.raises(FloatingPointError, match="training loss"):
        train_one_epoch(
            model,
            [make_batch(2)],
            optimizer,
            [model.weight],
            device="cpu",
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
        )


@pytest.mark.parametrize("steps", [0, -1, True])
def test_rejects_invalid_accumulation_steps(steps):
    model = ToyCausalModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        train_one_epoch(
            model,
            [make_batch(2)],
            optimizer,
            [model.weight],
            device="cpu",
            gradient_accumulation_steps=steps,
            max_grad_norm=1.0,
        )


def test_rejects_invalid_gradient_clip_value():
    model = ToyCausalModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    with pytest.raises(ValueError, match="max_grad_norm"):
        train_one_epoch(
            model,
            [make_batch(2)],
            optimizer,
            [model.weight],
            device="cpu",
            gradient_accumulation_steps=1,
            max_grad_norm=0.0,
        )
