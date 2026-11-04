"""Tests for token-weighted causal-LM loss accounting."""

import pytest
import torch

from finetunex.training.metrics import (
    TokenWeightedLoss,
    count_supervised_tokens,
    normalize_microbatch_loss,
)


def test_counts_only_shifted_supervised_tokens():
    labels = torch.tensor(
        [
            [-100, -100, 10, 11, -100],
            [-100, 20, 21, 22, 23],
        ]
    )

    # Position zero is excluded by causal shifting.
    assert count_supervised_tokens(labels) == 6


def test_rejects_invalid_label_tensor():
    with pytest.raises(ValueError, match="two sequence positions"):
        count_supervised_tokens(torch.tensor([[-100]]))

    with pytest.raises(ValueError, match="shape"):
        count_supervised_tokens(torch.tensor([-100, 10]))


def test_normalizes_microbatch_loss_by_window_tokens():
    loss = torch.tensor(2.0, requires_grad=True)

    scaled = normalize_microbatch_loss(
        loss,
        microbatch_tokens=2,
        accumulation_window_tokens=10,
    )

    assert scaled.item() == pytest.approx(0.4)

    scaled.backward()

    assert loss.grad.item() == pytest.approx(0.2)


def test_partial_accumulation_window_uses_actual_token_total():
    first = normalize_microbatch_loss(
        torch.tensor(2.0),
        microbatch_tokens=3,
        accumulation_window_tokens=5,
    )
    second = normalize_microbatch_loss(
        torch.tensor(4.0),
        microbatch_tokens=2,
        accumulation_window_tokens=5,
    )

    assert (first + second).item() == pytest.approx(2.8)


@pytest.mark.parametrize(
    ("microbatch_tokens", "window_tokens"),
    [
        (0, 5),
        (-1, 5),
        (6, 5),
        (True, 5),
        (2, 0),
        (2, False),
    ],
)
def test_rejects_invalid_token_counts(microbatch_tokens, window_tokens):
    with pytest.raises(ValueError):
        normalize_microbatch_loss(
            torch.tensor(1.0),
            microbatch_tokens=microbatch_tokens,
            accumulation_window_tokens=window_tokens,
        )


def test_rejects_non_finite_loss():
    with pytest.raises(ValueError, match="finite"):
        normalize_microbatch_loss(
            torch.tensor(float("nan")),
            microbatch_tokens=1,
            accumulation_window_tokens=1,
        )


def test_token_weighted_metric():
    metric = TokenWeightedLoss()
    metric.update(2.0, 3)
    metric.update(4.0, 2)

    assert metric.token_count == 5
    assert metric.mean == pytest.approx(2.8)


def test_metric_rejects_empty_mean():
    with pytest.raises(ValueError, match="No supervised tokens"):
        _ = TokenWeightedLoss().mean


@pytest.mark.parametrize(
    ("loss", "tokens"),
    [
        (float("nan"), 1),
        (float("inf"), 1),
        (1.0, 0),
        (1.0, -1),
        (1.0, True),
    ],
)
def test_metric_rejects_invalid_updates(loss, tokens):
    with pytest.raises((TypeError, ValueError)):
        TokenWeightedLoss().update(loss, tokens)
