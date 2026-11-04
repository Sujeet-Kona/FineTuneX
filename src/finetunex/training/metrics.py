"""Token-weighted loss accounting for causal language-model training."""

from __future__ import annotations

import math

import torch


def count_supervised_tokens(labels: torch.Tensor) -> int:
    """Count target tokens after the causal shift, excluding ignored labels."""
    if not isinstance(labels, torch.Tensor):
        raise TypeError("labels must be a torch.Tensor")

    if labels.ndim != 2:
        raise ValueError("labels must have shape [batch_size, sequence_length]")

    if labels.shape[1] < 2:
        raise ValueError("labels must contain at least two sequence positions")

    return int((labels[:, 1:] != -100).sum().item())


def normalize_microbatch_loss(
    mean_loss: torch.Tensor,
    *,
    microbatch_tokens: int,
    accumulation_window_tokens: int,
) -> torch.Tensor:
    """Scale a mean microbatch loss to its share of the accumulation window."""
    if not isinstance(mean_loss, torch.Tensor) or mean_loss.numel() != 1:
        raise TypeError("mean_loss must be a scalar torch.Tensor")

    if not torch.isfinite(mean_loss.detach()).item():
        raise ValueError("mean_loss must be finite")

    if (
        isinstance(microbatch_tokens, bool)
        or not isinstance(microbatch_tokens, int)
        or microbatch_tokens <= 0
    ):
        raise ValueError("microbatch_tokens must be a positive integer")

    if (
        isinstance(accumulation_window_tokens, bool)
        or not isinstance(accumulation_window_tokens, int)
        or accumulation_window_tokens <= 0
    ):
        raise ValueError("accumulation_window_tokens must be a positive integer")

    if microbatch_tokens > accumulation_window_tokens:
        raise ValueError("microbatch token count exceeds accumulation window")

    return mean_loss * (microbatch_tokens / accumulation_window_tokens)


class TokenWeightedLoss:
    """Accumulate mean losses weighted by their supervised-token counts."""

    def __init__(self) -> None:
        self.loss_sum = 0.0
        self.token_count = 0

    def update(self, mean_loss: float, token_count: int) -> None:
        if isinstance(mean_loss, bool) or not isinstance(mean_loss, (int, float)):
            raise TypeError("mean_loss must be numeric")

        if not math.isfinite(mean_loss):
            raise ValueError("mean_loss must be finite")

        if (
            isinstance(token_count, bool)
            or not isinstance(token_count, int)
            or token_count <= 0
        ):
            raise ValueError("token_count must be a positive integer")

        self.loss_sum += float(mean_loss) * token_count
        self.token_count += token_count

    @property
    def mean(self) -> float:
        if self.token_count == 0:
            raise ValueError("No supervised tokens have been accumulated")

        return self.loss_sum / self.token_count
