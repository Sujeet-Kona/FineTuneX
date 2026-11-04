"""Training and validation loops for causal language-model fine-tuning."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice
from typing import Any

import torch
from torch import nn

from finetunex.training.metrics import (
    TokenWeightedLoss,
    count_supervised_tokens,
    normalize_microbatch_loss,
)


@dataclass(frozen=True)
class EpochMetrics:
    """Metrics collected during one training or validation epoch."""

    mean_loss: float
    supervised_tokens: int
    microbatches: int
    optimizer_steps: int = 0
    mean_gradient_norm: float | None = None


def _move_model_inputs(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Move model tensors to the target device, excluding metadata."""
    required = ("input_ids", "attention_mask", "labels")
    missing = [key for key in required if key not in batch]

    if missing:
        raise ValueError(f"Batch is missing required fields: {missing}")

    model_inputs: dict[str, torch.Tensor] = {}

    for key in required:
        value = batch[key]

        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Batch field {key!r} must be a torch.Tensor")

        model_inputs[key] = value.to(device)

    return model_inputs


def _windows(
    iterable: Iterable[dict[str, Any]],
    window_size: int,
):
    """Yield accumulation windows, including a possibly incomplete final one."""
    iterator = iter(iterable)

    while window := list(islice(iterator, window_size)):
        yield window


def _validate_training_arguments(
    gradient_accumulation_steps: int,
    max_grad_norm: float,
) -> None:
    if (
        isinstance(gradient_accumulation_steps, bool)
        or not isinstance(gradient_accumulation_steps, int)
        or gradient_accumulation_steps <= 0
    ):
        raise ValueError("gradient_accumulation_steps must be a positive integer")

    if (
        isinstance(max_grad_norm, bool)
        or not isinstance(max_grad_norm, (int, float))
        or max_grad_norm <= 0
    ):
        raise ValueError("max_grad_norm must be a positive finite number")

    if not torch.isfinite(torch.tensor(float(max_grad_norm))).item():
        raise ValueError("max_grad_norm must be a positive finite number")


def train_one_epoch(
    model: nn.Module,
    dataloader: Iterable[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    trainable_parameters: list[nn.Parameter],
    *,
    device: torch.device | str,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    scheduler: Any | None = None,
) -> EpochMetrics:
    """Train for one epoch using token-weighted gradient accumulation."""
    _validate_training_arguments(
        gradient_accumulation_steps,
        max_grad_norm,
    )

    if not trainable_parameters:
        raise ValueError("trainable_parameters cannot be empty")

    device = torch.device(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    metric = TokenWeightedLoss()
    gradient_norm_sum = 0.0
    optimizer_steps = 0
    microbatches = 0

    for window in _windows(dataloader, gradient_accumulation_steps):
        token_counts = [count_supervised_tokens(batch["labels"]) for batch in window]
        window_token_count = sum(token_counts)

        if window_token_count <= 0:
            raise ValueError("Accumulation window contains no supervised tokens")

        # Every microbatch contributes its share of the same token denominator.
        for batch, token_count in zip(window, token_counts, strict=True):
            model_inputs = _move_model_inputs(batch, device)
            outputs = model(**model_inputs)
            loss = outputs.loss

            if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
                raise TypeError("Model outputs.loss must be a scalar tensor")

            if not torch.isfinite(loss.detach()).item():
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError("Non-finite training loss encountered")

            scaled_loss = normalize_microbatch_loss(
                loss,
                microbatch_tokens=token_count,
                accumulation_window_tokens=window_token_count,
            )
            scaled_loss.backward()

            metric.update(float(loss.detach().item()), token_count)
            microbatches += 1

        # Clip only after all microbatches in this window have contributed.
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            max_norm=float(max_grad_norm),
            error_if_nonfinite=True,
        )

        if not torch.isfinite(gradient_norm).item():
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("Non-finite gradient norm encountered")

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        optimizer.zero_grad(set_to_none=True)

        gradient_norm_sum += float(gradient_norm.item())
        optimizer_steps += 1

    if microbatches == 0:
        raise ValueError("Training dataloader produced no batches")

    return EpochMetrics(
        mean_loss=metric.mean,
        supervised_tokens=metric.token_count,
        microbatches=microbatches,
        optimizer_steps=optimizer_steps,
        mean_gradient_norm=gradient_norm_sum / optimizer_steps,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: Iterable[dict[str, Any]],
    *,
    device: torch.device | str,
) -> EpochMetrics:
    """Evaluate token-weighted loss without updating model parameters."""
    device = torch.device(device)
    was_training = model.training
    model.eval()

    metric = TokenWeightedLoss()
    microbatches = 0

    try:
        for batch in dataloader:
            token_count = count_supervised_tokens(batch["labels"])

            if token_count <= 0:
                raise ValueError("Evaluation batch has no supervised tokens")

            model_inputs = _move_model_inputs(batch, device)
            outputs = model(**model_inputs)
            loss = outputs.loss

            if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
                raise TypeError("Model outputs.loss must be a scalar tensor")

            if not torch.isfinite(loss).item():
                raise FloatingPointError("Non-finite validation loss encountered")

            metric.update(float(loss.item()), token_count)
            microbatches += 1
    finally:
        model.train(was_training)

    if microbatches == 0:
        raise ValueError("Evaluation dataloader produced no batches")

    return EpochMetrics(
        mean_loss=metric.mean,
        supervised_tokens=metric.token_count,
        microbatches=microbatches,
    )
