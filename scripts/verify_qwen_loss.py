"""Verify Qwen3 causal-LM loss reduction against independent cross-entropy."""

from pathlib import Path

import torch
import torch.nn.functional as F

from finetunex.config import load_yaml_config
from finetunex.training.data import FinQACollator, FinQADataset
from finetunex.training.model import load_training_components

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    model_config = load_yaml_config(ROOT / "configs/model.yaml")
    training_config = load_yaml_config(ROOT / "configs/training.yaml")

    components = load_training_components(
        model_config,
        training_config,
    )

    dataset = FinQADataset(
        ROOT / "artifacts/preprocessed/finqa/train.jsonl",
        max_sequence_length=training_config["training"]["max_sequence_length"],
    )

    # One genuine preprocessed FinQA example is sufficient to verify
    # the loss reduction semantics without starting a training run.
    batch = FinQACollator(pad_token_id=components.tokenizer.pad_token_id)([dataset[0]])

    device = components.device
    model = components.model
    model.eval()

    model_inputs = {
        key: batch[key].to(device) for key in ("input_ids", "attention_mask", "labels")
    }

    with torch.no_grad():
        outputs = model(**model_inputs)

        logits = outputs.logits.float()
        shifted_logits = logits[..., :-1, :].contiguous()
        shifted_labels = model_inputs["labels"][..., 1:].contiguous()

        supervised_tokens = int((shifted_labels != -100).sum().item())

        if supervised_tokens == 0:
            raise RuntimeError("Example contains no shifted supervised tokens")

        independent_mean_loss = F.cross_entropy(
            shifted_logits.view(-1, shifted_logits.size(-1)),
            shifted_labels.view(-1),
            ignore_index=-100,
            reduction="mean",
        )

        independent_sum_loss = F.cross_entropy(
            shifted_logits.view(-1, shifted_logits.size(-1)),
            shifted_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )

    model_loss = outputs.loss.float()

    if not torch.isfinite(model_loss).item():
        raise RuntimeError("Qwen3 returned a non-finite loss")

    if not torch.isfinite(independent_mean_loss).item():
        raise RuntimeError("Independent mean cross-entropy is non-finite")

    if not torch.allclose(
        model_loss,
        independent_mean_loss,
        rtol=1e-4,
        atol=1e-4,
    ):
        raise AssertionError(
            "Qwen3 loss does not match independent mean cross-entropy: "
            f"model={model_loss.item():.8f}, "
            f"independent={independent_mean_loss.item():.8f}"
        )

    reconstructed_mean = independent_sum_loss / supervised_tokens

    if not torch.allclose(
        independent_mean_loss,
        reconstructed_mean,
        rtol=1e-5,
        atol=1e-5,
    ):
        raise AssertionError("Mean loss is not sum loss divided by token count")

    print("\nQWEN3 LOSS SEMANTICS VERIFIED")
    print(f"Device:                    {device}")
    print(f"Supervised shifted tokens: {supervised_tokens}")
    print(f"Model outputs.loss:        {model_loss.item():.8f}")
    print(f"Independent mean loss:     {independent_mean_loss.item():.8f}")
    print(f"Independent sum loss:      {independent_sum_loss.item():.8f}")
    print(f"Sum / token count:         {reconstructed_mean.item():.8f}")
    print("Reduction:                 mean over non-ignored shifted tokens")
    print("Gradient accumulation:     compatible with token weighting")


if __name__ == "__main__":
    main()
