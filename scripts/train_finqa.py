"""Config-driven LoRA training entry point for FinQA."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finetunex.artifacts.checkpoint import save_best_checkpoint  # noqa: E402
from finetunex.config import load_yaml_config  # noqa: E402
from finetunex.training.data import FinQACollator, FinQADataset  # noqa: E402
from finetunex.training.engine import evaluate, train_one_epoch  # noqa: E402
from finetunex.training.model import load_training_components  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3-0.6B with LoRA on FinQA."
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Limit training batches per epoch for a smoke run.",
    )
    parser.add_argument(
        "--max-validation-batches",
        type=int,
        default=None,
        help="Limit validation batches for a smoke run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the checkpoint output directory.",
    )
    return parser.parse_args()


def validate_limit(name: str, value: int | None) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read JSON artifact: {path}") from exc

    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")

    return value


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def seed_everything(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("training.seed must be a non-negative integer")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_dataset_provenance(
    manifest_dataset: Any,
    dataset: dict[str, Any],
) -> None:
    """Verify that preprocessing artifacts came from the configured dataset."""
    if not isinstance(manifest_dataset, dict):
        raise ValueError("Preprocessing manifest has no dataset provenance")

    expected_name = dataset["name"]
    expected_source = dataset["source"]
    actual_name = manifest_dataset.get("name")

    if (
        not isinstance(actual_name, str)
        or actual_name.casefold() != expected_name.casefold()
    ):
        raise ValueError("Preprocessed artifacts use a different dataset")

    if (
        manifest_dataset.get("repository") != expected_source["repository"]
        or manifest_dataset.get("revision") != expected_source["revision"]
    ):
        raise ValueError("Preprocessed artifacts use a different dataset revision")


def validate_fresh_output_dir(output_dir: Path) -> None:
    """Reject existing output paths and symlinks before model initialization."""
    absolute_path = output_dir.absolute()

    for candidate in (absolute_path, *absolute_path.parents):
        if candidate.is_symlink():
            raise ValueError(
                f"Checkpoint output path cannot contain symlinks: {candidate}"
            )

    if absolute_path.resolve() == PROJECT_ROOT.resolve():
        raise ValueError("Checkpoint output cannot be the project root")

    if absolute_path.exists():
        raise FileExistsError(
            f"Checkpoint output already exists: {absolute_path}. "
            "This entrypoint starts a fresh training run and does not resume "
            "existing model or optimizer state. Choose a new output directory."
        )


def processed_example_count(
    dataset_size: int,
    microbatches: int,
    batch_size: int,
) -> int:
    """Return the exact number of examples consumed by ordinary DataLoader batches."""
    return min(dataset_size, microbatches * batch_size)


def validate_manifest(
    manifest: dict[str, Any],
    *,
    dataset_config: dict[str, Any],
    model_config: dict[str, Any],
    preprocessing_config: dict[str, Any],
    training_config: dict[str, Any],
    train_path: Path,
    validation_path: Path,
) -> None:
    dataset = dataset_config["dataset"]
    expected = dataset["expected"]
    model = model_config["model"]
    preprocessing = preprocessing_config["preprocessing"]
    training = training_config["training"]

    expected_counts = {
        "train": expected["train_examples"],
        "validation": expected["validation_examples"],
    }

    for split_name, count in expected_counts.items():
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"dataset.expected.{split_name}_examples must be positive")

    if manifest.get("schema_version") != 2:
        raise ValueError("Unsupported preprocessing manifest schema")

    validate_dataset_provenance(manifest.get("dataset"), dataset)

    if manifest.get("model") != {
        "name": model["name"],
        "revision": model["revision"],
    }:
        raise ValueError("Preprocessed artifacts use a different base model")

    tokenizer = manifest.get("tokenizer", {})

    if (
        tokenizer.get("name") != model["tokenizer"]["name"]
        or tokenizer.get("revision") != model["tokenizer"]["revision"]
    ):
        raise ValueError("Preprocessed artifacts use a different tokenizer")

    preprocessing_metadata = manifest.get("preprocessing", {})

    if preprocessing_metadata.get("test_split_included") is not False:
        raise ValueError("Training artifacts must exclude the test split")

    if preprocessing_metadata.get("evidence_budget") != preprocessing.get(
        "evidence", {}
    ).get("token_budget"):
        raise ValueError("Training and preprocessing evidence budgets differ")

    if (
        preprocessing_metadata.get("max_sequence_length")
        != training["max_sequence_length"]
    ):
        raise ValueError("Training and preprocessing sequence limits differ")

    if preprocessing_metadata.get("max_prompt_tokens") != preprocessing.get(
        "sequence", {}
    ).get("max_prompt_length"):
        raise ValueError("Training and preprocessing prompt budgets differ")

    if preprocessing_metadata.get("generation_reserve_tokens") != preprocessing.get(
        "generation", {}
    ).get("max_new_tokens"):
        raise ValueError("Training and preprocessing generation reserves differ")

    if preprocessing_metadata.get("prompt_version") != preprocessing.get(
        "prompt_version"
    ):
        raise ValueError("Prompt version does not match preprocessing config")

    splits = manifest.get("splits")

    if not isinstance(splits, dict):
        raise ValueError("Preprocessing manifest has no split metadata")

    for split_name, path in (
        ("train", train_path),
        ("validation", validation_path),
    ):
        split = splits.get(split_name)

        if not isinstance(split, dict):
            raise ValueError(f"Missing {split_name} split in manifest")

        if not path.is_file():
            raise FileNotFoundError(f"Missing {split_name} artifact: {path}")

        if sha256_file(path) != split.get("output_sha256"):
            raise ValueError(f"{split_name} artifact SHA-256 mismatch")

    for split_name, expected_count in expected_counts.items():
        if splits[split_name].get("record_count") != expected_count:
            raise ValueError(f"Unexpected {split_name} count in preprocessing manifest")

    if preprocessing.get("splits") != ["train", "validation"]:
        raise ValueError("Only train and validation splits are supported")

    if training.get("method") != "lora":
        raise ValueError("This entry point supports LoRA only")


def package_versions() -> dict[str, str]:
    import peft
    import transformers
    import yaml

    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "numpy": np.__version__,
        "PyYAML": yaml.__version__,
    }


def main() -> int:
    args = parse_args()

    validate_limit("max_train_batches", args.max_train_batches)
    validate_limit("max_validation_batches", args.max_validation_batches)

    dataset_config = load_yaml_config(PROJECT_ROOT / "configs/dataset.yaml")
    model_config = load_yaml_config(PROJECT_ROOT / "configs/model.yaml")
    training_config = load_yaml_config(PROJECT_ROOT / "configs/training.yaml")
    preprocessing_config = load_yaml_config(PROJECT_ROOT / "configs/preprocessing.yaml")

    dataset_settings = dataset_config["dataset"]
    model_settings = model_config["model"]
    training_settings = training_config["training"]
    preprocessing_settings = preprocessing_config["preprocessing"]

    seed = training_settings["seed"]
    epochs = training_settings["epochs"]
    batch_size = training_settings["batch_size"]
    accumulation_steps = training_settings["gradient_accumulation_steps"]
    max_sequence_length = training_settings["max_sequence_length"]
    learning_rate = training_settings["learning_rate"]

    for name, value in (
        ("epochs", epochs),
        ("batch_size", batch_size),
        ("gradient_accumulation_steps", accumulation_steps),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"training.{name} must be a positive integer")

    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not np.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("training.learning_rate must be positive and finite")

    if args.output_dir is not None:
        output_dir = resolve_project_path(str(args.output_dir))
    elif args.max_train_batches is not None or args.max_validation_batches is not None:
        output_dir = PROJECT_ROOT / "artifacts/checkpoints/smoke"
    else:
        configured_output = resolve_project_path(training_settings["output_dir"])
        output_dir = configured_output / "finqa-qwen3-0.6b"

    validate_fresh_output_dir(output_dir)

    seed_everything(seed)

    data_dir = resolve_project_path(preprocessing_settings["output"]["directory"])
    train_path = data_dir / "train.jsonl"
    validation_path = data_dir / "validation.jsonl"
    manifest_path = data_dir / "manifest.json"

    manifest = load_json(manifest_path)

    validate_manifest(
        manifest,
        dataset_config=dataset_config,
        model_config=model_config,
        preprocessing_config=preprocessing_config,
        training_config=training_config,
        train_path=train_path,
        validation_path=validation_path,
    )

    print("\n===== DATASET VALIDATION =====")
    print("Train artifact:", train_path)
    print("Validation artifact:", validation_path)
    print("Train SHA-256:", sha256_file(train_path))
    print("Validation SHA-256:", sha256_file(validation_path))
    print("Test split: excluded")

    train_dataset = FinQADataset(
        train_path,
        max_sequence_length=max_sequence_length,
    )
    validation_dataset = FinQADataset(
        validation_path,
        max_sequence_length=max_sequence_length,
    )

    expected = dataset_settings["expected"]

    if len(train_dataset) != expected["train_examples"]:
        raise ValueError("Loaded training count does not match dataset config")

    if len(validation_dataset) != expected["validation_examples"]:
        raise ValueError("Loaded validation count does not match dataset config")

    components = load_training_components(model_config, training_config)

    collator = FinQACollator(pad_token_id=components.tokenizer.pad_token_id)
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=collator,
        pin_memory=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
        pin_memory=False,
    )

    optimizer = torch.optim.AdamW(
        components.trainable_parameters,
        lr=float(learning_rate),
        weight_decay=float(training_settings.get("weight_decay", 0.01)),
    )

    history: list[dict[str, Any]] = []
    global_step = 0
    best_validation_loss = float("inf")

    config_paths = (
        PROJECT_ROOT / "configs/dataset.yaml",
        PROJECT_ROOT / "configs/model.yaml",
        PROJECT_ROOT / "configs/training.yaml",
        PROJECT_ROOT / "configs/preprocessing.yaml",
    )

    config_hashes = {path.name: sha256_file(path) for path in config_paths}

    print("\n===== TRAINING PLAN =====")
    print(
        "Run mode:",
        "LIMITED SMOKE RUN"
        if args.max_train_batches is not None or args.max_validation_batches is not None
        else "FULL DATASET RUN",
    )
    print("Checkpoint selection: lowest token-weighted validation loss")
    print("Epochs:", epochs)
    print("Batch size:", batch_size)
    print("Gradient accumulation:", accumulation_steps)
    print("Learning rate:", learning_rate)
    print("Weight decay:", training_settings.get("weight_decay", 0.01))
    print("Gradient clipping:", 1.0)
    print("Output directory:", output_dir)
    print("Training batches per epoch:", args.max_train_batches or len(train_loader))
    print(
        "Validation batches:",
        args.max_validation_batches or len(validation_loader),
    )

    for epoch in range(1, epochs + 1):
        print(f"\n===== EPOCH {epoch}/{epochs} =====")

        train_batches = (
            islice(train_loader, args.max_train_batches)
            if args.max_train_batches is not None
            else train_loader
        )

        train_metrics = train_one_epoch(
            components.model,
            train_batches,
            optimizer,
            components.trainable_parameters,
            device=components.device,
            gradient_accumulation_steps=accumulation_steps,
            max_grad_norm=1.0,
        )

        global_step += train_metrics.optimizer_steps

        validation_batches = (
            islice(validation_loader, args.max_validation_batches)
            if args.max_validation_batches is not None
            else validation_loader
        )

        validation_metrics = evaluate(
            components.model,
            validation_batches,
            device=components.device,
        )

        epoch_record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_metrics.mean_loss,
            "train_supervised_tokens": train_metrics.supervised_tokens,
            "train_microbatches": train_metrics.microbatches,
            "train_optimizer_steps": train_metrics.optimizer_steps,
            "mean_gradient_norm": train_metrics.mean_gradient_norm,
            "validation_loss": validation_metrics.mean_loss,
            "validation_supervised_tokens": validation_metrics.supervised_tokens,
            "validation_microbatches": validation_metrics.microbatches,
        }

        history.append(epoch_record)

        print(f"Train loss:      {train_metrics.mean_loss:.6f}")
        print(f"Validation loss: {validation_metrics.mean_loss:.6f}")
        print("Optimizer steps:", train_metrics.optimizer_steps)
        print("Gradient norm:", train_metrics.mean_gradient_norm)

        metadata = {
            "run_type": (
                "limited_smoke"
                if args.max_train_batches is not None
                or args.max_validation_batches is not None
                else "full_dataset"
            ),
            "checkpoint_selection_metric": "token_weighted_validation_loss",
            "model": model_settings["name"],
            "model_revision": model_settings["revision"],
            "tokenizer": model_settings["tokenizer"]["name"],
            "tokenizer_revision": model_settings["tokenizer"]["revision"],
            "dataset": dataset_settings["name"],
            "prompt_version": manifest["preprocessing"]["prompt_version"],
            "dataset_repository": dataset_settings["source"]["repository"],
            "dataset_revision": dataset_settings["source"]["revision"],
            "preprocessing_manifest_sha256": sha256_file(manifest_path),
            "train_artifact_sha256": sha256_file(train_path),
            "validation_artifact_sha256": sha256_file(validation_path),
            "configuration_sha256": config_hashes,
            "package_versions": package_versions(),
            "device": str(components.device),
            "base_dtype": model_settings["loading"]["dtype"],
            "seed": seed,
            "epoch": epoch,
            "global_step": global_step,
            "validation_loss": validation_metrics.mean_loss,
            "trainable_parameter_count": components.trainable_parameter_count,
            "total_parameter_count": components.total_parameter_count,
            "training_config": {
                "method": training_settings["method"],
                "epochs": epochs,
                "learning_rate": learning_rate,
                "weight_decay": training_settings.get("weight_decay", 0.01),
                "batch_size": batch_size,
                "gradient_accumulation_steps": accumulation_steps,
                "max_sequence_length": max_sequence_length,
                "max_grad_norm": 1.0,
                "lora": training_settings["lora"],
            },
            "evaluation_scope": {
                "train_batches_limited": args.max_train_batches is not None,
                "validation_batches_limited": args.max_validation_batches is not None,
                "max_train_batches_per_epoch": args.max_train_batches,
                "max_validation_batches_per_epoch": args.max_validation_batches,
                "training_examples_per_epoch": processed_example_count(
                    len(train_dataset),
                    train_metrics.microbatches,
                    batch_size,
                ),
                "validation_examples": processed_example_count(
                    len(validation_dataset),
                    validation_metrics.microbatches,
                    batch_size,
                ),
            },
            "created_at_utc": datetime.now(UTC).isoformat(),
        }

        saved = save_best_checkpoint(
            output_dir,
            components.model,
            components.tokenizer,
            metadata,
            history,
        )

        if saved:
            best_validation_loss = validation_metrics.mean_loss
            print("Best checkpoint updated:", output_dir)
        else:
            print("Checkpoint unchanged; validation did not improve.")

        print("Best validation loss:", f"{best_validation_loss:.6f}")

    print("\n===== TRAINING COMPLETE =====")
    print("Best validation loss:", best_validation_loss)
    print("Checkpoint directory:", output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
