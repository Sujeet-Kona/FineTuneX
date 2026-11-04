"""Build reproducible tokenized FinQA training and validation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from finetunex.config import load_yaml_config
from finetunex.data.finqa_artifacts import (
    preprocess_finqa_dataset,
    sha256_file,
)
from finetunex.data.finqa_preprocessing import PROMPT_VERSION

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping")
    return value


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _resolve_project_path(value: str, name: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def validate_configs(
    dataset_config: dict[str, Any],
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    preprocessing_config: dict[str, Any],
) -> dict[str, Any]:
    """Validate cross-configuration contracts and return resolved settings."""
    dataset = _require_mapping(dataset_config.get("dataset"), "dataset")
    source = _require_mapping(dataset.get("source"), "dataset.source")
    expected = _require_mapping(dataset.get("expected"), "dataset.expected")

    model = _require_mapping(model_config.get("model"), "model")
    tokenizer_config = _require_mapping(model.get("tokenizer"), "model.tokenizer")

    training = _require_mapping(training_config.get("training"), "training")
    preprocessing = _require_mapping(
        preprocessing_config.get("preprocessing"),
        "preprocessing",
    )

    input_config = _require_mapping(preprocessing.get("input"), "preprocessing.input")
    output_config = _require_mapping(
        preprocessing.get("output"), "preprocessing.output"
    )
    evidence_config = _require_mapping(
        preprocessing.get("evidence"), "preprocessing.evidence"
    )
    sequence_config = _require_mapping(
        preprocessing.get("sequence"), "preprocessing.sequence"
    )

    repository = _require_string(source.get("repository"), "dataset.source.repository")
    dataset_revision = _require_string(
        source.get("revision"), "dataset.source.revision"
    )

    model_name = _require_string(model.get("name"), "model.name")
    model_revision = _require_string(model.get("revision"), "model.revision")

    tokenizer_name = _require_string(
        tokenizer_config.get("name"), "model.tokenizer.name"
    )
    tokenizer_revision = _require_string(
        tokenizer_config.get("revision"), "model.tokenizer.revision"
    )

    if tokenizer_revision != model_revision:
        raise ValueError("Tokenizer revision must match model revision")

    if tokenizer_name != model_name:
        raise ValueError("Tokenizer name must match model name")

    if preprocessing.get("prompt_version") != PROMPT_VERSION:
        raise ValueError(
            f"Unsupported prompt version: {preprocessing.get('prompt_version')!r}"
        )

    if evidence_config.get("strategy") != "complete-table-first":
        raise ValueError("Only complete-table-first evidence strategy is supported")

    splits = preprocessing.get("splits")
    if splits != ["train", "validation"]:
        raise ValueError("Preprocessing must include exactly train and validation")

    if training.get("method") != "lora":
        raise ValueError("This preprocessing configuration expects LoRA training")

    expected_counts = _require_mapping(
        preprocessing.get("expected_counts"),
        "preprocessing.expected_counts",
    )

    if set(expected_counts) != {"train", "validation"}:
        raise ValueError(
            "Preprocessing expected_counts must contain train and validation"
        )

    train_count = _require_positive_int(
        expected.get("train_examples"),
        "dataset.expected.train_examples",
    )
    validation_count = _require_positive_int(
        expected.get("validation_examples"),
        "dataset.expected.validation_examples",
    )

    resolved_counts = {
        "train": _require_positive_int(
            expected_counts.get("train"),
            "preprocessing.expected_counts.train",
        ),
        "validation": _require_positive_int(
            expected_counts.get("validation"),
            "preprocessing.expected_counts.validation",
        ),
    }

    if resolved_counts != {
        "train": train_count,
        "validation": validation_count,
    }:
        raise ValueError(
            "Preprocessing expected counts do not match dataset configuration"
        )

    evidence_budget = _require_positive_int(
        evidence_config.get("token_budget"),
        "preprocessing.evidence.token_budget",
    )

    max_sequence_length = _require_positive_int(
        sequence_config.get("max_length"),
        "preprocessing.sequence.max_length",
    )

    training_max_length = _require_positive_int(
        training.get("max_sequence_length"),
        "training.max_sequence_length",
    )

    if max_sequence_length != training_max_length:
        raise ValueError(
            "Preprocessing and training max_sequence_length values must match"
        )

    padding_side = tokenizer_config.get("padding_side")
    if padding_side not in {"left", "right"}:
        raise ValueError("model.tokenizer.padding_side must be left or right")

    trust_remote_code = model.get("trust_remote_code")
    if not isinstance(trust_remote_code, bool):
        raise ValueError("model.trust_remote_code must be a boolean")

    input_directory = _resolve_project_path(
        _require_string(input_config.get("directory"), "preprocessing.input.directory"),
        "preprocessing.input.directory",
    )

    output_directory = _resolve_project_path(
        _require_string(
            output_config.get("directory"),
            "preprocessing.output.directory",
        ),
        "preprocessing.output.directory",
    )

    return {
        "source_repository": repository,
        "source_revision": dataset_revision,
        "expected_counts": resolved_counts,
        "model_name": model_name,
        "model_revision": model_revision,
        "tokenizer_name": tokenizer_name,
        "tokenizer_revision": tokenizer_revision,
        "trust_remote_code": trust_remote_code,
        "padding_side": padding_side,
        "evidence_budget": evidence_budget,
        "max_sequence_length": max_sequence_length,
        "input_dir": input_directory,
        "output_dir": output_directory,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess FinQA into reproducible tokenized artifacts."
    )
    parser.add_argument(
        "--dataset-config",
        default="configs/dataset.yaml",
    )
    parser.add_argument(
        "--model-config",
        default="configs/model.yaml",
    )
    parser.add_argument(
        "--training-config",
        default="configs/training.yaml",
    )
    parser.add_argument(
        "--config",
        default="configs/preprocessing.yaml",
        help="Preprocessing configuration",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory after successful staging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_paths = {
        "dataset.yaml": _resolve_project_path(args.dataset_config, "dataset config"),
        "model.yaml": _resolve_project_path(args.model_config, "model config"),
        "training.yaml": _resolve_project_path(args.training_config, "training config"),
        "preprocessing.yaml": _resolve_project_path(
            args.config, "preprocessing config"
        ),
    }

    configs = {name: load_yaml_config(path) for name, path in config_paths.items()}

    settings = validate_configs(
        configs["dataset.yaml"],
        configs["model.yaml"],
        configs["training.yaml"],
        configs["preprocessing.yaml"],
    )

    for name, path in config_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")

    print("Configuration validation passed.")
    print(f"Dataset: {settings['source_repository']}")
    print(f"Dataset revision: {settings['source_revision']}")
    print(f"Model: {settings['model_name']}")
    print(f"Model revision: {settings['model_revision']}")
    print(f"Input: {settings['input_dir']}")
    print(f"Output: {settings['output_dir']}")
    print(f"Evidence budget: {settings['evidence_budget']}")
    print(f"Maximum sequence length: {settings['max_sequence_length']}")

    if not settings["input_dir"].is_dir():
        raise FileNotFoundError(
            f"Processed FinQA directory not found: {settings['input_dir']}. "
            "Run scripts/prepare_dataset.py first."
        )

    print("\nLoading tokenizer from local Hugging Face cache...")

    tokenizer = AutoTokenizer.from_pretrained(
        settings["tokenizer_name"],
        revision=settings["tokenizer_revision"],
        trust_remote_code=settings["trust_remote_code"],
        local_files_only=True,
    )

    tokenizer.padding_side = settings["padding_side"]

    if tokenizer.eos_token_id is None or tokenizer.eos_token is None:
        raise ValueError("Tokenizer must define an EOS token")

    config_hashes = {name: sha256_file(path) for name, path in config_paths.items()}

    print("Tokenizer loaded.")
    print(f"EOS token: {tokenizer.eos_token!r}")
    print(f"EOS token ID: {tokenizer.eos_token_id}")
    print("\nStarting train/validation preprocessing...")

    manifest = preprocess_finqa_dataset(
        input_dir=settings["input_dir"],
        output_dir=settings["output_dir"],
        source_repository=settings["source_repository"],
        source_revision=settings["source_revision"],
        expected_counts=settings["expected_counts"],
        tokenizer=tokenizer,
        tokenizer_name=settings["tokenizer_name"],
        tokenizer_revision=settings["tokenizer_revision"],
        model_name=settings["model_name"],
        model_revision=settings["model_revision"],
        evidence_budget=settings["evidence_budget"],
        max_sequence_length=settings["max_sequence_length"],
        config_hashes=config_hashes,
        overwrite=args.overwrite,
    )

    print("\nPreprocessing completed successfully.")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nArtifacts written to: {settings['output_dir']}")


if __name__ == "__main__":
    main()
