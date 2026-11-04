from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any


class DatasetValidationError(ValueError):
    """Raised when a dataset contains invalid examples."""


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetValidationError(
                    f"Invalid JSON on line {line_number}: {exc.msg}"
                ) from exc

            if not isinstance(record, dict):
                raise DatasetValidationError(
                    f"Line {line_number} must contain a JSON object."
                )

            records.append(record)

    return records


def validate_records(
    records: list[dict[str, Any]],
    columns: dict[str, str],
) -> tuple[list[dict[str, str]], int]:
    required = ("instruction", "output")

    for field in required:
        if field not in columns:
            raise ValueError(f"Missing column mapping: {field}")

    input_column = columns.get("input")

    validated = []
    seen = set()
    duplicates_removed = 0
    errors = []

    for index, record in enumerate(records, start=1):
        instruction = record.get(columns["instruction"])
        output = record.get(columns["output"])
        input_text = record.get(input_column, "") if input_column else ""

        if not isinstance(instruction, str) or not instruction.strip():
            errors.append(f"Example {index}: instruction must be a non-empty string.")
            continue

        if not isinstance(output, str) or not output.strip():
            errors.append(f"Example {index}: output must be a non-empty string.")
            continue

        if not isinstance(input_text, str):
            errors.append(f"Example {index}: input must be a string.")
            continue

        example = {
            "instruction": instruction.strip(),
            "input": input_text.strip(),
            "output": output.strip(),
        }

        fingerprint = (
            example["instruction"],
            example["input"],
            example["output"],
        )

        if fingerprint in seen:
            duplicates_removed += 1
            continue

        seen.add(fingerprint)
        validated.append(example)

    if errors:
        preview = "\n".join(errors[:10])
        remaining = len(errors) - 10

        if remaining > 0:
            preview += f"\n... and {remaining} more validation errors"

        raise DatasetValidationError(preview)

    if len(validated) < 3:
        raise DatasetValidationError(
            "At least three unique valid examples are required for splitting."
        )

    return validated, duplicates_removed


def format_example(example: dict[str, str]) -> str:
    parts = [f"### Instruction:\n{example['instruction']}"]

    if example["input"]:
        parts.append(f"### Input:\n{example['input']}")

    parts.append(f"### Response:\n{example['output']}")

    return "\n\n".join(parts)


def split_dataset(
    records: list[dict[str, str]],
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, str]]]:
    ratios = (train_ratio, validation_ratio, test_ratio)

    if any(ratio <= 0 for ratio in ratios):
        raise ValueError("All split ratios must be greater than zero.")

    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("Dataset split ratios must sum to 1.0.")

    if len(records) < 3:
        raise ValueError("At least three records are required for splitting.")

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)

    validation_count = max(1, round(len(shuffled) * validation_ratio))
    test_count = max(1, round(len(shuffled) * test_ratio))
    train_count = len(shuffled) - validation_count - test_count

    if train_count < 1:
        raise ValueError("Not enough examples to create three non-empty splits.")

    return {
        "train": shuffled[:train_count],
        "validation": shuffled[train_count:train_count + validation_count],
        "test": shuffled[train_count + validation_count:],
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_dataset(
    input_path: str | Path,
    output_dir: str | Path,
    columns: dict[str, str],
    train_ratio: float = 0.80,
    validation_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> dict[str, Any]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    source_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()

    raw_records = load_jsonl(input_path)

    validated, duplicates_removed = validate_records(raw_records, columns)

    splits = split_dataset(
        validated,
        train_ratio,
        validation_ratio,
        test_ratio,
        seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, examples in splits.items():
        formatted = [
            {**example, "text": format_example(example)}
            for example in examples
        ]
        _write_jsonl(output_dir / f"{split_name}.jsonl", formatted)

    manifest = {
        "source_file": input_path.name,
        "source_sha256": source_hash,
        "examples_loaded": len(raw_records),
        "examples_validated": len(validated),
        "duplicates_removed": duplicates_removed,
        "seed": seed,
        "split_ratios": {
            "train": train_ratio,
            "validation": validation_ratio,
            "test": test_ratio,
        },
        "split_counts": {
            name: len(examples)
            for name, examples in splits.items()
        },
    }

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    return manifest
