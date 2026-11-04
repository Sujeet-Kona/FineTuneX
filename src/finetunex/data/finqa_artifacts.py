"""Reproducible serialization and provenance for preprocessed FinQA data."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from finetunex.data.finqa_preprocessing import (
    PROMPT_VERSION,
    preprocess_record,
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a JSON-compatible value using canonical serialization."""
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(serialized).hexdigest()


def package_versions() -> dict[str, str]:
    """Return versions of packages that affect preprocessing."""
    packages = (
        "finetunex",
        "transformers",
        "tokenizers",
        "torch",
        "PyYAML",
    )

    result = {}

    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed-as-package"

    return result


def _read_jsonl(path: Path, expected_count: int) -> list[dict[str, Any]]:
    """Read and validate a JSONL source split."""
    if not path.is_file():
        raise FileNotFoundError(f"Source split not found: {path}")

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL line")

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc

            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")

            records.append(record)

    if len(records) != expected_count:
        raise ValueError(
            f"{path}: expected {expected_count} records, found {len(records)}"
        )

    return records


def _validate_source_manifest(
    manifest: dict[str, Any],
    *,
    repository: str,
    revision: str,
    expected_counts: dict[str, int],
    input_dir: Path,
) -> dict[str, str]:
    """Verify source provenance and return source file hashes."""
    if manifest.get("dataset") != "FinQA":
        raise ValueError("Source manifest does not identify the FinQA dataset")

    if manifest.get("official_splits_preserved") is not True:
        raise ValueError("Source manifest does not confirm official split preservation")

    if manifest.get("examples_removed") != 0:
        raise ValueError("Source manifest reports removed examples")

    if manifest.get("repository") != repository:
        raise ValueError("Source dataset repository does not match config")

    if manifest.get("revision") != revision:
        raise ValueError("Source dataset revision does not match config")

    split_counts = manifest.get("split_counts")
    processed_hashes = manifest.get("processed_sha256")

    if not isinstance(split_counts, dict):
        raise ValueError("Source manifest has invalid split_counts")

    if not isinstance(processed_hashes, dict):
        raise ValueError("Source manifest has invalid processed_sha256")

    actual_hashes = {}

    for split, expected_count in expected_counts.items():
        if split not in {"train", "validation"}:
            raise ValueError(f"Unsupported preprocessing split: {split}")

        if split_counts.get(split) != expected_count:
            raise ValueError(f"Source manifest count mismatch for {split}")

        source_path = input_dir / f"{split}.jsonl"

        if not source_path.is_file():
            raise FileNotFoundError(f"Source split not found: {source_path}")

        actual_hash = sha256_file(source_path)

        if processed_hashes.get(split) != actual_hash:
            raise ValueError(f"{split}.jsonl does not match its source manifest hash")

        actual_hashes[split] = actual_hash

    return actual_hashes


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write deterministic JSONL."""
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write a deterministic, human-readable JSON manifest."""
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")


def _replace_directory(
    staging_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool,
) -> None:
    """Replace an output directory with rollback on replacement failure."""
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_dir}. Pass --overwrite to replace it."
        )

    if output_dir.is_symlink() or (output_dir.exists() and not output_dir.is_dir()):
        raise ValueError(f"Output path must be a regular directory: {output_dir}")

    backup_dir = None

    if output_dir.exists():
        backup_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{output_dir.name}.backup-",
                dir=output_dir.parent,
            )
        )
        backup_dir.rmdir()
        os.replace(output_dir, backup_dir)

    try:
        os.replace(staging_dir, output_dir)
    except Exception:
        if backup_dir is not None and backup_dir.exists():
            os.replace(backup_dir, output_dir)
        raise
    else:
        if backup_dir is not None:
            shutil.rmtree(backup_dir)


def preprocess_finqa_dataset(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    source_repository: str,
    source_revision: str,
    expected_counts: dict[str, int],
    tokenizer: Any,
    tokenizer_name: str,
    tokenizer_revision: str,
    model_name: str,
    model_revision: str,
    evidence_budget: int,
    max_sequence_length: int,
    config_hashes: dict[str, str],
    overwrite: bool = False,
) -> dict[str, Any]:
    """Preprocess train and validation and atomically publish artifacts."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not isinstance(expected_counts, dict):
        raise ValueError("expected_counts must be a dictionary")

    if set(expected_counts) != {"train", "validation"}:
        raise ValueError("expected_counts must contain exactly train and validation")

    for split, count in expected_counts.items():
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"Expected count for {split} must be a positive integer")

    if (
        isinstance(evidence_budget, bool)
        or not isinstance(evidence_budget, int)
        or evidence_budget <= 0
    ):
        raise ValueError("evidence_budget must be a positive integer")

    if (
        isinstance(max_sequence_length, bool)
        or not isinstance(max_sequence_length, int)
        or max_sequence_length <= 0
    ):
        raise ValueError("max_sequence_length must be a positive integer")

    if output_dir.is_symlink():
        raise ValueError("Output directory cannot be a symbolic link")

    resolved_input = input_dir.resolve(strict=True)
    resolved_output = output_dir.resolve()

    if (
        resolved_output == resolved_input
        or resolved_output.is_relative_to(resolved_input)
        or resolved_input.is_relative_to(resolved_output)
    ):
        raise ValueError("Input and output directories must not overlap")

    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_dir}. Pass --overwrite to replace it."
        )

    source_manifest_path = input_dir / "manifest.json"

    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"Source manifest not found: {source_manifest_path}")

    source_manifest_bytes = source_manifest_path.read_bytes()

    try:
        source_manifest = json.loads(source_manifest_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError("Source manifest is invalid JSON") from exc

    if not isinstance(source_manifest, dict):
        raise ValueError("Source manifest must be a JSON object")

    source_hashes = _validate_source_manifest(
        source_manifest,
        repository=source_repository,
        revision=source_revision,
        expected_counts=expected_counts,
        input_dir=input_dir,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-",
            dir=output_dir.parent,
        )
    )

    try:
        split_manifest = {}
        all_sequence_lengths = []

        for split in ("train", "validation"):
            records = _read_jsonl(
                input_dir / f"{split}.jsonl",
                expected_counts[split],
            )

            seen_ids = set()
            processed_records = []
            sequence_lengths = []
            prompt_lengths = []
            target_lengths = []
            evidence_lengths = []
            table_strategies = Counter()

            for record in records:
                processed = preprocess_record(
                    record,
                    tokenizer,
                    split=split,
                    tokenizer_name=tokenizer_name,
                    tokenizer_revision=tokenizer_revision,
                    evidence_budget=evidence_budget,
                    max_sequence_length=max_sequence_length,
                )

                if not isinstance(processed, dict):
                    raise RuntimeError(f"{split}: preprocessor returned a non-dict")

                metadata = processed.get("metadata")
                if not isinstance(metadata, dict):
                    raise RuntimeError(f"{split}: missing preprocessing metadata")

                record_id = metadata.get("record_id")
                if not isinstance(record_id, str) or not record_id:
                    raise RuntimeError(f"{split}: invalid record ID")

                input_ids = processed.get("input_ids")
                attention_mask = processed.get("attention_mask")
                labels = processed.get("labels")

                if not all(
                    isinstance(values, list)
                    for values in (input_ids, attention_mask, labels)
                ):
                    raise RuntimeError(f"{record_id}: token fields must be lists")

                if not input_ids or not (
                    len(input_ids) == len(attention_mask) == len(labels)
                ):
                    raise RuntimeError(
                        f"{record_id}: token fields have invalid lengths"
                    )

                if any(
                    isinstance(value, bool) or value not in (0, 1)
                    for value in attention_mask
                ):
                    raise RuntimeError(
                        f"{record_id}: attention mask must contain only 0 or 1"
                    )

                prompt_length = metadata.get("prompt_tokens")
                if (
                    isinstance(prompt_length, bool)
                    or not isinstance(prompt_length, int)
                    or not 0 < prompt_length < len(input_ids)
                ):
                    raise RuntimeError(f"{record_id}: invalid prompt token count")

                if record_id in seen_ids:
                    raise ValueError(f"{split}: duplicate record ID {record_id}")

                seen_ids.add(record_id)

                if len(processed["input_ids"]) != metadata["sequence_length"]:
                    raise RuntimeError(
                        f"{record_id}: serialized sequence length mismatch"
                    )

                if len(processed["input_ids"]) > max_sequence_length:
                    raise RuntimeError(
                        f"{record_id}: sequence exceeds configured limit"
                    )

                if processed["input_ids"][-1] != tokenizer.eos_token_id:
                    raise RuntimeError(f"{record_id}: EOS token missing")

                if labels[:prompt_length] != [-100] * prompt_length:
                    raise RuntimeError(f"{record_id}: prompt labels are not masked")

                if labels[prompt_length:] != input_ids[prompt_length:]:
                    raise RuntimeError(f"{record_id}: target labels are misaligned")

                if labels[-1] != tokenizer.eos_token_id:
                    raise RuntimeError(f"{record_id}: EOS label is not supervised")

                processed_records.append(processed)
                sequence_lengths.append(len(input_ids))
                prompt_lengths.append(prompt_length)
                target_lengths.append(metadata["target_tokens_including_eos"])
                evidence_lengths.append(metadata["evidence_tokens"])
                table_strategies[metadata["table_strategy"]] += 1

            if len(seen_ids) != expected_counts[split]:
                raise RuntimeError(f"{split}: processed record count mismatch")

            output_path = staging_dir / f"{split}.jsonl"
            _write_jsonl(output_path, processed_records)

            all_sequence_lengths.extend(sequence_lengths)

            split_manifest[split] = {
                "record_count": len(processed_records),
                "output_file": output_path.name,
                "output_sha256": sha256_file(output_path),
                "sequence_length": {
                    "min": min(sequence_lengths),
                    "max": max(sequence_lengths),
                    "mean": round(
                        sum(sequence_lengths) / len(sequence_lengths),
                        4,
                    ),
                },
                "prompt_tokens": {
                    "mean": round(
                        sum(prompt_lengths) / len(prompt_lengths),
                        4,
                    ),
                },
                "target_tokens_including_eos": {
                    "mean": round(
                        sum(target_lengths) / len(target_lengths),
                        4,
                    ),
                },
                "evidence_tokens": {
                    "mean": round(
                        sum(evidence_lengths) / len(evidence_lengths),
                        4,
                    ),
                    "max": max(evidence_lengths),
                },
                "table_strategies": dict(sorted(table_strategies.items())),
            }

        manifest = {
            "schema_version": 1,
            "dataset": {
                "name": "FinQA",
                "repository": source_repository,
                "revision": source_revision,
                "source_manifest_sha256": hashlib.sha256(
                    source_manifest_bytes
                ).hexdigest(),
                "source_split_sha256": source_hashes,
            },
            "model": {
                "name": model_name,
                "revision": model_revision,
            },
            "tokenizer": {
                "name": tokenizer_name,
                "revision": tokenizer_revision,
                "eos_token": tokenizer.eos_token,
                "eos_token_id": tokenizer.eos_token_id,
            },
            "preprocessing": {
                "prompt_version": PROMPT_VERSION,
                "evidence_budget": evidence_budget,
                "max_sequence_length": max_sequence_length,
                "splits": ["train", "validation"],
                "test_split_included": False,
                "truncation": False,
                "labels": "prompt masked with -100; target and EOS supervised",
            },
            "configuration_sha256": dict(sorted(config_hashes.items())),
            "package_versions": package_versions(),
            "splits": split_manifest,
            "total_records": sum(expected_counts.values()),
            "overall_sequence_length": {
                "min": min(all_sequence_lengths),
                "max": max(all_sequence_lengths),
            },
        }

        _write_manifest(staging_dir / "manifest.json", manifest)

        # Verify staged files before publishing.
        for split in ("train", "validation"):
            staged_path = staging_dir / f"{split}.jsonl"

            if sha256_file(staged_path) != split_manifest[split]["output_sha256"]:
                raise RuntimeError(f"{split}: staged artifact hash verification failed")

        _replace_directory(
            staging_dir,
            output_dir,
            overwrite=overwrite,
        )

        return manifest

    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
