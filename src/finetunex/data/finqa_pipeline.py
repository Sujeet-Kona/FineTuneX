from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_dataset


class FinQADatasetError(ValueError):
    """Raised when the FinQA dataset violates its expected contract."""


REQUIRED_FIELDS = {
    "id",
    "pre_text",
    "post_text",
    "table",
    "table_ori",
    "question",
    "answer",
    "exe_ans",
    "explanation",
    "ann_table_rows",
    "ann_text_rows",
    "steps",
    "program",
    "program_re",
    "gold_inds",
}

EXPECTED_SPLITS = {"train", "validation", "test"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _sha256_records(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()

    for record in records:
        serialized = json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest.update(serialized.encode("utf-8"))
        digest.update(b"\n")

    return digest.hexdigest()


def validate_dataset(
    dataset: DatasetDict,
    expected_counts: dict[str, int] | None = None,
) -> None:
    """Validate the official FinQA split structure and required fields."""
    actual_splits = set(dataset.keys())

    if actual_splits != EXPECTED_SPLITS:
        raise FinQADatasetError(
            f"Expected splits {sorted(EXPECTED_SPLITS)}, "
            f"received {sorted(actual_splits)}."
        )

    for split_name, split in dataset.items():
        missing = REQUIRED_FIELDS - set(split.column_names)

        if missing:
            raise FinQADatasetError(
                f"{split_name} is missing required fields: {sorted(missing)}"
            )

        if expected_counts is not None:
            expected = expected_counts[split_name]

            if len(split) != expected:
                raise FinQADatasetError(
                    f"{split_name} has {len(split)} records; expected {expected}."
                )

        ids = split["id"]

        if any(not isinstance(value, str) or not value.strip() for value in ids):
            raise FinQADatasetError(
                f"{split_name} contains an empty or invalid example ID."
            )

        if len(ids) != len(set(ids)):
            raise FinQADatasetError(
                f"{split_name} contains duplicate example IDs."
            )

    split_ids = {
        name: set(split["id"])
        for name, split in dataset.items()
    }

    for first in EXPECTED_SPLITS:
        for second in EXPECTED_SPLITS:
            if first >= second:
                continue

            overlap = split_ids[first] & split_ids[second]

            if overlap:
                raise FinQADatasetError(
                    f"ID overlap between {first} and {second}: "
                    f"{len(overlap)} records."
                )


def audit_dataset(
    records_by_split: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Report data anomalies without removing or modifying examples."""
    question_locations: dict[str, list[str]] = defaultdict(list)
    duplicate_ids: dict[str, list[str]] = defaultdict(list)

    report: dict[str, Any] = {
        "splits": {},
        "cross_split_duplicate_questions": [],
        "cross_split_duplicate_ids": [],
    }

    for split_name, records in records_by_split.items():
        operations = Counter()
        blank_answers = []
        final_step_mismatches = []
        final_step_unavailable = []

        for record in records:
            example_id = record["id"]
            question = record["question"].strip().casefold()

            question_locations[question].append(
                f"{split_name}:{example_id}"
            )
            duplicate_ids[example_id].append(split_name)

            if not record["answer"].strip():
                blank_answers.append(example_id)

            for operation in record["program"].split(","):
                operation = operation.strip()

                if "(" in operation:
                    operations[operation.split("(", 1)[0]] += 1

            steps = record["steps"]
            results = steps.get("res", []) if isinstance(steps, dict) else []

            if not results:
                final_step_unavailable.append(example_id)
            elif results[-1].strip() != record["exe_ans"].strip():
                final_step_mismatches.append(example_id)

        report["splits"][split_name] = {
            "examples": len(records),
            "blank_answer_count": len(blank_answers),
            "blank_answer_ids": blank_answers,
            "final_step_raw_mismatch_count": len(final_step_mismatches),
            "final_step_raw_mismatch_ids": final_step_mismatches,
            "final_step_unavailable_count": len(final_step_unavailable),
            "operation_counts": dict(sorted(operations.items())),
        }

    for question, locations in question_locations.items():
        if len(locations) > 1:
            split_names = {location.split(":", 1)[0] for location in locations}

            if len(split_names) > 1:
                report["cross_split_duplicate_questions"].append({
                    "question_sha256": hashlib.sha256(
                        question.encode("utf-8")
                    ).hexdigest(),
                    "locations": locations,
                })

    for example_id, split_names in duplicate_ids.items():
        if len(split_names) > 1:
            report["cross_split_duplicate_ids"].append({
                "id": example_id,
                "splits": split_names,
            })

    report["cross_split_duplicate_question_count"] = len(
        report["cross_split_duplicate_questions"]
    )
    report["cross_split_duplicate_id_count"] = len(
        report["cross_split_duplicate_ids"]
    )

    return report


def prepare_finqa_dataset(
    repository: str,
    revision: str,
    output_dir: str | Path,
    expected_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Download, validate and serialize the official FinQA splits."""
    dataset = load_dataset(repository, revision=revision)

    validate_dataset(dataset, expected_counts)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records_by_split: dict[str, list[dict[str, Any]]] = {}
    output_hashes = {}

    for split_name, split in dataset.items():
        records = [dict(record) for record in split]
        records_by_split[split_name] = records

        output_path = output_dir / f"{split_name}.jsonl"

        with output_path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

        output_hashes[split_name] = _sha256_file(output_path)

    audit = audit_dataset(records_by_split)

    manifest = {
        "dataset": "FinQA",
        "repository": repository,
        "revision": revision,
        "split_counts": {
            name: len(records)
            for name, records in records_by_split.items()
        },
        "split_fingerprints": {
            name: getattr(dataset[name], "_fingerprint", None)
            for name in dataset
        },
        "processed_sha256": output_hashes,
        "audit": audit,
        "examples_removed": 0,
        "official_splits_preserved": True,
    }

    manifest_path = output_dir / "manifest.json"

    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)

    return manifest
