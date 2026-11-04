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


def _validate_evidence_annotations(record: dict[str, Any]) -> dict[str, int]:
    """Validate gold evidence references and annotation-array consistency."""
    import re

    example_id = record.get("id", "<unknown>")
    annotations_raw = record.get("gold_inds")

    try:
        annotations = (
            json.loads(annotations_raw)
            if isinstance(annotations_raw, str)
            else annotations_raw
        )
    except json.JSONDecodeError as exc:
        raise FinQADatasetError(f"{example_id}: gold_inds is not valid JSON.") from exc

    if not isinstance(annotations, dict):
        raise FinQADatasetError(f"{example_id}: gold_inds must be a JSON object.")

    pre_text = record.get("pre_text")
    post_text = record.get("post_text")
    table = record.get("table")

    if not isinstance(pre_text, list) or not isinstance(post_text, list):
        raise FinQADatasetError(f"{example_id}: pre_text and post_text must be lists.")

    if not isinstance(table, list):
        raise FinQADatasetError(f"{example_id}: table must be a list.")

    combined_text = pre_text + post_text
    gold_text_indices = set()
    gold_table_indices = set()

    for key, description in annotations.items():
        text_match = re.fullmatch(r"text_(-?\d+)", key)
        table_match = re.fullmatch(r"table_(\d+)", key)

        if text_match:
            index = int(text_match.group(1))
            resolved_index = index if index >= 0 else len(combined_text) + index

            if not 0 <= resolved_index < len(combined_text):
                raise FinQADatasetError(
                    f"{example_id}: {key} is out of range for "
                    f"{len(combined_text)} text passages."
                )

            source_text = combined_text[resolved_index]

            if not isinstance(description, str) or not isinstance(source_text, str):
                raise FinQADatasetError(
                    f"{example_id}: {key} and its source must be strings."
                )

            def normalize(value: str) -> str:
                return " ".join(value.split())

            if normalize(description) != normalize(source_text):
                raise FinQADatasetError(
                    f"{example_id}: {key} description does not match "
                    "its referenced source passage."
                )

            gold_text_indices.add(index)

        elif table_match:
            index = int(table_match.group(1))

            if index >= len(table):
                raise FinQADatasetError(
                    f"{example_id}: {key} is out of range for {len(table)} table rows."
                )

            if not isinstance(description, str) or not description.strip():
                raise FinQADatasetError(
                    f"{example_id}: {key} must have a non-empty description."
                )

            gold_table_indices.add(index)

        else:
            raise FinQADatasetError(
                f"{example_id}: unsupported gold evidence key {key!r}."
            )

    ann_text = record.get("ann_text_rows")
    ann_table = record.get("ann_table_rows")

    for field_name, indices in (
        ("ann_text_rows", ann_text),
        ("ann_table_rows", ann_table),
    ):
        if not isinstance(indices, list) or any(
            type(index) is not int for index in indices
        ):
            raise FinQADatasetError(
                f"{example_id}: {field_name} must be a list of integers."
            )

    ann_text_indices = set(ann_text)
    ann_table_indices = set(ann_table)

    if not ann_text_indices <= gold_text_indices:
        raise FinQADatasetError(
            f"{example_id}: ann_text_rows contains indices absent from gold_inds."
        )

    if not ann_table_indices <= gold_table_indices:
        raise FinQADatasetError(
            f"{example_id}: ann_table_rows contains indices absent from gold_inds."
        )

    return {
        "gold_text_annotations": len(gold_text_indices),
        "gold_table_annotations": len(gold_table_indices),
        "ann_text_annotations": len(ann_text),
        "ann_table_annotations": len(ann_table),
    }


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

        for record in split:
            _validate_evidence_annotations(dict(record))

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
            raise FinQADatasetError(f"{split_name} contains duplicate example IDs.")

    split_ids = {name: set(split["id"]) for name, split in dataset.items()}

    for first in EXPECTED_SPLITS:
        for second in EXPECTED_SPLITS:
            if first >= second:
                continue

            overlap = split_ids[first] & split_ids[second]

            if overlap:
                raise FinQADatasetError(
                    f"ID overlap between {first} and {second}: {len(overlap)} records."
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
        evidence_counts = Counter()

        for record in records:
            evidence_counts.update(_validate_evidence_annotations(record))
            example_id = record["id"]
            question = record["question"].strip().casefold()

            question_locations[question].append(f"{split_name}:{example_id}")
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
            "evidence_annotations": dict(sorted(evidence_counts.items())),
        }

    for question, locations in question_locations.items():
        if len(locations) > 1:
            split_names = {location.split(":", 1)[0] for location in locations}

            if len(split_names) > 1:
                report["cross_split_duplicate_questions"].append(
                    {
                        "question_sha256": hashlib.sha256(
                            question.encode("utf-8")
                        ).hexdigest(),
                        "locations": locations,
                    }
                )

    for example_id, split_names in duplicate_ids.items():
        if len(split_names) > 1:
            report["cross_split_duplicate_ids"].append(
                {
                    "id": example_id,
                    "splits": split_names,
                }
            )

    report["cross_split_duplicate_question_count"] = len(
        report["cross_split_duplicate_questions"]
    )
    report["cross_split_duplicate_id_count"] = len(report["cross_split_duplicate_ids"])

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
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

        output_hashes[split_name] = _sha256_file(output_path)

    audit = audit_dataset(records_by_split)

    manifest = {
        "dataset": "FinQA",
        "repository": repository,
        "revision": revision,
        "split_counts": {
            name: len(records) for name, records in records_by_split.items()
        },
        "split_fingerprints": {
            name: getattr(dataset[name], "_fingerprint", None) for name in dataset
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
