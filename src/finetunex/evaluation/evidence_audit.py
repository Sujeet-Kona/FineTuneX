"""Measure whether inference-time retrieval includes annotated FinQA evidence.

Gold annotations are consumed only by this offline audit. They are never passed
to the retriever, prompt builder, training target, or inference path.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


_TEXT_KEY = re.compile(r"text_(-?\d+)\Z")
_TABLE_KEY = re.compile(r"table_(\d+)\Z")


def _gold_indices(record: dict[str, Any]) -> tuple[set[int], set[int]]:
    """Resolve gold_inds keys to zero-based source passage and table indices."""
    raw = record.get("gold_inds")
    if isinstance(raw, str):
        import json

        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{record.get('id')}: invalid gold_inds JSON") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"{record.get('id')}: gold_inds must be an object")

    passages = record.get("pre_text", []) + record.get("post_text", [])
    table = record.get("table", [])
    text_indices: set[int] = set()
    table_indices: set[int] = set()

    for key in raw:
        if not isinstance(key, str):
            raise ValueError(f"{record.get('id')}: gold evidence keys must be strings")

        text_match = _TEXT_KEY.fullmatch(key)
        table_match = _TABLE_KEY.fullmatch(key)

        if text_match:
            original_index = int(text_match.group(1))
            resolved_index = (
                original_index
                if original_index >= 0
                else len(passages) + original_index
            )
            if not 0 <= resolved_index < len(passages):
                raise ValueError(
                    f"{record.get('id')}: gold evidence {key} is out of range"
                )
            text_indices.add(resolved_index)
        elif table_match:
            index = int(table_match.group(1))
            if not 0 <= index < len(table):
                raise ValueError(
                    f"{record.get('id')}: gold evidence {key} is out of range"
                )
            table_indices.add(index)
        else:
            raise ValueError(
                f"{record.get('id')}: unsupported gold evidence key {key!r}"
            )

    return text_indices, table_indices


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _new_bucket() -> dict[str, int]:
    return {
        "examples": 0,
        "examples_with_gold_evidence": 0,
        "complete_evidence_examples": 0,
        "gold_text_units": 0,
        "retrieved_text_units": 0,
        "gold_table_units": 0,
        "retrieved_table_units": 0,
    }


def _finish_bucket(bucket: dict[str, int]) -> dict[str, Any]:
    gold_units = bucket["gold_text_units"] + bucket["gold_table_units"]
    retrieved_units = (
        bucket["retrieved_text_units"] + bucket["retrieved_table_units"]
    )
    examples_with_gold = bucket["examples_with_gold_evidence"]
    return {
        **bucket,
        "complete_evidence_coverage": _ratio(
            bucket["complete_evidence_examples"], examples_with_gold
        ),
        "evidence_unit_recall": _ratio(retrieved_units, gold_units),
        "text_evidence_recall": _ratio(
            bucket["retrieved_text_units"], bucket["gold_text_units"]
        ),
        "table_evidence_recall": _ratio(
            bucket["retrieved_table_units"], bucket["gold_table_units"]
        ),
    }


def audit_evidence_records(
    source_records: list[dict[str, Any]],
    processed_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate gold evidence recall from raw and tokenized validation records."""
    if len(source_records) != len(processed_records):
        raise ValueError(
            "Source and preprocessed record counts differ: "
            f"{len(source_records)} != {len(processed_records)}"
        )
    if not source_records:
        raise ValueError("Cannot audit an empty validation split")

    overall = _new_bucket()
    by_strategy: dict[str, dict[str, int]] = defaultdict(_new_bucket)
    missing_examples: list[dict[str, Any]] = []

    for position, (source, processed) in enumerate(
        zip(source_records, processed_records), start=1
    ):
        example_id = source.get("id")
        metadata = processed.get("metadata")

        if not isinstance(example_id, str) or not example_id:
            raise ValueError(f"Source record {position} has no valid ID")
        if not isinstance(metadata, dict):
            raise ValueError(f"{example_id}: preprocessing metadata is missing")
        if metadata.get("record_id") != example_id:
            raise ValueError(
                f"Record alignment mismatch at position {position}: {example_id}"
            )

        gold_text, gold_table = _gold_indices(source)
        selected_text = set(metadata.get("selected_passage_indices", []))
        selected_table = set(metadata.get("selected_table_row_indices", []))

        if any(type(index) is not int for index in selected_text | selected_table):
            raise ValueError(
                f"{example_id}: selected evidence indices must be integers"
            )

        strategy = metadata.get("table_strategy", "unknown")
        if strategy not in {"complete", "selected_rows"}:
            raise ValueError(f"{example_id}: unsupported table strategy {strategy!r}")

        # A complete table always contains its header. In selected-row mode,
        # the header is present whenever at least one row candidate was selected.
        if strategy == "complete" or selected_table:
            selected_table.add(0)

        retrieved_text = gold_text & selected_text
        retrieved_table = gold_table & selected_table
        missing_text = sorted(gold_text - selected_text)
        missing_table = sorted(gold_table - selected_table)
        has_gold = bool(gold_text or gold_table)
        complete = not missing_text and not missing_table

        for bucket in (overall, by_strategy[strategy]):
            bucket["examples"] += 1
            bucket["examples_with_gold_evidence"] += int(has_gold)
            bucket["complete_evidence_examples"] += int(has_gold and complete)
            bucket["gold_text_units"] += len(gold_text)
            bucket["retrieved_text_units"] += len(retrieved_text)
            bucket["gold_table_units"] += len(gold_table)
            bucket["retrieved_table_units"] += len(retrieved_table)

        if not complete:
            missing_examples.append(
                {
                    "id": example_id,
                    "table_strategy": strategy,
                    "missing_text_indices": missing_text,
                    "missing_table_indices": missing_table,
                }
            )

    return {
        "schema_version": 1,
        "split": "validation",
        "examples": len(source_records),
        "overall": _finish_bucket(overall),
        "by_table_strategy": {
            strategy: _finish_bucket(bucket)
            for strategy, bucket in sorted(by_strategy.items())
        },
        "missing_evidence_examples": missing_examples,
    }
