"""Deterministic, source-verified export of FinQA annotation audit samples."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

CATEGORY_FLAGS: dict[str, tuple[str, ...]] = {
    "exe_ans_disagreements": ("program_exe_ans_disagreement",),
    "exe_ans_disagreement_human_answer_consistent": (
        "exe_ans_disagreement_human_answer_consistent",
    ),
    "human_answer_program_disagreements": ("program_human_answer_disagreement",),
    "step_annotation_confirmed_mismatches": ("steps_program_disagreement",),
    "step_annotation_ambiguous": ("steps_annotation_ambiguous",),
    "unexecutable_programs": ("program_parse_error", "program_execution_error"),
    "program_re_disagreements": ("program_re_disagreement",),
    "program_re_invalid": ("program_re_invalid",),
    "program_re_missing": (),
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
            if not isinstance(record, dict) or "id" not in record:
                raise ValueError(f"{path}:{line_number}: expected object with an id")
            records.append(record)
    return records


def _has_category(audit_record: dict[str, Any], category: str) -> bool:
    if category == "program_re_missing":
        return audit_record.get("program_re_audit", {}).get("status") == "missing"
    flags = set(audit_record.get("flags", []))
    return bool(flags.intersection(CATEGORY_FLAGS[category]))


def export_audit_samples(
    *,
    source_dir: Path,
    audit_path: Path,
    output_dir: Path,
    limit_per_category: int = 10,
) -> dict[str, Any]:
    """Export up to N deterministic examples per category after source verification.

    Each JSONL item contains the untouched source record alongside its audit
    result. The source split's SHA-256 and record count must match the audit
    report before any samples are written.
    """
    if limit_per_category < 1:
        raise ValueError("limit_per_category must be at least 1")

    report = json.loads(audit_path.read_text(encoding="utf-8"))
    if report.get("audit") != "finqa_annotation_integrity":
        raise ValueError("Unsupported annotation audit report")
    split_reports = report.get("splits")
    if not isinstance(split_reports, dict):
        raise ValueError("Audit report is missing split reports")

    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_provenance: dict[str, dict[str, Any]] = {}

    for split in ("train", "validation", "test"):
        split_report = split_reports.get(split)
        source_path = source_dir / f"{split}.jsonl"
        if not isinstance(split_report, dict) or not source_path.is_file():
            raise ValueError(f"Missing audit metadata or source split: {split}")

        raw = source_path.read_bytes()
        source_hash = hashlib.sha256(raw).hexdigest()
        if source_hash != split_report.get("sha256"):
            raise ValueError(
                f"{split}: source SHA-256 differs from the audited input; "
                "rerun the annotation audit before exporting"
            )

        source_records = _read_jsonl(source_path)
        audit_records = split_report.get("records")
        if len(source_records) != split_report.get("examples"):
            raise ValueError(f"{split}: source count differs from audit report")
        if not isinstance(audit_records, list) or len(audit_records) != len(
            source_records
        ):
            raise ValueError(f"{split}: audit record count differs from source")

        source_by_id: dict[str, dict[str, Any]] = {}
        for record in source_records:
            record_id = str(record["id"])
            if record_id in source_by_id:
                raise ValueError(f"{split}: duplicate source id {record_id!r}")
            source_by_id[record_id] = record

        for audit_record in audit_records:
            record_id = str(audit_record.get("id"))
            source_record = source_by_id.get(record_id)
            if source_record is None:
                raise ValueError(f"{split}: audit id {record_id!r} not in source")
            joined = {
                "split": split,
                "source_record": source_record,
                "audit_record": audit_record,
            }
            for category in CATEGORY_FLAGS:
                if _has_category(audit_record, category):
                    candidates[category].append(joined)

        source_provenance[split] = {
            "sha256": source_hash,
            "record_count": len(source_records),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_categories: dict[str, Any] = {}
    for category in CATEGORY_FLAGS:
        ordered = sorted(
            candidates[category],
            key=lambda item: (
                item["split"],
                str(item["source_record"]["id"]),
            ),
        )
        selected = ordered[:limit_per_category]
        output_path = output_dir / f"{category}.jsonl"
        with output_path.open("w", encoding="utf-8") as file:
            for item in selected:
                file.write(json.dumps(item, ensure_ascii=False) + "\n")
        manifest_categories[category] = {
            "available_count": len(ordered),
            "exported_count": len(selected),
            "output_file": output_path.name,
            "sample_ids": [
                f"{item['split']}:{item['source_record']['id']}" for item in selected
            ],
        }

    manifest = {
        "schema_version": 1,
        "audit": "finqa_annotation_integrity",
        "audit_report": str(audit_path),
        "limit_per_category": limit_per_category,
        "selection": "first N by split order train, validation, test then string ID",
        "source_splits": source_provenance,
        "categories": manifest_categories,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest
