"""Tests for source-verified FinQA annotation sample exports."""

import hashlib
import json

import pytest

from finetunex.evaluation.finqa_audit_samples import export_audit_samples


def _write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _audit_record(record_id, flags=(), program_re_status="consistent"):
    return {
        "id": record_id,
        "flags": list(flags),
        "program_re_audit": {"status": program_re_status},
    }


def _fixture(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    split_reports = {}
    for split in ("train", "validation", "test"):
        records = [
            {"id": f"{split}-1", "question": "Q1", "table": [["x", "1"]]},
            {"id": f"{split}-2", "question": "Q2", "table": [["y", "2"]]},
        ]
        source_path = source_dir / f"{split}.jsonl"
        _write_jsonl(source_path, records)
        split_reports[split] = {
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "examples": len(records),
            "records": [
                _audit_record(
                    f"{split}-1",
                    ("program_exe_ans_disagreement", "steps_program_disagreement"),
                ),
                _audit_record(f"{split}-2", program_re_status="missing"),
            ],
        }

    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "audit": "finqa_annotation_integrity",
                "splits": split_reports,
            }
        ),
        encoding="utf-8",
    )
    return source_dir, audit_path


def test_exports_original_source_record_with_audit_result(tmp_path):
    source_dir, audit_path = _fixture(tmp_path)
    output_dir = tmp_path / "samples"

    manifest = export_audit_samples(
        source_dir=source_dir,
        audit_path=audit_path,
        output_dir=output_dir,
        limit_per_category=2,
    )

    sample = json.loads(
        (output_dir / "exe_ans_disagreements.jsonl").read_text().splitlines()[0]
    )
    assert sample["source_record"]["question"] == "Q1"
    assert sample["source_record"]["table"] == [["x", "1"]]
    assert sample["audit_record"]["flags"] == [
        "program_exe_ans_disagreement",
        "steps_program_disagreement",
    ]
    assert manifest["categories"]["exe_ans_disagreements"]["exported_count"] == 2


def test_exports_all_available_when_category_has_fewer_than_limit(tmp_path):
    source_dir, audit_path = _fixture(tmp_path)
    output_dir = tmp_path / "samples"

    manifest = export_audit_samples(
        source_dir=source_dir,
        audit_path=audit_path,
        output_dir=output_dir,
        limit_per_category=10,
    )

    details = manifest["categories"]["program_re_missing"]
    assert details["available_count"] == 3
    assert details["exported_count"] == 3


def test_refuses_to_export_if_source_changed_after_audit(tmp_path):
    source_dir, audit_path = _fixture(tmp_path)
    source_path = source_dir / "train.jsonl"
    source_path.write_text(source_path.read_text() + "{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 differs"):
        export_audit_samples(
            source_dir=source_dir,
            audit_path=audit_path,
            output_dir=tmp_path / "samples",
        )


def test_rejects_zero_sample_limit(tmp_path):
    source_dir, audit_path = _fixture(tmp_path)
    with pytest.raises(ValueError, match="at least 1"):
        export_audit_samples(
            source_dir=source_dir,
            audit_path=audit_path,
            output_dir=tmp_path / "samples",
            limit_per_category=0,
        )


def test_sample_selection_is_stratified_across_splits(tmp_path):
    source_dir, audit_path = _fixture(tmp_path)
    output_dir = tmp_path / "samples"

    manifest = export_audit_samples(
        source_dir=source_dir,
        audit_path=audit_path,
        output_dir=output_dir,
        limit_per_category=3,
    )

    details = manifest["categories"]["exe_ans_disagreements"]
    assert details["exported_split_counts"] == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }
    assert {item.split(":")[0] for item in details["sample_ids"]} == {
        "train",
        "validation",
        "test",
    }


def test_sample_selection_is_deterministic(tmp_path):
    source_dir, audit_path = _fixture(tmp_path)
    first = export_audit_samples(
        source_dir=source_dir,
        audit_path=audit_path,
        output_dir=tmp_path / "first",
        limit_per_category=2,
    )
    second = export_audit_samples(
        source_dir=source_dir,
        audit_path=audit_path,
        output_dir=tmp_path / "second",
        limit_per_category=2,
    )

    assert (
        first["categories"]["exe_ans_disagreements"]["sample_ids"]
        == second["categories"]["exe_ans_disagreements"]["sample_ids"]
    )
