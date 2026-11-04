import hashlib
import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import evaluate_finqa_model as evaluator  # noqa: E402
from evaluate_finqa_model import (  # noqa: E402
    evaluate_prediction,
    inference_only_record,
    load_validation_records,
    validate_preprocessing_provenance,
)


def test_inference_record_excludes_gold_fields():
    record = {
        "id": "example-1",
        "question": "What is the result?",
        "pre_text": ["Revenue was 100."],
        "post_text": [],
        "table": [["Year", "Revenue"], ["2024", "100"]],
        "program": "add(1, 2)",
        "exe_ans": "3",
        "answer": "3",
        "gold_inds": {"text_0": "Revenue was 100."},
    }

    result = inference_only_record(record)

    assert set(result) == {
        "id",
        "question",
        "pre_text",
        "post_text",
        "table",
    }
    assert "program" not in result
    assert "exe_ans" not in result
    assert "gold_inds" not in result

def test_correct_generated_program():
    result = evaluate_prediction(
        "add(100, 27.4)",
        [["Year", "Revenue"]],
        "127.4",
    )

    assert result["parse_status"] == "success"
    assert result["execution_status"] == "success"
    assert result["predicted_answer"] == 127.4
    assert result["answer_match"] is True

def test_valid_program_with_wrong_answer():
    result = evaluate_prediction(
        "add(100, 20)",
        [["Year", "Revenue"]],
        "127.4",
    )

    assert result["parse_status"] == "success"
    assert result["execution_status"] == "success"
    assert result["answer_match"] is False

def test_malformed_program_is_parse_failure():
    result = evaluate_prediction(
        "this is not a FinQA program",
        [["Year", "Revenue"]],
        "127.4",
    )

    assert result["parse_status"] == "failed"
    assert result["execution_status"] == "not_attempted"
    assert result["answer_match"] is False
    assert result["parse_error"]

def test_execution_error_is_not_a_parse_error():
    result = evaluate_prediction(
        "divide(10, 0)",
        [["Year", "Revenue"]],
        "5",
    )

    assert result["parse_status"] == "success"
    assert result["execution_status"] == "failed"
    assert result["answer_match"] is False
    assert result["execution_error"]

def test_table_program_executes_against_source_table():
    result = evaluate_prediction(
        "table_sum(Revenue, none)",
        [["Revenue", "100", "200"]],
        "300",
    )

    assert result["execution_status"] == "success"
    assert result["predicted_answer"] == 300
    assert result["answer_match"] is True

def test_validation_provenance_accepts_matching_source_and_artifact(
    tmp_path, monkeypatch
):
    processed_dir = tmp_path / "artifacts" / "preprocessed" / "finqa"
    processed_dir.mkdir(parents=True)
    source_path = tmp_path / "data" / "processed" / "finqa" / "validation.jsonl"
    source_path.parent.mkdir(parents=True)
    source_path.write_text('{"id":"record-1"}\\n', encoding="utf-8")
    artifact_path = processed_dir / "validation.jsonl"
    artifact_path.write_text('{"input_ids":[1,2]}\\n', encoding="utf-8")

    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    artifact_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 2,
        "dataset": {"source_split_sha256": {"validation": source_hash}},
        "splits": {"validation": {"output_sha256": artifact_hash}},
    }
    manifest_path = processed_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(evaluator, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(evaluator, "VALIDATION_PATH", source_path)

    manifest_hash, actual_source_hash, actual_artifact_hash = (
        validate_preprocessing_provenance()
    )

    assert manifest_hash == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert actual_source_hash == source_hash
    assert actual_artifact_hash == artifact_hash

def test_validation_provenance_rejects_wrong_source_hash(tmp_path, monkeypatch):
    processed_dir = tmp_path / "artifacts" / "preprocessed" / "finqa"
    processed_dir.mkdir(parents=True)
    source_path = tmp_path / "data" / "processed" / "finqa" / "validation.jsonl"
    source_path.parent.mkdir(parents=True)
    source_path.write_text('{"id":"record-1"}\\n', encoding="utf-8")
    (processed_dir / "validation.jsonl").write_text(
        '{"input_ids":[1,2]}\\n', encoding="utf-8"
    )
    (processed_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "dataset": {"source_split_sha256": {"validation": "wrong"}},
                "splits": {"validation": {"output_sha256": "wrong"}},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(evaluator, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(evaluator, "VALIDATION_PATH", source_path)

    with pytest.raises(ValueError, match="Raw validation split"):
        validate_preprocessing_provenance()

def test_validation_loader_defaults_to_the_complete_file(tmp_path, monkeypatch):
    path = tmp_path / "validation.jsonl"
    records = [
        {
            "id": f"record-{index}",
            "question": "Question?",
            "pre_text": [],
            "post_text": [],
            "table": [],
            "exe_ans": "1",
        }
        for index in range(3)
    ]
    path.write_text(
        "".join(json.dumps(record) + "\\n" for record in records),
        encoding="utf-8",
    )
    monkeypatch.setattr(evaluator, "VALIDATION_PATH", path)

    assert len(load_validation_records()) == 3
    assert len(load_validation_records(1)) == 1

def test_validation_sampling_is_reproducible_and_id_based(tmp_path, monkeypatch):
    path = tmp_path / "validation.jsonl"
    records = [
        {
            "id": f"record-{index}",
            "question": "Question?",
            "pre_text": [],
            "post_text": [],
            "table": [],
            "exe_ans": "1",
        }
        for index in range(10)
    ]
    path.write_text(
        "".join(json.dumps(record) + "\\n" for record in records),
        encoding="utf-8",
    )
    monkeypatch.setattr(evaluator, "VALIDATION_PATH", path)

    first = load_validation_records(4, sample_seed=42)
    second = load_validation_records(4, sample_seed=42)
    other_seed = load_validation_records(4, sample_seed=7)

    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert len({row["id"] for row in first}) == 4
    assert [row["id"] for row in first] != [row["id"] for row in other_seed]



def test_validation_loader_rejects_duplicate_record_ids(tmp_path, monkeypatch):
    path = tmp_path / "validation.jsonl"
    record = {
        "id": "duplicate",
        "question": "Question?",
        "pre_text": [],
        "post_text": [],
        "table": [],
        "exe_ans": "1",
    }
    path.write_text(
        json.dumps(record) + "\\n" + json.dumps(record) + "\\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(evaluator, "VALIDATION_PATH", path)

    with pytest.raises(ValueError, match="Duplicate validation record ID"):
        load_validation_records()
