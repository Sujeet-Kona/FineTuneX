import json

import pytest
from datasets import Dataset, DatasetDict

from finetunex.data.finqa_pipeline import (
    FinQADatasetError,
    _validate_evidence_annotations,
    audit_dataset,
    prepare_finqa_dataset,
    validate_dataset,
)


def make_record(example_id="example-1", question="What is the change?"):
    return {
        "id": example_id,
        "pre_text": ["Financial report context"],
        "post_text": [],
        "table": [["Year", "Revenue"], ["2022", "100"], ["2023", "125"]],
        "table_ori": [["Year", "Revenue"], ["2022", "100"], ["2023", "125"]],
        "question": question,
        "answer": "25%",
        "exe_ans": "0.25",
        "explanation": "Revenue increased by 25%.",
        "ann_table_rows": [1, 2],
        "ann_text_rows": [],
        "steps": {
            "op": ["subtract", "divide"],
            "arg1": ["125", "#0"],
            "arg2": ["100", "100"],
            "res": ["25", "25%"],
        },
        "program": "subtract(125, 100), divide(#0, 100)",
        "program_re": "divide(subtract(125, 100), 100)",
        "gold_inds": json.dumps({
            "table_1": "revenue row",
            "table_2": "comparison row",
        }),
    }


def test_evidence_validator_accepts_negative_text_index():
    record = make_record()
    record["pre_text"] = ["First passage"]
    record["post_text"] = ["Last passage"]
    record["gold_inds"] = json.dumps({
        "text_-1": "Last passage",
        "table_1": "revenue row",
        "table_2": "comparison row",
    })
    record["ann_text_rows"] = [-1]

    counts = _validate_evidence_annotations(record)

    assert counts == {
        "gold_text_annotations": 1,
        "gold_table_annotations": 2,
        "ann_text_annotations": 1,
        "ann_table_annotations": 2,
    }


def test_evidence_validator_rejects_out_of_range_text_index():
    record = make_record()
    record["gold_inds"] = json.dumps({"text_1": "Missing passage"})

    with pytest.raises(FinQADatasetError, match="text_1 is out of range"):
        _validate_evidence_annotations(record)


def test_evidence_validator_rejects_mismatched_text_description():
    record = make_record()
    record["gold_inds"] = json.dumps({"text_0": "Different passage"})

    with pytest.raises(FinQADatasetError, match="does not match"):
        _validate_evidence_annotations(record)


def test_evidence_validator_rejects_out_of_range_table_index():
    record = make_record()
    record["gold_inds"] = json.dumps({"table_3": "missing row"})

    with pytest.raises(FinQADatasetError, match="table_3 is out of range"):
        _validate_evidence_annotations(record)


def test_evidence_validator_rejects_malformed_gold_inds():
    record = make_record()
    record["gold_inds"] = "{invalid json"

    with pytest.raises(FinQADatasetError, match="not valid JSON"):
        _validate_evidence_annotations(record)


def test_evidence_validator_rejects_annotation_not_in_gold_inds():
    record = make_record()
    record["ann_table_rows"] = [0]

    with pytest.raises(FinQADatasetError, match="absent from gold_inds"):
        _validate_evidence_annotations(record)


def make_dataset():
    return DatasetDict({
        "train": Dataset.from_list([make_record("train-1")]),
        "validation": Dataset.from_list([make_record("validation-1")]),
        "test": Dataset.from_list([make_record("test-1")]),
    })


def test_validate_accepts_complete_official_split_structure():
    dataset = make_dataset()

    validate_dataset(
        dataset,
        {
            "train": 1,
            "validation": 1,
            "test": 1,
        },
    )


def test_validate_rejects_missing_split():
    dataset = make_dataset()
    del dataset["test"]

    with pytest.raises(FinQADatasetError, match="Expected splits"):
        validate_dataset(dataset)


def test_validate_rejects_missing_required_field():
    dataset = make_dataset()
    dataset["train"] = dataset["train"].remove_columns(["program"])

    with pytest.raises(FinQADatasetError, match="missing required fields"):
        validate_dataset(dataset)


def test_validate_rejects_duplicate_ids_within_split():
    dataset = make_dataset()
    dataset["train"] = Dataset.from_list([
        make_record("duplicate"),
        make_record("duplicate"),
    ])

    with pytest.raises(FinQADatasetError, match="duplicate example IDs"):
        validate_dataset(dataset)


def test_validate_rejects_cross_split_id_overlap():
    dataset = make_dataset()
    dataset["validation"] = Dataset.from_list([
        make_record("train-1"),
    ])

    with pytest.raises(FinQADatasetError, match="ID overlap"):
        validate_dataset(dataset)


def test_validate_rejects_unexpected_split_count():
    dataset = make_dataset()

    with pytest.raises(FinQADatasetError, match="expected"):
        validate_dataset(
            dataset,
            {"train": 2, "validation": 1, "test": 1},
        )


def test_audit_reports_blank_answers_without_removing_records():
    record = make_record()
    record["answer"] = ""

    audit = audit_dataset({"train": [record]})

    assert audit["splits"]["train"]["examples"] == 1
    assert audit["splits"]["train"]["blank_answer_count"] == 1
    assert audit["splits"]["train"]["blank_answer_ids"] == ["example-1"]


def test_audit_reports_raw_final_step_mismatch():
    record = make_record()
    record["exe_ans"] = "0.25"

    audit = audit_dataset({"train": [record]})

    assert audit["splits"]["train"]["final_step_raw_mismatch_count"] == 1


def test_audit_detects_cross_split_duplicate_questions():
    first = make_record("train-1", "What is the change?")
    second = make_record("test-1", "  WHAT IS THE CHANGE? ")

    audit = audit_dataset({
        "train": [first],
        "test": [second],
    })

    assert audit["cross_split_duplicate_question_count"] == 1


def test_prepare_writes_original_records_and_manifest(tmp_path, monkeypatch):
    dataset = make_dataset()

    monkeypatch.setattr(
        "finetunex.data.finqa_pipeline.load_dataset",
        lambda repository, revision: dataset,
    )

    output_dir = tmp_path / "finqa"

    manifest = prepare_finqa_dataset(
        repository="bevaya/FinQA",
        revision="test-revision",
        output_dir=output_dir,
        expected_counts={
            "train": 1,
            "validation": 1,
            "test": 1,
        },
    )

    assert manifest["split_counts"] == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }
    assert manifest["examples_removed"] == 0
    assert manifest["official_splits_preserved"] is True

    record = json.loads(
        (output_dir / "train.jsonl").read_text(encoding="utf-8")
    )

    assert record["program"] == "subtract(125, 100), divide(#0, 100)"
    assert record["program_re"] == "divide(subtract(125, 100), 100)"
    assert record["exe_ans"] == "0.25"
    assert (output_dir / "manifest.json").exists()


def test_evidence_validator_preserves_duplicate_annotation_indices():
    record = {
        "id": "duplicate-evidence",
        "pre_text": [],
        "post_text": [],
        "table": [["Revenue"], ["Profit"], ["Assets"], ["Debt"], ["Credit"]],
        "gold_inds": json.dumps({
            "table_4": "credit derivatives row",
        }),
        "ann_text_rows": [],
        "ann_table_rows": [4, 4],
    }

    original_indices = record["ann_table_rows"].copy()

    counts = _validate_evidence_annotations(record)

    assert record["ann_table_rows"] == original_indices
    assert counts["gold_table_annotations"] == 1
    assert counts["ann_table_annotations"] == 2
