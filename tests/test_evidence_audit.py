"""Tests for FinQA evidence audit."""

import pytest

from finetunex.evaluation.evidence_audit import audit_evidence_records


def make_source():
    return {
        "id": "example-1",
        "pre_text": ["first", "second"],
        "post_text": ["third passage"],
        "table": [["Year", "Revenue"], ["2022", "100"], ["2023", "125"]],
        "gold_inds": {"text_-1": "third passage", "table_2": "2023 | 125"},
    }


def make_processed(passages=None, rows=None, strategy="selected_rows"):
    return {
        "metadata": {
            "record_id": "example-1",
            "selected_passage_indices": passages or [],
            "selected_table_row_indices": rows or [],
            "table_strategy": strategy,
        }
    }


def test_negative_text_index_and_table_evidence_are_resolved():
    report = audit_evidence_records(
        [make_source()], [make_processed(passages=[2], rows=[2])]
    )
    assert report["overall"]["complete_evidence_coverage"] == 1.0
    assert report["overall"]["evidence_unit_recall"] == 1.0


def test_missing_evidence_is_reported_by_type():
    report = audit_evidence_records([make_source()], [make_processed(rows=[1])])
    assert report["overall"]["text_evidence_recall"] == 0.0
    assert report["overall"]["table_evidence_recall"] == 0.0
    assert report["missing_evidence_examples"][0]["missing_text_indices"] == [2]
    assert report["missing_evidence_examples"][0]["missing_table_indices"] == [2]


def test_complete_table_includes_header_and_all_rows():
    report = audit_evidence_records(
        [make_source()], [make_processed(rows=[1, 2], strategy="complete")]
    )
    assert report["overall"]["table_evidence_recall"] == 1.0


def test_rejects_record_alignment_mismatch():
    item = make_processed()
    item["metadata"]["record_id"] = "wrong-id"
    with pytest.raises(ValueError, match="alignment mismatch"):
        audit_evidence_records([make_source()], [item])


def test_rejects_out_of_range_gold_indices():
    record = make_source()
    record["gold_inds"] = {"text_99": "missing"}
    with pytest.raises(ValueError, match="out of range"):
        audit_evidence_records([record], [make_processed()])


def test_rejects_empty_or_mismatched_splits():
    with pytest.raises(ValueError, match="empty"):
        audit_evidence_records([], [])
    with pytest.raises(ValueError, match="counts differ"):
        audit_evidence_records([make_source()], [])
