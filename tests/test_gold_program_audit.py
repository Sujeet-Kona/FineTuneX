"""Tests for the offline FinQA gold-program audit."""

import json

import pytest

from finetunex.evaluation.finqa_executor import FinQAExecutionError
from scripts import evaluate_gold_programs as audit


def test_reference_audit_runs_even_when_custom_executor_fails(tmp_path, monkeypatch):
    data_dir = tmp_path / "finqa"
    data_dir.mkdir()
    record = {
        "id": "record-1",
        "program": "add(1, 2)",
        "table": [],
        "exe_ans": 3,
        "answer": "3",
    }
    (data_dir / "validation.jsonl").write_text(json.dumps(record) + "\n")
    monkeypatch.setattr(audit, "DATA_DIR", data_dir)

    def custom_failure(program, table):
        raise FinQAExecutionError("deliberate custom-executor failure")

    monkeypatch.setattr(audit, "execute_program", custom_failure)
    monkeypatch.setattr(audit, "reference_execute_program", lambda program, table: 3)

    result = audit.evaluate_split("validation")

    assert result["successful"] == 0
    assert len(result["errors"]) == 1
    assert result["reference_errors"] == []
    assert result["reference_mismatches"] == []
    assert len(result["executor_disagreements"]) == 1
    assert result["executor_disagreements"][0]["custom_predicted"] is None
    assert result["executor_disagreements"][0]["reference_predicted"] == 3


@pytest.mark.parametrize(
    ("predicted", "expected", "matches"),
    [
        (0.33333, "0.333333", False),\n        (0.33333, "0.33333", True),
        (0.33331, "0.333333", False),
        ("yes", "yes", True),
        ("no", "yes", False),
    ],
)
def test_answers_match_rounds_prediction_only_then_compares_exactly(predicted, expected, matches):
    assert audit.answers_match(predicted, expected) is matches
