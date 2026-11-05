"""Tests for the non-destructive FinQA annotation-integrity audit."""

from finetunex.evaluation.finqa_data_integrity import audit_record


def _record(**overrides):
    record = {
        "id": "example-1",
        "program": "add(100, 200), divide(#0, 2)",
        "table": [],
        "exe_ans": "150",
        "answer": "150",
        "steps": {
            "op": ["add1-1", "divide0-0"],
            "arg1": ["100", "#0"],
            "arg2": ["200", "2"],
            "res": ["300", "150"],
        },
    }
    record.update(overrides)
    return record


def test_consistent_record_is_not_flagged():
    result = audit_record(_record())
    assert result["classification"] == "internally_consistent"
    assert result["flags"] == []
    assert result["program_result"] == 150


def test_detects_step_result_that_does_not_follow_program():
    record = _record()
    record["steps"]["res"][1] = "91"
    result = audit_record(record)
    assert "steps_program_disagreement" in result["flags"]


def test_detects_step_argument_that_does_not_follow_program():
    record = _record()
    record["steps"]["arg2"][0] = "92500"
    result = audit_record(record)
    assert "steps_program_disagreement" in result["flags"]


def test_detects_program_answer_conflict():
    result = audit_record(_record(answer="91000", exe_ans="150"))
    assert "program_human_answer_disagreement" in result["flags"]


def test_identifies_rounded_exe_ans_when_answer_agrees():
    result = audit_record(
        _record(
            program="multiply(423897556, 87.32)",
            exe_ans="3.70147e+10",
            answer="37014734589.9",
            steps={
                "op": ["multiply1-1"],
                "arg1": ["423897556"],
                "arg2": ["87.32"],
                "res": ["37014734589.9"],
            },
        )
    )
    assert "program_exe_ans_disagreement" in result["flags"]
    assert "likely_exe_ans_precision_or_format_issue" in result["flags"]
    assert result["human_answer_match"] is True


def test_percentage_human_answer_is_normalized():
    result = audit_record(
        _record(
            program="divide(28, 100)",
            exe_ans="0.28",
            answer="28%",
            steps={
                "op": ["divide0-0"],
                "arg1": ["28"],
                "arg2": ["100"],
                "res": ["28%"],
            },
        )
    )
    assert result["human_answer_match"] is True


def test_ambiguous_human_answer_is_not_guessed():
    result = audit_record(_record(answer="$386,797,190 or $386.8 million"))
    assert result["human_answer_match"] is None
    assert "human_answer_requires_manual_review" in result["flags"]


def test_detects_program_step_mismatch_in_anomaly_examples():
    result = audit_record(
        _record(
            program="add(100000, 92500), add(#0, #0), add(#1, const_3), divide(#2, const_2)",
            exe_ans="192502",
            answer="91000",
            steps={
                "op": ["add1-1", "add1-2", "add1-3", "divide0-0"],
                "arg1": ["100000", "#0", "#1", "#2"],
                "arg2": ["92500", "#0", "const_3", "const_2"],
                "res": ["192500", "273000", "182000.0", "91000"],
            },
        )
    )
    assert "steps_program_disagreement" in result["flags"]
    assert "program_human_answer_disagreement" in result["flags"]
