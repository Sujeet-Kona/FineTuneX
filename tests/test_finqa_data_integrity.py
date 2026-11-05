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
            program=(
                "add(100000, 92500), add(#0, #0), add(#1, const_3), divide(#2, const_2)"
            ),
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


def test_text_result_is_compared_without_numeric_parsing():
    result = audit_record(
        _record(
            program="greater(10, 5)",
            exe_ans="yes",
            answer="yes",
            steps={
                "op": ["greater0-0"],
                "arg1": ["10"],
                "arg2": ["5"],
                "res": ["yes"],
            },
        )
    )
    assert result["exe_ans_match"] is True
    assert result["human_answer_match"] is True
    assert result["classification"] == "internally_consistent"


def test_detects_hig_sign_conflict():
    result = audit_record(
        _record(
            program="multiply(306.7, -8.99)",
            exe_ans="-2757.23",
            answer="2757.2",
            steps={
                "op": ["multiply1-1"],
                "arg1": ["306.7"],
                "arg2": ["-8.99"],
                "res": ["2757.2"],
            },
        )
    )
    assert "steps_program_disagreement" in result["flags"]
    assert "program_human_answer_disagreement" in result["flags"]


def test_detects_awk_program_steps_and_answer_conflict():
    result = audit_record(
        _record(
            program="multiply(11.4, const_1000000), multiply(3158226, #0)",
            exe_ans="3.60038e+13",
            answer="28%",
            steps={
                "op": ["multiply2-1", "multiply2-2"],
                "arg1": ["11.4", "3158226"],
                "arg2": ["const_1000000", "#0"],
                "res": ["11400000", "28%"],
            },
        )
    )
    assert "steps_annotation_ambiguous" in result["flags"]
    assert "steps_program_disagreement" not in result["flags"]
    assert "program_human_answer_disagreement" in result["flags"]


def test_source_minus_annotation_maps_to_subtract():
    result = audit_record(
        _record(
            program="subtract(10, 3)",
            exe_ans="7",
            answer="7",
            steps={
                "op": ["minus1-1"],
                "arg1": ["10"],
                "arg2": ["3"],
                "res": ["7"],
            },
        )
    )
    assert result["steps_audit"]["status"] == "consistent"
    assert result["classification"] == "internally_consistent"


def test_source_compare_larger_annotation_maps_to_greater():
    result = audit_record(
        _record(
            program="greater(10, 3)",
            exe_ans="yes",
            answer="yes",
            steps={
                "op": ["compare_larger1-1"],
                "arg1": ["10"],
                "arg2": ["3"],
                "res": ["yes"],
            },
        )
    )
    assert result["steps_audit"]["status"] == "consistent"
    assert result["classification"] == "internally_consistent"


def test_step_annotation_rounding_at_recorded_decimal_precision():
    result = audit_record(
        _record(
            program="add(75.0, 72.7), divide(#0, 20)",
            exe_ans="7.385",
            answer="7.4",
            steps={
                "op": ["add1-1", "divide1-2"],
                "arg1": ["75.0", "#0"],
                "arg2": ["72.7", "20"],
                "res": ["147.7", "7.4"],
            },
        )
    )
    assert result["steps_audit"]["status"] == "consistent"


def test_percentage_step_annotation_rounding_at_recorded_precision():
    result = audit_record(
        _record(
            program="divide(1786.2, 2807.0)",
            exe_ans="0.63634",
            answer="63.6%",
            steps={
                "op": ["divide1-1"],
                "arg1": ["1786.2"],
                "arg2": ["2807.0"],
                "res": ["63.6%"],
            },
        )
    )
    assert result["steps_audit"]["status"] == "consistent"


def test_parse_error_is_classified_without_crashing():
    result = audit_record(_record(program="not-a-program"))
    assert result["program_status"] == "parse_error"
    assert result["classification"] == "requires_manual_review"
    assert result["steps_audit"]["status"] == "not_checked"


def test_audit_record_retains_raw_and_parsed_step_diagnostics():
    result = audit_record(_record())
    assert result["raw_steps"] == _record()["steps"]
    assert result["parsed_steps"] == [
        {"operation": "add", "arg1": "100", "arg2": "200"},
        {"operation": "divide", "arg1": "#0", "arg2": "2"},
    ]


def test_human_answer_is_compared_at_its_displayed_precision():
    result = audit_record(
        _record(
            program="add(75.0, 72.7), divide(#0, 20)",
            exe_ans="7.385",
            answer="7.4",
            steps={
                "op": ["add1-1", "divide1-2"],
                "arg1": ["75.0", "#0"],
                "arg2": ["72.7", "20"],
                "res": ["147.7", "7.4"],
            },
        )
    )
    assert result["human_answer_match"] is True


def test_human_percentage_answer_is_compared_in_percentage_units():
    result = audit_record(
        _record(
            program="divide(1786.2, 2807.0)",
            exe_ans="0.63634",
            answer="63.6%",
            steps={
                "op": ["divide1-1"],
                "arg1": ["1786.2"],
                "arg2": ["2807.0"],
                "res": ["63.6%"],
            },
        )
    )
    assert result["human_answer_match"] is True


def test_magnitude_units_are_not_automatically_rescaled():
    result = audit_record(_record(answer="150 million"))
    assert result["human_answer_match"] is None
    assert result["human_answer_parse_status"] == "answer_requires_unit_context"


def test_exe_ans_is_not_rounded_before_official_comparison():
    result = audit_record(
        _record(
            program="divide(1, 3)",
            exe_ans="0.333333",
            answer="0.3333",
            steps={
                "op": ["divide1-1"],
                "arg1": ["1"],
                "arg2": ["3"],
                "res": ["0.3333"],
            },
        )
    )
    assert result["exe_ans_match"] is False
    assert "program_exe_ans_disagreement" in result["flags"]


def test_program_re_matches_flat_program_expression():
    result = audit_record(_record(program_re="divide(add(100, 200), 2)"))
    assert result["program_re_audit"]["status"] == "consistent"


def test_program_re_disagreement_is_reported():
    result = audit_record(_record(program_re="divide(add(100, 200), 3)"))
    assert result["program_re_audit"]["status"] == "mismatch"
    assert "program_re_disagreement" in result["flags"]


def test_malformed_program_re_is_reported_for_manual_review():
    result = audit_record(_record(program_re="divide(add(100, 200), 2"))
    assert result["program_re_audit"]["status"] == "invalid"
    assert "program_re_invalid" in result["flags"]


def test_program_re_parser_handles_thousands_separators():
    result = audit_record(
        _record(
            program="add(1,000, 25)",
            program_re="add(1,000, 25)",
        )
    )
    assert result["program_re_audit"]["status"] == "consistent"


def test_program_re_parser_handles_table_operations():
    result = audit_record(
        _record(
            program="table_sum(revenue, none)",
            program_re="table_sum(revenue, none)",
            answer="30",
            exe_ans="30",
            table=[["revenue", "10", "20"]],
            steps={
                "op": ["table_sum1-1"],
                "arg1": ["revenue"],
                "arg2": ["none"],
                "res": ["30"],
            },
        )
    )
    assert result["program_re_audit"]["status"] == "consistent"


def test_table_average_step_alias_maps_to_table_operation():
    result = audit_record(
        _record(
            program="table_average(revenue, none)",
            table=[["revenue", "10", "20"]],
            exe_ans="15",
            answer="15",
            steps={
                "op": ["average1-1"],
                "arg1": ["revenue"],
                "arg2": ["none"],
                "res": ["15"],
            },
        )
    )
    assert result["steps_audit"]["status"] == "consistent"
    assert "steps_program_disagreement" not in result["flags"]


def test_integer_step_annotation_accepts_explicit_integer_rounding():
    result = audit_record(
        _record(
            program="multiply(607, 18.13)",
            exe_ans="11004.91",
            answer="11004.91",
            steps={
                "op": ["multiply1-1"],
                "arg1": ["607"],
                "arg2": ["18.13"],
                "res": ["11005"],
            },
        )
    )
    assert result["steps_audit"]["status"] == "consistent"


def test_percentage_step_annotation_is_ambiguous_when_scale_does_not_match():
    result = audit_record(
        _record(
            program="divide(100, 100)",
            exe_ans="1",
            answer="1",
            steps={
                "op": ["divide1-1"],
                "arg1": ["100"],
                "arg2": ["100"],
                "res": ["1%"],
            },
        )
    )
    assert result["steps_audit"]["status"] == "ambiguous"
    assert "steps_annotation_ambiguous" in result["flags"]
    assert "steps_program_disagreement" not in result["flags"]


def test_percentage_step_annotation_matches_ratio_to_percent_convention():
    result = audit_record(
        _record(
            program="divide(1786.2, 2807.0)",
            exe_ans="0.63634",
            answer="63.6%",
            steps={
                "op": ["divide1-1"],
                "arg1": ["1786.2"],
                "arg2": ["2807.0"],
                "res": ["63.6%"],
            },
        )
    )
    assert result["steps_audit"]["status"] == "consistent"


def test_percentage_step_sign_conflict_is_ambiguous_not_confirmed_mismatch():
    result = audit_record(
        _record(
            program="subtract(440.9, 442.3), divide(#0, 442.3)",
            exe_ans="-0.00317",
            answer="0.3%",
            steps={
                "op": ["minus2-1", "divide2-2"],
                "arg1": ["440.9", "#0"],
                "arg2": ["442.3", "442.3"],
                "res": ["-1.4", "0.3%"],
            },
        )
    )
    assert result["steps_audit"]["status"] == "ambiguous"
    assert "steps_annotation_ambiguous" in result["flags"]


def test_plain_numeric_step_disagreement_remains_confirmed():
    result = audit_record(
        _record(
            program="divide(2.5, 1.6)",
            exe_ans="1.5625",
            answer="1.5625",
            steps={
                "op": ["divide1-1"],
                "arg1": ["2.5"],
                "arg2": ["1.6"],
                "res": ["0.9"],
            },
        )
    )
    assert result["steps_audit"]["status"] == "mismatch"
    assert "steps_program_disagreement" in result["flags"]
