"""Tests for the FinQA deterministic executor."""

import pytest

from finetunex.evaluation.finqa_executor import (
    FinQAExecutionError,
    execute_program,
    parse_number,
)
from finetunex.evaluation.finqa_reference import reference_execute_program


def test_parse_integer_and_decimal():
    assert parse_number("1,234.56") == 1234.56


def test_parse_percentage():
    assert parse_number("23.6%") == pytest.approx(0.236)


def test_parse_negative_constant():
    assert parse_number("const_m1") == -1.0


@pytest.mark.parametrize(
    ("program", "expected"),
    [
        ("add(10, 20)", 30),
        ("subtract(10, 20)", -10),
        ("multiply(10, 20)", 200),
        ("divide(10, 4)", 2.5),
        ("exp(2, 3)", 8),
        ("greater(10, 5)", "yes"),
        ("greater(5, 10)", "no"),
    ],
)
def test_arithmetic_operations(program, expected):
    assert execute_program(program, []) == expected


def test_intermediate_reference():
    program = "subtract(125, 100), divide(#0, 5)"
    assert execute_program(program, []) == 5


def test_multiple_intermediate_references():
    program = "add(10, 20), multiply(#0, 3), subtract(#1, 5)"
    assert execute_program(program, []) == 85


def test_percentage_in_arithmetic():
    assert execute_program("divide(100, 25%)", []) == 400


def test_rounds_final_numeric_result_to_five_decimals():
    assert execute_program("divide(1, 3)", []) == 0.33333


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        ("table_max", 30),
        ("table_min", 10),
        ("table_sum", 60),
        ("table_average", 20),
    ],
)


def test_table_operations(operation, expected):
    table = [["revenue", "$ 10", "$ 20", "$ 30"]]
    assert execute_program(f"{operation}(revenue, none)", table) == expected


def test_table_percentages_and_parenthetical_annotations():
    table = [["margin", "10% ( 10 % )", "20%", "30%"]]
    assert execute_program("table_average(margin, none)", table) == 0.2


def test_table_currency_and_thousands_separators():
    table = [["assets", "$ 1,000", "$ 2,000"]]
    assert execute_program("table_sum(assets, none)", table) == 3000


def test_duplicate_table_labels_use_last_matching_row():
    table = [
        ["first quarter", "$ 62.42", "$ 54.10"],
        ["first quarter", "$ 38.02", "$ 34.19"],
    ]
    assert execute_program("table_max(first quarter, none)", table) == 38.02


def test_table_operation_can_be_followed_by_arithmetic():
    table = [["revenue", "10", "20", "30"]]
    assert execute_program("table_sum(revenue, none), divide(#0, 3)", table) == 20


@pytest.mark.parametrize(
    "program",
    [
        "divide(10, 0)",
        "add(#0, 1)",
        "add(#1, 1)",
        "add(#-1, 1)",
        "add(#0abc, 1)",
        "add(yes, 1), add(#0, 1)",
        "table_sum(missing, none)",
    ],
)


def test_invalid_execution_raises_error(program):
    table = [["revenue", "10", "20"]]
    with pytest.raises(FinQAExecutionError):
        execute_program(program, table)


def test_table_operation_ignores_second_argument_like_official_evaluator():
    table = [["revenue", "10", "20"]]
    assert execute_program("table_sum(revenue, invalid)", table) == 30


def test_rejects_invalid_table_cell():
    table = [["revenue", "10", "not-a-number"]]
    with pytest.raises(FinQAExecutionError):
        execute_program("table_sum(revenue, none)", table)


def test_rejects_empty_table_row():
    table = [["revenue"]]
    with pytest.raises(FinQAExecutionError):
        execute_program("table_sum(revenue, none)", table)


def test_rejects_non_finite_number():
    with pytest.raises(FinQAExecutionError):
        execute_program("multiply(1e308, 1e308)", [])


def test_rejects_non_sequence_table():
    with pytest.raises(FinQAExecutionError):
        execute_program("add(1, 2)", "not-a-table")


def test_rejects_complex_exponentiation_result():
    with pytest.raises(FinQAExecutionError, match="complex result"):
        execute_program("exp(-1, 0.5)", [])


def test_rejects_exponentiation_overflow():
    with pytest.raises(FinQAExecutionError):
        execute_program("exp(10, 10000)", [])


def test_reference_executor_matches_custom_arithmetic():
    program = "subtract(125, 100), divide(#0, 5)"
    assert reference_execute_program(program, []) == execute_program(program, [])


def test_reference_executor_matches_custom_table_aggregation():
    table = [["Revenue", "$ 100", "$ 200", "$ 300"]]
    program = "table_sum(Revenue, none)"
    assert reference_execute_program(program, table) == execute_program(program, table)

def test_reference_executor_resolves_table_row_references():
    table = [["yes", "10", "20"]]
    program = "greater(2, 1), table_sum(#0, none)"

    assert reference_execute_program(program, table) == 30
    assert execute_program(program, table) == 30
