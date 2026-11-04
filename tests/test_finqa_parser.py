"""Tests for the FinQA program parser."""

import pytest

from finetunex.evaluation.finqa_parser import (
    FinQAProgramParseError,
    ProgramStep,
    parse_program,
)


def test_parse_single_arithmetic_step():
    assert parse_program("add(75.0, 72.7)") == (
        ProgramStep("add", "75.0", "72.7"),
    )


def test_parse_multiple_steps_and_references():
    assert parse_program(
        "subtract(125, 100), divide(#0, 100)"
    ) == (
        ProgramStep("subtract", "125", "100"),
        ProgramStep("divide", "#0", "100"),
    )


def test_parse_percentage_argument():
    assert parse_program("divide(9896, 23.6%)") == (
        ProgramStep("divide", "9896", "23.6%"),
    )


def test_parse_negative_constant():
    assert parse_program("multiply(10, const_m1)") == (
        ProgramStep("multiply", "10", "const_m1"),
    )


@pytest.mark.parametrize(
    ("operation", "row"),
    [
        ("table_average", "2016"),
        ("table_max", "cumulative foreign currency translation"),
        ("table_min", "expected volatility"),
        ("table_sum", "cash flows on retained interests"),
    ],
)
def test_parse_table_operations(operation, row):
    assert parse_program(f"{operation}({row}, none)") == (
        ProgramStep(operation, row, "none"),
    )


def test_parse_whitespace():
    assert parse_program(
        "  add(10, 20) ,  multiply(#0, 3)  "
    ) == (
        ProgramStep("add", "10", "20"),
        ProgramStep("multiply", "#0", "3"),
    )


@pytest.mark.parametrize(
    "program",
    [
        "",
        "   ",
        "unknown(1, 2)",
        "add(1)",
        "add(, 2)",
        "add(1, )",
        "add(1, 2",
        "add(1, 2),",
        "add(1, 2) multiply(3, 4)",
        "add(1, 2), unknown(3, 4)",
        "add(1, 2),, multiply(3, 4)",
    ],
)
def test_reject_invalid_programs(program):
    with pytest.raises(FinQAProgramParseError):
        parse_program(program)


def test_reject_non_string_input():
    with pytest.raises(FinQAProgramParseError):
        parse_program(None)
