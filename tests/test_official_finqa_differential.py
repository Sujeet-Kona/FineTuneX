"""Tests for the vendored official FinQA evaluator oracle."""

import pytest

from finetunex.evaluation.finqa_executor import execute_program
from finetunex.evaluation.official_finqa_oracle import official_eval_program


@pytest.mark.parametrize(
    ("program", "table"),
    [
        ("add(1,000, 25)", []),
        ("subtract(125, 100), divide(#0, 5)", []),
        ("multiply(10, 20)", []),
        ("divide(1, 3)", []),
        ("exp(2, 3)", []),
        ("greater(10, 5)", []),
        ("table_max(revenue, ignored)", [["revenue", "$ 10", "$ 20", "$ 30"]]),
        ("table_min(revenue, ignored)", [["revenue", "$ 10", "$ 20", "$ 30"]]),
        ("table_sum(revenue, ignored)", [["revenue", "$ 10", "$ 20", "$ 30"]]),
        ("table_average(revenue, ignored)", [["revenue", "$ 10", "$ 20", "$ 30"]]),
        ("divide(100, 25%)", []),
        ("multiply(10, const_m1)", []),
        (
            "table_average(margin, ignored)",
            [["margin", "10% ( 10 % )", "20%", "30%"]],
        ),
        ("divide(1, 3), multiply(#0, 3)", []),
    ],
)
def test_custom_executor_matches_official_oracle(program, table):
    invalid, expected = official_eval_program(program, table)
    assert invalid == 0
    assert execute_program(program, table) == expected


def test_official_oracle_marks_missing_table_row_invalid():
    invalid, _ = official_eval_program(
        "table_sum(missing, ignored)",
        [["revenue", "10", "20"]],
    )
    assert invalid == 1
