"""Tests for the canonical FinQA answer comparison rule."""

import pytest

from finetunex.evaluation.answer_comparison import (
    ANSWER_COMPARISON_RULE,
    answers_match,
    format_significant,
)
from finetunex.evaluation.finqa_executor import execute_program


def test_rule_label_names_the_precision_actually_used():
    assert ANSWER_COMPARISON_RULE == "six_significant_digits_of_exe_ans@6"
    assert format_significant(41932.20339) == "41932.2"
    assert format_significant(26138609.0) == "2.61386e+07"


@pytest.mark.parametrize(
    ("predicted", "expected"),
    [
        (127.4, "127.4"),
        (300.0, "300"),
        (0.5, "0.5"),
        (995.0, "995"),
        (150.0, "150.0"),
    ],
)
def test_exact_numeric_matches(predicted, expected):
    assert answers_match(predicted, expected) is True


@pytest.mark.parametrize(
    ("predicted", "expected"),
    [
        # Scientific-notation rendering of the same value (316 real records).
        (26138609.0, "2.61386e+07"),
        (37014734589.92, "3.70147e+10"),
        (2e-05, "2e-05"),
        # A trailing ".0" is dropped by the dataset's renderer.
        (150.0, "150"),
        (300.0, "300"),
    ],
)
def test_equivalent_numeric_formats_match(predicted, expected):
    assert answers_match(predicted, expected) is True


@pytest.mark.parametrize(
    ("predicted", "expected"),
    [
        (41932.20339, "41932.2"),
        (96.55172, "96.5517"),
        (56.57146, "56.5715"),
        (93670.33333, "93670.3"),
        (204.11765, "204.118"),
        (331.2253, "331.225"),
        (11.33333, "11.3333"),
        (154.54545, "154.545"),
        (192501.5, "192502"),
    ],
)
def test_display_precision_differences_are_not_false_negatives(predicted, expected):
    assert answers_match(predicted, expected) is True


@pytest.mark.parametrize(
    ("predicted", "expected", "matches"),
    [
        (-0.306, "-0.306", True),
        (-2757.23, "-2757.23", True),
        (-0.00317, "-0.00317", True),
        (-0.03219, "-3.219e-02", True),
        (-0.306, "0.306", False),
        (0.306, "-0.306", False),
        (192501.5, "-192502", False),
    ],
)
def test_negative_values(predicted, expected, matches):
    assert answers_match(predicted, expected) is matches


def test_negative_and_percentage_program_arguments():
    # const_m1 is the dataset's negative-one constant.
    assert execute_program("multiply(3176, const_m1), divide(#0, 10379)", []) == -0.306
    # A percentage argument is a ratio, so 23.6% divides as 0.236.
    assert execute_program("divide(9896, 23.6%)", []) == 41932.20339


def test_accounting_negative_table_cells_are_parsed():
    table = [
        ["Line", "2009", "2008"],
        ["Operating income", "$ -23158 ( 23158 )", "$ 9457"],
    ]
    assert execute_program("table_min(Operating income, none)", table) == -23158.0
    assert answers_match(
        execute_program("table_min(Operating income, none)", table), "-23158"
    )


@pytest.mark.parametrize(
    ("predicted", "expected", "matches"),
    [
        # Genuinely different financial answers must stay incorrect.
        (129.4, "127.4", False),
        (0.83803, "0.935", False),
        (200.0, "0.935", False),
        (688.0, "995", False),
        (0.33333, "0.333333", False),
        (0.33331, "0.333333", False),
        (0.33333, "0.33333", True),
    ],
)
def test_incorrect_answers_stay_incorrect(predicted, expected, matches):
    assert answers_match(predicted, expected) is matches


@pytest.mark.parametrize(
    ("predicted", "expected", "matches"),
    [
        ("yes", "yes", True),
        ("no", "no", True),
        ("yes", "no", False),
        ("no", "yes", False),
        ("YES", "yes", True),
    ],
)
def test_greater_comparison_answers(predicted, expected, matches):
    assert answers_match(predicted, expected) is matches


@pytest.mark.parametrize(
    ("predicted", "expected"),
    [
        (127.4, ""),
        (127.4, "not a number"),
        (127.4, None),
        (127.4, "127,4"),
        ("", "127.4"),
        ("yes", ""),
    ],
)
def test_invalid_and_empty_expected_answers_never_match(predicted, expected):
    assert answers_match(predicted, expected) is False


@pytest.mark.parametrize(
    ("program", "exe_ans"),
    [
        ("divide(9896, 23.6%)", "41932.2"),
        ("subtract(34.8, 1.2), divide(#0, 34.8), multiply(#1, const_100)", "96.5517"),
        ("multiply(1327657, 42.61), divide(#0, const_1000000)", "56.5715"),
        ("subtract(426390481, 400251872)", "2.61386e+07"),
        ("divide(34.7, 17), multiply(#0, const_100)", "204.118"),
        ("divide(838, 2.53)", "331.225"),
        ("add(9, 10), add(#0, 15), divide(#1, const_3)", "11.3333"),
        ("divide(170, 1.10)", "154.545"),
        ("multiply(3176, const_m1), divide(#0, 10379)", "-0.306"),
        (
            "add(100000, 92500), add(#0, #0), add(#1, const_3), divide(#2, const_2)",
            "192502",
        ),
    ],
)
def test_real_gold_programs_reproduce_their_stored_exe_ans(program, exe_ans):
    """Regression: these FinQA records were scored wrong under the old rule."""
    assert answers_match(execute_program(program, []), exe_ans) is True


def test_percentage_program_answer_is_a_ratio_not_percentage_points():
    # divide(60, 243), multiply(#0, const_100) -> 24.6914 (percentage points)
    assert answers_match(execute_program("divide(60, 243)", []), "0.24691") is True
    assert answers_match(execute_program("divide(60, 243)", []), "24.6914") is False
