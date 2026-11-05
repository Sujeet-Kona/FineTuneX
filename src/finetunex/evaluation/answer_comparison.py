"""Canonical comparison between a computed FinQA answer and ``exe_ans``.

``bevaya/FinQA`` stores ``exe_ans`` as a string rendered at six significant
digits, not as ``repr(round(value, 5))``. Comparing the executor's five-decimal
result directly against the parsed ``exe_ans`` therefore marks a *correct*
program as wrong whenever the two renderings differ, e.g. ``41932.20339``
against ``"41932.2"`` or ``26138609.0`` against ``"2.61386e+07"``.

Normalising both sides to the dataset's own six-significant-digit rendering
removes that false-negative class while keeping genuinely different financial
answers distinct. Across all 8,281 pinned records every gold program reproduces
``exe_ans`` under this rule.
"""

from __future__ import annotations

from finetunex.evaluation.finqa_executor import FinQAExecutionError, parse_number

SIGNIFICANT_DIGITS = 6

ANSWER_COMPARISON_RULE = f"six_significant_digits_of_exe_ans@{SIGNIFICANT_DIGITS}"


def format_significant(value: float) -> str:
    """Render a number the way the FinQA dataset renders ``exe_ans``."""
    return f"{value:.{SIGNIFICANT_DIGITS}g}"


def answers_match(predicted: float | str, expected: object) -> bool:
    """Return True when a computed answer equals the stored ``exe_ans``.

    ``predicted`` is the output of ``execute_program``: a float already rounded
    to five decimals, or a string such as ``"yes"`` for ``greater``.
    """
    if isinstance(predicted, str):
        return predicted.strip().casefold() == str(expected).strip().casefold()

    try:
        expected_number = parse_number(str(expected))
    except (FinQAExecutionError, TypeError, ValueError):
        return False

    return format_significant(predicted) == format_significant(expected_number)
