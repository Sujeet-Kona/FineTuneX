"""Safe deterministic execution of FinQA arithmetic programs."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from finetunex.evaluation.finqa_parser import ProgramStep, parse_program


class FinQAExecutionError(ValueError):
    """Raised when a FinQA program cannot be executed safely."""


_CONSTANT_PATTERN = re.compile(r"const_(m?\d+)\Z")
_REFERENCE_PATTERN = re.compile(r"#(\d+)\Z")


def parse_number(value: str) -> float:
    """Parse FinQA numeric literals, percentages and named constants."""
    if not isinstance(value, str):
        raise FinQAExecutionError("Numeric values must be strings.")

    normalized = value.strip().replace(",", "")

    if not normalized:
        raise FinQAExecutionError("Numeric value cannot be empty.")

    constant_match = _CONSTANT_PATTERN.fullmatch(normalized)
    if constant_match:
        constant = constant_match.group(1)
        normalized = f"-{constant[1:]}" if constant.startswith("m") else constant

    is_percentage = normalized.endswith("%")
    if is_percentage:
        normalized = normalized[:-1].strip()

    try:
        number = float(normalized)
    except ValueError as exc:
        raise FinQAExecutionError(f"Invalid numeric value: {value!r}.") from exc

    if not math.isfinite(number):
        raise FinQAExecutionError(f"Non-finite numeric value: {value!r}.")

    if is_percentage:
        number /= 100.0

    return number


def _resolve_argument(
    argument: str,
    results: Sequence[float | str],
    current_step: int,
) -> float:
    """Resolve a numeric literal or a reference to an earlier step."""
    reference_match = _REFERENCE_PATTERN.fullmatch(argument)

    if reference_match:
        reference_index = int(reference_match.group(1))

        if reference_index >= current_step:
            raise FinQAExecutionError(
                f"Reference #{reference_index} must point to an earlier step."
            )

        if reference_index >= len(results):
            raise FinQAExecutionError(f"Reference #{reference_index} does not exist.")

        result = results[reference_index]

        if isinstance(result, str):
            raise FinQAExecutionError(f"Reference #{reference_index} is not numeric.")

        return result

    if "#" in argument:
        raise FinQAExecutionError(f"Malformed result reference: {argument!r}.")

    return parse_number(argument)


def _parse_table_row(row: Sequence[object]) -> list[float]:
    """Parse all numeric cells in a selected table row."""
    if len(row) < 2:
        raise FinQAExecutionError(
            "Table aggregation requires at least one numeric cell."
        )

    values: list[float] = []

    for cell in row[1:]:
        if not isinstance(cell, str):
            raise FinQAExecutionError(f"Table cell must be a string: {cell!r}.")

        cleaned = cell.replace("$", "").strip()
        cleaned = cleaned.split("(", maxsplit=1)[0].strip()

        values.append(parse_number(cleaned))

    return values


def _execute_step(
    step: ProgramStep,
    table: Sequence[Sequence[object]],
    results: Sequence[float | str],
    step_index: int,
) -> float | str:
    """Execute one validated operation."""
    operation = step.operation

    if operation.startswith("table_"):
        # The official FinQA evaluator ignores the second table-operation argument.
        if "#" in step.arg1:
            raise FinQAExecutionError(
                "Table aggregation requires a row-label argument."
            )

        selected_row: Sequence[object] | None = None

        # Match the original FinQA evaluator: the last duplicate label wins.
        for row in table:
            if row and str(row[0]).strip() == step.arg1:
                selected_row = row

        if selected_row is None:
            raise FinQAExecutionError(f"Table row not found: {step.arg1!r}.")

        values = _parse_table_row(selected_row)

        if operation == "table_max":
            return max(values)
        if operation == "table_min":
            return min(values)
        if operation == "table_sum":
            return sum(values)
        if operation == "table_average":
            return sum(values) / len(values)

        raise FinQAExecutionError(f"Unsupported table operation: {operation!r}.")

    left = _resolve_argument(step.arg1, results, step_index)
    right = _resolve_argument(step.arg2, results, step_index)

    if operation == "add":
        return left + right
    if operation == "subtract":
        return left - right
    if operation == "multiply":
        return left * right
    if operation == "divide":
        if right == 0:
            raise FinQAExecutionError("Division by zero.")
        return left / right
    if operation == "exp":
        try:
            result = left**right
        except (ArithmeticError, ValueError) as exc:
            raise FinQAExecutionError(
                "Exponentiation produced an invalid result."
            ) from exc

        if isinstance(result, complex):
            raise FinQAExecutionError("Exponentiation produced a complex result.")

        return result
    if operation == "greater":
        return "yes" if left > right else "no"

    raise FinQAExecutionError(f"Unsupported operation: {operation!r}.")


def execute_program(
    program: str,
    table: Sequence[Sequence[object]],
) -> float | str:
    """Execute a FinQA program and return its final result.

    Numeric results are rounded to five decimal places, matching
    the official FinQA evaluator.
    """
    if not isinstance(table, Sequence) or isinstance(table, (str, bytes)):
        raise FinQAExecutionError("Table must be a sequence of rows.")

    steps = parse_program(program)
    results: list[float | str] = []

    for step_index, step in enumerate(steps):
        try:
            result = _execute_step(step, table, results, step_index)
        except FinQAExecutionError:
            raise
        except (ArithmeticError, TypeError, ValueError) as exc:
            raise FinQAExecutionError(
                f"Execution failed at step {step_index}: {exc}."
            ) from exc

        if isinstance(result, float) and not math.isfinite(result):
            raise FinQAExecutionError(
                f"Step {step_index} produced a non-finite result."
            )

        results.append(result)

    final_result = results[-1]

    if isinstance(final_result, str):
        return final_result

    return round(final_result, 5)
