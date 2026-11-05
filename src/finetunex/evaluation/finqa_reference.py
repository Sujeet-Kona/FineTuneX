"""Independent execution-semantics port for the upstream FinQA evaluator.

It reuses the validated project parser but separately implements arithmetic
and table-cell behavior from FinQA's public evaluator. It audits gold programs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from finetunex.evaluation.finqa_parser import parse_program


class ReferenceExecutionError(ValueError):
    """Raised when the reference-compatible evaluator rejects a program."""


def _reference_number(value: str) -> float:
    text = value.replace(",", "")
    try:
        number = float(text)
    except ValueError:
        if "%" in text:
            try:
                number = float(text.replace("%", "")) / 100.0
            except ValueError as exc:
                raise ReferenceExecutionError(
                    f"Invalid reference number: {value!r}"
                ) from exc
        elif "const" in text:
            text = text.replace("const_", "")
            if text == "m1":
                text = "-1"
            try:
                number = float(text)
            except ValueError as exc:
                raise ReferenceExecutionError(
                    f"Invalid reference constant: {value!r}"
                ) from exc
        else:
            raise ReferenceExecutionError(
                f"Invalid reference number: {value!r}"
            )

    if not math.isfinite(number):
        raise ReferenceExecutionError(
            f"Non-finite reference number: {value!r}"
        )
    return number


def _reference_argument(
    value: str,
    results: dict[int, float | str],
) -> float:
    if "#" in value:
        try:
            result = results[int(value.replace("#", ""))]
        except (ValueError, KeyError) as exc:
            raise ReferenceExecutionError(
                f"Invalid reference: {value!r}"
            ) from exc
        if isinstance(result, str):
            raise ReferenceExecutionError(
                "A text result cannot be used as a number"
            )
        return result
    return _reference_number(value)


def reference_execute_program(
    program: str,
    table: Sequence[Sequence[object]],
) -> float | str:
    """Execute using the upstream evaluator's operational semantics."""
    if not isinstance(table, Sequence) or isinstance(table, (str, bytes)):
        raise ReferenceExecutionError("Table must be a sequence")

    results: dict[int, float | str] = {}
    steps = parse_program(program)

    for index, step in enumerate(steps):
        operation = step.operation
        if operation.startswith("table_"):
            table_by_label = {row[0]: row[1:] for row in table if row}
            row_label: float | str = step.arg1
            if "#" in step.arg1:
                try:
                    row_label = results[int(step.arg1.replace("#", ""))]
                except (ValueError, KeyError) as exc:
                    raise ReferenceExecutionError(
                        f"Invalid table row reference: {step.arg1!r}"
                    ) from exc

            if row_label not in table_by_label:
                raise ReferenceExecutionError(
                    f"Table row not found: {row_label!r}"
                )

            numeric_cells = []
            for cell in table_by_label[row_label]:
                if not isinstance(cell, str):
                    raise ReferenceExecutionError(
                        "Table cells must be strings"
                    )
                cleaned = (
                    cell.replace("$", "")
                    .strip()
                    .split("(", maxsplit=1)[0]
                    .strip()
                )
                numeric_cells.append(_reference_number(cleaned))

            if not numeric_cells:
                raise ReferenceExecutionError(
                    "Table row contains no numeric cells"
                )
            if operation == "table_max":
                result = max(numeric_cells)
            elif operation == "table_min":
                result = min(numeric_cells)
            elif operation == "table_sum":
                result = sum(numeric_cells)
            elif operation == "table_average":
                result = sum(numeric_cells) / len(numeric_cells)
            else:
                raise ReferenceExecutionError(
                    f"Unsupported operation: {operation}"
                )
        else:
            left = _reference_argument(step.arg1, results)
            right = _reference_argument(step.arg2, results)
            if operation == "add":
                result = left + right
            elif operation == "subtract":
                result = left - right
            elif operation == "multiply":
                result = left * right
            elif operation == "divide":
                result = left / right
            elif operation == "exp":
                result = left**right
            elif operation == "greater":
                result = "yes" if left > right else "no"
            else:
                raise ReferenceExecutionError(
                    f"Unsupported operation: {operation}"
                )

        if isinstance(result, complex):
            raise ReferenceExecutionError(
                "Reference execution produced a complex result"
            )
        if isinstance(result, float) and not math.isfinite(result):
            raise ReferenceExecutionError(
                "Reference execution produced a non-finite result"
            )
        results[index] = result

    final = results[len(steps) - 1]
    return final if isinstance(final, str) else round(final, 5)
