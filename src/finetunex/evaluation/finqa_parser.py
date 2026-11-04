"""Parser for the FinQA arithmetic-program language."""

from __future__ import annotations

import re
from dataclasses import dataclass

SUPPORTED_OPERATIONS = frozenset(
    {
        "add",
        "subtract",
        "multiply",
        "divide",
        "exp",
        "greater",
        "table_max",
        "table_min",
        "table_sum",
        "table_average",
    }
)

_OPERATION_START = re.compile(r"([a-z_]+)\s*\(")


class FinQAProgramParseError(ValueError):
    """Raised when a FinQA program does not follow the expected syntax."""


@dataclass(frozen=True, slots=True)
class ProgramStep:
    """One operation and its two raw arguments."""

    operation: str
    arg1: str
    arg2: str


def parse_program(program: str) -> tuple[ProgramStep, ...]:
    """Parse a comma-separated sequence of two-argument FinQA operations.

    Examples:
        add(10, 20)
        subtract(100, 25), divide(#0, 5)
        table_sum(revenue, none)

    Arguments remain strings so parsing stays separate from execution.
    """
    if not isinstance(program, str):
        raise FinQAProgramParseError("Program must be a string.")

    source = program.strip()
    if not source:
        raise FinQAProgramParseError("Program cannot be empty.")

    steps: list[ProgramStep] = []
    position = 0

    while position < len(source):
        while position < len(source) and source[position].isspace():
            position += 1

        match = _OPERATION_START.match(source, position)
        if match is None:
            raise FinQAProgramParseError(
                f"Expected an operation at position {position}."
            )

        operation = match.group(1)
        if operation not in SUPPORTED_OPERATIONS:
            raise FinQAProgramParseError(
                f"Unsupported operation: {operation!r}."
            )

        arguments_start = match.end()
        separator = source.find(",", arguments_start)

        if separator == -1:
            raise FinQAProgramParseError(
                f"Missing argument separator in {operation!r}."
            )

        closing = source.find(")", separator + 1)
        if closing == -1:
            raise FinQAProgramParseError(
                f"Missing closing parenthesis in {operation!r}."
            )

        arg1 = source[arguments_start:separator].strip()
        arg2 = source[separator + 1 : closing].strip()

        if not arg1 or not arg2:
            raise FinQAProgramParseError(
                f"Both arguments are required for {operation!r}."
            )

        if "(" in arg1 or "(" in arg2 or ")" in arg2:
            raise FinQAProgramParseError(
                f"Nested or malformed arguments in {operation!r}."
            )

        steps.append(
            ProgramStep(
                operation=operation,
                arg1=arg1,
                arg2=arg2,
            )
        )

        position = closing + 1

        while position < len(source) and source[position].isspace():
            position += 1

        if position == len(source):
            break

        if source[position] != ",":
            raise FinQAProgramParseError(
                f"Expected a comma between steps at position {position}."
            )

        position += 1

        if position == len(source):
            raise FinQAProgramParseError(
                "Program cannot end with a step separator."
            )

    if not steps:
        raise FinQAProgramParseError("Program must contain at least one step.")

    return tuple(steps)
