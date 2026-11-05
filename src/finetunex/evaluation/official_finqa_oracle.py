"""Test oracle extracted from the official FinQA execution evaluator.

Source: https://github.com/czyssrs/FinQA/blob/dfc5b72c01ee17c442d28d5201b82a1f4e95d5af/code/evaluate/evaluate.py
Upstream file blob: 4829a4ca58a5725fb3f7a46b3b23b7f7e848a512

This intentionally preserves the upstream execution path, including its
five-decimal final-result rounding and table-reference behavior. Do not use
this module for production inference; it exists only for differential tests.
"""

from __future__ import annotations

ALL_OPERATIONS = (
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
)


def _str_to_num(text: str) -> float | str:
    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        if "%" in text:
            text = text.replace("%", "")
            try:
                return float(text) / 100.0
            except ValueError:
                return "n/a"
        if "const" in text:
            text = text.replace("const_", "")
            if text == "m1":
                text = "-1"
            return float(text)
        return "n/a"


def _process_row(row: list[str]) -> list[float] | str:
    values: list[float] = []
    for cell in row:
        cleaned = cell.replace("$", "").strip()
        cleaned = cleaned.split("(")[0].strip()
        number = _str_to_num(cleaned)
        if number == "n/a":
            return "n/a"
        values.append(number)
    return values


def _program_tokenization(program: str) -> list[str]:
    tokens: list[str] = []
    for operation in program.split(", "):
        current = ""
        for character in operation:
            if character == ")":
                if current:
                    tokens.append(current)
                    current = ""
            current += character
            if character in ("(", ")"):
                tokens.append(current)
                current = ""
        if current:
            tokens.append(current)
    tokens.append("EOF")
    return tokens


def official_eval_program(
    program: str, table: list[list[str]]
) -> tuple[int, float | str]:
    """Return (invalid_flag, result), matching upstream eval_program()."""
    invalid_flag = 0
    result: float | str = "n/a"
    try:
        tokens = _program_tokenization(program)[:-1]
        for index, token in enumerate(tokens):
            if index % 4 == 0 and token.strip("(") not in ALL_OPERATIONS:
                return 1, "n/a"
            if (index + 1) % 4 == 0 and token != ")":
                return 1, "n/a"

        tokenized_program = "|".join(tokens)
        steps = tokenized_program.split(")")[:-1]
        results: dict[int, float | str] = {}

        for index, step in enumerate(steps):
            step = step.strip()
            if len(step.split("(")) > 2:
                invalid_flag = 1
                break

            operation = step.split("(")[0].strip("|").strip()
            arguments = step.split("(")[1].strip("|").strip()
            arg1, arg2 = (part.strip() for part in arguments.split("|")[:2])

            if operation in {
                "add",
                "subtract",
                "multiply",
                "divide",
                "exp",
                "greater",
            }:
                if "#" in arg1:
                    left = results[int(arg1.replace("#", ""))]
                else:
                    left = _str_to_num(arg1)
                    if left == "n/a":
                        invalid_flag = 1
                        break

                if "#" in arg2:
                    right = results[int(arg2.replace("#", ""))]
                else:
                    right = _str_to_num(arg2)
                    if right == "n/a":
                        invalid_flag = 1
                        break

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
                else:
                    result = "yes" if left > right else "no"
                results[index] = result

            elif "table" in operation:
                table_dict = {row[0]: row[1:] for row in table}
                if "#" in arg1:
                    arg1 = results[int(arg1.replace("#", ""))]
                else:
                    if arg1 not in table_dict:
                        invalid_flag = 1
                        break
                    numeric_row = _process_row(table_dict[arg1])

                # Keep the upstream control flow: for a referenced row,
                # numeric_row is unassigned or retains a previous table row.
                if numeric_row == "n/a":
                    invalid_flag = 1
                    break

                if operation == "table_max":
                    result = max(numeric_row)
                elif operation == "table_min":
                    result = min(numeric_row)
                elif operation == "table_sum":
                    result = sum(numeric_row)
                elif operation == "table_average":
                    result = sum(numeric_row) / len(numeric_row)
                results[index] = result

        if result not in ("yes", "no", "n/a"):
            result = round(result, 5)
    except Exception:
        invalid_flag = 1

    return invalid_flag, result
