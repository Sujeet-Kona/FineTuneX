"""Non-destructive integrity audit for FinQA gold annotations.

The audit keeps source records untouched. It distinguishes executable-program
results, recorded intermediate steps, execution answers, and human answers.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from finetunex.evaluation.finqa_executor import (
    FinQAExecutionError,
    _execute_step,
    parse_number,
)
from finetunex.evaluation.finqa_parser import (
    FinQAProgramParseError,
    parse_program,
)

_STEP_OP = re.compile(r"^([a-z_]+)\d+-\d+$")
_STEP_OPERATION_ALIASES = {
    "minus": "subtract",
    "compare_larger": "greater",
}
_ANSWER = re.compile(
    r"^\s*[$€£]?\s*([-+]?(?:\d[\d,]*\.?\d*|\.\d+))"
    r"\s*(%|million|billion|thousand)?\s*$",
    re.IGNORECASE,
)


def _close(left: float, right: float, *, rel_tol: float = 1e-5) -> bool:
    return math.isclose(left, right, rel_tol=rel_tol, abs_tol=1e-5)


def _step_result_comparison(value: float | str, annotation: Any) -> dict[str, Any]:
    """Classify a step annotation without guessing percentage conventions."""
    if isinstance(value, str):
        matched = value.strip().casefold() == str(annotation).strip().casefold()
        return {"status": "match" if matched else "mismatch"}

    raw = str(annotation).strip().replace(",", "")
    raw = raw.replace("$", "").replace("€", "").replace("£", "").strip()
    is_percentage = raw.endswith("%")
    numeric_text = raw[:-1].strip() if is_percentage else raw

    try:
        parsed = parse_number(numeric_text)
    except FinQAExecutionError:
        return {"status": "unparseable"}

    decimal_places = (
        len(numeric_text.split(".", maxsplit=1)[1]) if "." in numeric_text else 0
    )
    if is_percentage:
        actual_percentage = value * 100
        matched = round(actual_percentage, decimal_places) == round(
            parsed, decimal_places
        )
        if matched:
            return {"status": "match", "convention": "ratio_to_percent"}
        return {
            "status": "ambiguous",
            "reason": "percentage_scale_or_sign_convention",
            "computed_ratio": value,
            "computed_percent": actual_percentage,
            "annotated_percent": parsed,
        }

    matched = round(value, decimal_places) == round(parsed, decimal_places)
    return {
        "status": "match" if matched else "mismatch",
        "computed": value,
        "annotated": parsed,
        "displayed_decimal_places": decimal_places,
    }


def _normalize_step_operation(annotation: Any, program_operation: str) -> str | None:
    if not isinstance(annotation, str):
        return None
    match = _STEP_OP.fullmatch(annotation)
    if match is None:
        return None

    operation = _STEP_OPERATION_ALIASES.get(match.group(1), match.group(1))
    if program_operation.startswith("table_"):
        table_alias = f"table_{operation}"
        if table_alias == program_operation:
            return table_alias
    return operation


def _parse_human_answer(value: Any) -> tuple[float | None, str]:
    """Parse only unambiguous single-number answers; never guess alternatives."""
    if not isinstance(value, str):
        return None, "answer_not_string"
    match = _ANSWER.fullmatch(value)
    if match is None:
        return None, "answer_not_single_numeric_value"
    number_text, unit = match.groups()
    try:
        number = parse_number(number_text.replace(",", ""))
    except FinQAExecutionError:
        return None, "answer_numeric_parse_error"
    unit = (unit or "").lower()
    if unit in {"thousand", "million", "billion"}:
        return None, "answer_requires_unit_context"
    if unit == "%":
        number /= 100.0
    return number, "parsed"


def _human_answer_matches(
    value: float, answer: str, *, percentage_context: bool = False
) -> bool:
    """Compare at displayed precision while allowing either explicit percent scale.

    FinQA programs may return a ratio (0.28) or percentage points (28.0), for
    example after an explicit multiply-by-100 step. A percent sign in the
    human answer therefore cannot determine the program's output scale alone.
    An unmarked answer is interpreted as percentage points only when the
    question or recorded final step explicitly establishes percentage context.
    """
    match = _ANSWER.fullmatch(answer.strip())
    if match is None:
        return False

    number_text, unit = match.groups()
    decimal_places = (
        len(number_text.replace(",", "").split(".", maxsplit=1)[1])
        if "." in number_text
        else 0
    )
    try:
        expected = parse_number(number_text.replace(",", ""))
    except FinQAExecutionError:
        return False

    is_percentage = (unit or "").lower() == "%"
    candidates = [value, value * 100] if is_percentage else [value]
    if percentage_context and not is_percentage:
        candidates.append(value * 100)

    return any(
        round(candidate, decimal_places) == round(expected, decimal_places)
        for candidate in candidates
    )


def _normalize_program_atom(
    value: str, operation: str, argument_index: int
) -> tuple[str, Any]:
    value = value.strip()
    if operation.startswith("table_") and argument_index == 0:
        return ("label", value)
    if value.startswith("#"):
        return ("reference", value)
    try:
        return ("number", parse_number(value))
    except FinQAExecutionError:
        return ("symbol", value)


def _parse_nested_program(program_re: str) -> tuple[Any, ...]:
    """Parse the fully nested program representation into a small AST."""
    position = 0

    def skip_whitespace() -> None:
        nonlocal position
        while position < len(program_re) and program_re[position].isspace():
            position += 1

    def parse_argument(operation: str, argument_index: int) -> Any:
        nonlocal position
        skip_whitespace()
        if re.match(r"[a-z_]+\s*\(", program_re[position:]):
            return parse_expression()

        start = position
        while position < len(program_re):
            if program_re[position] == ")":
                break
            if (
                program_re[position] == ","
                and position + 1 < len(program_re)
                and program_re[position + 1].isspace()
            ):
                break
            position += 1

        atom = program_re[start:position].strip()
        if not atom:
            raise ValueError("Nested program contains an empty argument.")
        return _normalize_program_atom(atom, operation, argument_index)

    def parse_expression() -> tuple[Any, ...]:
        nonlocal position
        skip_whitespace()
        match = re.match(r"([a-z_]+)\s*\(", program_re[position:])
        if match is None:
            raise ValueError(f"Expected an operation at offset {position}.")

        operation = match.group(1)
        position += match.end()
        arg1 = parse_argument(operation, 0)
        skip_whitespace()
        if position >= len(program_re) or program_re[position] != ",":
            raise ValueError(f"Expected argument separator at offset {position}.")
        position += 1
        arg2 = parse_argument(operation, 1)
        skip_whitespace()
        if position >= len(program_re) or program_re[position] != ")":
            raise ValueError(f"Expected closing parenthesis at offset {position}.")
        position += 1
        return (operation, arg1, arg2)

    expression = parse_expression()
    skip_whitespace()
    if position != len(program_re):
        raise ValueError(f"Unexpected trailing text at offset {position}.")
    return expression


def _linear_program_expression(steps: tuple[Any, ...]) -> tuple[Any, ...]:
    expressions: list[tuple[Any, ...]] = []
    for index, step in enumerate(steps):
        arguments: list[Any] = []
        for argument_index, argument in enumerate((step.arg1, step.arg2)):
            reference = re.fullmatch(r"#(\d+)", argument)
            if reference is not None:
                reference_index = int(reference.group(1))
                if reference_index >= index:
                    raise ValueError(
                        f"Step {index} has invalid reference {argument!r}."
                    )
                arguments.append(expressions[reference_index])
            else:
                arguments.append(
                    _normalize_program_atom(argument, step.operation, argument_index)
                )
        expressions.append((step.operation, *arguments))
    return expressions[-1]


def _compare_program_re(
    program_re: Any, program_steps: tuple[Any, ...]
) -> dict[str, Any]:
    if not isinstance(program_re, str) or not program_re.strip():
        return {"status": "missing"}

    try:
        nested_expression = _parse_nested_program(program_re)
        linear_expression = _linear_program_expression(program_steps)
    except (FinQAExecutionError, ValueError, TypeError) as exc:
        return {"status": "invalid", "error": str(exc)}

    return {
        "status": (
            "consistent" if nested_expression == linear_expression else "mismatch"
        ),
        "nested_expression": nested_expression,
        "linear_expression": linear_expression,
    }


def _compare_step_annotations(
    record: dict[str, Any], program_steps: tuple[Any, ...], table: Any
) -> dict[str, Any]:
    annotations = record.get("steps")
    if not isinstance(annotations, dict):
        return {"status": "missing_or_invalid", "issues": ["steps_not_object"]}

    keys = ("op", "arg1", "arg2", "res")
    arrays = {key: annotations.get(key) for key in keys}
    if any(not isinstance(value, list) for value in arrays.values()):
        return {"status": "missing_or_invalid", "issues": ["step_fields_not_lists"]}

    lengths = {key: len(value) for key, value in arrays.items()}
    if len(set(lengths.values())) != 1 or lengths["op"] != len(program_steps):
        return {
            "status": "mismatch",
            "issues": ["step_array_lengths_or_program_step_count_differ"],
            "array_lengths": lengths,
            "program_step_count": len(program_steps),
        }

    results: list[float | str] = []
    issues: list[dict[str, Any]] = []
    for index, step in enumerate(program_steps):
        op_annotation = arrays["op"][index]
        annotated_operation = _normalize_step_operation(op_annotation, step.operation)
        if annotated_operation != step.operation:
            issues.append(
                {
                    "step": index,
                    "kind": "operation_mismatch",
                    "program": step.operation,
                    "annotated": op_annotation,
                    "normalized_annotation": annotated_operation,
                }
            )

        for field, expected in (("arg1", step.arg1), ("arg2", step.arg2)):
            actual = arrays[field][index]
            if str(actual).strip() != expected:
                issues.append(
                    {
                        "step": index,
                        "kind": f"{field}_mismatch",
                        "program": expected,
                        "annotated": actual,
                    }
                )

        try:
            value = _execute_step(step, table, results, index)
            results.append(value)
        except (FinQAExecutionError, ArithmeticError, TypeError, ValueError) as exc:
            issues.append(
                {
                    "step": index,
                    "kind": "program_step_execution_error",
                    "error": str(exc),
                }
            )
            break

        annotated_result = arrays["res"][index]
        comparison = _step_result_comparison(value, annotated_result)
        if comparison["status"] == "mismatch":
            issues.append(
                {
                    "step": index,
                    "kind": "step_result_mismatch",
                    "program_result": value,
                    "annotated_result": annotated_result,
                    "comparison": comparison,
                }
            )
        elif comparison["status"] == "ambiguous":
            issues.append(
                {
                    "step": index,
                    "kind": "step_result_ambiguous",
                    "program_result": value,
                    "annotated_result": annotated_result,
                    "comparison": comparison,
                }
            )
        elif comparison["status"] == "unparseable":
            issues.append(
                {
                    "step": index,
                    "kind": "step_result_unparseable",
                    "program_result": value,
                    "annotated_result": annotated_result,
                }
            )

    issue_kinds = {issue["kind"] for issue in issues}
    confirmed_mismatch = any(
        kind in issue_kinds
        for kind in (
            "operation_mismatch",
            "arg1_mismatch",
            "arg2_mismatch",
            "step_result_mismatch",
            "step_result_unparseable",
            "program_step_execution_error",
        )
    )
    ambiguous = "step_result_ambiguous" in issue_kinds
    status = (
        "mismatch" if confirmed_mismatch else "ambiguous" if ambiguous else "consistent"
    )
    return {
        "status": status,
        "issues": issues,
        "program_intermediate_results": results,
    }


def audit_record(record: dict[str, Any]) -> dict[str, Any]:
    """Audit one record without changing any source field."""
    example_id = record.get("id", "<missing-id>")
    result: dict[str, Any] = {
        "id": example_id,
        "program": record.get("program"),
        "program_re": record.get("program_re"),
        "raw_steps": record.get("steps"),
        "answer": record.get("answer"),
        "exe_ans": record.get("exe_ans"),
        "flags": [],
    }

    try:
        steps = parse_program(record.get("program"))
    except (FinQAProgramParseError, TypeError) as exc:
        result.update({"program_status": "parse_error", "program_error": str(exc)})
        result["flags"].append("program_parse_error")
        result["steps_audit"] = {
            "status": "not_checked",
            "issues": ["program_parse_error"],
        }
        result["program_re_audit"] = {"status": "not_checked"}
        result["classification"] = "requires_manual_review"
        return result

    result["program_step_count"] = len(steps)
    result["parsed_steps"] = [
        {
            "operation": step.operation,
            "arg1": step.arg1,
            "arg2": step.arg2,
        }
        for step in steps
    ]
    program_re_audit = _compare_program_re(record.get("program_re"), steps)
    result["program_re_audit"] = program_re_audit
    if program_re_audit["status"] == "mismatch":
        result["flags"].append("program_re_disagreement")
    elif program_re_audit["status"] == "invalid":
        result["flags"].append("program_re_invalid")
    table = record.get("table", [])
    intermediate: list[float | str] = []
    try:
        for index, step in enumerate(steps):
            intermediate.append(_execute_step(step, table, intermediate, index))
        computed = intermediate[-1]
        result["program_result"] = computed
        result["program_status"] = "executed"
    except (FinQAExecutionError, ArithmeticError, TypeError, ValueError) as exc:
        result.update({"program_status": "execution_error", "program_error": str(exc)})
        result["flags"].append("program_execution_error")
        computed = None

    step_audit = _compare_step_annotations(record, steps, table)
    result["steps_audit"] = step_audit
    if step_audit["status"] == "mismatch":
        result["flags"].append("steps_program_disagreement")
    elif step_audit["status"] == "ambiguous":
        result["flags"].append("steps_annotation_ambiguous")

    if computed is not None:
        expected = record.get("exe_ans")
        if isinstance(computed, str):
            exe_match = computed.strip().casefold() == str(expected).strip().casefold()
            result["exe_ans_match"] = exe_match
            if not exe_match:
                result["flags"].append("program_exe_ans_disagreement")
        else:
            try:
                expected_number = parse_number(str(expected))
                # The official evaluator rounds only the computed final result.
                exe_match = round(computed, 5) == expected_number
                result["exe_ans_match"] = exe_match
                if not exe_match:
                    result["flags"].append("program_exe_ans_disagreement")
            except FinQAExecutionError:
                result["exe_ans_match"] = False
                result["flags"].append("exe_ans_unparseable")

        answer_number, answer_status = _parse_human_answer(record.get("answer"))
        result["human_answer_parse_status"] = answer_status
        if isinstance(computed, str):
            normalized_answer = str(record.get("answer", "")).strip().casefold()
            if normalized_answer in {"yes", "no"}:
                answer_match = computed.casefold() == normalized_answer
                result["human_answer_match"] = answer_match
                if not answer_match:
                    result["flags"].append("program_human_answer_disagreement")
            else:
                result["human_answer_match"] = None
                result["flags"].append("human_answer_requires_manual_review")
        elif answer_number is None:
            result["human_answer_match"] = None
            result["flags"].append("human_answer_requires_manual_review")
        else:
            steps = record.get("steps", {})
            step_results = steps.get("res", []) if isinstance(steps, dict) else []
            percentage_context = (
                "percent" in str(record.get("question", "")).casefold()
                or "percentage" in str(record.get("question", "")).casefold()
                or any("%" in str(item) for item in step_results)
            )
            answer_match = _human_answer_matches(
                computed,
                str(record.get("answer", "")),
                percentage_context=percentage_context,
            )
            result["human_answer_match"] = answer_match
            if not answer_match:
                result["flags"].append("program_human_answer_disagreement")

        if (
            result.get("exe_ans_match") is False
            and result.get("human_answer_match") is True
        ):
            result["flags"].append("likely_exe_ans_precision_or_format_issue")

    result["classification"] = (
        "requires_manual_review" if result["flags"] else "internally_consistent"
    )
    return result


def audit_files(paths: list[Path]) -> dict[str, Any]:
    """Audit JSONL splits and include file hashes for reproducibility."""
    split_reports: dict[str, Any] = {}
    all_results: list[dict[str, Any]] = []

    for path in paths:
        split = path.stem
        raw = path.read_bytes()
        records = [
            json.loads(line)
            for line in raw.decode("utf-8").splitlines()
            if line.strip()
        ]
        results = [audit_record(record) for record in records]
        all_results.extend(results)
        flag_counts: Counter[str] = Counter(
            flag for item in results for flag in item["flags"]
        )
        split_reports[split] = {
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "examples": len(records),
            "internally_consistent": sum(
                item["classification"] == "internally_consistent" for item in results
            ),
            "requires_manual_review": sum(
                item["classification"] == "requires_manual_review" for item in results
            ),
            "flag_counts": dict(sorted(flag_counts.items())),
            "records": results,
        }

    category_counts = {
        "exe_ans_disagreements": sum(
            "program_exe_ans_disagreement" in item["flags"] for item in all_results
        ),
        "likely_exe_ans_precision_or_format_issues": sum(
            "likely_exe_ans_precision_or_format_issue" in item["flags"]
            for item in all_results
        ),
        "human_answer_program_disagreements": sum(
            "program_human_answer_disagreement" in item["flags"] for item in all_results
        ),
        "step_annotation_confirmed_mismatches": sum(
            "steps_program_disagreement" in item["flags"] for item in all_results
        ),
        "step_annotation_ambiguous": sum(
            "steps_annotation_ambiguous" in item["flags"] for item in all_results
        ),
        "unexecutable_programs": sum(
            any(
                flag in item["flags"]
                for flag in ("program_parse_error", "program_execution_error")
            )
            for item in all_results
        ),
        "program_re_disagreements": sum(
            "program_re_disagreement" in item["flags"] for item in all_results
        ),
        "program_re_invalid": sum(
            "program_re_invalid" in item["flags"] for item in all_results
        ),
        "program_re_missing": sum(
            item.get("program_re_audit", {}).get("status") == "missing"
            for item in all_results
        ),
    }

    return {
        "audit": "finqa_annotation_integrity",
        "schema_version": 1,
        "policy": {
            "source_records_modified": False,
            "automatic_exclusions": 0,
            "human_answer_parser": (
                "scalar numeric; explicit percentages allow ratio or percentage-point program output; unmarked values use percentage-point scale only with question/step context; magnitude units require manual review"
            ),
            "step_numeric_comparison": (
                "plain numbers rounded to displayed decimal precision; percentage "
                "annotations are compared as ratio-to-percent and otherwise marked "
                "ambiguous, never auto-converted to another scale"
            ),
            "exe_ans_numeric_tolerance": (
                "round computed final result to 5 decimals; compare exactly to exe_ans"
            ),
            "ambiguous_human_answers": "flagged for manual review; never guessed",
        },
        "total_examples": len(all_results),
        "category_counts": category_counts,
        "internally_consistent": sum(
            item["classification"] == "internally_consistent" for item in all_results
        ),
        "requires_manual_review": sum(
            item["classification"] == "requires_manual_review" for item in all_results
        ),
        "flag_counts": dict(
            sorted(
                Counter(flag for item in all_results for flag in item["flags"]).items()
            )
        ),
        "splits": split_reports,
    }
