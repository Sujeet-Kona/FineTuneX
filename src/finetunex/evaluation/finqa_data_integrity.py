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


def _step_result_matches(value: float | str, annotation: Any) -> bool:
    """Compare a step result at the precision recorded by FinQA annotations."""
    if isinstance(value, str):
        return value.strip().casefold() == str(annotation).strip().casefold()

    raw = str(annotation).strip().replace(",", "")
    raw = raw.replace("$", "").replace("€", "").replace("£", "").strip()
    is_percentage = raw.endswith("%")
    numeric_text = raw[:-1].strip() if is_percentage else raw

    if "e" in numeric_text.casefold():
        try:
            return _close(value, parse_number(raw))
        except FinQAExecutionError:
            return False

    try:
        parsed = parse_number(raw)
    except FinQAExecutionError:
        return False

    decimal_places = (
        len(numeric_text.split(".", maxsplit=1)[1])
        if "." in numeric_text
        else 0
    )
    actual = value * 100 if is_percentage else value
    expected = parsed * 100 if is_percentage else parsed
    return round(actual, decimal_places) == round(expected, decimal_places)


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


def _human_answer_matches(value: float, answer: str) -> bool:
    """Compare a human answer at its displayed precision without guessing scale."""
    match = _ANSWER.fullmatch(answer.strip())
    if match is None:
        return False

    number_text, unit = match.groups()
    decimal_places = (
        len(number_text.replace(",", "").split(".", maxsplit=1)[1])
        if "." in number_text
        else 0
    )
    is_percentage = (unit or "").lower() == "%"
    actual = value * 100 if is_percentage else value
    parsed, status = _parse_human_answer(answer)
    if status != "parsed" or parsed is None:
        return False
    expected = parsed * 100 if is_percentage else parsed
    return round(actual, decimal_places) == round(expected, decimal_places)


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
        op_match = (
            _STEP_OP.fullmatch(op_annotation)
            if isinstance(op_annotation, str)
            else None
        )
        annotated_operation = (
            _STEP_OPERATION_ALIASES.get(op_match.group(1), op_match.group(1))
            if op_match is not None
            else None
        )
        if annotated_operation != step.operation:
            issues.append({
                "step": index,
                "kind": "operation_mismatch",
                "program": step.operation,
                "annotated": op_annotation,
                "normalized_annotation": annotated_operation,
            })

        for field, expected in (("arg1", step.arg1), ("arg2", step.arg2)):
            actual = arrays[field][index]
            if str(actual).strip() != expected:
                issues.append({
                    "step": index,
                    "kind": f"{field}_mismatch",
                    "program": expected,
                    "annotated": actual,
                })

        try:
            value = _execute_step(step, table, results, index)
            results.append(value)
        except (FinQAExecutionError, ArithmeticError, TypeError, ValueError) as exc:
            issues.append({
                "step": index,
                "kind": "program_step_execution_error",
                "error": str(exc),
            })
            break

        annotated_result = arrays["res"][index]
        if not _step_result_matches(value, annotated_result):
            issues.append({
                "step": index,
                "kind": "step_result_mismatch",
                "program_result": value,
                "annotated_result": annotated_result,
            })

    return {
        "status": "consistent" if not issues else "mismatch",
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
    if step_audit["status"] != "consistent":
        result["flags"].append("steps_program_disagreement")

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
                exe_match = round(computed, 5) == round(expected_number, 5)
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
            answer_match = _human_answer_matches(computed, str(record.get("answer", "")))
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
            "program_exe_ans_disagreement" in item["flags"]
            for item in all_results
        ),
        "likely_exe_ans_precision_or_format_issues": sum(
            "likely_exe_ans_precision_or_format_issue" in item["flags"]
            for item in all_results
        ),
        "human_answer_program_disagreements": sum(
            "program_human_answer_disagreement" in item["flags"]
            for item in all_results
        ),
        "step_annotation_issues": sum(
            "steps_program_disagreement" in item["flags"]
            for item in all_results
        ),
        "unexecutable_programs": sum(
            any(
                flag in item["flags"]
                for flag in ("program_parse_error", "program_execution_error")
            )
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
                "single numeric scalar with optional %, thousand, million, billion"
            ),
            "step_numeric_tolerance": "relative 1e-5, absolute 1e-5",
            "exe_ans_numeric_tolerance": "exact after five-decimal rounding semantics",
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
        "flag_counts": dict(sorted(Counter(
            flag for item in all_results for flag in item["flags"]
        ).items())),
        "splits": split_reports,
    }
