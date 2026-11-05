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
    if unit == "%":
        number /= 100.0
    elif unit == "thousand":
        number *= 1_000
    elif unit == "million":
        number *= 1_000_000
    elif unit == "billion":
        number *= 1_000_000_000
    return number, "parsed"


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
        try:
            parsed_annotation = parse_number(str(annotated_result))
            if isinstance(value, str) or not _close(value, parsed_annotation):
                issues.append({
                    "step": index,
                    "kind": "step_result_mismatch",
                    "program_result": value,
                    "annotated_result": annotated_result,
                })
        except FinQAExecutionError:
            if str(value) != str(annotated_result):
                issues.append({
                    "step": index,
                    "kind": "step_result_unparseable_or_mismatch",
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
        "answer": record.get("answer"),
        "exe_ans": record.get("exe_ans"),
        "flags": [],
    }

    try:
        steps = parse_program(record.get("program"))
    except (FinQAProgramParseError, TypeError) as exc:
        result.update({"program_status": "parse_error", "program_error": str(exc)})
        result["flags"].append("program_parse_error")
        return result

    result["program_step_count"] = len(steps)
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
            answer_match = _close(computed, answer_number)
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
    records_by_split: dict[str, list[dict[str, Any]]] = {}
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
        records_by_split[split] = records
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
