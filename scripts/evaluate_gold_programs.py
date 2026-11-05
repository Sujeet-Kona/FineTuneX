"""Evaluate FinQA gold programs against their execution answers."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from finetunex.evaluation.finqa_executor import (
    FinQAExecutionError,
    execute_program,
    parse_number,
)
from finetunex.evaluation.finqa_parser import parse_program
from finetunex.evaluation.finqa_reference import reference_execute_program

DATA_DIR = Path("data/processed/finqa")
SPLITS = ("train", "validation")
ERROR_EXAMPLES = 20


def answers_match(predicted: float | str, expected: object) -> bool:
    """Compare using the official evaluator's numeric result semantics."""
    if isinstance(predicted, str):
        return predicted == expected

    try:
        expected_number = parse_number(str(expected))
    except (FinQAExecutionError, TypeError, ValueError):
        return False

    # execute_program already rounds the prediction to five decimals. The\n    # official evaluator compares that value directly to the unrounded exe_ans.\n    return predicted == expected_number


def evaluate_split(split: str) -> dict[str, object]:
    path = DATA_DIR / f"{split}.jsonl"

    total = 0
    successful = 0
    matched = 0
    errors: list[dict[str, object]] = []
    mismatches: list[dict[str, object]] = []
    reference_errors: list[dict[str, object]] = []
    reference_mismatches: list[dict[str, object]] = []
    executor_disagreements: list[dict[str, object]] = []
    operation_counts: Counter[str] = Counter()
    gold_answer_types: Counter[str] = Counter()

    with path.open() as file:
        for line_number, line in enumerate(file, start=1):
            record = json.loads(line)
            total += 1

            program = record["program"]
            expected = record["exe_ans"]
            gold_answer_types[type(expected).__name__] += 1

            for step in parse_program(program):
                operation_counts[step.operation] += 1

            predicted: float | str | None = None
            reference_predicted: float | str | None = None

            try:
                predicted = execute_program(program, record["table"])
                successful += 1
            except (FinQAExecutionError, ArithmeticError, ValueError) as exc:
                errors.append(
                    {
                        "line": line_number,
                        "id": record["id"],
                        "program": program,
                        "expected": expected,
                        "error": str(exc),
                    }
                )

            try:
                reference_predicted = reference_execute_program(
                    program, record["table"]
                )
            except (ArithmeticError, TypeError, ValueError) as exc:
                reference_errors.append(
                    {
                        "line": line_number,
                        "id": record["id"],
                        "program": program,
                        "expected": expected,
                        "error": str(exc),
                    }
                )
            else:
                if not answers_match(reference_predicted, expected):
                    reference_mismatches.append(
                        {
                            "line": line_number,
                            "id": record["id"],
                            "program": program,
                            "reference_predicted": reference_predicted,
                            "expected": expected,
                        }
                    )

            if predicted is not None:
                if answers_match(predicted, expected):
                    matched += 1
                else:
                    mismatches.append(
                        {
                            "line": line_number,
                            "id": record["id"],
                            "program": program,
                            "predicted": predicted,
                            "expected": expected,
                            "answer": record.get("answer"),
                        }
                    )

            if predicted != reference_predicted:
                executor_disagreements.append(
                    {
                        "line": line_number,
                        "id": record["id"],
                        "program": program,
                        "custom_predicted": predicted,
                        "reference_predicted": reference_predicted,
                        "expected": expected,
                    }
                )

    return {
        "split": split,
        "total": total,
        "successful": successful,
        "matched": matched,
        "errors": errors,
        "mismatches": mismatches,
        "reference_errors": reference_errors,
        "reference_mismatches": reference_mismatches,
        "executor_disagreements": executor_disagreements,
        "operation_counts": operation_counts,
        "gold_answer_types": gold_answer_types,
    }


def main() -> None:
    results = [evaluate_split(split) for split in SPLITS]

    total_examples = 0
    total_successful = 0
    total_matched = 0
    all_errors: list[dict[str, object]] = []
    all_mismatches: list[dict[str, object]] = []
    total_operations: Counter[str] = Counter()

    print("FINETUNEX — FINQA GOLD PROGRAM EVALUATION")
    print("=" * 64)

    for result in results:
        split = str(result["split"])
        total = int(result["total"])
        successful = int(result["successful"])
        matched = int(result["matched"])
        errors = result["errors"]
        mismatches = result["mismatches"]

        total_examples += total
        total_successful += successful
        total_matched += matched
        all_errors.extend(errors)
        all_mismatches.extend(mismatches)
        total_operations.update(result["operation_counts"])

        execution_accuracy = matched / total if total else 0.0
        execution_success_rate = successful / total if total else 0.0

        print(f"\n{split.upper()}")
        print(f"  Examples:             {total}")
        print(f"  Executed:             {successful}")
        print(f"  Execution errors:     {len(errors)}")
        print(f"  Exact answer matches: {matched}")
        print(f"  Answer mismatches:    {len(mismatches)}")
        print(f"  Execution success:    {execution_success_rate:.4%}")
        print(f"  Execution accuracy:   {execution_accuracy:.4%}")
        print(f"  Gold answer types:    {dict(result['gold_answer_types'])}")

    print("\n" + "=" * 64)
    print("OVERALL")
    print(f"  Examples:             {total_examples}")
    print(f"  Executed:             {total_successful}")
    print(f"  Execution errors:     {len(all_errors)}")
    print(f"  Exact answer matches: {total_matched}")
    print(f"  Answer mismatches:    {len(all_mismatches)}")
    reference_mismatch_count = sum(
        len(result["reference_mismatches"]) for result in results
    )
    print(
        "  Reference execution errors:",
        sum(len(result["reference_errors"]) for result in results),
    )
    print("  Reference mismatches:", reference_mismatch_count)
    executor_disagreement_count = sum(
        len(result["executor_disagreements"]) for result in results
    )
    print("  Executor differences:", executor_disagreement_count)
    print(f"  Execution success:    {total_successful / total_examples:.4%}")
    print(f"  Execution accuracy:   {total_matched / total_examples:.4%}")

    print("\nOPERATION COUNTS")
    for operation, count in total_operations.most_common():
        print(f"  {operation:16} {count}")

    if all_errors:
        print("\nFIRST EXECUTION ERRORS")
        for item in all_errors[:ERROR_EXAMPLES]:
            print(json.dumps(item, ensure_ascii=False))

    if all_mismatches:
        print("\nFIRST ANSWER MISMATCHES")
        for item in all_mismatches[:ERROR_EXAMPLES]:
            print(json.dumps(item, ensure_ascii=False))

    report = {
        "evaluation": "finqa_gold_program_execution",
        "comparison": "exact_after_five_decimal_rounding",
        "total": total_examples,
        "executed": total_successful,
        "execution_errors": len(all_errors),
        "exact_matches": total_matched,
        "answer_mismatches": len(all_mismatches),
        "reference_execution_errors": sum(
            len(result["reference_errors"]) for result in results
        ),
        "reference_answer_mismatches": reference_mismatch_count,
        "custom_reference_disagreements": executor_disagreement_count,
        "execution_success_rate": (
            total_successful / total_examples if total_examples else 0
        ),
        "execution_accuracy": (total_matched / total_examples if total_examples else 0),
        "splits": [
            {
                "split": result["split"],
                "total": result["total"],
                "executed": result["successful"],
                "exact_matches": result["matched"],
                "execution_errors": len(result["errors"]),
                "answer_mismatches": len(result["mismatches"]),
                "reference_execution_errors": len(result["reference_errors"]),
                "reference_answer_mismatches": len(result["reference_mismatches"]),
                "custom_reference_disagreements": len(result["executor_disagreements"]),
            }
            for result in results
        ],
        "errors": all_errors,
        "mismatches": all_mismatches,
        "reference_errors": [
            item for result in results for item in result["reference_errors"]
        ],
        "reference_mismatches": [
            item for result in results for item in result["reference_mismatches"]
        ],
        "executor_disagreements": [
            item for result in results for item in result["executor_disagreements"]
        ],
    }

    report_path = Path("experiments/finqa_gold_execution_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print(f"\nDetailed report: {report_path}")


if __name__ == "__main__":
    main()
