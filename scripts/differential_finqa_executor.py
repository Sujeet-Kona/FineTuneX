"""Differentially compare our executor with the official FinQA evaluator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finetunex.evaluation.finqa_executor import (  # noqa: E402
    FinQAExecutionError,
    execute_program,
)
from finetunex.evaluation.official_finqa_oracle import (  # noqa: E402
    official_eval_program,
)


def compare_record(record: dict[str, Any]) -> dict[str, Any] | None:
    program = record["program"]
    table = record["table"]
    official_invalid, official_result = official_eval_program(program, table)

    try:
        actual_result = execute_program(program, table)
        actual_invalid = 0
        actual_error = None
    except (FinQAExecutionError, ArithmeticError, TypeError, ValueError) as exc:
        actual_result = None
        actual_invalid = 1
        actual_error = str(exc)

    if actual_invalid == official_invalid and (
        actual_invalid or actual_result == official_result
    ):
        return None

    return {
        "id": record.get("id"),
        "program": program,
        "official_invalid": official_invalid,
        "official_result": official_result,
        "custom_invalid": actual_invalid,
        "custom_result": actual_result,
        "custom_error": actual_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/processed/finqa"),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation"],
        choices=["train", "validation", "test"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/finqa_official_executor_differential.json"),
    )
    args = parser.parse_args()

    source_dir = args.source_dir
    if not source_dir.is_absolute():
        source_dir = PROJECT_ROOT / source_dir
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output

    report: dict[str, Any] = {
        "reference": {
            "repository": "czyssrs/FinQA",
            "path": "code/evaluate/evaluate.py",
            "commit": "dfc5b72c01ee17c442d28d5201b82a1f4e95d5af",
            "file_blob": "4829a4ca58a5725fb3f7a46b3b23b7f7e848a512",
        },
        "comparison": "official invalid flag and final result equality",
        "splits": {},
        "differences": [],
    }

    for split in args.splits:
        path = source_dir / f"{split}.jsonl"
        if not path.is_file():
            parser.error(f"Missing source split: {path}")

        count = 0
        differences: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                count += 1
                difference = compare_record(record)
                if difference is not None:
                    difference["split"] = split
                    difference["line"] = line_number
                    differences.append(difference)

        report["splits"][split] = {
            "examples": count,
            "identical": count - len(differences),
            "differences": len(differences),
        }
        report["differences"].extend(differences)

    report["total_examples"] = sum(
        item["examples"] for item in report["splits"].values()
    )
    report["total_differences"] = len(report["differences"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("OFFICIAL FINQA EXECUTOR DIFFERENTIAL")
    print("=" * 52)
    for split, result in report["splits"].items():
        print(
            f"{split}: {result['identical']}/{result['examples']} identical; "
            f"{result['differences']} differences"
        )
    print(f"Total: {report['total_examples']} examples")
    print(f"Total differences: {report['total_differences']}")
    print(f"Report: {output}")
    if report["total_differences"]:
        print("First 20 differences:")
        for item in report["differences"][:20]:
            print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    main()
