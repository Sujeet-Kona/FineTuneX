"""Run a non-destructive integrity audit of FinQA source annotations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finetunex.evaluation.finqa_data_integrity import audit_files  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/processed/finqa"),
        help="Directory containing the original train.jsonl, validation.jsonl, and test.jsonl.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/finqa_annotation_integrity.json"),
    )
    args = parser.parse_args()

    source_dir = args.source_dir
    if not source_dir.is_absolute():
        source_dir = PROJECT_ROOT / source_dir
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output

    paths = [\n        source_dir / "train.jsonl",\n        source_dir / "validation.jsonl",\n        source_dir / "test.jsonl",\n    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        parser.error("Missing source split(s): " + ", ".join(missing))

    report = audit_files(paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("FINQA ANNOTATION INTEGRITY AUDIT")
    print("=" * 48)
    print(f"Examples: {report['total_examples']}")
    print(f"Internally consistent: {report['internally_consistent']}")
    print(f"Requires manual review: {report['requires_manual_review']}")
    print("\nFlag counts:")
    for flag, count in report["flag_counts"].items():
        print(f"  {flag}: {count}")
    print(f"\nReport: {output}")
    print("Source JSONL files were not modified.")


if __name__ == "__main__":
    main()
