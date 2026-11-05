"""Export source-verified representative FinQA annotation audit samples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finetunex.evaluation.finqa_audit_samples import (  # noqa: E402
    export_audit_samples,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/processed/finqa"),
        help="Directory containing train.jsonl, validation.jsonl, and test.jsonl.",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("experiments/finqa_annotation_integrity.json"),
        help="Existing output from scripts/audit_finqa_annotations.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/audit_samples"),
    )
    parser.add_argument("--limit-per-category", type=int, default=10)
    args = parser.parse_args()

    def project_path(path: Path) -> Path:
        return path if path.is_absolute() else PROJECT_ROOT / path

    manifest = export_audit_samples(
        source_dir=project_path(args.source_dir),
        audit_path=project_path(args.audit),
        output_dir=project_path(args.output_dir),
        limit_per_category=args.limit_per_category,
    )
    print("FINQA AUDIT SAMPLE EXPORT")
    print("=" * 48)
    for category, details in manifest["categories"].items():
        print(
            f"{category}: {details['exported_count']}/{details['available_count']} "
            f"-> {details['output_file']}"
        )
    print(f"Manifest: {project_path(args.output_dir) / 'manifest.json'}")
    print("Source JSONL files were not modified.")


if __name__ == "__main__":
    main()
