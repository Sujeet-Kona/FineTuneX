"""Audit FinQA evidence retrieval against gold annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finetunex.evaluation.evidence_audit import audit_evidence_records  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/processed/finqa/validation.jsonl"),
    )
    parser.add_argument(
        "--processed",
        type=Path,
        default=Path("artifacts/preprocessed/finqa/validation.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/preprocessed/finqa/manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/finqa_evidence_retrieval_audit.json"),
    )
    args = parser.parse_args()

    def project_path(path: Path) -> Path:
        return path if path.is_absolute() else PROJECT_ROOT / path

    source, processed, manifest_path, output = map(
        project_path,
        (args.source, args.processed, args.manifest, args.output),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise ValueError("Unsupported preprocessing manifest schema")

    source_hash = sha256(source)
    processed_hash = sha256(processed)
    if source_hash != manifest["dataset"]["source_split_sha256"]["validation"]:
        raise ValueError(
            "Raw validation hash does not match preprocessing manifest"
        )
    if processed_hash != manifest["splits"]["validation"]["output_sha256"]:
        raise ValueError(
            "Tokenized validation hash does not match preprocessing manifest"
        )

    source_records = read_jsonl(source)
    processed_records = read_jsonl(processed)
    expected_count = manifest["splits"]["validation"]["record_count"]
    if (
        len(source_records) != expected_count
        or len(processed_records) != expected_count
    ):
        raise ValueError(
            "Validation record count does not match preprocessing manifest"
        )

    report = audit_evidence_records(source_records, processed_records)
    report["provenance"] = {
        "source_validation_sha256": source_hash,
        "preprocessed_validation_sha256": processed_hash,
        "manifest_sha256": sha256(manifest_path),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    metrics = report["overall"]
    print("FINQA EVIDENCE RETRIEVAL AUDIT")
    print(f"Examples: {report['examples']}")
    print(
        "Complete evidence coverage: "
        f"{metrics['complete_evidence_coverage']:.2%}"
    )
    print(f"Evidence-unit recall: {metrics['evidence_unit_recall']:.2%}")
    print(f"Text evidence recall: {metrics['text_evidence_recall']:.2%}")
    print(f"Table evidence recall: {metrics['table_evidence_recall']:.2%}")
    print(
        "Examples with missing evidence: "
        f"{len(report['missing_evidence_examples'])}"
    )
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
