"""Compare paired base-model and LoRA FinQA evaluation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def load_predictions(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank line in {path}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Invalid prediction in {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No predictions in {path}")
    return rows


def compare(
    base_summary: dict[str, Any],
    base_rows: list[dict[str, Any]],
    adapter_summary: dict[str, Any],
    adapter_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fail closed unless both evaluations used identical benchmark inputs."""
    if base_summary.get("model_source", {}).get("type") != "base_model":
        raise ValueError("First run must be base_model")
    if adapter_summary.get("model_source", {}).get("type") != "lora_adapter":
        raise ValueError("Second run must be lora_adapter")

    for key in (
        "examples_evaluated",
        "sample_method",
        "sample_seed",
        "max_new_tokens",
        "max_prompt_tokens",
        "evidence_budget",
        "generation",
        "answer_reference",
        "answer_comparison",
    ):
        base_value = base_summary.get("scope", {}).get(key)
        adapter_value = adapter_summary.get("scope", {}).get(key)
        if base_value != adapter_value:
            raise ValueError(f"Scope mismatch: {key}")

    for key in (
        "validation_sha256",
        "preprocessed_validation_sha256",
        "model_config_sha256",
        "training_config_sha256",
        "preprocessing_config_sha256",
        "preprocessing_manifest_sha256",
    ):
        base_value = base_summary.get("inputs", {}).get(key)
        adapter_value = adapter_summary.get("inputs", {}).get(key)
        if base_value != adapter_value:
            raise ValueError(f"Input provenance mismatch: {key}")

    base_model = base_summary["model_source"]
    adapter_model = adapter_summary["model_source"]
    for key in ("model", "model_revision"):
        if base_model.get(key) != adapter_model.get(key):
            raise ValueError(f"Base model mismatch: {key}")

    if len(base_rows) != len(adapter_rows):
        raise ValueError("Prediction counts differ")

    for index, (base, adapter) in enumerate(zip(base_rows, adapter_rows), 1):
        if base.get("record_id") != adapter.get("record_id"):
            raise ValueError(f"Record order mismatch at row {index}")
        prompt_hash = base.get("prompt_sha256")
        if not prompt_hash or prompt_hash != adapter.get("prompt_sha256"):
            raise ValueError(f"Prompt hash mismatch at row {index}")
        if base.get("expected_answer") != adapter.get("expected_answer"):
            raise ValueError(f"Expected answer mismatch at row {index}")

    base_metrics = base_summary["metrics"]
    adapter_metrics = adapter_summary["metrics"]
    metric_names = ("parse_rate", "execution_rate", "answer_accuracy")
    return {
        "comparison": "paired_base_vs_lora_finqa_validation",
        "examples": len(base_rows),
        "identical_prompts_and_record_order": True,
        # Recorded so a reader can tell which scoring rule produced these numbers.
        "answer_comparison": base_summary["scope"]["answer_comparison"],
        "base_metrics": base_metrics,
        "adapter_metrics": adapter_metrics,
        "adapter_minus_base": {
            metric: adapter_metrics[metric] - base_metrics[metric]
            for metric in metric_names
        },
        "base_model": base_model,
        "adapter_model": adapter_model,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("artifacts/evaluation/v4-base-full"),
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=Path("artifacts/evaluation/v4-adapter-full"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluation/v4-paired-full-comparison.json"),
    )
    args = parser.parse_args()

    base_summary = load_json(args.base_dir / "summary.json")
    adapter_summary = load_json(args.adapter_dir / "summary.json")
    result = compare(
        base_summary,
        load_predictions(args.base_dir / "validation_predictions.jsonl"),
        adapter_summary,
        load_predictions(args.adapter_dir / "validation_predictions.jsonl"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    print("Comparison written to:", args.output)


if __name__ == "__main__":
    main()
