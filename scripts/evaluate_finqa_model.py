"""Evaluate generated FinQA programs against execution answers."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_gold_programs import answers_match  # noqa: E402

from finetunex.config import load_yaml_config  # noqa: E402
from finetunex.evaluation.finqa_executor import (  # noqa: E402
    FinQAExecutionError,
    execute_program,
)
from finetunex.evaluation.finqa_parser import (  # noqa: E402
    FinQAProgramParseError,
    parse_program,
)
from finetunex.inference.generator import (  # noqa: E402
    generate_program,
    load_base_inference_components,
    load_inference_components,
)
from finetunex.inference.prompt import build_inference_prompt  # noqa: E402

VALIDATION_PATH = PROJECT_ROOT / "data/processed/finqa/validation.jsonl"
MODEL_CONFIG_PATH = PROJECT_ROOT / "configs/model.yaml"
TRAINING_CONFIG_PATH = PROJECT_ROOT / "configs/training.yaml"
PREPROCESSING_CONFIG_PATH = PROJECT_ROOT / "configs/preprocessing.yaml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_validation_records(
    limit: int | None = None,
    *,
    sample_seed: int = 42,
) -> list[dict[str, Any]]:
    """Load the complete split or a deterministic, ID-hash sample."""
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0
    ):
        raise ValueError("limit must be a positive integer or None")
    if (
        isinstance(sample_seed, bool)
        or not isinstance(sample_seed, int)
        or sample_seed < 0
    ):
        raise ValueError("sample_seed must be a non-negative integer")

    records: list[dict[str, Any]] = []
    record_ids: set[str] = set()
    with VALIDATION_PATH.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                raise ValueError(f"Blank validation record at line {line_number}")
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(
                    f"Validation line {line_number} must contain an object"
                )
            required = (
                "id",
                "question",
                "pre_text",
                "post_text",
                "table",
                "exe_ans",
            )
            missing = [key for key in required if key not in record]
            if missing:
                raise ValueError(f"Validation line {line_number} is missing {missing}")
            record_id = record["id"]
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"Validation line {line_number} has an invalid ID")
            if record_id in record_ids:
                raise ValueError(f"Duplicate validation record ID: {record_id}")
            record_ids.add(record_id)
            records.append(record)

    if not records:
        raise ValueError("No validation records were loaded")
    if limit is None or limit >= len(records):
        return records

    ranked_indices = sorted(
        range(len(records)),
        key=lambda index: hashlib.sha256(
            f"{sample_seed}:{records[index]['id']}".encode()
        ).hexdigest(),
    )[:limit]
    return [records[index] for index in sorted(ranked_indices)]


def inference_only_record(record: dict[str, Any]) -> dict[str, Any]:
    """Explicitly exclude all gold targets and annotations from prompting."""
    return {
        key: record[key] for key in ("id", "question", "pre_text", "post_text", "table")
    }


def validate_preprocessing_provenance() -> tuple[str, str, str]:
    """Verify raw evaluation data is the exact source of the tokenized artifacts."""
    manifest_path = PROJECT_ROOT / "artifacts/preprocessed/finqa/manifest.json"
    preprocessed_validation_path = (
        PROJECT_ROOT / "artifacts/preprocessed/finqa/validation.jsonl"
    )

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Preprocessing manifest not found: {manifest_path}")
    if not preprocessed_validation_path.is_file():
        raise FileNotFoundError(
            f"Preprocessed validation artifact not found: "
            f"{preprocessed_validation_path}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 2:
        raise ValueError("Unsupported or invalid preprocessing manifest")

    source_validation_sha256 = sha256_file(VALIDATION_PATH)
    preprocessed_validation_sha256 = sha256_file(preprocessed_validation_path)
    manifest_source_sha256 = (
        manifest.get("dataset", {}).get("source_split_sha256", {}).get("validation")
    )
    manifest_artifact_sha256 = (
        manifest.get("splits", {}).get("validation", {}).get("output_sha256")
    )

    if manifest_source_sha256 != source_validation_sha256:
        raise ValueError(
            "Raw validation split does not match the source validation hash "
            "recorded in the preprocessing manifest"
        )
    if manifest_artifact_sha256 != preprocessed_validation_sha256:
        raise ValueError(
            "Tokenized validation artifact does not match its preprocessing manifest"
        )

    return (
        sha256_file(manifest_path),
        source_validation_sha256,
        preprocessed_validation_sha256,
    )


def evaluate_prediction(
    normalized_output: str,
    table: list[list[str]],
    expected: object,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "parse_status": "failed",
        "execution_status": "not_attempted",
        "answer_match": False,
        "predicted_answer": None,
        "parse_error": None,
        "execution_error": None,
        "operations": [],
    }

    try:
        steps = parse_program(normalized_output)
    except FinQAProgramParseError as exc:
        result["parse_error"] = str(exc)
        return result

    result["parse_status"] = "success"
    result["operations"] = [step.operation for step in steps]

    try:
        predicted = execute_program(normalized_output, table)
    except (FinQAExecutionError, ArithmeticError, ValueError) as exc:
        result["execution_status"] = "failed"
        result["execution_error"] = str(exc)
        return result

    result["execution_status"] = "success"
    result["predicted_answer"] = predicted
    result["answer_match"] = answers_match(predicted, expected)

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate FineTuneX generated FinQA programs."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="LoRA checkpoint directory (required unless --base-model is set).",
    )
    parser.add_argument(
        "--base-model",
        action="store_true",
        help="Evaluate the unadapted pinned base model instead of a LoRA checkpoint.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional deterministic validation sample size; "
            "default evaluates all records."
        ),
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Seed for reproducible ID-hash sampling when --limit is set.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (defaults to a full-validation run directory).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum generated tokens per example (configured reserve: 128).",
    )
    parser.add_argument(
        "--evidence-budget",
        type=int,
        default=1024,
        help="Maximum evidence tokens supplied to the prompt.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing evaluation output files.",
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.sample_seed < 0:
        parser.error("--sample-seed must be non-negative")

    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")

    if args.evidence_budget <= 0:
        parser.error("--evidence-budget must be positive")

    return args


def main() -> None:
    args = parse_args()

    checkpoint_path = None
    if not args.base_model:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for adapter evaluation")
        checkpoint_path = args.checkpoint.expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_ROOT / checkpoint_path

    output_dir = args.output_dir or Path(
        "artifacts/evaluation/v4-base-full"
        if args.base_model
        else "artifacts/evaluation/v4-adapter-full"
    )
    output_dir = output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    predictions_path = output_dir / "validation_predictions.jsonl"
    summary_path = output_dir / "summary.json"

    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.overwrite and (predictions_path.exists() or summary_path.exists()):
        raise FileExistsError(
            f"Evaluation outputs already exist in {output_dir}. "
            "Choose another --output-dir or pass --overwrite."
        )

    records = load_validation_records(args.limit, sample_seed=args.sample_seed)

    model_config = load_yaml_config(MODEL_CONFIG_PATH)
    training_config = load_yaml_config(TRAINING_CONFIG_PATH)
    preprocessing_config = load_yaml_config(PREPROCESSING_CONFIG_PATH)["preprocessing"]
    max_prompt_tokens = preprocessing_config["sequence"]["max_prompt_length"]
    configured_max_new_tokens = preprocessing_config["generation"]["max_new_tokens"]

    if args.max_new_tokens > configured_max_new_tokens:
        raise ValueError(
            f"--max-new-tokens cannot exceed configured generation reserve "
            f"({configured_max_new_tokens})"
        )

    max_sequence_length = training_config["training"]["max_sequence_length"]
    if max_prompt_tokens + args.max_new_tokens > max_sequence_length:
        raise ValueError("Prompt budget and generation limit exceed sequence limit")

    evidence_budget = args.evidence_budget

    print("FINETUNEX — FINQA MODEL EVALUATION")
    print("=" * 64)
    print("Model source:", "base model" if args.base_model else "LoRA adapter")
    if not args.base_model:
        print("Checkpoint:", checkpoint_path)
    print("Validation examples:", len(records))
    print("Max new tokens:", args.max_new_tokens)
    print("Evidence budget:", evidence_budget)
    print("Maximum prompt tokens:", max_prompt_tokens)
    (
        preprocessing_manifest_sha256,
        source_validation_sha256,
        preprocessed_validation_sha256,
    ) = validate_preprocessing_provenance()

    print("Output directory:", output_dir)
    print("Source validation SHA-256:", source_validation_sha256)
    print(
        "Preprocessed validation artifact SHA-256:",
        preprocessed_validation_sha256,
    )
    print("Preprocessing manifest SHA-256:", preprocessing_manifest_sha256)

    if args.base_model:
        components = load_base_inference_components(model_config, training_config)
    else:
        components = load_inference_components(
            model_config,
            training_config,
            checkpoint_path,
        )

    if (
        not args.base_model
        and components.checkpoint_metadata.get("preprocessing_manifest_sha256")
        != preprocessing_manifest_sha256
    ):
        raise ValueError(
            "Checkpoint preprocessing manifest does not match current artifacts"
        )

    counters: Counter[str] = Counter()
    operation_counts: Counter[str] = Counter()

    predictions_temp = predictions_path.with_suffix(".jsonl.tmp")
    summary_temp = summary_path.with_suffix(".json.tmp")

    try:
        with predictions_temp.open("w", encoding="utf-8") as predictions_file:
            for index, record in enumerate(records, start=1):
                safe_record = inference_only_record(record)

                prompt_result = build_inference_prompt(
                    safe_record,
                    components.tokenizer,
                    evidence_budget=evidence_budget,
                    max_prompt_tokens=max_prompt_tokens,
                )

                generation = generate_program(
                    components,
                    prompt_result["prompt"],
                    max_new_tokens=args.max_new_tokens,
                )

                evaluation = evaluate_prediction(
                    generation["normalized_output"],
                    record["table"],
                    record["exe_ans"],
                )

                counters["total"] += 1
                counters["generated"] += 1
                counters[f"parse_{evaluation['parse_status']}"] += 1
                counters[f"execution_{evaluation['execution_status']}"] += 1

                if evaluation["answer_match"]:
                    counters["correct"] += 1
                else:
                    counters["incorrect"] += 1

                if evaluation["parse_status"] == "failed":
                    counters["parse_failures"] += 1

                if evaluation["execution_status"] == "failed":
                    counters["execution_failures"] += 1

                operation_counts.update(evaluation["operations"])

                prediction = {
                    "record_id": record["id"],
                    "prompt_sha256": hashlib.sha256(
                        prompt_result["prompt"].encode("utf-8")
                    ).hexdigest(),
                    "split": "validation",
                    "raw_output": generation["raw_output"],
                    "normalized_output": generation["normalized_output"],
                    "generated_token_ids": generation["generated_token_ids"],
                    "prompt_tokens": generation["prompt_tokens"],
                    "generated_tokens": generation["generated_tokens"],
                    "ended_with_eos": generation["ended_with_eos"],
                    "prompt_metadata": prompt_result["metadata"],
                    "parse_status": evaluation["parse_status"],
                    "execution_status": evaluation["execution_status"],
                    "predicted_answer": evaluation["predicted_answer"],
                    "expected_answer": record["exe_ans"],
                    "answer_match": evaluation["answer_match"],
                    "parse_error": evaluation["parse_error"],
                    "execution_error": evaluation["execution_error"],
                    "operations": evaluation["operations"],
                }

                predictions_file.write(
                    json.dumps(prediction, ensure_ascii=False) + "\n"
                )
                predictions_file.flush()

                print(
                    f"[{index:>3}/{len(records)}] "
                    f"{record['id']} | "
                    f"parse={evaluation['parse_status']} | "
                    f"execution={evaluation['execution_status']} | "
                    f"match={evaluation['answer_match']} | "
                    f"tokens={generation['generated_tokens']}"
                )

        total = counters["total"]
        summary = {
            "evaluation": "finetunex_finqa_generated_programs",
            "evaluation_schema_version": 1,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "split": "validation",
            "scope": {
                "examples_evaluated": total,
                "full_validation_examples": 883,
                "limited_run": total < 883,
                "sample_method": (
                    "sha256_record_id"
                    if total < 883
                    else "complete_official_order"
                ),
                "sample_seed": args.sample_seed if total < 883 else None,
                "max_new_tokens": args.max_new_tokens,
                "evidence_budget": evidence_budget,
                "max_prompt_tokens": max_prompt_tokens,
                "generation": "greedy",
                "answer_reference": "exe_ans",
                "answer_comparison": "exact_after_five_decimal_rounding",
            },
            "model_source": {
                "type": "base_model" if args.base_model else "lora_adapter",
                "checkpoint_path": None if args.base_model else str(checkpoint_path),
                "checkpoint_metadata_sha256": (
                    None
                    if args.base_model
                    else sha256_file(checkpoint_path / "checkpoint_metadata.json")
                ),
                "model": components.checkpoint_metadata["model"],
                "model_revision": components.checkpoint_metadata["model_revision"],
                "training_evaluation_scope": (
                    None
                    if args.base_model
                    else components.checkpoint_metadata["evaluation_scope"]
                ),
            },
            "inputs": {
                "validation_path": str(VALIDATION_PATH),
                "validation_sha256": source_validation_sha256,
                "preprocessed_validation_sha256": preprocessed_validation_sha256,
                "model_config_sha256": sha256_file(MODEL_CONFIG_PATH),
                "training_config_sha256": sha256_file(TRAINING_CONFIG_PATH),
                "preprocessing_config_sha256": sha256_file(PREPROCESSING_CONFIG_PATH),
                "preprocessing_manifest_sha256": preprocessing_manifest_sha256,
            },
            "metrics": {
                "total": total,
                "generated": counters["generated"],
                "parse_success": counters["parse_success"],
                "parse_failures": counters["parse_failures"],
                "parse_rate": counters["parse_success"] / total,
                "execution_success": counters["execution_success"],
                "execution_failures": counters["execution_failures"],
                "execution_rate": counters["execution_success"] / total,
                "correct": counters["correct"],
                "incorrect": counters["incorrect"],
                "answer_accuracy": counters["correct"] / total,
            },
            "operation_counts": dict(operation_counts),
            "artifacts": {
                "predictions": predictions_path.name,
                "summary": summary_path.name,
            },
        }

        summary_temp.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        predictions_temp.replace(predictions_path)
        summary_temp.replace(summary_path)

    except Exception:
        predictions_temp.unlink(missing_ok=True)
        summary_temp.unlink(missing_ok=True)
        raise

    print("\n" + "=" * 64)
    print("EVALUATION SUMMARY")
    print(f"  Examples:          {total}")
    print(f"  Parse success:     {counters['parse_success']}")
    print(f"  Parse failures:    {counters['parse_failures']}")
    print(f"  Execution success: {counters['execution_success']}")
    print(f"  Execution failures:{counters['execution_failures']}")
    print(f"  Correct answers:   {counters['correct']}")
    print(f"  Answer accuracy:   {summary['metrics']['answer_accuracy']:.2%}")
    print(f"\nPredictions: {predictions_path}")
    print(f"Summary:     {summary_path}")


if __name__ == "__main__":
    main()
