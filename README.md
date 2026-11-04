# FineTuneX

**Reproducible Financial Reasoning LLM Fine-Tuning and Evaluation Pipeline**

FineTuneX explores the lifecycle of adapting a pretrained language model
to numerical reasoning over financial reports.

## Benchmark

FinQA — a financial question-answering benchmark containing report
context, tables, numerical questions and annotated reasoning programs.

## Install

Python 3.11 or 3.12 is required.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[ml,api,dev]"
```

The `ml` extra installs PyTorch, Transformers, PEFT, Accelerate and
Safetensors. The `api` extra installs FastAPI and Uvicorn. The `dev` extra
installs the test and lint tools.

FineTuneX loads its pinned model and tokenizer from the local Hugging Face
cache during training and inference. Dataset preparation may require network
access to retrieve the pinned FinQA dataset revision. Training requires a
supported CUDA or Apple MPS accelerator; CPU-only environments can run the
unit tests and non-training utilities but cannot run the configured training
workflow.

## Pipeline

1. Financial dataset validation and preprocessing.
2. Reproducible dataset and experiment management.
3. Base-model and LoRA adapter evaluation on identical validation inputs.
4. Parameter-efficient LoRA fine-tuning.
5. Structured reasoning-program generation.
6. Deterministic program validation and execution.
7. Baseline comparison and error analysis.
8. Adapter packaging and provenance.
9. FastAPI inference.
10. Docker and automated testing.

## Controlled Base-Model Benchmark

The evaluator supports both the unadapted pinned base model and a saved LoRA
checkpoint. Both modes use the same validation records, inference-only record
projection, prompt builder, evidence budget, greedy decoding, parser, executor,
and answer matcher. Each prediction stores a SHA-256 hash of the exact prompt
so paired runs can be checked before comparing metrics.

Run the unadapted model:

```bash
python scripts/evaluate_finqa_model.py \\
  --base-model \\
  --limit 32 \\
  --max-new-tokens 128 \\
  --evidence-budget 1024 \\
  --output-dir artifacts/evaluation/base-chat-32
```

Run the adapter with the same evaluation settings:

```bash
python scripts/evaluate_finqa_model.py \\
  --checkpoint artifacts/checkpoints/<checkpoint-name> \\
  --limit 32 \\
  --max-new-tokens 128 \\
  --evidence-budget 1024 \\
  --output-dir artifacts/evaluation/adapter-chat-32
```

After both runs finish, verify the paired comparison:

```bash
python scripts/compare_finqa_evaluations.py \\
  --base-dir artifacts/evaluation/base-chat-32 \\
  --adapter-dir artifacts/evaluation/adapter-chat-32 \\
  --output artifacts/evaluation/paired-comparison.json
```

The comparison utility fails closed if validation/configuration provenance,
evaluation settings, record ordering, expected answers, or exact prompt hashes
differ. It reports adapter-minus-base deltas; it does not claim statistical
significance from a small pilot subset.

Do not compare the runs unless their ordered record IDs and per-record
`prompt_sha256` values match. A 32-example prefix is a smoke benchmark, not
a full validation result and not a test-set result. Keep the test split held
out until model and evaluation decisions are frozen.

## Prompt contract and checkpoint compatibility

The current prompt contract is `finqa-program-v3-reasoning`. It explicitly
defines the supported FinQA operations, zero-based intermediate-result
references, table aggregation syntax, operand order, percentage calculations,
and unit/scale handling. Training and inference import the same prompt
implementation to preserve prompt parity.

A checkpoint trained with an earlier prompt version is intentionally rejected
by the current inference loader. After changing the prompt contract, regenerate
the tokenized train/validation artifacts, train a new adapter, and evaluate
that adapter against a base-model run produced with the same prompt version.
Do not reuse the v2 pilot checkpoint as a v3 result.

## Engineering Principles

- Preserve official benchmark splits.
- Use deterministic, restricted program execution.
- Evaluate numerical results rather than fluent text alone.
- Keep baseline and fine-tuned evaluation comparable.
- Record dataset, model and experiment provenance.
- Report measured results without unsupported claims.

## Project Status

- [x] Repository foundation.
- [x] FinQA source inspection and initial data audit.
- [x] Dedicated FinQA dataset preparation and validation pipeline.
- [x] FinQA program parser and deterministic executor.
- [x] Gold-program execution audit across all 8,281 examples.
- [x] Model feasibility and base-model evaluation path.
- [x] Versioned explicit FinQA reasoning prompt contract.
- [ ] Re-preprocess data and train/evaluate a v3 adapter.
- [ ] Full-data LoRA training and comparative evaluation.
- [ ] Artifact packaging and inference API.
- [ ] Docker and automated CI validation.
