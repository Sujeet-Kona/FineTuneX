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
  --output-dir artifacts/evaluation/v3-base-32
```

Run the adapter with the same evaluation settings:

```bash
python scripts/evaluate_finqa_model.py \\
  --checkpoint artifacts/checkpoints/<checkpoint-name> \\
  --limit 32 \\
  --max-new-tokens 128 \\
  --evidence-budget 1024 \\
  --output-dir artifacts/evaluation/v3-adapter-32
```

After both runs finish, verify the paired comparison:

```bash
python scripts/compare_finqa_evaluations.py \\
  --base-dir artifacts/evaluation/v3-base-32 \\
  --adapter-dir artifacts/evaluation/v3-adapter-32 \\
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

The current prompt contract is `finqa-program-v4-fixed-prompt-budget`. It defines
the supported FinQA operations, zero-based intermediate-result references,
table aggregation syntax, operand order, percentage calculations, and
unit/scale handling. Training and inference share one evidence-selection and
prompt-rendering implementation.

The prompt ceiling is 1,408 tokenizer tokens and the generation reserve is 128
tokens within the 1,536-token sequence limit. Evidence is reduced using only
the question, passages, table, and tokenizer. Gold program length never affects
evidence selection. Preprocessing rejects a target that cannot fit in the
reserved generation window rather than truncating it. Evaluation and the API
load the same limits from `configs/preprocessing.yaml`.

A checkpoint trained with an earlier prompt version is intentionally rejected
by the current inference loader. After changing the prompt contract, regenerate
the tokenized train/validation artifacts, train a new adapter, and evaluate
that adapter against a base-model run produced with the same prompt version.
The v3 pilot checkpoint and its metrics are historical and must not be presented
as v4 results.

## FastAPI inference

The API loads one verified LoRA checkpoint at startup and serializes generation
requests so concurrent clients do not issue simultaneous inference calls to
the same local accelerator. Set the checkpoint path and start the server:

```bash
export FINETUNEX_CHECKPOINT=artifacts/checkpoints/<v4-checkpoint-name>
uvicorn finetunex.api.app:app --host 127.0.0.1 --port 8000
```

The model and tokenizer must already exist in the local Hugging Face cache.
The API does not download model weights at startup. It exposes:

- `GET /health/live` — process liveness.
- `GET /health/ready` — model readiness.
- `POST /v1/finqa/score` — generate a FinQA program and execute it against the supplied table.

Example request:

```bash
curl -X POST http://127.0.0.1:8000/v1/finqa/score \\
  -H 'Content-Type: application/json' \\
  -d '{
    "id": "example-1",
    "question": "What is revenue in 2023 divided by revenue in 2022?",
    "table": [["Year", "Revenue"], ["2022", "100"], ["2023", "125"]]
  }'
```

The request schema rejects gold answer/program fields. A syntactically invalid
model output is reported as a parse failure; it is not silently converted into
an answer. The returned numerical value is the deterministic executor result,
not a free-form model-generated answer.

## Docker deployment (NVIDIA GPU)

The container target is a Linux host with an NVIDIA GPU and the NVIDIA
Container Toolkit. Docker Desktop on Apple Silicon does not expose the Mac's
MPS accelerator to a Linux container; run the API directly on the Mac for MPS.

Build and start the service after exporting the local adapter and Hugging Face
cache directories:

```bash
export FINETUNEX_CHECKPOINT="$PWD/artifacts/checkpoints/<v3-checkpoint-name>"
export HF_CACHE_PATH="$HOME/.cache/huggingface"
docker compose up --build
```

The adapter and model cache are mounted read-only. The image runs in offline
Hugging Face mode and will fail at startup if the pinned model/tokenizer files
are not already present in the mounted cache. The service listens on port
8000; verify readiness with `curl http://localhost:8000/health/ready`.

## Engineering Principles

- Preserve official benchmark splits.
- Use deterministic, restricted program execution.
- Evaluate numerical results rather than fluent text alone.
- Keep baseline and fine-tuned evaluation comparable.
- Record dataset, model and experiment provenance.
- Report measured results without unsupported claims.

## Project Status

- [x] Repository foundation and pinned FinQA/model revisions.
- [x] FinQA preprocessing pipeline with train/validation provenance and test-split isolation.
- [x] FinQA program parser and deterministic executor.
- [x] Gold-program execution audit across all 8,281 examples.
- [x] Versioned v3 reasoning prompt and controlled 32-example pilot.
- [x] Identified and reproduced the v3 training/inference evidence-budget mismatch.
- [x] Implemented v4 target-independent prompt budgeting across preprocessing, evaluation, and API.
- [ ] Regenerate v4 train/validation artifacts and verify exact prompt parity on all 883 validation examples.
- [ ] Train a fresh v4 LoRA adapter and run paired base/adapter evaluation.
- [ ] Run a full 883-example validation comparison before making model-quality claims.
- [x] FastAPI inference implementation (real checkpoint-backed smoke test pending).
- [x] CUDA Docker deployment definition (Linux NVIDIA build/runtime validation pending).
- [ ] Confirm GitHub Actions CI on the final pull-request chain.

### Historical v3 pilot (superseded)

The earlier v3 32-example pilot produced these results:

| Metric | Base model | LoRA pilot |
| --- | ---: | ---: |
| Parse success | 18/32 (56.25%) | 32/32 (100%) |
| Execution success | 10/32 (31.25%) | 32/32 (100%) |
| Exact answer accuracy | 0/32 (0%) | 2/32 (6.25%) |

The pilot showed improved program formatting and executability, but weak
numerical reasoning. These measurements used the v3 prompt-budget policy and
are not v4 results. The v4 pipeline must be re-preprocessed, retrained, and
evaluated before any updated model-quality claim is made.
