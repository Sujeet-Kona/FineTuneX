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
installs the test and lint tools. Optional MLflow tracking is installed with
`python -m pip install -e ".[tracking]"` and is disabled by default.

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
11. Optional MLflow tracking of run parameters, epoch metrics, configuration files, and selected checkpoint artifacts.
12. Prometheus request-count and latency metrics for the FastAPI service.

## Controlled Base-Model Benchmark

The evaluator supports both the unadapted pinned base model and a saved LoRA
checkpoint. Both modes use the same validation records, inference-only record
projection, prompt builder, evidence budget, greedy decoding, parser, executor,
and answer matcher. Each prediction stores a SHA-256 hash of the exact prompt
so paired runs can be checked before comparing metrics.

Run the unadapted model on all 883 validation examples:

```bash
python scripts/evaluate_finqa_model.py \
  --base-model \
  --max-new-tokens 128 \
  --evidence-budget 1024 \
  --output-dir artifacts/evaluation/v4-base-full
```

Run the adapter with the same full-validation settings:

```bash
python scripts/evaluate_finqa_model.py \
  --checkpoint artifacts/checkpoints/<checkpoint-name> \
  --max-new-tokens 128 \
  --evidence-budget 1024 \
  --output-dir artifacts/evaluation/v4-adapter-full
```

For a quick smoke sample, add `--limit 32 --sample-seed 42` and use separate
output directories. The sample is selected deterministically by record ID;
it is not the first 32 records and must not be reported as full-validation
accuracy.

After both runs finish, verify the paired comparison:

```bash
python scripts/compare_finqa_evaluations.py \\
  --base-dir artifacts/evaluation/v4-base-full \\
  --adapter-dir artifacts/evaluation/v4-adapter-full \\
  --output artifacts/evaluation/v4-paired-full-comparison.json
```

The comparison utility fails closed if validation/configuration provenance,
evaluation settings, record ordering, expected answers, or exact prompt hashes
differ. It reports adapter-minus-base deltas; it does not claim statistical
significance from a small pilot subset.

Do not compare runs unless their ordered record IDs and per-record
`prompt_sha256` values match. A limited deterministic sample is a smoke
benchmark, not a full validation result or a test-set result. Keep the test
split held out until model and evaluation decisions are frozen.

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

## Training and evaluation interpretation

The trainer selects checkpoints by the lowest token-weighted validation loss.
That is a language-model objective, not FinQA answer accuracy. A selected
checkpoint must still pass generation-level evaluation on the same validation
records and prompts as the base model. Runs using
`--max-train-batches` or `--max-validation-batches` are limited smoke runs;
they are not full-dataset training or validation results.

## Gold-program executor audit

Run:

```bash
python scripts/evaluate_gold_programs.py
```

The audit compares the project executor with an independent interpreter of
the upstream FinQA evaluator's arithmetic and table semantics. Its report
separates expected-answer mismatches from disagreements between the two
executors. The current command is restricted to train and validation. A prior one-off
executor QA run also inspected test labels, so do not describe the test split
as a pristine untouched holdout or use it for model-selection decisions.

## Evidence retrieval audit

Run the offline audit after preprocessing:

```bash
python scripts/audit_finqa_evidence.py
```

It verifies the raw and tokenized validation hashes against the preprocessing
manifest, then reports complete gold-evidence coverage and separate text/table
evidence recall. Gold annotations are used only by this audit, never by the
retriever or inference prompt. Review the report before interpreting model
errors.

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
- `GET /metrics` — Prometheus exposition for request counts and latency, labeled only by HTTP method, route template, and status code.

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

The `/metrics` endpoint excludes itself from request metrics and never labels by request ID, question, table contents, or other user-provided values.

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
export FINETUNEX_CHECKPOINT="$PWD/artifacts/checkpoints/<v4-checkpoint-name>"
export HF_CACHE_PATH="$HOME/.cache/huggingface"
docker compose up --build
```

The adapter and model cache are mounted read-only. The image runs in offline
Hugging Face mode and will fail at startup if the pinned model/tokenizer files
are not already present in the mounted cache. The service listens on port
8000; verify readiness with `curl http://localhost:8000/health/ready`.

Validation status of the container path: **not built or run** — the Docker daemon
is unavailable in the development environment, and the image targets NVIDIA CUDA,
which this Apple Silicon host cannot exercise. The image's `transformers` bound was
corrected to `>=5.22.1,<6.0` to match `pyproject.toml`, because the pipeline calls
`from_pretrained(dtype=...)`, a Transformers 5 argument name; the previous `<5.0`
bound would have failed at model load. A build-time assertion now checks the
installed majors. One divergence remains open: the base image supplies torch 2.6.0
while `pyproject.toml` requires `torch>=2.15,<3.0`, so the first real build must
confirm the CUDA base image can serve these checkpoints. All MPS results in this
repository come from running directly on the Mac, outside Docker.

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
- [x] Gold-program executor audit now restricted to train and validation; prior one-off QA also inspected test labels.
- [x] Versioned v3 reasoning prompt and controlled 32-example pilot.
- [x] Identified and reproduced the v3 training/inference evidence-budget mismatch.
- [x] Implemented v4 target-independent prompt budgeting across preprocessing, evaluation, and API.
- [x] Regenerate v4 train/validation artifacts and verify exact prompt parity on all 883 validation examples.
- [x] Train a fresh v4 LoRA adapter and run paired base/adapter evaluation after retrieval and baseline diagnostics.
- [x] Full 883-example paired validation comparison measured (base 0.68% vs 8-step
      adapter 10.19%). No model-quality claim is made; see the limitations below.
- [x] Offline gold-evidence recall audit tooling (full validation report pending).
- [x] FastAPI inference implementation, validated live against the v4 adapter on MPS
      (health, a real 44-row validation record, 422 rejection paths, `/metrics`).
- [x] CUDA Docker deployment definition (Linux NVIDIA build/runtime validation pending;
      never built — the dependency pin was corrected for Transformers 5.x, see
      `docs/EXPERIMENT_RESULTS.md`).
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
the v3 exact-after-five-decimals scoring rule, which recorded correct answers as
wrong when the dataset rendered `exe_ans` at six significant digits. They are
kept only as history. V4 preprocessing and full prompt parity are complete.

### v5 paired pilot under the corrected scoring rule (measured)

Re-measured on the identical deterministic 32-record validation subset
(`--limit 32 --sample-seed 42`, SHA-256 record-ID sampling), greedy generation,
`max_new_tokens=128`, evidence budget 1024, prompt budget 1408, on Apple
Silicon MPS. Both runs are scored with
`six_significant_digits_of_exe_ans@6` and are checked for identical prompts,
record order and input provenance by `scripts/compare_finqa_evaluations.py`.

| Metric | Base model | v4 LoRA pilot | Delta |
| --- | ---: | ---: | ---: |
| Parse success | 15/32 (46.88%) | 32/32 (100%) | +53.12 pp |
| Execution success | 8/32 (25.00%) | 32/32 (100%) | +75.00 pp |
| Correct answers | 0/32 (0%) | 4/32 (12.50%) | +12.50 pp |
| Evaluation wall time | 49.69 s | 38.84 s | — |

Evidence: `artifacts/evaluation/v5-base-32/`,
`artifacts/evaluation/v5-adapter-32/`, and
`artifacts/evaluation/v5-paired-comparison.json`. Full configuration, provenance
digests, statistical caveats and reproduction commands are in
`docs/EXPERIMENT_RESULTS.md`.

Scope limits, stated deliberately:

- This is an **8-optimizer-step pilot adapter** (32 examples x 2 epochs, gradient
  accumulation 8), not a trained model. It does not support a claim about
  FineTuneX's accuracy on FinQA.
- The pilot's base-model 0% understates the real baseline. On the complete 883-record
  validation split the base model is correct on 6 records (0.68%); the 32 hash-sampled
  records contained none of them. The pilot's +12.50 pp gain is therefore inflated
  relative to the full measurement's +9.51 pp.
- The base model's near-zero score is explained, not unexplained: it emits `add`
  exclusively among parsed outputs and separates multi-step programs with newlines
  instead of the dataset's commas, so most outputs fail the strict program parser. The
  evaluator marks parse and execution failures as incorrect rather than silently
  accepting them. Re-scoring all 480 full-run parse failures with a maximally lenient
  newline-to-comma parser recovers 329 parses and 250 executions but **zero** correct
  answers, so the strictness of the scorer is not what suppresses the baseline.
- All four adapter hits were inspected manually; each generated program is
  byte-identical to its gold program, so they are not tolerance artifacts.
- The accuracy gain is **not statistically established at n = 32**. The paired
  comparison is 4 records gained and 0 lost, exact McNemar two-sided p = 0.125.
  The formatting and executability gains are large and unambiguous; the
  numeric-reasoning gain is not yet separable from noise.

### v5 full 883-example paired validation comparison (measured)

Both models scored the complete official validation split in file order (`limit=None`,
no sampling), same prompts, same record order, same
`six_significant_digits_of_exe_ans@6` rule, greedy decoding on Apple Silicon MPS.
Evidence: `artifacts/evaluation/v5-base-full/`,
`artifacts/evaluation/v5-adapter-full/`,
`artifacts/evaluation/v5-paired-full-comparison.json`.

| Metric | Base model | v4 LoRA pilot adapter | Wilson 95% CI (base) | Wilson 95% CI (adapter) |
| --- | ---: | ---: | ---: | ---: |
| Parse success | 403/883 (45.64%) | 879/883 (99.55%) | [42.38%, 48.94%] | [98.84%, 99.82%] |
| Execution success | 247/883 (27.97%) | 876/883 (99.21%) | [25.11%, 31.02%] | [98.37%, 99.62%] |
| Correct answers | 6/883 (0.68%) | 90/883 (10.19%) | [0.31%, 1.47%] | [8.37%, 12.36%] |
| Failure classes | 480 parse / 156 execution / 241 wrong | 4 parse / 3 execution / 786 wrong | — | — |
| Evaluation wall time | 1,547.6 s | 1,277.5 s | — | — |

Paired: 90 records gained, **6 lost**, 0 shared hits; exact McNemar two-sided
p = 2.5e-20 on 96 discordants. At this scale the improvement is statistically
established, unlike the pilot.

What the gain is not: the adapter emits only `divide` and `subtract` across all 883
records — never `add`, `multiply`, `greater` or any `table_*` operation — and writes
programs of at most two steps. It scores 0/73 on records whose gold program has three
or more steps, 69/566 (12.2%) where gold ends in `divide`, and 1/56 (1.8%) where gold
ends in `add`. The 6 regressions are precisely the `add` records the base model got
right; the adapter applies its learned `subtract(a, b), divide(#0, b)` percent-change
template to questions asking for a total, once returning `-7.0` for the sum of two
positive reserves. 88 of its 90 correct answers come from records whose gold program
it could even express; only 2 of 206 outside that subset. So 10.19% measures output-format acquisition plus
coverage of the two most frequent operations, not learned financial reasoning, and
**no claim about FineTuneX's accuracy on FinQA is supported** — neither on the test
split nor at a realistic training budget. Full per-operation tables, the regression
list, manual inspection of all 7 adapter parse/execution failures, and the
checkpoint-selection overlap (3 of 90 adapter hits lie in the 32 records used for
checkpoint selection, at most 0.34 pp of optimism) are in
`docs/EXPERIMENT_RESULTS.md`.
