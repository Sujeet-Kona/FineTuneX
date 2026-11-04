# FineTuneX

**Reproducible Financial Reasoning LLM Fine-Tuning and Evaluation Pipeline**

FineTuneX explores the lifecycle of adapting a pretrained language model
to numerical reasoning over financial reports.

## Benchmark

FinQA — a financial question-answering benchmark containing report
context, tables, numerical questions and annotated reasoning programs.

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
python scripts/evaluate_finqa_model.py \
  --base-model \
  --limit 32 \
  --max-new-tokens 128 \
  --evidence-budget 1024 \
  --output-dir artifacts/evaluation/base-chat-32
```

Run the adapter with the same evaluation settings:

```bash
python scripts/evaluate_finqa_model.py \
  --checkpoint artifacts/checkpoints/pilot-chat-32 \
  --limit 32 \
  --max-new-tokens 128 \
  --evidence-budget 1024 \
  --output-dir artifacts/evaluation/pilot-chat-32
```

Do not compare the runs unless their ordered record IDs and per-record
`prompt_sha256` values match. A 32-example prefix is a smoke benchmark, not
a full validation result and not a test-set result. Keep the test split held
out until model and evaluation decisions are frozen.

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
- [x] Model feasibility and base-model evaluation path.\n- [ ] Measured base-model benchmark and controlled adapter comparison.
- [x] LoRA pilot training and adapter inference.\n- [ ] Full-data LoRA training and comparative evaluation.
- [ ] Artifact packaging and inference API.
