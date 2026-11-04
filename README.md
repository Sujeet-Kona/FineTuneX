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
3. Base-model evaluation.
4. Parameter-efficient LoRA fine-tuning.
5. Structured reasoning-program generation.
6. Deterministic program validation and execution.
7. Baseline comparison and error analysis.
8. Adapter packaging and provenance.
9. FastAPI inference.
10. Docker and automated testing.

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
- [ ] Model feasibility and base-model baseline.
- [ ] LoRA fine-tuning and comparative evaluation.
- [ ] Artifact packaging and inference API.
