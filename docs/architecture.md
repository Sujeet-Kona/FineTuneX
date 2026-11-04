# FineTuneX — System Architecture

## 1. Objective

FineTuneX is a reproducible financial-reasoning LLM fine-tuning,
evaluation and inference pipeline.

It adapts a pretrained language model using LoRA, evaluates its
reasoning programs through a deterministic executor, records
experiment provenance and serves validated predictions through an API.

## 2. Benchmark

- Dataset: FinQA.
- Source: `bevaya/FinQA`.
- Revision: `3d6a736bc67e06bc15fbf3618d88204a57c5b25e`.
- Task: numerical reasoning over financial reports.
- Input: report context, financial table and natural-language question.
- Target: structured FinQA reasoning program.
- Execution reference: `exe_ans`.

Official splits:
- Train: 6,251.
- Validation: 883.
- Test: 1,147.

The official splits must remain unchanged.

## 3. Workflow

1. Configuration and reproducibility.
2. Dataset ingestion and validation.
3. Financial context and table preprocessing.
4. Base-model evaluation.
5. LoRA fine-tuning.
6. Structured program parsing and validation.
7. Deterministic program execution.
8. Comparative evaluation and error analysis.
9. Adapter packaging and experiment provenance.
10. FastAPI inference.
11. Docker delivery and automated testing.

## 4. Dataset Contract

Preserve all original FinQA fields.

The initial training target is `program`.

Keep `program_re`, `steps`, `answer`, `exe_ans`, `explanation`,
and evidence annotations available for validation and analysis.

Do not silently remove blank answers, duplicates or annotation
inconsistencies. Report them in the dataset manifest.

## 5. Reasoning Program Execution

Implement the FinQA operation semantics using a restricted,
deterministic interpreter.

Supported operation families include:

- add
- subtract
- multiply
- divide
- exp
- greater
- table_sum
- table_average
- table_max
- table_min

Support intermediate references such as `#0`, percentage values
and the constants defined by the original FinQA evaluator.

Do not use Python `eval` or execute generated source code.

The executor must be tested against the original FinQA evaluation
semantics before it is used for model evaluation.

## 6. Evaluation

Primary metric:
- Execution accuracy.

Additional metrics:
- Program accuracy.
- Program validity rate.
- Numerical answer accuracy.
- Error category counts.
- Inference latency.

Use the same examples and evaluator for base and fine-tuned models.

The official test split is reserved for final evaluation.
Validation data is used for development and model selection.

## 7. Model and Hardware

Development hardware: Apple Silicon MacBook Air, 16 GB unified memory.

Initial model candidate: Qwen3-0.6B, not yet locked.

Validate model loading, generation, MPS compatibility, PEFT
compatibility, training-step completion, memory usage and runtime
before finalizing the model configuration.

Do not assume CUDA-specific QLoRA support on Apple Silicon.

## 8. Technology

- Python
- PyTorch
- Hugging Face Transformers and Datasets
- PEFT / LoRA
- scikit-learn
- FastAPI and Pydantic
- pytest and Ruff
- Docker

Optional technologies require a demonstrated need.

## 9. Experiment Provenance

Record:

- Model identifier and revision.
- Dataset repository and revision.
- Dataset fingerprints and processed-file hashes.
- Train and validation configuration.
- Random seed.
- Tokenizer and prompt versions.
- LoRA configuration.
- Python and dependency versions.
- Hardware and device.
- Training runtime and loss.
- Adapter artifact identity.
- Evaluation metrics and prediction outputs.

## 10. Promotion Criteria

Training completion does not imply model promotion.

A candidate must:
1. Load successfully.
2. Generate parseable reasoning programs.
3. Execute safely.
4. Complete evaluation.
5. Be compared against the base model.
6. Meet predefined acceptance criteria.
7. Have its artifact and provenance recorded.

Report regressions honestly.

## 11. Out of Scope

- RAG applications.
- Chatbot frontend.
- Fraud classification.
- Autonomous agents.
- Distributed training.
- Kubernetes and Kafka.
- Unnecessary microservices.
- Unsupported production-scale claims.

## 12. Implementation Phases

| Phase | Deliverable |
|---|---|
| 0 | Hardware and model feasibility |
| 1 | Repository foundation |
| 2 | FinQA dataset pipeline |
| 3 | Official program executor and tests |
| 4 | Base-model baseline |
| 5 | LoRA fine-tuning |
| 6 | Evaluation and error analysis |
| 7 | Experiment and artifact management |
| 8 | FastAPI inference |
| 9 | Docker and reproducible release |
