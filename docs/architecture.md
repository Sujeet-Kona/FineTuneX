# FineTuneX — System Architecture

## 1. Project Objective

FineTuneX is a reproducible LLM fine-tuning, evaluation, and serving pipeline.

It takes a pretrained instruction model, fine-tunes it for a defined task using
parameter-efficient fine-tuning (LoRA), evaluates the result against the
original model, records experiment provenance, and exposes the validated model
through an inference API.

The project must demonstrate a complete and testable model lifecycle—not
merely a training script.

## 2. Initial Benchmark and Task

- Dataset: Banking77.
- Task: customer-support intent classification.
- Number of classes: 77.
- Input: a customer-support query.
- Output: exactly one label from the benchmark's permitted intent labels.

The benchmark's official test split must remain untouched.

The official training split will be divided into training and validation
subsets using stratified sampling and a fixed random seed.

The exact dataset source, revision, schema, and split integrity must be
verified during dataset implementation and recorded in the experiment
manifest.

Banking77 is a public customer-support benchmark. It is not proprietary
payment-provider data and is not a fraud-detection dataset.

## 3. System Workflow

1. Configuration
   - Model identifier and revision.
   - Dataset identifier and revision.
   - Random seed.
   - Training and evaluation parameters.
   - Hardware and environment metadata.

2. Dataset preparation
   - Load the verified dataset.
   - Validate required fields and labels.
   - Normalize text consistently.
   - Audit duplicates and split overlap.
   - Create a stratified train/validation split from official training data.
   - Preserve the official test split.
   - Tokenize and record dataset fingerprints.

3. Baseline evaluation
   - Run the pretrained model before fine-tuning.
   - Use a fixed prompt and output-parsing protocol.
   - Evaluate on the official test split.
   - Save predictions and metrics.

4. Fine-tuning
   - Load the selected pretrained model.
   - Apply LoRA adapters.
   - Train using the training split.
   - Use validation data for model and hyperparameter decisions.
   - Record training configuration, loss, runtime, and checkpoints.

5. Candidate evaluation
   - Evaluate the fine-tuned model on the same official test examples.
   - Use the same prompt, parser, and metrics as baseline evaluation.
   - Produce aggregate and per-class metrics.
   - Record errors and invalid predictions.
   - Do not claim improvement unless measured results support it.

6. Experiment and artifact management
   - Save adapter weights and tokenizer-related files as required.
   - Record base-model identifier and revision.
   - Record dataset fingerprint, seed, configuration, and package versions.
   - Associate evaluation results with the exact model artifact.
   - Record whether the candidate passed the promotion criteria.

7. Inference service
   - Load a selected, validated model artifact.
   - Expose a versioned FastAPI prediction endpoint.
   - Validate requests and model outputs.
   - Provide health and readiness endpoints.
   - Return the predicted intent and model/version metadata.
   - Do not fabricate confidence scores.

8. Release
   - Run unit and integration tests.
   - Build a Docker image.
   - Document setup, training, evaluation, and inference.
   - Provide a reproducible end-to-end workflow.

## 4. Evaluation Protocol

The evaluation pipeline will report:

- Accuracy.
- Macro F1.
- Weighted F1.
- Per-class precision, recall, and F1.
- Confusion matrix.
- Invalid prediction rate.
- Inference latency.

The baseline and fine-tuned model must use the same test examples,
prompting protocol, output parser, and metric implementation.

The official test split must not be used for training, prompt tuning,
hyperparameter selection, or model selection.

Training loss alone is not evidence of model improvement.

## 5. Model Output Contract

The model must return exactly one valid Banking77 intent label.

The output parser may normalize surrounding whitespace and label casing.
It must reject outputs that do not match the permitted label set.

Invalid outputs must be measured and reported, not silently converted into
valid predictions.

No confidence score will be returned unless a defensible confidence
estimation method is implemented and evaluated.

## 6. Technology

Core implementation:

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- PEFT / LoRA
- scikit-learn
- FastAPI
- Pydantic
- pytest
- Ruff
- Docker

Optional additions require a demonstrated need:

- QLoRA
- MLflow
- vLLM

The initial implementation will not depend on Kubernetes, Kafka,
distributed training, a frontend dashboard, or a microservice architecture.

## 7. Hardware and Model Feasibility

Development hardware: Apple Silicon MacBook Air with 16 GB unified memory.

Qwen3-0.6B is the initial model candidate, not a confirmed final selection.

Before locking the model, verify:

- Model and tokenizer load successfully.
- The model can generate a valid response.
- PyTorch detects and uses the available MPS device.
- PEFT can attach LoRA adapters to the model.
- A small training step completes.
- Memory use and runtime are acceptable.

Do not assume CUDA-specific quantization or QLoRA support is available on MPS.

If the candidate cannot be trained reliably on the target hardware, reassess
the model or training approach and document the decision.

## 8. Reproducibility Requirements

Every experiment must record:

- Model identifier and exact revision.
- Dataset identifier, revision, and fingerprint.
- Training, validation, and test split information.
- Random seed.
- Prompt and output-parser versions.
- Training and LoRA configuration.
- Python and relevant package versions.
- Hardware and device.
- Training duration and recorded loss.
- Adapter/artifact identity.
- Evaluation metrics and prediction outputs.
- Candidate validation and promotion outcome.

## 9. Quality Gates

A successful training run is not automatically a successful model release.

A candidate can be promoted only after:

1. Training completes successfully.
2. The artifact can be loaded.
3. Evaluation completes on the designated test set.
4. Predictions satisfy the output contract.
5. Results are compared against the baseline.
6. The agreed acceptance criteria are met.
7. The artifact and its provenance are recorded.

If the fine-tuned model performs worse, report that result honestly and do
not promote it as an improvement.

## 10. Explicitly Out of Scope

The initial project will not build:

- A RAG application or chatbot.
- An autonomous agent.
- A generic MLOps platform.
- Distributed or multi-node training.
- Kubernetes or Kafka infrastructure.
- A frontend dashboard.
- Unsupported claims about production scale or model quality.

## 11. Implementation Phases

| Phase | Deliverable |
|---|---|
| 0 | Hardware, model, and LoRA feasibility |
| 1 | Repository and configuration foundation |
| 2 | Verified Banking77 dataset pipeline |
| 3 | Pretrained-model baseline evaluation |
| 4 | LoRA fine-tuning |
| 5 | Fine-tuned evaluation and error analysis |
| 6 | Experiment tracking and artifact lifecycle |
| 7 | FastAPI inference service |
| 8 | Docker, integration tests, and reproducible release |

## 12. Decisions Still Requiring Validation

- Final base-model selection.
- Exact model and tokenizer revisions.
- Compatible PyTorch, Transformers, and PEFT versions.
- MPS training stability and memory requirements.
- Final training hyperparameters.
- Model promotion thresholds.

These decisions must be based on experiments, not assumptions.
