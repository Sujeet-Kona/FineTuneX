# Reference pipeline adoption

FineTuneX adopts the useful lifecycle and separation of concerns from
[Rishi625/LLM-Finetune-Pipeline](https://github.com/Rishi625/LLM-Finetune-Pipeline)
without copying its model, dataset, or hardware assumptions.

## Included stages

1. Dataset acquisition, validation, and preprocessing.
2. Config-driven LoRA fine-tuning.
3. Controlled base-model and adapter evaluation.
4. Adapter checkpointing and merged-model export.
5. FastAPI inference.
6. Docker Compose deployment.
7. Automated tests and linting.

## Deliberately omitted

- Kubernetes, Helm, Minikube, manifests, jobs, and autoscaling.
- bitsandbytes 4-bit QLoRA as a default: the primary development target is
  Apple Silicon MPS, where that CUDA-specific path is not supported.
- vLLM as the default serving backend: local inference uses Transformers.
- A second generic dataset/model configuration: FinQA and the pinned Qwen3
  revision remain the benchmark contract.

## Hardware profiles

- **Apple Silicon development:** run training and API directly on macOS using
  MPS; Docker Desktop does not expose the host MPS device to Linux containers.
- **NVIDIA Linux deployment:** use the Dockerfile and Compose GPU reservation.
- **CPU-only CI:** run tests, lint, configuration validation, and non-model
  utilities; do not claim this executes model training.

## Commands

See the repository Makefile. Training smoke runs are explicitly limited and
must not be reported as full training. Base and adapter generation evaluations
must use the same validation records and prompt hashes before comparison.
