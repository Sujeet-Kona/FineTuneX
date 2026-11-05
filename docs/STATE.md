# FineTuneX State Report

**Inspection date:** 2026-11-05  
**Repository:** https://github.com/Sujeet-Kona/FineTuneX  
**Inspected branch:** `fix/finqa-experiment-integrity`  
**Inspected PR #6 head before this document:** `ed086ec837894f93b32b25560e6b967d194c38ac`  
**Base branch:** `feat/inference-independent-prompt-budget` (`6d3ca14a4c8c96ddfd187124976624661a2d31c0`)

This is a read-only engineering assessment of the repository and its open pull-request stack, plus this state document. It does not claim that local model, dataset, or MPS workloads were executed.

## Executive status

FineTuneX has substantial pipeline code: source-data preparation, evidence selection, tokenized preprocessing, LoRA training, generation, program parsing/execution, evaluation, checkpoint handling, API serving, and a CUDA-oriented Docker definition.

**The project is not complete and is not currently validated end to end.** The latest completed CI run for the inspected code failed during Ruff parsing, before formatting or pytest. No committed dataset, tokenized artifacts, model checkpoints, predictions, or experiment reports are present in the inspected repository tree. Local test execution and all model/data workloads are therefore **MISSING** from this inspection.

## Pull-request stack

All six PRs are open, draft, and unmerged at inspection time. Preserve this dependency order; do not merge out of order.

| PR | Branch | Base | Purpose | Validation status recorded in PR |
|---|---|---|---|---|
| #1 | `feat/base-model-benchmark` | `main` | Base-model evaluation and paired comparison | Local Ruff/pytest and actual base/adapter evaluation required |
| #2 | `feat/finqa-prompt-contract-v3` | #1 branch | Explicit FinQA program-generation prompt contract | Local validation and MPS pilot required |
| #3 | `feat/fastapi-inference` | #2 branch | FastAPI inference using shared generation/execution path | Local tests and checkpoint-backed API smoke test required |
| #4 | `feat/docker-deployment` | #3 branch | NVIDIA CUDA container deployment | Linux NVIDIA build/runtime not run; not an MPS deployment |
| #5 | `feat/inference-independent-prompt-budget` | #4 branch | Target-independent shared prompt budget and v4 prompt contract | Full preprocessing and 883-record prompt parity required locally |
| #6 | `fix/finqa-experiment-integrity` | #5 branch | Evidence audit, gold executor audit, annotation-integrity diagnostics | Current CI failure; local audits still required |

PR #6 currently contains 19 changed files and includes the prior pipeline through its base branch. Do not treat the merge-commit metadata field as evidence that a draft PR has merged; GitHub reports each PR as `merged: false`.

## Dataset and split accounting

The pinned Hub dataset is `bevaya/FinQA`. The exact revision in `configs/dataset.yaml` is:

`3d6a736bc67e06bc15fbf3618d88204a57c5b25e` (40 hexadecimal characters).

The Hugging Face commit page resolves this exact revision:
https://huggingface.co/datasets/bevaya/FinQA/commit/3d6a736bc67e06bc15fbf3618d88204a57c5b25e

The dataset card reports these official split counts:

| Official split | Records | Intended use |
|---|---:|---|
| Train | 6,251 | Supervised training and train-side annotation diagnostics |
| Validation | 883 | Development, model selection, baseline/adapter comparison |
| Test | 1,147 | Final evaluation only, after decisions are frozen |
| **Total** | **8,281** | All official records |

The apparent 7,134 versus 8,281 discrepancy is arithmetic, not a dataset-count conflict: 6,251 train + 883 validation = 7,134. The README and dataset config correctly list 8,281 as the full total. Current tokenized preprocessing is configured for train and validation only (7,134); dataset preparation expects all three official splits. Do not silently repartition or drop the test split.

The README currently states that an earlier one-off executor QA run inspected test labels. Therefore the test split must not be described as a pristine, never-inspected holdout. Keep it out of model selection and development diagnostics; disclose the prior label inspection in the final limitations.

The source dataset is not committed in this repository. The remote tree contains only storage placeholders under `data/raw` and `data/processed`.

## Model and hardware configuration

- Model: `Qwen/Qwen3-0.6B`.
- Pinned model revision: `c1899de289a04d12100db370d81485cdf75e47ca` (40 characters); the Hugging Face revision page resolves.
- Tokenizer uses the same model revision.
- Current configured dtype: `float16`.
- Quantization: none.
- Training method: LoRA, rank 8, alpha 16, dropout 0.05.
- Configured LoRA targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- Configured sequence length: 1,536; prompt ceiling: 1,408; generation reserve: 128.
- Hardware target: Apple M4, 16 GB unified memory, MPS.

The repository code explicitly rejects bfloat16 on MPS in its current training loader. The configured float16 path has not been verified on the user's Mac in this inspection. No MPS training, memory measurement, one-batch overfit, or model generation is confirmed here. QLoRA/bitsandbytes is not configured and should not be introduced for MPS.

## Existing implementation inventory

| Area | Present in repository | Verified status |
|---|---|---|
| Dataset loading and manifest | `finqa_pipeline.py`, `prepare_dataset.py` | Source reviewed; local data load missing |
| Evidence retrieval | `finqa_retrieval.py` | Source reviewed; full validation audit report not present |
| Shared preprocessing/prompt construction | `finqa_preprocessing.py`, `inference/prompt.py` | Implementation present; full prompt parity not independently rerun here |
| Tokenized artifact generation | `finqa_artifacts.py`, `preprocess_finqa.py` | Implementation present; artifacts missing from remote tree |
| FinQA parser/executor | `finqa_parser.py`, `finqa_executor.py` | Implementation and unit tests present; differential equivalence to official evaluator not established |
| Reference executor | `finqa_reference.py` | A separately written reference-style port exists; it is not a substitute for importing/running the official evaluator source |
| Gold-program audit | `evaluate_gold_programs.py` | Script exists; local full-split report is not committed |
| Annotation integrity audit | `finqa_data_integrity.py`, `audit_finqa_annotations.py` | Currently syntactically broken; its aggregate output is not trustworthy |
| Base/adapter evaluation | `evaluate_finqa_model.py` | Script exists; no current v4 base/adapter reports in repository |
| Paired comparison | `compare_finqa_evaluations.py` | Script and tests exist; real paired runs missing |
| LoRA trainer | `training/model.py`, `training/engine.py`, `train_finqa.py` | Implementation present; no verified current v4 training run |
| Checkpoint packaging | `artifacts/checkpoint.py` | Implementation present; real v4 adapter package missing |
| FastAPI | `api/app.py` | Implementation and stub tests present; real checkpoint-backed smoke test missing |
| Docker | `Dockerfile`, `compose.yaml` | CUDA-oriented definition present; build/runtime validation missing |

## Confirmed defects and validation blockers

1. **Confirmed: annotation-audit module syntax error.** At the inspected head, `src/finetunex/evaluation/finqa_data_integrity.py` contains literal backslash-n sequences inside Python expressions around the operation-match expression (reported at line 88) and the policy string (reported at line 272). These are source text, not line breaks.
2. **Confirmed: latest completed CI did not reach pytest.** Run `41979466756` for commit `38bc78b24c6a6e3e5afa41c7b20e11d92d035bab` failed in `ruff check .`. Ruff reported invalid syntax in the annotation-audit module. Formatting and pytest steps were not reached.
3. **Confirmed: annotation audit's previous 7,134/7,134 step-disagreement aggregate is not accepted as evidence.** The checker implementation has not passed its own syntax/lint gate and has not been validated against a hand-verified fixture corpus. Treat all earlier aggregates as untrusted until repaired and independently checked.
4. **Confirmed: the current gold audit does not yet satisfy the requested official differential test.** The repository contains a separate reference-style implementation, but the official FinQA evaluator source has not yet been wired into a differential run across all gold programs.
5. **Confirmed: the remote repository does not contain the generated evidence audit, annotation audit, gold audit, preprocessing artifacts, base-model predictions, adapter predictions, or a trained checkpoint.** These outputs are missing from the inspected Git tree.
6. **Confirmed: CI dependency installation completes before lint.** The latest run installed the declared extras and then failed at Ruff parsing. This says nothing about runtime correctness on MPS.
7. **Historical README metrics are not current v4 results.** README labels the v3 32-example base/adapter pilot as historical and superseded. No v4 base-versus-adapter comparison is present.

## Tests and CI

- Test files present: 22 under `tests/`.
- Latest inspected completed CI run: `41979466756`, conclusion `failure`.
- Failure stage: `ruff check .`.
- `ruff format --check .`: **NOT REACHED** in that run.
- `pytest -q`: **NOT REACHED** in that run.
- Local `ruff check`, `ruff format --check`, and full `pytest` on the user's Mac: **MISSING**.
- Model/dataset integration tests and MPS workloads: **MISSING**.

A later run was triggered by the metadata/documentation commits made while orienting; at the time of writing its result is not yet confirmed. Do not infer success from an in-progress run.

## Required next gates

1. Repair only the annotation-audit syntax and add/verify fixture coverage; run Ruff and pytest before interpreting audit output.
2. Fetch the official FinQA evaluator source and build an independent differential test over every train/validation gold program. Preserve its arithmetic and rounding semantics.
3. Inspect approximately 20 raw records with raw program, raw steps, parsed steps, and independently computed results. Validate the audit on hand-verified fixtures before rerunning aggregates.
4. Produce separate, inspectable annotation categories and samples; then document the no-silent-exclusion data policy.
5. Verify raw-to-processed field preservation, shared prompt byte equality, target/EOS masking, Qwen3 thinking/template behavior, and full-dataset token-length distributions.
6. Validate the scorer on gold, corrupted, and garbage programs; only then run a fixed, frozen base-model validation sample and inspect raw outputs.
7. Run the MPS pilot only after all pre-flight gates pass. Do not increase training scale unless measured learning beats the baseline under identical evaluation.
8. Package and reproduce the adapter only after a real training run and paired evaluation.
9. Complete final CI and a fresh-clone reproduction. Keep all PRs draft until the stack is safe and the evidence is complete.

## Explicitly missing

- User-Mac full test results.
- Verified official-executor differential report.
- Trusted annotation-integrity aggregates and reviewed sample files.
- `docs/DATA_POLICY.md`, `docs/EVAL_SPEC.md`, and `docs/RESULTS.md`.
- Token-length percentile report and MPS memory feasibility evidence.
- Verified v4 base-model evaluation report and manually reviewed generations.
- Verified v4 LoRA pilot, loss curves, overfit test, and paired comparison.
- Adapter metadata/hash validation and deterministic inference reproduction.
- Real API smoke test, Docker runtime test, and fresh-clone reproduction.

**No source data was changed by this inspection. No model training or model evaluation was run.**
