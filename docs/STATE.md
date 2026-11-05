# FineTuneX State Report

**Inspection date:** 2026-11-05  
**Repository:** https://github.com/Sujeet-Kona/FineTuneX  
**Branch:** `fix/finqa-experiment-integrity` (PR #6, stacked on PR #5)

This report distinguishes source changes from executed validation. No local Mac dataset audit, model inference, or MPS training has been run by this inspection.

## Executive status

FineTuneX has a substantial end-to-end implementation, but it is **not complete**. The current work is concentrated on correctness of the FinQA parser/executor and annotation audit. No current base-model report, trained adapter, paired evaluation, or fresh-clone reproduction is available.

## Pull-request stack

All six PRs remain open, draft, and unmerged. Preserve this dependency order:

1. #1 `feat/base-model-benchmark`
2. #2 `feat/finqa-prompt-contract-v3`
3. #3 `feat/fastapi-inference`
4. #4 `feat/docker-deployment`
5. #5 `feat/inference-independent-prompt-budget`
6. #6 `fix/finqa-experiment-integrity` (current work)

Do not merge draft PRs out of order.

## Dataset and model pins

- Dataset: `bevaya/FinQA`
- Dataset revision: `3d6a736bc67e06bc15fbf3618d88204a57c5b25e` (40 hex characters; resolves on the Hub)
- Model: `Qwen/Qwen3-0.6B`
- Model revision: `c1899de289a04d12100db370d81485cdf75e47ca` (40 hex characters; resolves on the Hub)

The dataset card reports these official split sizes:

| Split | Expected records | Intended use |
|---|---:|---|
| Train | 6,251 | Supervised training |
| Validation | 883 | Development and model selection |
| Test | 1,147 | Final model evaluation after decisions are frozen |
| **Total** | **8,281** | |

The 7,134 figure is train plus validation, not the full dataset. Dataset preparation writes all three official splits. Tokenized preprocessing currently uses train and validation only.

The README records a prior one-off executor QA inspection of test labels. Therefore the test split is not a pristine, never-inspected holdout. Deterministic gold-program and annotation-integrity audits may read all splits, but the test split must not be used for prompt iteration, model selection, or hyperparameter tuning.

Actual local split counts and ID digests: **MISSING**.

## Implemented correctness repairs

1. **Annotation audit syntax:** the literal backslash-n source corruption was repaired.
2. **Step operation labels:** source labels `minus1-1` and `compare_larger1-4` are normalized to DSL operations `subtract` and `greater`.
3. **Step-result classification:** plain numeric step annotations are compared at displayed decimal precision; table-operation aliases are normalized in context. Percentage annotations that do not match ratio-to-percent semantics are marked ambiguous, not confirmed mismatches.
4. **Human answers:** answers are compared at displayed precision. Magnitude suffixes (thousand/million/billion) are not automatically rescaled; they require unit-context review.
5. **Parse-error handling:** malformed programs are reported as manual-review records rather than crashing the whole audit.
6. **Diagnostics:** audit JSON retains raw `program`, `program_re`, raw `steps`, parsed steps, intermediate results, and per-record flags.
7. **Official final-answer semantics:** only the computed final result is rounded to five decimals; that value is compared exactly with the unrounded `exe_ans`. The previous helper incorrectly rounded both sides.
8. **Flat-program parser:** argument separation now uses the official comma-space delimiter, preserving numeric literals such as `1,000`.
9. **Empty table rows:** `table_sum` of an empty row returns 0, matching the official evaluator. `table_max`, `table_min`, and `table_average` remain invalid for an empty row.
10. **Nested `program_re` audit:** the checker reconstructs an AST from the flat program and `#N` references, parses the nested expression, and reports consistent/mismatch/invalid/missing status. This is a structural comparison, not algebraic equivalence checking.

The official evaluator's table-row-reference branch contains an upstream defect: it reads `num_row` even when the reference branch did not assign it (or may reuse a previous value). The vendored test oracle preserves that behavior. The full gold-program differential will establish whether any official records exercise this edge case.

The vendored execution oracle is based on `czyssrs/FinQA/code/evaluate/evaluate.py`, upstream commit `dfc5b72c01ee17c442d28d5201b82a1f4e95d5af`, file blob `4829a4ca58a5725fb3f7a46b3b23b7f7e848a512`. Attribution is in `THIRD_PARTY_NOTICES.md`.

## Validation status

- Latest PR #6 CI run `41985206107`, commit `a41b85c2b709c1ce4d9984d264c34154bc38e240`: **success**.
- Local Mac validation at `a41b85c`: `ruff check .` passed, `ruff format --check .` passed, **341 tests passed**, 1 Starlette deprecation warning.
- Official executor differential: all 8,281 examples across train (6,251), validation (883), and test (1,147) matched the vendored official evaluator; zero differences.
- Full annotation audit generated locally: 8,281 examples; 6,559 internally consistent; 1,722 require manual review. Flag counts: 640 `exe_ans` disagreements, 479 cases where the human answer agrees despite an `exe_ans` disagreement, 893 human-answer/program disagreements, 626 confirmed step mismatches, 345 ambiguous step annotations, and zero unexecutable programs or `program_re` structural failures.
- Representative samples confirm the percentage fix for IPG and LLY while preserving the genuine ANSS program/answer mismatch. The former precision/format label has been replaced with the evidence-limited diagnostic name `exe_ans_disagreement_human_answer_consistent`; all 479 records remain manual-review cases.
- The full sample export and broader manual review are not yet complete. The full audit has not been frozen.
- No source data has been edited.

There are 23 test files in the current repository tree. CI tests are synthetic/offline tests; they do not establish correctness on the pinned downloaded dataset or MPS hardware.

## Remaining blockers

- Finish representative manual review: inspect at least 20 real records with raw program, raw `program_re`, raw steps, parsed steps, and computed values.
- Export at least 10 real examples per annotation category (or all if fewer) under `docs/audit_samples/`.
- Finalize `docs/DATA_POLICY.md` with the complete per-split counts and audit artifact provenance; current automatic exclusions remain zero.
- Verify raw-to-processed preservation, shared prompt byte equality, target/EOS masking, Qwen3 thinking/template behavior, and token-length percentiles.
- Validate the scorer with gold, corrupted, and garbage programs; manually inspect at least 20 base-model generations.
- Run the frozen validation baseline and only then the MPS LoRA pilot. No training has been run or claimed.
- Complete adapter packaging, deterministic inference, final comparison, and fresh-clone reproduction.

## Current policy

`docs/DATA_POLICY.md` defines all-record inclusion by default, no silent exclusions, official split use, and overall plus clean-subset reporting. Clean-subset counts are **MISSING** until the local audit runs.

**No model-quality claim is supported yet.** Historical v3 pilot numbers in the README are not current v4 results.
