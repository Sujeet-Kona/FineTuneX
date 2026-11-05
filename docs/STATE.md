# FineTuneX State Report

**Inspection date:** 2026-11-05  
**Repository:** https://github.com/Sujeet-Kona/FineTuneX  
**Branch:** `fix/finqa-experiment-integrity` (PR #6, stacked on PR #5)

This report distinguishes source changes from executed validation. The user has run the local annotation audit and validation commands; no base-model inference or MPS training has been run.

> **Superseded in part on 2026-11-05.** This report describes the PR #6 audit
> branch. Current work is on `feat/reference-pipeline-without-kubernetes`. The
> statement above that no base-model inference or MPS training has been run, the
> `MISSING` split digests below, and the claim that no base-model report, trained
> adapter, paired evaluation or fresh-clone reproduction is available, are no longer
> accurate: MPS training, base and adapter evaluations, the export and serving checks,
> and a clean-clone data/tokenization reproduction have since been executed. See
> `docs/EXPERIMENT_RESULTS.md` for the measured record.

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

Actual local split counts and ID digests, measured 2026-11-05 from
`data/processed/finqa/`. Counts match the expected values above, with no duplicate
IDs in any split. Digests are SHA-256 over the ordered JSON list of record IDs.

| Split | Records | Unique IDs | Ordered-ID SHA-256 |
|---|---:|---:|---|
| Train | 6,251 | 6,251 | `8bb63636e7bf2e0cd0f195ef0928eadb6f9d52ae2c6781614975ddb79ff45e35` |
| Validation | 883 | 883 | `7c8aad37cf933545beeed0ed35080eb42b3988e060a122366e2442af098a3d2e` |
| Test | 1,147 | 1,147 | `f0d1e2a743b3d6a6b155e15d63e79f8064574a28e0686c7fba90ef41245b30b6` |

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
- Local Mac validation at `15f352c` on `feat/reference-pipeline-without-kubernetes`, 2026-11-05: `ruff check .` passed, `ruff format --check .` passed (78 files), **428 tests passed**, same single Starlette deprecation warning.
- Official executor differential: all 8,281 examples across train (6,251), validation (883), and test (1,147) matched the vendored official evaluator; zero differences.
- Full annotation audit, **regenerated 2026-11-05 under the current scoring rule** (`experiments/finqa_annotation_integrity_v2_sixsig.json`): 8,281 examples; 7,011 internally consistent; 1,270 require manual review. Flag counts: 0 `exe_ans` disagreements, 0 `exe_ans_disagreement_human_answer_consistent`, 893 human-answer/program disagreements, 626 confirmed step mismatches, 345 ambiguous step annotations, 213 human answers needing parsing review, and zero unexecutable programs or `program_re` structural failures.
- Superseded: the earlier report (`experiments/finqa_annotation_integrity.json`, written 07:13 UTC before the fix landed) reported 6,559 consistent / 1,722 flagged including 640 `exe_ans` disagreements and 479 `exe_ans_disagreement_human_answer_consistent`. All 640 were precision artifacts of the pre-fix comparison; they are 0 under the current rule. Both reports are retained on disk; the older one is not evidence for the current rule.
- Representative samples confirm the percentage fix for IPG and LLY. `ANSS/2012/page_92.pdf-1` remains `requires_manual_review`, but for a narrower reason than previously recorded: its program result 192501.5 now matches `exe_ans` `192502` at six significant digits, while the human answer `91000` disagrees with the program (`program_human_answer_disagreement`). The `exe_ans_disagreement_human_answer_consistent` diagnostic no longer fires on any record.
- The audit sample export under `docs/audit_samples/` (10 per non-empty category, 50 records, untracked and preserved) exists and its manifest source hashes match the on-disk splits exactly (train `891f7f73…`, validation `8e564eb0…`, test `35b9b2c0…`), with all 50 audit/source joins consistent. It was exported from the stale pre-fix report, so its two `exe_ans` categories are no longer live review queues; the files are left unmodified rather than re-exported over.
- No source data has been edited.
- Full 883-example paired base-vs-adapter evaluation on MPS, 2026-11-05, working tree
  at `15f352c`: base parse 403/883, execution 247/883, correct 6/883 (0.68%); the
  8-optimizer-step v4 adapter parse 879/883, execution 876/883, correct 90/883
  (10.19%). Identical prompts, record order, expected answers and scoring rule were
  asserted by `scripts/compare_finqa_evaluations.py`. McNemar: 90 gained, 6 lost, exact
  two-sided p = 2.5e-20. The adapter emits only `divide` and `subtract` and scores 0/73
  on gold programs of three or more steps, so this remains a pipeline measurement and
  not a model-quality claim. Details, the regression list and per-failure inspection are
  in `docs/EXPERIMENT_RESULTS.md`.
- Scorer behaviour was exercised against real generations at scale rather than only
  synthetic tests: each of the 883 base and 883 adapter outputs is classified into
  exactly one of `parse_failed` / `execution_failed` / `wrong_answer`, and the 6 base
  hits, all 4 parse plus 3 execution adapter failures, and the 2 adapter answers that
  match `exe_ans` while the program differs from gold were inspected individually. A
  lenient re-scoring probe over all 480 base parse failures recovered 329 parses and 250
  executions but zero additional correct answers, confirming the strict parser is not
  suppressing the baseline.
- Experiment tracking was exercised against **real MLflow** (3.18.0) for the first time
  on 2026-11-05, which exposed a shipping defect: `configs/experiment.yaml` used
  `tracking_uri: mlruns`, and MLflow 3.x refuses the file-store backend outright, so
  enabling tracking could not work under any version allowed by the
  `mlflow>=2.20,<4.0` pin. The default is now `sqlite:///mlruns.db`, two tests cover the
  enabled path against the installed library (skipped when absent), and a real 2-batch
  MPS training run with tracking enabled completed with run status FINISHED, 11
  parameters, epoch metrics and all three artifact groups in the store. The config was
  restored to `enabled: false` afterwards. Training numbers matched the same run
  without tracking.
- Export parity and serving were re-checked after that environment change: the merged
  model and the adapter emit identical text **and identical token IDs** on a real
  validation record, and the FastAPI app still returns 200/200 on health, 200 with an
  executed program for a real record, 422 for injected gold / empty question / ragged
  table, and low-cardinality `finetunex_*` metrics only.
- Docker image: **NOT BUILT**. No Docker daemon is available on this machine and the
  base image is CUDA/NVIDIA-only (`pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`) on an
  arm64 Apple Silicon host. The Transformers pin inside the Dockerfile was corrected to
  `>=5.22.1,<6.0` (with a major-version assertion) because the code calls
  `from_pretrained(dtype=...)`, which requires Transformers 5.x; the residual torch
  divergence (image pins 2.6.0, `pyproject.toml` allows `>=2.15,<3.0`) is documented and
  unresolved.
- Quality gates at the current working tree: `ruff check .` passed, `ruff format --check .`
  passed (79 files), **430 tests passed** (428 before the two real-MLflow tests were
  added), 2 warnings (the Starlette deprecation plus an SQLAlchemy deprecation raised
  inside MLflow's own store).

There are 26 test files in the current repository tree. CI tests are synthetic/offline tests; they do not establish correctness on the pinned downloaded dataset or MPS hardware. The full 883-example MPS runs and the 7,134-record gold audit are the checks that exercise real data and real hardware.

## Remaining blockers

- ~~Finish representative manual review: inspect at least 20 real records with raw
  program, raw `program_re`, raw steps, parsed steps, and computed values.~~ Done on
  2026-11-05: 20 of the 50 exported samples (4 per non-empty category) were inspected
  with every field the blocker names present, including per-step computed values, and
  the findings are recorded in `docs/EXPERIMENT_RESULTS.md`. Broader review of the
  remaining 1,270 flagged records is still open.
- ~~Run `scripts/export_finqa_audit_samples.py` to export at least 10 real examples per
  annotation category (or all if fewer) under `docs/audit_samples/`; inspect the manifest
  and sample files.~~ Done: `docs/audit_samples/` holds 50 samples across the 5
  non-empty categories (the other 4 have zero available records), the manifest's three
  source-split SHA-256 values were re-verified against the files on disk, and all 50
  audit/source joins check out. Caveat: the export was produced from the pre-fix audit
  report, so its two `exe_ans` categories no longer exist under the current rule; the
  files are preserved unmodified rather than regenerated over the user's evidence.
- ~~Finalize `docs/DATA_POLICY.md` with the complete per-split counts and audit artifact provenance; current automatic exclusions remain zero.~~ Done on 2026-11-05: materialized counts (6,251 / 883 / 1,147) verified against the card, per-split file digests and ordered-record-ID digests recorded, and the current vs superseded audit reports identified. Automatic exclusions remain zero. Clean-subset counts are still missing (see Current policy).
- Verify raw-to-processed preservation, shared prompt byte equality, target/EOS masking, Qwen3 thinking/template behavior, and token-length percentiles.
- ~~Validate the scorer with gold, corrupted, and garbage programs; manually inspect at
  least 20 base-model generations.~~ Done on 2026-11-05: gold programs execute and match
  `exe_ans` on 7,134/7,134 train+validation records, the official-executor differential
  is 0 differences over 8,281, in-tree tests cover corrupted and garbage programs and
  assert that a parse failure can never score correct, and the full 883-record base run
  was inspected beyond the 20 recorded `failure_examples` (all 6 hits individually, all
  480 parse failures classified by shape, all 156 execution failures). See
  `docs/EXPERIMENT_RESULTS.md`.
- ~~Run the frozen validation baseline and only then the MPS LoRA pilot.~~ Done on
  2026-11-05: a base-model benchmark and a 32-example MPS LoRA pilot adapter exist,
  and both were evaluated under the corrected scoring rule. The pilot used 8
  optimizer steps on 32 examples, so it is a pipeline check, not a model-quality
  result. See `docs/EXPERIMENT_RESULTS.md`.
- ~~Run the full 883-example validation comparison.~~ Done on 2026-11-05 (numbers in
  Validation status above). Still open: no test-split run, and no adapter trained at a
  realistic budget, so no model-quality claim exists.
- Adapter packaging, deterministic inference and the final comparison are done and
  evidenced. ~~Fresh-clone reproduction is NOT RUN~~ Done on 2026-11-05: a clean
  `file://` clone of `5e8ea29` re-ran the test suite, `prepare_dataset.py` and
  `preprocess_finqa.py`, and produced byte-identical materialized splits and
  byte-identical tokenized train/validation artifacts. The tokenized `manifest.json`
  hash does differ, and the difference was isolated to 12 audit counters that only the
  newer script revision writes, plus the one digest key that covers them. Inference was
  not re-run in the clone. `artifacts/` and `experiments/` evidence remains gitignored,
  so it still does not travel with the repository. See the Fresh-clone reproduction
  section of `docs/EXPERIMENT_RESULTS.md`.

## Current policy

`docs/DATA_POLICY.md` defines all-record inclusion by default, no silent exclusions, official split use, and overall plus clean-subset reporting. Clean-subset counts are **MISSING** until records have been manually adjudicated and the clean-subset calculation is implemented.

**No model-quality claim is supported yet.** Historical v3 pilot numbers in the README are not current v4 results.
