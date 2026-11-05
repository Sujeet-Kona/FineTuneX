# FinQA Data Policy

**Status:** Policy defined; local full-split audit has been generated. Representative-record export and manual review are pending.  
**Dataset:** `bevaya/FinQA`  
**Pinned revision:** `3d6a736bc67e06bc15fbf3618d88204a57c5b25e`

## Source-of-truth and immutability

- The pinned Hugging Face dataset revision is the source of truth.
- Raw source files are immutable inputs. FineTuneX may create manifests, JSONL conversions, tokenized datasets, reports, and model artifacts, but must never rewrite or patch source records.
- Every derived dataset manifest must record the source revision, split, record count, ordered ID digest, and source-file hash where available.
- A preprocessing or audit error must stop the run with a record ID and reason. It must not silently skip a row.

## Official split use

The dataset card reports the following official split sizes. The local materialized
counts were verified on 2026-11-05 against `data/processed/finqa/*.jsonl` and match the
card exactly, with no records added, dropped or moved between splits.

| Split | Expected records | Local records | SHA-256 over ordered record IDs |
|---|---:|---:|---|
| Train | 6,251 | 6,251 | `8bb63636e7bf2e0cd0f195ef0928eadb6f9d52ae2c6781614975ddb79ff45e35` |
| Validation | 883 | 883 | `7c8aad37cf933545beeed0ed35080eb42b3988e060a122366e2442af098a3d2e` |
| Test | 1,147 | 1,147 | `f0d1e2a743b3d6a6b155e15d63e79f8064574a28e0686c7fba90ef41245b30b6` |
| **Total** | **8,281** | **8,281** | |

Per-split file digests, as recorded by the integrity audit and the audit-sample manifest
(both agree):

| Split | SHA-256 of the materialized split file |
|---|---|
| Train | `891f7f736f56a8dacee631a1c51d5e19f5d99651d3f91e8d08799dbb79e8e657` |
| Validation | `8e564eb0f0cb83957f737babd0631c3ede0d6185649803b524c9df5fc65c790a` |
| Test | `35b9b2c0265784d1a807280eff9fd28a910ea3edb0a4296532a05b1a7340cf62` |

Audit artifact provenance: `experiments/finqa_annotation_integrity_v2_sixsig.json` is the
current annotation-integrity report (8,281 examples; 7,011 internally consistent; 1,270
flagged; 0 `exe_ans` disagreements) and was generated with the shipped
`six_significant_digits_of_exe_ans@6` comparison. The earlier
`experiments/finqa_annotation_integrity.json` is retained but superseded — it predates
that fix and its 640 `exe_ans` disagreement flags were precision artifacts. Both reports
are local, gitignored evidence; `docs/audit_samples/` holds 50 exported samples (10 per
non-empty category) drawn proportionally from all three splits and was exported from the
superseded report, so its two `exe_ans` categories are not live review queues.

Permitted use by split is unchanged: train for supervised training and training-side
diagnostics; validation for development, prompt and scorer debugging, and baseline and
adapter selection; test for the final report only, after all decisions and
configurations are frozen.

The 7,134 figure is train plus validation (6,251 + 883), not the total dataset. Keep the official splits; do not repartition or move examples between them.

The repository README records an earlier one-off executor QA inspection of test labels. Accordingly, the test split must not be represented as a pristine, never-inspected holdout. It remains reserved for final evaluation and must not be used for model selection or iterative debugging.

## Inclusion and exclusion policy

**Default automatic exclusions: zero.**

- Keep every record in its official split, including records with questionable answers, malformed annotations, unexecutable gold programs, or disagreement between `answer`, `exe_ans`, `program`, `program_re`, and `steps`.
- Annotation flags are diagnostics, not permission to remove training examples.
- If a required field is missing or malformed such that preprocessing cannot construct the agreed input/target, fail the preprocessing run and report the affected IDs. Do not continue with a smaller dataset.
- Any future exclusion requires a separately reviewed policy change, a reason code, a count, and a complete list of affected IDs. The exclusion must be visible in the manifest and all reports.
- No source row may be altered to make it pass validation.

## Diagnostic clean subset

For transparent secondary metrics, FineTuneX will report both **overall** results (all records in the evaluated official split) and a **clean diagnostic subset**. A record belongs to the clean subset only when all of the following are independently verified:

1. The gold program parses and executes under the official FinQA evaluator.
2. Its final result matches `exe_ans` using the official five-decimal final-result semantics.
3. Its `program_re` nested expression agrees structurally with the expression reconstructed from the flat `program` and `#N` references.
4. Its recorded `steps.op`, `arg1`, `arg2`, and `res` agree with the linear gold program. Table-operation aliases are normalized in context. Plain numeric results are rounded to their explicitly displayed decimal precision. Percentage annotations are compared using ratio-to-percent semantics; when that interpretation does not match, the record is marked ambiguous rather than assigned a guessed scale or called a confirmed mismatch.
5. The human-readable `answer` can be parsed unambiguously under a documented normalization rule and agrees with the program result at its displayed precision. For an answer with `%`, compare both a ratio-valued program result and a percentage-point program result; the program may already contain an explicit multiply-by-100 step. For an unmarked numeric answer, consider percentage-point interpretation only when the question or recorded step results explicitly establish percentage context. Magnitude suffixes (thousand/million/billion) require unit context and are not automatically rescaled.

Ambiguous human answers and step percentage conventions are not guessed. Such examples remain in the overall set but are not counted in the clean diagnostic subset until manually adjudicated. Human-answer percentage comparison accepts either ratio or percentage-point program output for an explicitly marked percentage answer; an unmarked numeric answer is considered in percentage points only when the question or recorded steps establish that context. Step percentage annotations use ratio-to-percent comparison only when it matches at displayed precision; otherwise they are marked ambiguous. Magnitude suffixes (thousand/million/billion) are not automatically rescaled because their units may already be represented in the table; they require manual unit-context review. The clean subset is for stratified reporting only; it is not a filtered training dataset.

**Current clean-subset counts: MISSING.** The local audit has been generated, but records have not yet been manually adjudicated and a clean-subset count has not been calculated. Aggregate audit totals alone are not sufficient to declare records clean.

## Interpretation of execution-answer diagnostics

The per-record flag `exe_ans_disagreement_human_answer_consistent` means only that the computed program result disagrees with the recorded `exe_ans` under the official comparison, while the parseable human-readable answer agrees with the computed result. It is a diagnostic intersection, **not proof** that the `exe_ans` difference is caused by rounding or formatting. Records carrying this flag also retain `program_exe_ans_disagreement` and remain in `requires_manual_review`. Do not use this flag alone to include records in the clean diagnostic subset or to alter official evaluator semantics.

## Required reporting

For each split, reports must include:

- raw count, processed count, and any documented exclusions (currently zero by policy);
- ordered ID digest and source revision;
- counts for program parse/execution failures;
- `exe_ans` differences after official rounding;
- human-answer/program disagreements and ambiguous human answers;
- confirmed step-operation, argument, numeric-result, and malformed-annotation mismatches;
- ambiguous percentage step annotations, reported separately from confirmed mismatches;
- `program` versus `program_re` structural disagreements and malformed nested expressions;
- overall and clean-subset metrics, with sample counts.

No aggregate annotation result is considered trusted until the audit passes hand-verified fixtures and representative real-record review.
