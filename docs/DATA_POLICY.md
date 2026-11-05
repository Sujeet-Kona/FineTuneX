# FinQA Data Policy

**Status:** Policy defined; local data counts and annotation-audit findings are not yet verified.  
**Dataset:** `bevaya/FinQA`  
**Pinned revision:** `3d6a736bc67e06bc15fbf3618d88204a57c5b25e`

## Source-of-truth and immutability

- The pinned Hugging Face dataset revision is the source of truth.
- Raw source files are immutable inputs. FineTuneX may create manifests, JSONL conversions, tokenized datasets, reports, and model artifacts, but must never rewrite or patch source records.
- Every derived dataset manifest must record the source revision, split, record count, ordered ID digest, and source-file hash where available.
- A preprocessing or audit error must stop the run with a record ID and reason. It must not silently skip a row.

## Official split use

The dataset card reports the following official split sizes. These are expected counts; the exact local materialized counts remain **MISSING** until checked against the pinned revision.

| Split | Expected records | Permitted use |
|---|---:|---|
| Train | 6,251 | Supervised training; training-side diagnostics |
| Validation | 883 | Development, prompt/scorer debugging, baseline and adapter selection |
| Test | 1,147 | Final report only, after all decisions and configurations are frozen |
| **Total** | **8,281** | Complete source dataset |

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

**Current clean-subset counts: MISSING.** The integrity checker has been repaired but the local audit has not yet been rerun and reviewed.

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
