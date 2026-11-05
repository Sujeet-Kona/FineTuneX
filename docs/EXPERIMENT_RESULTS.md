# FineTuneX — measured results and reproduction record

**Report date:** 2026-11-05
**Branch:** `feat/reference-pipeline-without-kubernetes`
**Commit of the code that produced these numbers:** `15f352c`
(`fix: serve real FinQA records and align container dependencies`), on
`feat/reference-pipeline-without-kubernetes`, 2 commits ahead of `origin` and not
pushed. This report and `README.md` are working-tree changes on top of it.

This report contains only measurements actually executed on the development
machine. Every number traces to a summary file written by the evaluation script,
and each run records the SHA-256 of the inputs it consumed.

## Scope of the claim

Two measurements exist: a 32-example paired pilot and a full 883-example paired
validation comparison, both completed. The 883-example run measures this adapter on
this split; it does **not** support a claim about FineTuneX's accuracy on FinQA in
general. The adapter was trained for 8 optimizer steps on 32 examples of the train
split, so its behaviour on the test split, on larger training budgets, or as a
FineTuneX capability is unknown. These runs establish that the pipeline executes,
scores and compares correctly on real MPS hardware; they do not establish model
quality.

## Frozen configuration

| Item | Value |
|---|---|
| Base model | `Qwen/Qwen3-0.6B` |
| Model revision | `c1899de289a04d12100db370d81485cdf75e47ca` |
| Tokenizer | same name and revision as the model |
| Loading | `float16`, no quantization, `local_files_only: true`, `trust_remote_code: false` |
| Dataset | `bevaya/FinQA` |
| Dataset revision | `3d6a736bc67e06bc15fbf3618d88204a57c5b25e` |
| Split evaluated | validation, 883 records, official split preserved |
| Prompt version | `finqa-program-v4-fixed-prompt-budget` |
| Evidence strategy / budget | `complete-table-first`, 1024 tokens |
| Max prompt tokens | 1408 |
| Max generated tokens | 128 |
| Decoding | greedy (`do_sample=False`, `num_beams=1`), stop at EOS |
| Answer reference | `exe_ans` |
| Scoring rule | `six_significant_digits_of_exe_ans@6` |
| Device | Apple Silicon `mps`, 16 GB unified memory |
| Training seed / evaluation sample seed | 42 |

LoRA configuration for the measured adapter: rank 8, alpha 16, dropout 0.05,
target modules `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`,
learning rate 2e-4, batch size 1, gradient accumulation 8, 2 epochs, maximum
sequence length 1536, gradient checkpointing enabled (non-reentrant).

Checkpoints select on lowest token-weighted validation loss. Trainable parameters
are 5,046,272 of 601,096,192 (0.8395%), and all 392 saved adapter tensors are LoRA
tensors, so no full-model training occurred.

## Input provenance (SHA-256)

Recorded by the evaluation script and compared across paired runs.

| Input | SHA-256 |
|---|---|
| Raw validation split | `8e564eb0f0cb83957f737babd0631c3ede0d6185649803b524c9df5fc65c790a` |
| Tokenized validation artifact | `4c322338f2ecb6b4683885f9ea92421f4a8c1a1d236f3515c36eb0a2f6b7355f` |
| Preprocessing manifest | `772cd3486023a3d492a9968dfc1561cb4a1800e2df566281d9b7e038218a79d5` |
| `configs/model.yaml` | `080e176f6253096d9300886596c1217c7a20d87fb80e7832bb7f8e7bf9c91481` |
| `configs/training.yaml` | `06410fd5d98e5587908aff108400a3e52b2666b7e471a1ac87304de63ac40b3b` |
| `configs/preprocessing.yaml` | `fe54d9ee5786a0db38a2071222685d2a0df51a8ca2a36f202a873ec565995035` |
| v4-pilot-32 `checkpoint_metadata.json` | `71c444c83ceaf5e339629a01a9a9e4efafbebaccafb781d7a6b85ed3a34a8346` |

The validation record set is 883 unique IDs loaded in official file order.
SHA-256 over the ordered ID list:
`7c8aad37cf933545beeed0ed35080eb42b3988e060a122366e2442af098a3d2e`.

**Two files changed after the measured runs, and neither affects them.**
`configs/experiment.yaml` is now `fe2376eacfce4fe4b736383fe37bfe5208b8a75d91ac2bb776038eb0056670e3`
(SQLite tracking URI) and `.gitignore` is `c69a965b01f322525fdc5139c2c9ffc1a6f656d2f42716f6709e6953c29609a6`
(ignore rules for that store). Neither is in the six hashed inputs above, the evaluator
never reads either, and tracking was disabled during every measured training and
evaluation run, so the recorded provenance of the completed runs still holds. The
`pip install mlflow` that followed the runs changed only MLflow's own dependencies:
torch 2.15.0, numpy 2.4.6, transformers 5.22.1, peft 0.21.2 and accelerate 1.15.0 were
the same before and after, and it did not invalidate the completed evaluations.

## Pairwise comparability

`scripts/compare_finqa_evaluations.py` refuses to produce a report unless both runs
agree on sample size, sampling method and seed; prompt and generation budgets;
decoding mode; answer reference; scoring rule; all six input provenance hashes;
model name and revision; and, per record, record ID, prompt SHA-256 and expected
answer. The selected scoring rule is written into the comparison output so a report
produced under a superseded rule cannot be mistaken for a current one.

Full 883-example runs use `limit=None`, which loads the complete split in official
order, so no sampling is involved.

## Failure taxonomy

Failures are reported as three distinct classes and never collapsed into a score:

| `failure_reason` | Meaning | Counted as |
|---|---|---|
| `parse_failed` | output is not a valid program | incorrect |
| `execution_failed` | parses but cannot execute on the record's table | incorrect |
| `wrong_answer` | executes, value differs from `exe_ans` | incorrect |

Summaries carry `parse_success`, `parse_failures`, `execution_success`,
`execution_failures`, `valid_predictions`, `invalid_predictions`, `correct`,
`incorrect`, `answer_accuracy`, `answer_accuracy_over_valid_predictions` and a
`failure_reasons` histogram, plus up to 20 `failure_examples` carrying raw output,
normalized output, extracted prediction, expected answer and error text. A
prediction that fails to parse can never be scored correct.

## Validation of the evaluator

| Check | Result |
|---|---|
| Gold programs executed, train + validation | 7,134 / 7,134 succeeded |
| Gold answers matching `exe_ans` | 7,134 / 7,134, zero mismatches |
| Official-executor differential, all splits | 8,281 examples, 0 differences |
| Annotation integrity audit, current rule | 8,281 examples; 7,011 internally consistent; 1,270 flagged |
| Annotation integrity audit, superseded report | 8,281 examples; 6,559 consistent; 1,722 flagged |

### The on-disk annotation audit was stale; regenerating it changed the flag counts

`experiments/finqa_annotation_integrity.json` (written 07:13 UTC, before the two full
evaluation runs at 12:26 and 12:47 UTC) predates the
six-significant-digit comparison fix and reported 640 `program_exe_ans_disagreement`
flags and 479 `exe_ans_disagreement_human_answer_consistent` flags. Re-running
`scripts/audit_finqa_annotations.py` against the current code, writing to a **new** path
(`experiments/finqa_annotation_integrity_v2_sixsig.json`; the earlier report, the raw
splits and `docs/audit_samples/` were left untouched), gives:

| Count | Superseded report | Regenerated report |
|---|---:|---:|
| Internally consistent | 6,559 | 7,011 |
| Requires manual review | 1,722 | 1,270 |
| `program_exe_ans_disagreement` | 640 | **0** |
| `exe_ans_disagreement_human_answer_consistent` | 479 | **0** |
| `program_human_answer_disagreement` | 893 | 893 |
| `steps_program_disagreement` | 626 | 626 |
| `steps_annotation_ambiguous` | 345 | 345 |

So every `exe_ans` disagreement was a precision artifact of the old comparison, and the
claim now holds across all 8,281 records including the 1,147 test records: every gold
program reproduces its `exe_ans` at six significant digits. The human-answer and
step-annotation counts are unchanged because those comparisons are separate paths.
`docs/STATE.md` quoted the stale figures; corrected there.

Two previously recorded claims needed restating against the regenerated report:

- `ANSS/2012/page_92.pdf-1` is **still** `requires_manual_review`, but for a different
  reason than recorded: `program_result` 192501.5 vs `exe_ans` `192502` now matches,
  while the human `answer` `91000` disagrees with the program
  (`program_human_answer_disagreement`, `steps_program_disagreement`).
- The audit sample export in `docs/audit_samples/` (10 records per non-empty category,
  50 total) was generated from the stale report. Its manifest hashes were re-checked
  against the source splits and match exactly — train
  `891f7f736f56a8dacee631a1c51d5e19f5d99651d3f91e8d08799dbb79e8e657` (6,251), validation
  `8e564eb0f0cb83957f737babd0631c3ede0d6185649803b524c9df5fc65c790a` (883), test
  `35b9b2c0265784d1a807280eff9fd28a910ea3edb0a4296532a05b1a7340cf62` (1,147) — and all
  50 audit/source joins are consistent. The two categories that are now empty
  (`exe_ans_disagreements`, `exe_ans_disagreement_human_answer_consistent`) should not be
  treated as live review queues; the samples are preserved unmodified because they are
  the user's untracked files and re-exporting them would overwrite that evidence.

### Manual review of 20 real audit records

Completed against `docs/audit_samples/` (4 per non-empty category, 20 of the 50
exported). Each sample carries the full `source_record` (raw `program`, `program_re`,
`steps`, table, passages) joined to the full `audit_record` (`parsed_steps`,
`raw_steps`, `steps_audit` with per-step computed values, `program_result`,
`exe_ans_match`, `human_answer_match`, `flags`, `classification`). Representative
findings:

- Genuine step-annotation inconsistency: `BLL/2010/page_58.pdf-4` — program
  `divide(10.1, const_100), divide(46.2, #0)` computes 457.42574…, the annotated step
  result is `98.0`, and the human answer is `98.0` while `exe_ans` is `457.426`. The
  program and `exe_ans` agree; the annotation trail does not.
- Displayed-precision agreement, previously mis-flagged: `JPM/2008/page_177.pdf-3` —
  `divide(200265, 20.8)` computes 9628.125 against `exe_ans` `9628.12`; it is
  `internally_consistent` with no flags in the regenerated report and was a
  `program_exe_ans_disagreement` in the stale one.
- Percentage-annotation ambiguity is retained rather than silently resolved
  (`steps_annotation_ambiguous`, 345 records), e.g. `-66.67%` against a computed
  `-0.66667`.

No source record was edited, no record was dropped, and no exclusion was applied.

The gold-program audit excludes the 1,147 test records so the scoring rule is never
tuned against them. Reports live in `experiments/`, which is gitignored; they are
local evidence, not version-controlled artifacts.

## Measured result — 32-example paired pilot

Selection: `--limit 32 --sample-seed 42` (SHA-256 record-ID ranking), MPS, greedy,
identical prompts and order confirmed by the comparison report.

| Metric | Base model | v4 LoRA pilot | Wilson 95% CI (base) | Wilson 95% CI (LoRA) |
|---|---:|---:|---:|---:|
| Parse success | 15/32 (46.9%) | 32/32 (100%) | [0.309, 0.636] | [0.893, 1.000] |
| Execution success | 8/32 (25.0%) | 32/32 (100%) | [0.133, 0.421] | [0.893, 1.000] |
| Correct answers | 0/32 (0%) | 4/32 (12.5%) | [0.000, 0.107] | [0.050, 0.281] |
| Wall time | 49.69 s | 38.84 s | — | — |

Failure reasons, base: 17 `parse_failed`, 7 `execution_failed`, 8 `wrong_answer`.
Failure reasons, pilot: 28 `wrong_answer`, no parse or execution failures.

Paired analysis: 0 correct for both, 4 correct only for the pilot, 0 correct only
for the base model, 28 wrong for both. Exact McNemar two-sided p = 0.125. **The
accuracy gain is not statistically established at n = 32.** The formatting and
executability gain is large and unambiguous; the numeric-reasoning gain is not yet
distinguishable from noise.

Reading of the base model's 0%: it emits `add` almost exclusively (15 of 15 parsed
operations are `add`) and separates multi-step programs with newlines rather than
the dataset's comma delimiter, so most outputs fail the strict parser. This is a
genuine model limitation, not a scoring artifact — the inspected `wrong_answer`
cases are numerically far off (`492.0` against a gold `114`; `361000.0` against
`2.37383`). All four pilot hits were inspected manually: each generated program is
byte-identical to its gold program, so they are not tolerance artifacts.

**The pilot's `0%` for the base model was a sampling artifact.** On the complete 883
validation records the base model is correct on 6 of them (0.68%); the 32 hash-sampled
records simply contained none of the six. Reported here so the pilot figure is not
mistaken for the baseline.

## Measured result — full 883-example paired validation comparison

Both runs evaluated the complete official validation split in file order (`limit=None`,
no sampling), so every one of the 883 records is scored for both models. Identical
prompts, record order, expected answers and scoring rule were asserted by
`scripts/compare_finqa_evaluations.py`; the comparison output is
`artifacts/evaluation/v5-paired-full-comparison.json`.

| Metric | Base model | v4 LoRA pilot adapter | Wilson 95% CI (base) | Wilson 95% CI (adapter) |
|---|---:|---:|---:|---:|
| Parse success | 403/883 (45.64%) | 879/883 (99.55%) | [42.38%, 48.94%] | [98.84%, 99.82%] |
| Execution success | 247/883 (27.97%) | 876/883 (99.21%) | [25.11%, 31.02%] | [98.37%, 99.62%] |
| Correct answers | 6/883 (0.68%) | 90/883 (10.19%) | [0.31%, 1.47%] | [8.37%, 12.36%] |
| Correct of valid predictions | 6/247 (2.43%) | 90/876 (10.27%) | — | — |
| Wall time on MPS | 1,547.6 s (25.8 min) | 1,277.5 s (21.3 min) | — | — |

Failure taxonomy, base: 480 `parse_failed`, 156 `execution_failed`, 241 `wrong_answer`.
Failure taxonomy, adapter: 4 `parse_failed`, 3 `execution_failed`, 786 `wrong_answer`.
The three classes are mutually exclusive and reported separately; no failure was
silently skipped, and the adapter generated a completion for all 883 records
(`generated: 883`, `ended_with_eos` 883/883 versus 788/883 for the base model).

Absolute accuracy gain: **+9.51 percentage points** (0.68% to 10.19%).

Paired analysis over the same 883 records: 0 correct for both models, 90 correct only
for the adapter, 6 correct only for the base model, 787 wrong for both. Exact McNemar
two-sided p = 2.5 × 10⁻²⁰ on 96 discordant pairs. Unlike the pilot, the improvement is
statistically established at this scale — and the 6 base-only records are real
regressions, reported in full below rather than averaged away.

### The adapter's gain is mostly output-format learning, and it collapses onto two operations

Operation counts over successfully parsed outputs:

| Model | Operations emitted across the 883 outputs |
|---|---|
| Base | `add` only (408 steps) |
| Adapter | `divide` (640), `subtract` (433) — **no `add`, `multiply`, `greater` or any `table_*`** |

Neither model reproduces FinQA's operation vocabulary. The base model collapses onto
`add`; the 8-step adapter collapses onto `divide` and `subtract`, the two most frequent
operations in the training distribution. This bounds what the adapter can score, and
the constraint is visible in the results broken down by the gold final operation:

| Gold final operation | Records | Base correct | Adapter correct |
|---|---:|---:|---:|
| `divide` | 566 | 0 | 69 (12.2%) |
| `subtract` | 168 | 0 | 19 (11.3%) |
| `add` | 56 | 6 (10.7%) | 1 (1.8%) |
| `multiply` | 49 | 0 | 0 |
| `table_average` | 18 | 0 | 0 |
| `greater` | 10 | 0 | 0 |
| `table_max` | 7 | 0 | 0 |
| `table_min` | 5 | 0 | 1 (20.0%) |
| `table_sum` | 4 | 0 | 0 |

Program-length collapse shows the same effect. Adapter outputs are 685 one-step and 194
two-step programs; the gold split is 523 / 287 / 43 / 14 / 16 steps for 1 through 5
steps. Accuracy by gold program length: 82/523 (15.7%) for one step, 8/287 (2.8%) for
two steps, and **0/73 for every gold program of three or more steps**. Restricting to
records the adapter could even express (gold of at most two steps ending in `divide` or
`subtract`, 677 of 883), it is correct on 88/677 (13.0%); outside that subset it is
correct on 2 of 206. The 10.19% headline is therefore largely a measure of how much of
the split the collapsed output format can reach, not of financial reasoning.

The 2 hits outside the reachable subset are answer matches produced by different
programs, both inspected manually:

- `GS/2014/page_136.pdf-2` — gold `table_min(interest rate hedges, none)` → `-8683`;
  adapter emitted `subtract(1936, 8683), subtract(#0, 1936)`, which also evaluates to
  `-8683`. Correct answer, wrong reasoning.
- `SWKS/2006/page_81.pdf-1` — gold `multiply(351, const_m1), add(#0, 538)` → `187`;
  adapter emitted `subtract(786, 599)` → `187`. Same class.

FinQA scores the answer, so these count as correct under the dataset's own convention
and are reported as such. They are 2 of 90 and do not change any conclusion.

### Regressions: the 6 records the base model got and the adapter lost

All six have `add` as their gold final operation, which the adapter never emits, so each
is structurally unreachable rather than a decoding fluctuation. Inspected individually:

| Record ID | Question (abbreviated) | Base output | Adapter output | Gold | `exe_ans` |
|---|---|---|---|---|---|
| `AMT/2005/page_105.pdf-2` | total value of net operating loss carryforwards | `add(2157503, 2418012)` | `subtract(2418012, 2200000)` | `add(2157503, 2418012)` | `4.57552e+06` |
| `MRO/2007/page_134.pdf-2` | total tax benefits realized in excess of stock-based compensation | `add(30, 36)` | `subtract(30, 36), divide(#0, 36)` | `add(30, 36)` | `66` |
| `IPG/2008/page_93.pdf-4` | parent company guarantees combined for 2007 and 2008 | `add(255.7, 327.1)` | `subtract(327.1, 255.7), divide(#0, 255.7)` | `add(255.7, 327.1)` | `582.8` |
| `PKG/2009/page_65.pdf-1` | purchase commitments for 2010 and 2011 | `add(6951, 5942)` | `subtract(6951, 5942), divide(#0, 6951)` | `add(6951, 5942)` | `12893` |
| `PNC/2011/page_78.pdf-4` | total of the reserve for losses | `add(47, 54)` | `subtract(47, 54)` | `add(47, 54)` | `101` |
| `RE/2017/page_159.pdf-3` | estimated total cost to replace annuities in 2017 | `add(144618, 34444)` | `subtract(34444, 33860)` | `add(144618, 34444)` | `179062` |

The adapter's dominant training pattern is the percent-change
`subtract(a, b), divide(#0, b)`, and it applies that template even where the question
asks for a total — including `PNC/2011/page_78.pdf-4`, where it returns `-7.0` for a
sum of two positive reserves. This is the concrete cost of the collapse described
above, and it is the reason the adapter's 10.19% must not be read as "the model learned
FinQA".

### Adapter parse and execution failures, individually inspected

4 `parse_failed` — in three of them the model copied table text instead of writing a
program, all three from the same document (`AES/2015/page_117.pdf`), which suggests a
specific passage the prompt's evidence budget renders poorly:

- `AES/2015/page_117.pdf-2` → `less : proportional non-recoverable environmental capital expenditures ( 2 )`
- `AES/2015/page_117.pdf-4` → `less : proportional adjustment factor on operating cash activities … | -558 ( 558 ) | …`
- `AES/2015/page_117.pdf-1` → `less : proportional adjustment factor on operating cash activities ( 2 ) ( 3 ) | 1241, 891, 1271`
- `AMAT/2014/page_18.pdf-1` → `subtract(1400, 48% ( 48 % ))` (percent literal inside an argument)

3 `execution_failed` — well-formed operations with un-parseable numeric literals:

- `VLO/2018/page_25.pdf-2` → `divide(2.2 billion, 2.2 billion)` (`Invalid numeric value: '2.2 billion'`)
- `VLO/2018/page_25.pdf-3` → same output and error
- `UNP/2009/page_65.pdf-1` → `divide(60000, 1/3), divide(#0, 1/3)` (`Invalid numeric value: '1/3'`)

No record was dropped from the denominator because of these: all 883 are counted, and
each failure carries its raw output, error text, predicted and expected value in
`summary.json`.

### The strict parser is not hiding a better baseline

The base model separates steps with newlines instead of the dataset's comma-space
delimiter, so a fair challenge to the 45.64% parse rate is whether a lenient parser
would raise the baseline. Measured on all 480 base `parse_failed` records, rejoining
lines with `, ` and re-scoring through the same `evaluate_prediction()` path:

| Scope | Parse failures | Lenient parse success | Then executable | Then correct |
|---|---:|---:|---:|---:|
| 32-example pilot | 17 | 11 | 7 | **0** |
| Full 883 validation | 480 | 329 | 250 | **0** |

A maximally lenient delimiter parser would move the base parse rate from 45.64% to
82.9% (403 → 732 of 883) and the execution rate from 27.97% to 56.3% (247 → 497) while
adding **zero** correct answers, because the newline-delimited outputs are mostly
repeated `add(...)` steps over wrong operands, truncated programs, or the prompt's own
operation definitions recited back (`JPM/2008/page_85.pdf-3` reproduces the DSL
documentation verbatim). The remaining 151 outputs do not parse even leniently
(38 `Missing argument separator in 'add'`, and non-numeric arguments such as
`add(20ac750, 20ac1000)`). So the 0.68% baseline is a property of the model, not of the
scorer's strictness.

### Pilot versus full measurement

| Metric | Pilot (n = 32) | Full (n = 883) | Read |
|---|---:|---:|---|
| Base parse rate | 46.9% | 45.6% | Replicates |
| Base execution rate | 25.0% | 28.0% | Replicates |
| Base accuracy | 0.0% | 0.68% | Pilot missed all 6 hits |
| Adapter parse rate | 100% | 99.55% | Replicates |
| Adapter execution rate | 100% | 99.21% | Replicates |
| Adapter accuracy | 12.5% | 10.19% | Pilot CI [5.0%, 28.1%] contains 10.19% |
| Reported accuracy gain | +12.5 pp | +9.51 pp | Pilot overstated the gain |
| McNemar p | 0.125 (4 discordants) | 2.5 × 10⁻²⁰ (96 discordants) | Only the full run establishes the effect |

All 4 pilot adapter hits replicate in the full run (`HOLX/2010/page_124.pdf-1`,
`ETFC/2013/page_26.pdf-2`, `AES/2016/page_185.pdf-2`, `NKE/2016/page_37.pdf-1`), and no
pilot record is among the 6 regressions. The pilot's rate estimates were therefore
consistent with the full measurement, but its *effect-size* claim was inflated: with no
base hits in the sample, the pilot could only report gains, and it could not see the
`add`-collapse regressions at all. That is the concrete reason not to read a 32-example
pilot as a performance estimate.

## Training actually executed

`artifacts/checkpoints/v4-pilot-32`: 32 training micro-batches per epoch, batch size
1, gradient accumulation 8, 2 epochs — therefore **8 optimizer steps** and 1,200
supervised tokens. Train loss 1.0075 to 0.5454, validation loss 0.6037 to 0.5033,
mean gradient norm 9.67 to 3.02. Optimizer updates, finite gradients, and checkpoint
save/reload are confirmed by loading this checkpoint and generating under MPS.

Loss semantics were verified independently of the trainer: the model's reported loss
(`2.25773263`) equals a hand-computed cross-entropy mean over the non-ignored shifted
tokens (21 supervised tokens) for a real training record. Prompt positions are
asserted masked to `-100`, labels asserted to align with inputs, padding masked to
`-100`; the loader raises rather than training on a record that violates this.

A 2-batch MPS smoke run wrote `artifacts/checkpoints/smoke-mps-verify`: trainable
5,046,272 / 601,096,192, train loss 2.5077 to 1.1316, validation loss 1.8940 to
1.4993, gradient norms 20.67 and 12.09, reloaded successfully for generation.

## Export and serving

`artifacts/exported/v4-pilot-32-merged` loads independently of PEFT and produced
output byte-identical to the adapter on the same prompt (`divide(5.0, 50.3)`).
Re-verified after the environment change on a different real record
(`JPM/2008/page_85.pdf-3`): adapter and merged model both emit
`divide(136104, 88746)` with **identical generated token IDs**, not merely identical
text. Live FastAPI checks on MPS with the v4 adapter: `/health/live` 200,
`/health/ready` 200, a real 44-row validation record returns 200 with an executed
program, and injected gold fields, an empty question and a ragged table each return
422. `/metrics` exposes `finetunex_http_requests_total` and
`finetunex_http_request_duration_seconds` labelled only by method, route and status
code; no request content appears in metric labels. Re-run after the MLflow install:
`/health/live` 200, `/health/ready` 200, a real 6-row validation record scored
`divide(136104, 88746)` → 1.53364 with parse and execution status `success`, the three
invalid payloads again returned 422, and `/metrics` again returned only low-cardinality
`finetunex_*` series.

## Experiment tracking (MLflow)

Until this point, tracking had only ever been tested against a fake `mlflow` module
injected by `monkeypatch`, with `experiment.tracking.enabled: false` in the shipped
config. Installing the pinned extra (`mlflow 3.18.0`, which left torch 2.15.0,
numpy 2.4.6, transformers 5.22.1, peft 0.21.2 and accelerate 1.15.0 unchanged) and
exercising the enabled path found a real defect:

**`configs/experiment.yaml` shipped `tracking_uri: mlruns`, and MLflow 3.x refuses the
file-store backend outright.** `tracker.start()` raised
`MlflowException: The filesystem tracking backend (e.g., './mlruns') is in maintenance
mode … set MLFLOW_ALLOW_FILE_STORE=true` from inside `set_experiment`, before the run
was marked started, so enabling tracking as configured could not work with any MLflow
version the `mlflow>=2.20,<4.0` pin allows. The fake-module tests could not detect this
because they never let MLflow interpret the URI.

Fix: the default URI is now `sqlite:///mlruns.db` (same working-directory-relative
semantics as the old path, supported by MLflow 2.x and 3.x), `mlruns.db*` is gitignored,
and two tests exercise the enabled path against the installed library rather than a
stand-in: one asserts a real run lands in the store with `FINISHED` status, parameters
(`None`-valued params excluded), per-epoch metrics and the `config`/`training`/
`checkpoint` artifact trees; the other asserts a failure inside `start()` leaves a
`FAILED`/`KILLED` run rather than a phantom successful one. Both skip cleanly when
MLflow is not installed, so CI without the extra is unaffected.

The full trainer path was then run for real with tracking temporarily enabled
(`--max-train-batches 2 --max-validation-batches 2`, config restored to `enabled: false`
immediately afterwards): `scripts/train_finqa.py` exited 0 and wrote run
`finqa-20261105T131207Z` with status **FINISHED**, 11 parameters including the pinned
model and dataset revisions, epoch metrics (`train_loss` 1.13158, `validation_loss`
1.49934, `mean_gradient_norm` 12.0862, `train_optimizer_steps` 1), and all three
artifact groups — 5 config files, `training_history.json`, `checkpoint_metadata.json`.
The logged numbers match the same 2-batch smoke run performed earlier without
tracking, so enabling tracking did not change training.

No secret material is written to the tracking store: the logged parameters are model,
dataset, revision, seed, epoch and batch settings, and the uploaded artifacts are the
five YAML configs, which contain no credentials.

## Known limitations and possible biases

- **Adapter operation collapse.** The adapter emitted only `divide` and `subtract`
  across 883 records and never `add`, `multiply`, `greater` or a `table_*` operation.
  It scored 0/73 on records whose gold program has three or more steps. Its 10.19% is
  a lower bound limited by output format, not a measure of financial reasoning, and an
  8-step adapter says nothing about what FineTuneX achieves with a real training budget.
- **Regressions exist and are reported.** The base model's 6 correct answers are all
  lost by the adapter (McNemar: 90 gained, 6 lost). They are listed individually above.
- **Pilot scale.** 8 optimizer steps on 32 examples. The parse-rate improvement
  largely reflects learning the output format.
- **Checkpoint-selection overlap.** The pilot selected its checkpoint using the
  first 32 validation records. The 32-example evaluation is hash-sampled and overlaps
  that scope by 3 records (`AMAT/2013/page_18.pdf-2`, `STT/2011/page_83.pdf-1`,
  `V/2008/page_17.pdf-1`); none of the 3 is correct in either run, so the measured
  pilot accuracy carries no selection bias. The full 883-example evaluation contains
  all 32 selection records, and **3 of the adapter's 90 correct answers are among
  them** (`AAPL/2004/page_36.pdf-2`, `ABMD/2006/page_75.pdf-1`,
  `PNC/2013/page_207.pdf-1`), so at most 3/883 = 0.34 pp of the 10.19% headline is
  optimism from selecting on records that are also scored — inside the reported CI,
  but real and not hidden. One of the 6 regressions (`AMT/2005/page_105.pdf-2`) is
  also a selection-scope record, which is direct evidence that this adapter did not
  memorise those 32 records in 8 steps. Selection used validation records that are
  also evaluated.
- **Single seed, single machine.** One run per model, greedy decoding, seed 42, M4 MPS.
  No repeated-seed variance estimate exists, so run-to-run variability is unmeasured.
- **No training on validation or test records.** Training consumed the train split
  only, and the training entry point raises if artifacts include the test split.
- **Prompts were not tuned against test data.**
- **The test split is not a pristine holdout.** Historical one-off inspection of test
  labels is recorded in `docs/STATE.md`; test stays excluded from scoring-rule
  tuning, model selection and hyperparameter decisions.
- **Dataset annotation quality.** 1,270 of 8,281 records are flagged for manual review
  by the integrity audit under the current six-significant-digit rule
  (`experiments/finqa_annotation_integrity_v2_sixsig.json`; the superseded pre-fix report
  counted 1,722); nothing is silently dropped. The measured 10.19%
  is over all 883 records; no clean-subset accuracy is reported because the clean
  subset has not been adjudicated.
- **Evidence is not in Git.** `artifacts/` and `experiments/` are gitignored, so the
  JSON reports cited here exist only on the machine that produced them.
- **One latent defect was found and deliberately left.** In
  `scripts/evaluate_finqa_model.py`, `load_validation_records()` validates the tuple
  `("id", "question", "pre_text", "post_text", "table", "exe_ans")` but line 462 then
  writes `record["program"]`, so `program` is a de-facto required field that the check
  does not list. It cannot affect any measured run — all 883 records in the pinned split
  carry `program`, `configs/dataset.yaml` already lists it in `required_fields`, and the
  failure mode would be a plain `KeyError` rather than a silently wrong score — so the
  cost of the fix (churning three test fixtures that build minimal records) buys only a
  better error message. It is recorded here instead of being patched around finished
  runs.

## Reproduction

```bash
cd ~/ForgeTune
source .venv/bin/activate

# 32-example paired pilot
python scripts/evaluate_finqa_model.py --base-model --limit 32 --sample-seed 42 \
  --output-dir artifacts/evaluation/v5-base-32
python scripts/evaluate_finqa_model.py --checkpoint artifacts/checkpoints/v4-pilot-32 \
  --limit 32 --sample-seed 42 --output-dir artifacts/evaluation/v5-adapter-32
python scripts/compare_finqa_evaluations.py \
  --base-dir artifacts/evaluation/v5-base-32 \
  --adapter-dir artifacts/evaluation/v5-adapter-32 \
  --output artifacts/evaluation/v5-paired-comparison.json

# full 883-example paired validation comparison
python scripts/evaluate_finqa_model.py --base-model \
  --output-dir artifacts/evaluation/v5-base-full
python scripts/evaluate_finqa_model.py --checkpoint artifacts/checkpoints/v4-pilot-32 \
  --output-dir artifacts/evaluation/v5-adapter-full
python scripts/compare_finqa_evaluations.py \
  --base-dir artifacts/evaluation/v5-base-full \
  --adapter-dir artifacts/evaluation/v5-adapter-full \
  --output artifacts/evaluation/v5-paired-full-comparison.json

# scorer validation, training smoke test, loss semantics, quality gates
python scripts/evaluate_gold_programs.py
python scripts/train_finqa.py --max-train-batches 2 --max-validation-batches 2 \
  --output-dir artifacts/checkpoints/smoke-mps-verify
python scripts/verify_qwen_loss.py
ruff check . && ruff format --check . && pytest -q
```

`--overwrite` is required to regenerate an existing evaluation directory; the
scripts refuse to replace prior results by default.

## Fresh-clone reproduction

Executed on 2026-11-05 in `/tmp/ft_clone`, cloned from the local repository over
`file://` at commit `5e8ea29` (94 tracked files, `.git` 468 KB, largest tracked blob
33 KB — this document; no weights, datasets or generated artifacts are tracked). The
clone used the same interpreter environment but its own source tree
(`PYTHONPATH=/tmp/ft_clone/src`, verified: `import finetunex` resolves inside the clone).

| Step | Result |
|---|---|
| `pytest -q` from the clone | **PASS** — 430 passed, 2 warnings |
| `scripts/prepare_dataset.py` | **PASS** — 6,251 / 883 / 1,147 records, 0 removed |
| Materialized split files | **Byte-identical** to the originals: train `891f7f73…`, validation `8e564eb0…`, test `35b9b2c0…` |
| `scripts/preprocess_finqa.py` | **PASS** — 7,134 tokenized records |
| Tokenized train artifact | **Byte-identical**: `8809f4db5ce6129b…` |
| Tokenized validation artifact | **Byte-identical**: `4c322338f2ecb6b4…` — exactly the hash recorded in Input provenance above |
| Tokenized `manifest.json` | **Differs** (`a008de85…` vs the recorded `772cd348…`) for one transitive reason, below |
| Base/adapter evaluation from the clone | **NOT RUN** — the inference step was already measured on the primary tree; re-running it in the clone would add no new evidence and costs about 47 minutes of MPS time |

The manifest difference was isolated rather than waved away: of 6,235 leaf values in the
prep manifest, exactly 12 differ (4 evidence-annotation counts × 3 splits), and in every
case the clone has a number where the original has `null`. The on-disk original was
produced by an earlier revision of `prepare_dataset.py`, before the evidence-annotation
audit block existed. The tokenized manifest then differs only because it hashes that
prep manifest (`dataset.source_manifest_sha256`: clone `5adf9d4b…`, original `fb21cba1…`)
— 1 differing key out of 61. So the *records* reproduce byte-for-byte, and the only
incompatibility is a recorded digest that transitively covers non-record metadata.

The cross-check was executed in the clone rather than reasoned about: importing
`validate_preprocessing_provenance()` from `/tmp/ft_clone/scripts` with `PROJECT_ROOT`
resolving inside the clone returned without raising, yielding manifest `a008de85…`,
source validation `8e564eb0…`, tokenized validation `4c322338…`. The two hashes the
function actually compares — the manifest's recorded `source_split_sha256.validation`
and `splits.validation.output_sha256` against the on-disk files — are byte-for-byte the
values in Input provenance above. Only the manifest's own file hash differs.

Practical consequence for a reviewer: a clean checkout regenerates the artifact hashes
the evaluator checks, so evaluation is reproducible from tracked code plus the pinned
dataset. Quoting `772cd348…` as a compatibility requirement across checkouts would be
wrong; it identifies the specific run recorded here, not a stable content fingerprint.

The paired statistics in this report (Wilson 95% intervals, the exact McNemar test on
discordant pairs, the per-gold-operation and per-length breakdowns, the regression
list, and the lenient-delimiter probe) were computed from the two
`validation_predictions.jsonl` files with throwaway analysis scripts under `/tmp`,
which are **not version-controlled**. They are not part of the shipped pipeline, so a
reviewer must re-derive them. Wilson: `p̂` inverted-score interval at z = 1.95996.
McNemar: exact two-sided binomial on the discordant count at p = 0.5, here
`2 · Σ_{i=0}^{6} C(96, i) / 2⁹⁶` for 90 versus 6 discordants. The lenient probe calls the
in-tree `evaluate_prediction()` on the line-rejoined output, so it scores with the same
parser, executor and comparison rule as the runs it analyses — no separate scoring path
was introduced for it.
