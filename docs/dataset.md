# FinQA Dataset Specification

## Source

- Repository: `bevaya/FinQA`
- Revision: `3d6a736bc67e06bc15fbf3618d88204a57c5b25e`
- Task: numerical reasoning over financial reports
- Language: English
- Benchmark paper: FinQA, EMNLP 2021
- Paper: https://aclanthology.org/2021.emnlp-main.300/

The repository revision is pinned to make dataset preparation reproducible.
Do not silently replace it with the latest `main` revision.

## Official Splits

| Split | Examples |
|---|---:|
| Train | 6,251 |
| Validation | 883 |
| Test | 1,147 |
| Total | 8,281 |

All official splits must be preserved. FineTuneX must not randomly
repartition the dataset.

## Source Schema

The complete source record includes:

- `id`
- `filename`
- `pre_text`
- `post_text`
- `table`
- `table_ori`
- `question`
- `answer`
- `exe_ans`
- `explanation`
- `ann_table_rows`
- `ann_text_rows`
- `steps`
- `program`
- `program_re`
- `gold_inds`

All source fields are retained in processed JSONL records.

## Annotation Semantics

- `question`: financial question.
- `pre_text` and `post_text`: report context.
- `table`: normalized table representation.
- `table_ori`: original table representation.
- `program`: flat gold reasoning program used as the initial
  supervised-generation target.
- `program_re`: nested representation of the reasoning program.
- `steps`: annotated intermediate operations and results.
- `exe_ans`: reference result used by the official execution evaluator.
- `answer`: human-readable answer.
- `explanation`: additional annotation.
- `ann_table_rows` and `ann_text_rows`: annotated evidence locations.
- `gold_inds`: primary mapping of evidence keys to evidence descriptions.

### Evidence Reference Semantics

`gold_inds` is a JSON object whose keys identify source evidence:

- `text_N` refers to index `N` in the concatenation of `pre_text`
  followed by `post_text`.
- Negative text indices use Python indexing semantics. For example,
  `text_-1` refers to the final passage in the combined list.
- `table_N` refers directly to row index `N` in `table`.

Text evidence descriptions are checked against their referenced source
passages after whitespace normalization. Table evidence descriptions
are synthesized annotations and are not expected to equal the raw table
row representation.

`ann_text_rows` and `ann_table_rows` are annotation index arrays.
They represent subsets of the corresponding evidence references in
`gold_inds`; they are not complete replacements for `gold_inds`.
Duplicate indices in these arrays are preserved because the pipeline
retains source annotations as supplied. Audit counts distinguish
unique `gold_inds` references from source annotation entries.

**Evidence leakage policy:** `gold_inds`, `ann_text_rows`, and
`ann_table_rows` are ground-truth annotations for offline validation
and evidence-recall evaluation. They must not be included in model
inputs or used to select evidence at inference time.

These fields are not interchangeable.

In particular, `answer`, `exe_ans`, and the last value in `steps.res`
can differ because of rounding, units, percentage formatting, or
annotation inconsistencies.

Blank human-readable answers must not automatically cause examples
to be removed.

## Processing Policy

1. Load the pinned source revision.
2. Require train, validation, and test splits.
3. Validate the expected source fields.
4. Validate unique IDs within each split.
5. Check ID overlap between splits.
6. Preserve all original annotations.
7. Audit duplicate normalized questions.
8. Audit blank human-readable answers.
9. Report raw final-step versus `exe_ans` differences.
10. Write the original official splits without random repartitioning.

No examples are silently removed.

## Processed Output

The output directory is `data/processed/finqa/`.

- `train.jsonl`
- `validation.jsonl`
- `test.jsonl`
- `manifest.json`

Each JSONL line is a complete source example. The processing pipeline
does not discard fields or rewrite gold annotations.

## Manifest

The manifest records:

- Dataset repository and pinned revision.
- Split counts.
- Source dataset fingerprints.
- SHA-256 hashes of processed split files.
- Duplicate and annotation anomaly audit results.
- Confirmation that official splits were preserved.
- Number of removed examples.

## Evaluation Policy

The initial supervised target is `program`.

Execution evaluation must follow the official FinQA evaluator's
operation semantics and numerical comparison rules.

`exe_ans` is retained as the execution reference. Human-readable
`answer` and `steps.res` are retained for diagnostics.

Raw string inequality is an anomaly indicator, not proof of an
incorrect annotation.

## Attribution and Licensing

The dataset source and its underlying financial-table data have
separate licensing and attribution requirements.

- FinQA annotation licensing: MIT, as documented by the selected
  dataset card.
- Underlying FinTabNet table data: CDLA-Permissive-1.0, as documented
  by the selected dataset card.

Review and preserve the original dataset-card attribution and license
notices when redistributing processed data.

The benchmark paper is:

Zhu et al. (2021), FinQA: A Dataset of Numerical Reasoning over
Financial Data.

https://aclanthology.org/2021.emnlp-main.300/
