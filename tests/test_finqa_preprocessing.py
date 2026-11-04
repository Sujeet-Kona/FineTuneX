from copy import deepcopy

import pytest

from finetunex.data.finqa_preprocessing import (
    PROMPT_VERSION,
    SYSTEM_INSTRUCTION,
    build_prompt,
    preprocess_record,
)


class FakeTokenizer:
    """Small deterministic tokenizer for preprocessing contract tests."""

    eos_token = "<eos>"
    eos_token_id = 999999

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    ):
        assert tokenize is False
        assert enable_thinking is False

        rendered = "".join(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            for message in messages
        )

        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
            if enable_thinking:
                rendered += "<think>\n"

        return rendered

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False

        if text.endswith(self.eos_token):
            content = text[: -len(self.eos_token)]
            return list(range(len(content[::5]))) + [self.eos_token_id]

        return list(range(len(text[::5])))


def make_record():
    return {
        "id": "TEST/2024/page_1.pdf-1",
        "question": "What was revenue in 2023?",
        "pre_text": [
            "The company reported revenue growth.",
            "Operating expenses increased.",
        ],
        "post_text": [
            "Revenue increased during the financial year.",
        ],
        "table": [
            ["Year", "Revenue"],
            ["2022", "100"],
            ["2023", "125"],
        ],
        "program": "divide(125, 100)",
        "gold_inds": '{"text_0": "Revenue growth"}',
        "ann_text_rows": [0],
        "ann_table_rows": [2],
    }


def preprocess(record=None, **kwargs):
    kwargs.setdefault("max_sequence_length", 4096)
    return preprocess_record(
        record or make_record(),
        FakeTokenizer(),
        split="train",
        tokenizer_name="Qwen/Qwen3-0.6B",
        tokenizer_revision="c1899de289a04d12100db370d81485cdf75e47ca",
        **kwargs,
    )

def test_prompt_contract_defines_operations_and_reasoning_constraints():
    assert PROMPT_VERSION == "finqa-program-v4-fixed-prompt-budget"
    assert "subtract(a, b) computes a minus b" in SYSTEM_INSTRUCTION
    assert "divide(a, b) computes a divided by b" in SYSTEM_INSTRUCTION
    assert "zero-based" in SYSTEM_INSTRUCTION
    assert "percentage change" in SYSTEM_INSTRUCTION
    assert "do not subtract the count" in SYSTEM_INSTRUCTION
    assert "preserve" in SYSTEM_INSTRUCTION and "units" in SYSTEM_INSTRUCTION
    assert "Output the program only." in SYSTEM_INSTRUCTION

def test_builds_tokenized_example_with_target_only_labels():
    result = preprocess()

    input_ids = result["input_ids"]
    labels = result["labels"]
    metadata = result["metadata"]

    prompt_length = metadata["prompt_tokens"]

    assert len(input_ids) == len(labels)
    assert len(result["attention_mask"]) == len(input_ids)
    assert labels[:prompt_length] == [-100] * prompt_length
    assert labels[prompt_length:] == input_ids[prompt_length:]
    assert labels[-1] == FakeTokenizer.eos_token_id
    assert input_ids[-1] == FakeTokenizer.eos_token_id

def test_records_reproducibility_metadata():
    metadata = preprocess()["metadata"]

    assert metadata["record_id"] == "TEST/2024/page_1.pdf-1"
    assert metadata["split"] == "train"
    assert metadata["prompt_version"] == PROMPT_VERSION
    assert metadata["tokenizer_name"] == "Qwen/Qwen3-0.6B"
    assert metadata["tokenizer_revision"] == "c1899de289a04d12100db370d81485cdf75e47ca"
    assert metadata["evidence_budget"] == 1024
    assert metadata["sequence_length"] > 0

def test_complete_table_is_preserved_when_it_fits():
    metadata = preprocess()["metadata"]

    assert metadata["table_strategy"] == "complete"
    assert metadata["selected_table_row_indices"] == [1, 2]

def test_reduces_evidence_budget_to_fit_sequence_without_truncation():
    record = make_record()
    record["pre_text"] = ["Revenue " + "growth " * 600]

    no_passages_record = {
        **record,
        "pre_text": [],
        "post_text": [],
    }
    no_passages = preprocess(no_passages_record, evidence_budget=1024)["metadata"]
    full_evidence = preprocess(record, evidence_budget=1024)["metadata"]
    assert full_evidence["sequence_length"] > no_passages["sequence_length"]

    max_sequence_length = (
        no_passages["sequence_length"] + full_evidence["sequence_length"]
    ) // 2
    result = preprocess(
        record,
        evidence_budget=1024,
        max_sequence_length=max_sequence_length,
    )

    metadata = result["metadata"]

    assert metadata["effective_evidence_budget"] < metadata["evidence_budget"]
    assert metadata["evidence_budget_reductions"] > 0
    assert metadata["evidence_tokens"] <= metadata["effective_evidence_budget"]
    assert metadata["sequence_length"] <= max_sequence_length

def test_oversized_table_uses_header_grounded_row_fallback():
    record = make_record()
    record["table"] = [
        ["Year", "Revenue"],
        ["2020", "100 revenue"],
        ["2021", "110 revenue"],
        ["2022", "120 revenue"],
        ["2023", "125 revenue"],
    ]

    result = preprocess(record, evidence_budget=10)

    metadata = result["metadata"]

    assert metadata["table_strategy"] == "selected_rows"
    assert metadata["evidence_tokens"] <= 10
    assert all(index > 0 for index in metadata["selected_table_row_indices"])

def test_empty_passages_are_supported():
    record = make_record()
    record["pre_text"] = []
    record["post_text"] = []

    result = preprocess(record)

    assert result["metadata"]["selected_passage_indices"] == []
    assert len(result["input_ids"]) == len(result["labels"])

def test_gold_annotations_are_not_returned():
    result = preprocess()

    assert "gold_inds" not in result
    assert "ann_text_rows" not in result
    assert "ann_table_rows" not in result
    assert "gold_inds" not in result["metadata"]

def test_does_not_mutate_source_record():
    record = make_record()
    original = deepcopy(record)

    preprocess(record)

    assert record == original

def test_rejects_sequence_over_maximum_without_truncation():
    record = make_record()
    record["program"] = "x" * 1000

    with pytest.raises(ValueError, match="generation reserve"):
        preprocess(record, max_sequence_length=1536, max_prompt_tokens=1408)

def test_rejects_unsupported_split():
    with pytest.raises(ValueError, match="split"):
        preprocess_record(
            make_record(),
            FakeTokenizer(),
            split="test",
            tokenizer_name="Qwen/Qwen3-0.6B",
            tokenizer_revision="pinned-revision",
        )

def test_rejects_missing_program():
    record = make_record()
    del record["program"]

    with pytest.raises(ValueError, match="program"):
        preprocess(record)

def test_rejects_empty_program():
    record = make_record()
    record["program"] = "  "

    with pytest.raises(ValueError, match="program"):
        preprocess(record)

def test_rejects_invalid_evidence_budget():
    with pytest.raises(ValueError, match="evidence_budget"):
        preprocess(evidence_budget=0)

def test_generation_prompt_respects_disabled_thinking_flag():
    prompt = build_prompt(
        "What was revenue?",
        [],
        "Year | Revenue\\n2023 | 125",
        FakeTokenizer(),
    )
    assert "<think>" not in prompt
    assert prompt.endswith("<|im_start|>assistant\\n")
