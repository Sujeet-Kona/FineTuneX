from copy import deepcopy

import pytest

from finetunex.data.finqa_preprocessing import preprocess_record
from finetunex.inference.prompt import build_inference_prompt


class FakeTokenizer:
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
            rendered += "<|im_start|>assistant\n<think>\n\n</think>\n\n"

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
        "exe_ans": "1.25",
        "gold_inds": '{"text_0": "Revenue growth"}',
        "ann_text_rows": [0],
        "ann_table_rows": [2],
    }


def test_builds_prompt_without_requiring_gold_program():
    record = make_record()
    del record["program"]

    result = build_inference_prompt(record, FakeTokenizer())

    assert result["prompt"].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert "What was revenue in 2023?" in result["prompt"]
    assert "2023 | 125" in result["prompt"]
    assert result["metadata"]["record_id"] == record["id"]


def test_ignores_gold_fields_even_when_present():
    record = make_record()
    original = deepcopy(record)

    result = build_inference_prompt(record, FakeTokenizer())

    assert record == original
    assert record["program"] not in result["prompt"]
    assert record["exe_ans"] not in result["prompt"]
    assert "gold_inds" not in result["metadata"]
    assert "ann_text_rows" not in result["metadata"]
    assert "ann_table_rows" not in result["metadata"]


def test_matches_training_evidence_selection_metadata():
    record = make_record()
    tokenizer = FakeTokenizer()

    inference = build_inference_prompt(record, tokenizer)
    training = preprocess_record(
        record,
        tokenizer,
        split="validation",
        tokenizer_name="Qwen/Qwen3-0.6B",
        tokenizer_revision="c1899de289a04d12100db370d81485cdf75e47ca",
        max_sequence_length=4096,
    )

    inference_metadata = inference["metadata"]
    training_metadata = training["metadata"]

    assert inference_metadata["table_strategy"] == training_metadata["table_strategy"]
    assert (
        inference_metadata["selected_passage_indices"]
        == training_metadata["selected_passage_indices"]
    )
    assert (
        inference_metadata["selected_table_row_indices"]
        == training_metadata["selected_table_row_indices"]
    )
    assert inference_metadata["evidence_tokens"] == training_metadata["evidence_tokens"]
    assert inference_metadata["prompt_tokens"] == training_metadata["prompt_tokens"]


def test_oversized_table_uses_same_fallback_as_training():
    record = make_record()
    record["table"] = [
        ["Year", "Revenue"],
        ["2020", "100 revenue"],
        ["2021", "110 revenue"],
        ["2022", "120 revenue"],
        ["2023", "125 revenue"],
    ]

    tokenizer = FakeTokenizer()
    inference = build_inference_prompt(
        record,
        tokenizer,
        evidence_budget=10,
    )
    training = preprocess_record(
        record,
        tokenizer,
        split="validation",
        tokenizer_name="Qwen/Qwen3-0.6B",
        tokenizer_revision="c1899de289a04d12100db370d81485cdf75e47ca",
        evidence_budget=10,
        max_sequence_length=4096,
    )

    assert inference["metadata"]["table_strategy"] == "selected_rows"
    assert (
        inference["metadata"]["selected_table_row_indices"]
        == training["metadata"]["selected_table_row_indices"]
    )
    assert (
        inference["metadata"]["selected_passage_indices"]
        == training["metadata"]["selected_passage_indices"]
    )
    assert inference["metadata"]["evidence_tokens"] <= 10


def test_prompt_budget_is_target_independent_and_matches_training():
    record = make_record()
    record["pre_text"] = [
        "Revenue growth financial performance " * 35 + f"section {index}"
        for index in range(4)
    ]
    tokenizer = FakeTokenizer()
    settings = {"evidence_budget": 1024, "max_prompt_tokens": 700}

    inference = build_inference_prompt(record, tokenizer, **settings)
    training = preprocess_record(
        record,
        tokenizer,
        split="train",
        tokenizer_name="Qwen/Qwen3-0.6B",
        tokenizer_revision="c1899de289a04d12100db370d81485cdf75e47ca",
        max_sequence_length=4000,
        **settings,
    )

    prompt_length = training["metadata"]["prompt_tokens"]
    inference_ids = tokenizer.encode(inference["prompt"], add_special_tokens=False)
    assert inference_ids == training["input_ids"][:prompt_length]

    longer_target = {**record, "program": "add(1, 2), " * 40 + "add(1, 2)"}
    longer_training = preprocess_record(
        longer_target,
        tokenizer,
        split="train",
        tokenizer_name="Qwen/Qwen3-0.6B",
        tokenizer_revision="c1899de289a04d12100db370d81485cdf75e47ca",
        max_sequence_length=4000,
        **settings,
    )
    assert (
        longer_training["input_ids"][:prompt_length]
        == training["input_ids"][:prompt_length]
    )
    assert (
        longer_training["metadata"]["effective_evidence_budget"]
        == training["metadata"]["effective_evidence_budget"]
    )
    assert inference["metadata"]["evidence_budget_reductions"] > 0
    assert inference["metadata"]["prompt_tokens"] <= settings["max_prompt_tokens"]


def test_supports_empty_passages():
    record = make_record()
    record["pre_text"] = []
    record["post_text"] = []

    result = build_inference_prompt(record, FakeTokenizer())

    assert "Financial report passages:\nNone." in result["prompt"]
    assert result["metadata"]["selected_passage_indices"] == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("id", "", "record id"),
        ("question", "", "record question"),
        ("pre_text", "invalid", "must be lists"),
        ("post_text", "invalid", "must be lists"),
        ("table", "invalid", "table must be a list"),
    ],
)
def test_rejects_invalid_record_fields(field, value, message):
    record = make_record()
    record[field] = value

    with pytest.raises(ValueError, match=message):
        build_inference_prompt(record, FakeTokenizer())


def test_rejects_missing_inference_field():
    record = make_record()
    del record["question"]

    with pytest.raises(ValueError, match="question"):
        build_inference_prompt(record, FakeTokenizer())


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_rejects_invalid_evidence_budget(budget):
    with pytest.raises(ValueError, match="evidence_budget"):
        build_inference_prompt(
            make_record(),
            FakeTokenizer(),
            evidence_budget=budget,
        )
