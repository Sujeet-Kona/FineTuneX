"""Tests for FinQA training datasets and dynamic padding."""

import json

import pytest
import torch

from finetunex.training.data import FinQACollator, FinQADataset


def make_record(record_id, input_ids, prompt_tokens):
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * prompt_tokens + input_ids[prompt_tokens:],
        "metadata": {
            "record_id": record_id,
            "prompt_tokens": prompt_tokens,
        },
    }


def write_dataset(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def test_loads_jsonl_dataset(tmp_path):
    path = write_dataset(
        tmp_path / "train.jsonl",
        [make_record("a", [10, 11, 12], 2)],
    )

    dataset = FinQADataset(path, max_sequence_length=10)

    assert len(dataset) == 1
    assert dataset[0]["input_ids"] == [10, 11, 12]


def test_rejects_sequence_exceeding_limit(tmp_path):
    path = write_dataset(
        tmp_path / "train.jsonl",
        [make_record("a", [10, 11, 12], 2)],
    )

    with pytest.raises(ValueError, match="exceeds configured limit"):
        FinQADataset(path, max_sequence_length=2)


def test_rejects_misaligned_arrays(tmp_path):
    record = make_record("a", [10, 11, 12], 2)
    record["labels"] = [-100, 12]

    path = write_dataset(tmp_path / "train.jsonl", [record])

    with pytest.raises(ValueError, match="misaligned"):
        FinQADataset(path, max_sequence_length=10)


def test_rejects_unmasked_prompt(tmp_path):
    record = make_record("a", [10, 11, 12], 2)
    record["labels"][0] = 10

    path = write_dataset(tmp_path / "train.jsonl", [record])

    with pytest.raises(ValueError, match="prompt labels must be masked"):
        FinQADataset(path, max_sequence_length=10)


def test_dynamic_padding_preserves_supervised_labels():
    collator = FinQACollator(pad_token_id=0)

    batch = collator(
        [
            make_record("a", [10, 11, 12], 2),
            make_record("b", [20, 21, 22, 23, 24], 3),
        ]
    )

    assert batch["input_ids"].shape == (2, 5)
    assert batch["attention_mask"].tolist() == [
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
    ]
    assert batch["labels"].tolist() == [
        [-100, -100, 12, -100, -100],
        [-100, -100, -100, 23, 24],
    ]
    assert batch["metadata"][0]["record_id"] == "a"

    assert batch["input_ids"].dtype == torch.long
    assert batch["attention_mask"].dtype == torch.long
    assert batch["labels"].dtype == torch.long


def test_rejects_empty_batch():
    collator = FinQACollator(pad_token_id=0)

    with pytest.raises(ValueError, match="empty batch"):
        collator([])


def test_rejects_empty_dataset(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        FinQADataset(path, max_sequence_length=10)
