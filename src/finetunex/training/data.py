"""Dataset loading and dynamic padding for preprocessed FinQA artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class FinQADataset(Dataset):
    """Load preprocessed FinQA examples from a JSONL artifact."""

    def __init__(
        self,
        path: str | Path,
        *,
        max_sequence_length: int,
    ) -> None:
        self.path = Path(path)

        if not self.path.is_file():
            raise FileNotFoundError(f"Dataset artifact not found: {self.path}")

        if (
            isinstance(max_sequence_length, bool)
            or not isinstance(max_sequence_length, int)
            or max_sequence_length <= 0
        ):
            raise ValueError("max_sequence_length must be a positive integer")

        self.max_sequence_length = max_sequence_length
        self.records: list[dict[str, Any]] = []

        with self.path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    raise ValueError(f"{self.path}:{line_number}: blank JSONL line")

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{self.path}:{line_number}: invalid JSON"
                    ) from exc

                self._validate_record(record, line_number)
                self.records.append(record)

        if not self.records:
            raise ValueError(f"Dataset artifact is empty: {self.path}")

    def _validate_record(self, record: Any, line_number: int) -> None:
        """Validate the serialized token sequence contract."""
        location = f"{self.path}:{line_number}"

        if not isinstance(record, dict):
            raise ValueError(f"{location}: expected a JSON object")

        required = {"input_ids", "attention_mask", "labels", "metadata"}

        if set(record) != required:
            raise ValueError(f"{location}: unexpected record schema")

        input_ids = record["input_ids"]
        attention_mask = record["attention_mask"]
        labels = record["labels"]
        metadata = record["metadata"]

        if not all(
            isinstance(value, list)
            for value in (
                input_ids,
                attention_mask,
                labels,
            )
        ):
            raise ValueError(f"{location}: token fields must be lists")

        if not input_ids or not (len(input_ids) == len(attention_mask) == len(labels)):
            raise ValueError(f"{location}: invalid or misaligned token arrays")

        if len(input_ids) > self.max_sequence_length:
            raise ValueError(f"{location}: sequence exceeds configured limit")

        if any(
            isinstance(token, bool) or not isinstance(token, int) for token in input_ids
        ):
            raise ValueError(f"{location}: input_ids must contain integers")

        if any(isinstance(mask, bool) or mask not in (0, 1) for mask in attention_mask):
            raise ValueError(f"{location}: invalid attention mask")

        if any(
            isinstance(label, bool) or not isinstance(label, int) for label in labels
        ):
            raise ValueError(f"{location}: labels must contain integers")

        if not isinstance(metadata, dict):
            raise ValueError(f"{location}: metadata must be an object")

        prompt_tokens = metadata.get("prompt_tokens")

        if (
            isinstance(prompt_tokens, bool)
            or not isinstance(prompt_tokens, int)
            or not 0 < prompt_tokens < len(input_ids)
        ):
            raise ValueError(f"{location}: invalid prompt token count")

        if any(label != -100 for label in labels[:prompt_tokens]):
            raise ValueError(f"{location}: prompt labels must be masked")

        if labels[prompt_tokens:] != input_ids[prompt_tokens:]:
            raise ValueError(f"{location}: target labels do not match inputs")

        if not isinstance(metadata.get("record_id"), str):
            raise ValueError(f"{location}: missing record ID")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]

        return {
            "input_ids": record["input_ids"],
            "attention_mask": record["attention_mask"],
            "labels": record["labels"],
            "metadata": record["metadata"],
        }


class FinQACollator:
    """Dynamically right-pad token sequences within each batch."""

    def __init__(self, *, pad_token_id: int) -> None:
        if (
            isinstance(pad_token_id, bool)
            or not isinstance(pad_token_id, int)
            or pad_token_id < 0
        ):
            raise ValueError("pad_token_id must be a non-negative integer")

        self.pad_token_id = pad_token_id

    def __call__(
        self,
        examples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not examples:
            raise ValueError("Cannot collate an empty batch")

        max_length = max(len(example["input_ids"]) for example in examples)

        input_ids = []
        attention_masks = []
        labels = []
        metadata = []

        for example in examples:
            padding_length = max_length - len(example["input_ids"])

            input_ids.append(
                example["input_ids"] + [self.pad_token_id] * padding_length
            )

            attention_masks.append(example["attention_mask"] + [0] * padding_length)

            labels.append(example["labels"] + [-100] * padding_length)

            metadata.append(example["metadata"])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(
                attention_masks,
                dtype=torch.long,
            ),
            "labels": torch.tensor(labels, dtype=torch.long),
            "metadata": metadata,
        }
