"""Inference-safe FinQA prompt construction."""

from __future__ import annotations

from typing import Any

from finetunex.data.finqa_preprocessing import (
    PROMPT_VERSION,
    build_prompt_with_evidence,
)


def build_inference_prompt(
    record: dict[str, Any],
    tokenizer: Any,
    *,
    evidence_budget: int = 1024,
    max_prompt_tokens: int = 1408,
) -> dict[str, Any]:
    """Build a target-independent prompt under the shared prompt-token ceiling.

    Only question, source passages, and the financial table are used.
    Gold programs, execution answers, and evidence annotations are ignored.
    The exact same evidence-budgeting function is used during training.
    """
    if not isinstance(record, dict):
        raise TypeError("record must be a dictionary")

    required = ("id", "question", "pre_text", "post_text", "table")
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(f"record is missing required fields: {missing}")

    if not isinstance(record["id"], str) or not record["id"].strip():
        raise ValueError("record id must be a non-empty string")

    question = record["question"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError("record question must be a non-empty string")

    pre_text = record["pre_text"]
    post_text = record["post_text"]

    if not isinstance(pre_text, list) or not isinstance(post_text, list):
        raise ValueError("pre_text and post_text must be lists")

    passages = pre_text + post_text
    if any(not isinstance(passage, str) for passage in passages):
        raise ValueError("all passages must be strings")

    table = record["table"]
    if not isinstance(table, list):
        raise ValueError("table must be a list")

    if any(
        not isinstance(row, list) or any(not isinstance(cell, str) for cell in row)
        for row in table
    ):
        raise ValueError("table must contain lists of strings")

    if table:
        width = len(table[0])
        if width == 0 or any(len(row) != width for row in table):
            raise ValueError("table rows must have a consistent non-zero width")

    result = build_prompt_with_evidence(
        question,
        passages,
        table,
        tokenizer,
        evidence_budget=evidence_budget,
        max_prompt_tokens=max_prompt_tokens,
    )
    result["metadata"]["record_id"] = record["id"]
    result["metadata"]["prompt_version"] = PROMPT_VERSION
    return result
