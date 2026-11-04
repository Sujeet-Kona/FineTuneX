"""Inference-safe FinQA prompt construction."""

from __future__ import annotations

from functools import cache
from typing import Any

from finetunex.data.finqa_preprocessing import PROMPT_VERSION, build_prompt
from finetunex.data.finqa_retrieval import select_evidence


def build_inference_prompt(
    record: dict[str, Any],
    tokenizer: Any,
    *,
    evidence_budget: int = 1024,
) -> dict[str, Any]:
    """Build a FinQA generation prompt without reading gold targets.

    Only question, source passages, and the financial table are used.
    Gold programs, execution answers, and evidence annotations are ignored.
    The returned prompt ends at the same chat-template generation
    prefix used during training.
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

    if (
        not isinstance(evidence_budget, int)
        or isinstance(evidence_budget, bool)
        or evidence_budget <= 0
    ):
        raise ValueError("evidence_budget must be a positive integer")

    if not callable(getattr(tokenizer, "encode", None)):
        raise ValueError("tokenizer must provide an encode method")

    @cache
    def count_tokens(text: str) -> int:
        count = len(tokenizer.encode(text, add_special_tokens=False))
        return count

    complete_table = "\n".join(" | ".join(row) for row in table)
    complete_table_tokens = count_tokens(complete_table)

    if complete_table_tokens <= evidence_budget:
        selection = select_evidence(
            question,
            passages,
            [],
            token_budget=evidence_budget - complete_table_tokens,
            token_counter=count_tokens,
        )

        selected_passages = [(item.index, item.content) for item in selection.items]
        table_context = complete_table
        table_strategy = "complete"
        selected_table_rows = list(range(1, len(table)))
        evidence_tokens = complete_table_tokens + selection.used_tokens
    else:
        selection = select_evidence(
            question,
            passages,
            table,
            token_budget=evidence_budget,
            token_counter=count_tokens,
        )

        selected_passages = [
            (item.index, item.content)
            for item in selection.items
            if item.kind == "text"
        ]
        selected_table_items = [
            item for item in selection.items if item.kind == "table"
        ]

        table_context = "\n".join(item.content for item in selected_table_items)
        table_strategy = "selected_rows"
        selected_table_rows = [item.index for item in selected_table_items]
        evidence_tokens = selection.used_tokens

    if evidence_tokens > evidence_budget:
        raise RuntimeError(
            f"{record['id']}: evidence budget exceeded "
            f"({evidence_tokens} > {evidence_budget})"
        )

    prompt = build_prompt(question, selected_passages, table_context, tokenizer)
    prompt_token_count = len(tokenizer.encode(prompt, add_special_tokens=False))

    return {
        "prompt": prompt,
        "metadata": {
            "record_id": record["id"],
            "prompt_version": PROMPT_VERSION,
            "evidence_budget": evidence_budget,
            "evidence_tokens": evidence_tokens,
            "table_strategy": table_strategy,
            "selected_passage_indices": [index for index, _ in selected_passages],
            "selected_table_row_indices": selected_table_rows,
            "prompt_tokens": prompt_token_count,
        },
    }
