"""Deterministic prompt construction and tokenization for FinQA."""

from __future__ import annotations

from functools import cache
from typing import Any

from finetunex.data.finqa_retrieval import select_evidence

PROMPT_VERSION = "finqa-program-v4-fixed-prompt-budget"

SYSTEM_INSTRUCTION = (
    "You solve financial questions by generating FinQA reasoning programs. "
    "Return only a comma-separated sequence of valid FinQA operations; never "
    "return prose, explanations, markdown, or a final answer. "
    "Use exactly these two-argument operations: "
    "add(a, b) adds; subtract(a, b) computes a minus b; "
    "multiply(a, b) multiplies; divide(a, b) computes a divided by b; "
    "exp(a, b) raises a to power b; greater(a, b) returns yes if a is greater "
    "than b and no otherwise; table_max(row, none), table_min(row, none), "
    "table_sum(row, none), and table_average(row, none) aggregate the numeric "
    "cells in the table row whose first cell exactly matches row. "
    "Arguments for arithmetic operations are numeric values, constants such "
    "as const_100 or const_m1, or references such as #0 to the result of an "
    "earlier operation. References are zero-based: #0 is the first operation, "
    "#1 the second, and a step may reference only earlier steps. "
    "For table operations, use the exact row label as the first argument and "
    "none as the second argument. "
    "Reason before emitting the program: identify the requested quantity, "
    "locate the relevant values using both row labels and column headers, "
    "and preserve the order and units of the operands. "
    "For a difference, subtract the earlier or comparison value from the "
    "later or target value in the order requested. For a ratio, divide the "
    "numerator by the denominator. For a percentage change, compute "
    "(new value minus old value) divided by old value, then multiply by 100. "
    "For a percentage of a total, divide the part by the stated total and "
    "multiply by 100 only when the question asks for a percent. "
    "For an average, divide the relevant total by the matching count; do not "
    "subtract the count. Do not confuse a requested change with a percentage "
    "change. "
    "Use values exactly as represented in the evidence, including their "
    "reported scale and units; do not invent values, approximate a denominator, "
    "or apply an extra scale conversion unless the question requires it. "
    "Use multiple operations only when the calculation requires them, and "
    "reference their results with #0, #1, and so on. "
    "Check that the final operation answers the question and that its units "
    "match the requested quantity. Output the program only."
)


def build_prompt(
    question: str,
    passages: list[tuple[int, str]],
    table_context: str,
    tokenizer: Any,
) -> str:
    """Render the versioned FinQA prompt using the model chat template."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    parts = ["Question:", question, "", "Financial report passages:"]

    if passages:
        for index, content in passages:
            parts.extend([f"Passage {index}:", content])
    else:
        parts.append("None.")

    parts.extend(
        [
            "",
            "Financial table:",
            table_context or "No table evidence selected.",
            "",
            "FinQA program:",
        ]
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_INSTRUCTION,
        },
        {
            "role": "user",
            "content": "\n".join(parts),
        },
    ]

    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)

    if not callable(apply_chat_template):
        raise ValueError("tokenizer must provide apply_chat_template")

    prompt = apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    if not isinstance(prompt, str) or not prompt:
        raise RuntimeError("Tokenizer produced an invalid chat prompt")

    return prompt


def _validate_record(record: dict[str, Any]) -> None:
    required = ("id", "question", "pre_text", "post_text", "table", "program")

    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(f"record is missing required fields: {missing}")

    if not isinstance(record["id"], str) or not record["id"].strip():
        raise ValueError("record id must be a non-empty string")

    if not isinstance(record["question"], str) or not record["question"].strip():
        raise ValueError("record question must be a non-empty string")

    if not isinstance(record["pre_text"], list) or not isinstance(
        record["post_text"], list
    ):
        raise ValueError("pre_text and post_text must be lists")

    passages = record["pre_text"] + record["post_text"]
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

    if not isinstance(record["program"], str) or not record["program"].strip():
        raise ValueError("program must be a non-empty string")


def build_prompt_with_evidence(
    question: str,
    passages: list[str],
    table: list[list[str]],
    tokenizer: Any,
    *,
    evidence_budget: int,
    max_prompt_tokens: int,
) -> dict[str, Any]:
    """Select evidence under a target-independent prompt-token ceiling.

    The same deterministic function is used by preprocessing and inference.
    Its budget reduction depends only on the inference-visible prompt, never
    on the gold program or its length.
    """
    for name, value in (
        ("evidence_budget", evidence_budget),
        ("max_prompt_tokens", max_prompt_tokens),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    if not callable(getattr(tokenizer, "encode", None)):
        raise ValueError("tokenizer must provide an encode method")

    @cache
    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    complete_table = "\n".join(" | ".join(row) for row in table)
    complete_table_tokens = count_tokens(complete_table)
    effective_evidence_budget = evidence_budget
    evidence_budget_reductions = 0

    while True:
        if complete_table_tokens <= effective_evidence_budget:
            selection = select_evidence(
                question,
                passages,
                [],
                token_budget=effective_evidence_budget - complete_table_tokens,
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
                token_budget=effective_evidence_budget,
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

        if evidence_tokens > effective_evidence_budget:
            raise RuntimeError(
                f"Evidence budget exceeded ({evidence_tokens} > "
                f"{effective_evidence_budget})"
            )

        prompt = build_prompt(question, selected_passages, table_context, tokenizer)
        prompt_tokens = count_tokens(prompt)

        if prompt_tokens <= max_prompt_tokens:
            break

        if effective_evidence_budget == 0:
            raise ValueError(
                "Prompt exceeds max_prompt_tokens even with no retrieved evidence "
                f"({prompt_tokens} > {max_prompt_tokens})"
            )

        reduction = max(8, prompt_tokens - max_prompt_tokens + 8)
        effective_evidence_budget = max(0, effective_evidence_budget - reduction)
        evidence_budget_reductions += 1

    return {
        "prompt": prompt,
        "metadata": {
            "prompt_version": PROMPT_VERSION,
            "evidence_budget": evidence_budget,
            "effective_evidence_budget": effective_evidence_budget,
            "evidence_budget_reductions": evidence_budget_reductions,
            "evidence_tokens": evidence_tokens,
            "table_strategy": table_strategy,
            "selected_passage_indices": [index for index, _ in selected_passages],
            "selected_table_row_indices": selected_table_rows,
            "prompt_tokens": prompt_tokens,
            "max_prompt_tokens": max_prompt_tokens,
        },
    }


def preprocess_record(
    record: dict[str, Any],
    tokenizer: Any,
    *,
    split: str,
    tokenizer_name: str,
    tokenizer_revision: str,
    evidence_budget: int = 1024,
    max_sequence_length: int = 1536,
    max_prompt_tokens: int | None = None,
) -> dict[str, Any]:
    """Convert one FinQA record into target-only supervised token IDs.

    Evidence selection is bounded by a fixed prompt-token ceiling and is
    independent of the gold target length. The target must fit in the
    generation reserve left by that ceiling.
    """
    _validate_record(record)

    if split not in {"train", "validation"}:
        raise ValueError("split must be 'train' or 'validation'")

    if not isinstance(tokenizer_name, str) or not tokenizer_name.strip():
        raise ValueError("tokenizer_name must be a non-empty string")

    if not isinstance(tokenizer_revision, str) or not tokenizer_revision.strip():
        raise ValueError("tokenizer_revision must be a non-empty string")

    for name, value in (
        ("evidence_budget", evidence_budget),
        ("max_sequence_length", max_sequence_length),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    if max_prompt_tokens is None:
        max_prompt_tokens = max_sequence_length - 128

    if (
        isinstance(max_prompt_tokens, bool)
        or not isinstance(max_prompt_tokens, int)
        or max_prompt_tokens <= 0
    ):
        raise ValueError("max_prompt_tokens must be a positive integer")

    if max_prompt_tokens >= max_sequence_length:
        raise ValueError("max_prompt_tokens must be smaller than max_sequence_length")

    if not callable(getattr(tokenizer, "encode", None)):
        raise ValueError("tokenizer must provide an encode method")

    eos_token = getattr(tokenizer, "eos_token", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    if not isinstance(eos_token, str) or not eos_token:
        raise ValueError("tokenizer must define a non-empty eos_token")

    if not isinstance(eos_token_id, int):
        raise ValueError("tokenizer must define an integer eos_token_id")

    passages = record["pre_text"] + record["post_text"]
    prompt_result = build_prompt_with_evidence(
        record["question"],
        passages,
        record["table"],
        tokenizer,
        evidence_budget=evidence_budget,
        max_prompt_tokens=max_prompt_tokens,
    )
    prompt = prompt_result["prompt"]
    prompt_prefix_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = tokenizer.encode(
        prompt + record["program"] + eos_token,
        add_special_tokens=False,
    )
    prompt_length = len(prompt_prefix_ids)

    if input_ids[:prompt_length] != prompt_prefix_ids:
        raise RuntimeError(f"{record['id']}: tokenizer prompt-prefix boundary mismatch")
    if not input_ids or input_ids[-1] != eos_token_id:
        raise RuntimeError(f"{record['id']}: EOS token is missing or misplaced")
    if prompt_length >= len(input_ids):
        raise RuntimeError(f"{record['id']}: no supervised target tokens")

    target_tokens = len(input_ids) - prompt_length
    generation_reserve = max_sequence_length - max_prompt_tokens

    if target_tokens > generation_reserve:
        raise ValueError(
            f"{record['id']}: target requires {target_tokens} tokens, "
            f"but the configured generation reserve is {generation_reserve}"
        )

    sequence_length = len(input_ids)
    if sequence_length > max_sequence_length:
        raise RuntimeError(
            f"{record['id']}: sequence length {sequence_length} exceeds "
            f"maximum {max_sequence_length}"
        )

    labels = [-100] * prompt_length + input_ids[prompt_length:]

    if len(labels) != sequence_length:
        raise RuntimeError(f"{record['id']}: input and label lengths differ")

    if labels[:prompt_length] != [-100] * prompt_length:
        raise RuntimeError(f"{record['id']}: prompt labels are not masked")

    if labels[prompt_length:] != input_ids[prompt_length:]:
        raise RuntimeError(f"{record['id']}: target labels do not match inputs")

    if labels[-1] != eos_token_id:
        raise RuntimeError(f"{record['id']}: EOS is not supervised")

    metadata = dict(prompt_result["metadata"])
    metadata.update(
        {
            "record_id": record["id"],
            "split": split,
            "tokenizer_name": tokenizer_name,
            "tokenizer_revision": tokenizer_revision,
            "target_tokens_including_eos": target_tokens,
            "sequence_length": sequence_length,
        }
    )

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * sequence_length,
        "labels": labels,
        "metadata": metadata,
    }
