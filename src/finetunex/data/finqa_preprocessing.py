"""Deterministic prompt construction and tokenization for FinQA."""

from __future__ import annotations

from functools import cache
from typing import Any

from finetunex.data.finqa_retrieval import select_evidence

PROMPT_VERSION = "finqa-program-v3-reasoning"

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
    table = record["table"]
    complete_table = "\n".join(" | ".join(row) for row in table)
    complete_table_tokens = count_tokens(complete_table)

    # Treat the configured evidence budget as a ceiling. The fixed system
    # prompt, question, chat-template tokens, and target also consume context,
    # so reduce retrieved evidence only as much as needed to fit the complete
    # supervised sequence. Never truncate the prompt or target.
    configured_evidence_budget = evidence_budget
    effective_evidence_budget = evidence_budget
    evidence_budget_reductions = 0

    while True:
        if complete_table_tokens <= effective_evidence_budget:
            # Preserve the complete table and use the remaining budget for text.
            selection = select_evidence(
                record["question"],
                passages,
                [],
                token_budget=effective_evidence_budget - complete_table_tokens,
                token_counter=count_tokens,
            )

            selected_passages = [
                (item.index, item.content) for item in selection.items
            ]
            table_context = complete_table
            table_strategy = "complete"
            selected_table_rows = list(range(1, len(table)))
            evidence_tokens = complete_table_tokens + selection.used_tokens

        else:
            # Oversized-table fallback: select header-grounded data rows.
            selection = select_evidence(
                record["question"],
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

            table_context = "\n".join(
                item.content for item in selected_table_items
            )
            table_strategy = "selected_rows"
            selected_table_rows = [
                item.index for item in selected_table_items
            ]
            evidence_tokens = selection.used_tokens

        if evidence_tokens > effective_evidence_budget:
            raise RuntimeError(
                f"{record['id']}: evidence budget exceeded "
                f"({evidence_tokens} > {effective_evidence_budget})"
            )

        prompt = build_prompt(
            record["question"],
            selected_passages,
            table_context,
            tokenizer,
        )
        target = record["program"]

        # The chat template already provides the assistant generation prefix.
        # Append the target directly so training and inference share one boundary.
        prompt_prefix_ids = tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )
        full_text = prompt + target + eos_token
        input_ids = tokenizer.encode(
            full_text,
            add_special_tokens=False,
        )

        prompt_length = len(prompt_prefix_ids)

        if input_ids[:prompt_length] != prompt_prefix_ids:
            raise RuntimeError(
                f"{record['id']}: tokenizer prompt-prefix boundary mismatch"
            )

        if not input_ids or input_ids[-1] != eos_token_id:
            raise RuntimeError(f"{record['id']}: EOS token is missing or misplaced")

        if prompt_length >= len(input_ids):
            raise RuntimeError(f"{record['id']}: no supervised target tokens")

        sequence_length = len(input_ids)

        if sequence_length <= max_sequence_length:
            break

        if effective_evidence_budget == 0:
            raise ValueError(
                f"{record['id']}: sequence length {sequence_length} exceeds "
                f"maximum {max_sequence_length} even with no retrieved evidence"
            )

        overflow = sequence_length - max_sequence_length
        next_budget = max(0, effective_evidence_budget - overflow - 8)
        if next_budget == effective_evidence_budget:
            next_budget = effective_evidence_budget - 1

        effective_evidence_budget = next_budget
        evidence_budget_reductions += 1

    labels = [-100] * prompt_length + input_ids[prompt_length:]

    if len(labels) != sequence_length:
        raise RuntimeError(f"{record['id']}: input and label lengths differ")

    if labels[:prompt_length] != [-100] * prompt_length:
        raise RuntimeError(f"{record['id']}: prompt labels are not masked")

    if labels[prompt_length:] != input_ids[prompt_length:]:
        raise RuntimeError(f"{record['id']}: target labels do not match inputs")

    if labels[-1] != eos_token_id:
        raise RuntimeError(f"{record['id']}: EOS is not supervised")

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * sequence_length,
        "labels": labels,
        "metadata": {
            "record_id": record["id"],
            "split": split,
            "prompt_version": PROMPT_VERSION,
            "tokenizer_name": tokenizer_name,
            "tokenizer_revision": tokenizer_revision,
            "evidence_budget": configured_evidence_budget,
            "effective_evidence_budget": effective_evidence_budget,
            "evidence_budget_reductions": evidence_budget_reductions,
            "evidence_tokens": evidence_tokens,
            "table_strategy": table_strategy,
            "selected_passage_indices": [index for index, _ in selected_passages],
            "selected_table_row_indices": selected_table_rows,
            "prompt_tokens": prompt_length,
            "target_tokens_including_eos": sequence_length - prompt_length,
            "sequence_length": sequence_length,
        },
    }
