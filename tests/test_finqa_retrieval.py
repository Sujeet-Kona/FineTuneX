import inspect

import pytest

from finetunex.data.finqa_retrieval import select_evidence


def word_count(text):
    return len(text.split())


def test_selects_relevant_passage():
    result = select_evidence(
        question="What was the revenue growth?",
        passages=[
            "The company expanded its international workforce.",
            "Revenue increased from 100 to 125 million.",
        ],
        table=[],
        token_budget=100,
        token_counter=word_count,
    )

    assert len(result.items) == 1
    assert result.items[0].kind == "text"
    assert result.items[0].index == 1
    assert "Revenue increased" in result.items[0].content


def test_table_candidate_includes_original_header():
    result = select_evidence(
        question="What was revenue in 2023?",
        passages=[],
        table=[
            ["Year", "Revenue"],
            ["2022", "100"],
            ["2023", "125"],
        ],
        token_budget=100,
        token_counter=word_count,
    )

    assert result.items
    selected = next(
        item for item in result.items if item.kind == "table" and item.index == 2
    )
    assert selected.content == "Year | Revenue\n2023 | 125"


def test_skips_table_rows_with_only_empty_cells():
    result = select_evidence(
        question="What was revenue?",
        passages=[],
        table=[
            ["Year", "Revenue"],
            ["", ""],
            ["2023", "125"],
        ],
        token_budget=100,
        token_counter=word_count,
    )

    assert all(item.index != 1 for item in result.items)
    assert any(item.kind == "table" and item.index == 2 for item in result.items)


def test_header_only_table_produces_no_candidates():
    result = select_evidence(
        question="What was revenue?",
        passages=[],
        table=[["Year", "Revenue"]],
        token_budget=100,
        token_counter=word_count,
    )

    assert result.items == ()
    assert result.used_tokens == 0


def test_respects_token_budget():
    result = select_evidence(
        question="revenue growth",
        passages=[
            "revenue increased substantially",
            "revenue increased from one hundred to two hundred",
        ],
        table=[],
        token_budget=3,
        token_counter=word_count,
    )

    assert result.used_tokens <= 3
    assert result.remaining_tokens == result.budget - result.used_tokens
    assert all(item.token_count <= 3 for item in result.items)


def test_skips_oversized_candidate_and_considers_smaller_candidates():
    result = select_evidence(
        question="revenue",
        passages=[
            (
                "revenue revenue revenue revenue increased "
                "significantly across the business"
            ),
            "revenue increased",
        ],
        table=[],
        token_budget=2,
        token_counter=word_count,
    )

    assert len(result.items) == 1
    assert result.items[0].index == 1
    assert result.used_tokens == 2


def test_ties_preserve_source_order():
    result = select_evidence(
        question="revenue",
        passages=[
            "revenue increased",
            "revenue declined",
        ],
        table=[],
        token_budget=2,
        token_counter=lambda _: 2,
    )

    assert len(result.items) == 1
    assert result.items[0].index == 0


def test_selector_interface_does_not_accept_gold_annotations():
    parameters = inspect.signature(select_evidence).parameters

    assert "gold_inds" not in parameters
    assert "ann_text_rows" not in parameters
    assert "ann_table_rows" not in parameters


@pytest.mark.parametrize(
    ("question", "passages", "table", "budget", "message"),
    [
        ("", [], [], 10, "question"),
        ("revenue", ["valid"], [], -1, "non-negative"),
        ("revenue", ["valid"], [["Year"], ["2023", "125"]], 10, "consistent width"),
        ("revenue", ["valid"], [], 10, "token_counter"),
    ],
)
def test_rejects_invalid_inputs(
    question,
    passages,
    table,
    budget,
    message,
):
    counter = None if message == "token_counter" else word_count

    with pytest.raises(ValueError, match=message):
        select_evidence(
            question,
            passages,
            table,
            token_budget=budget,
            token_counter=counter,
        )
