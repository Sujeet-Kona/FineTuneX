import json

import pytest

from finetunex.data.pipeline import (
    DatasetValidationError,
    format_example,
    load_jsonl,
    prepare_dataset,
    split_dataset,
    validate_records,
)

COLUMNS = {
    "instruction": "instruction",
    "input": "input",
    "output": "output",
}


def make_records(count=10):
    return [
        {
            "instruction": f"Explain payment status {i}",
            "input": f"Transaction reference {i}",
            "output": f"Payment status explanation {i}",
        }
        for i in range(count)
    ]


def test_load_jsonl(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        '{"instruction":"Explain","input":"","output":"Done"}\n',
        encoding="utf-8",
    )

    assert len(load_jsonl(path)) == 1


def test_invalid_json_raises_error(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{invalid json}\n", encoding="utf-8")

    with pytest.raises(DatasetValidationError):
        load_jsonl(path)


def test_validation_removes_exact_duplicates():
    records = make_records(3)
    records.append(records[0])

    validated, duplicates = validate_records(records, COLUMNS)

    assert len(validated) == 3
    assert duplicates == 1


def test_validation_rejects_empty_instruction():
    records = make_records(3)
    records[0]["instruction"] = "  "

    with pytest.raises(DatasetValidationError):
        validate_records(records, COLUMNS)


def test_validation_rejects_non_string_output():
    records = make_records(3)
    records[0]["output"] = 123

    with pytest.raises(DatasetValidationError):
        validate_records(records, COLUMNS)


def test_format_example():
    example = {
        "instruction": "Explain a refund",
        "input": "Refund requested",
        "output": "The refund is being processed.",
    }

    result = format_example(example)

    assert "### Instruction:" in result
    assert "### Input:" in result
    assert "### Response:" in result


def test_split_is_deterministic():
    records = make_records(20)

    first = split_dataset(records, 0.8, 0.1, 0.1, 42)
    second = split_dataset(records, 0.8, 0.1, 0.1, 42)

    assert first == second


def test_split_has_no_overlap():
    records = make_records(20)
    splits = split_dataset(records, 0.8, 0.1, 0.1, 42)

    all_instructions = [
        example["instruction"]
        for examples in splits.values()
        for example in examples
    ]

    assert len(all_instructions) == len(set(all_instructions))


def test_prepare_dataset_writes_manifest_and_splits(tmp_path):
    source = tmp_path / "input.jsonl"
    output = tmp_path / "processed"

    records = make_records(20)
    source.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    manifest = prepare_dataset(source, output, COLUMNS)

    assert manifest["examples_validated"] == 20
    assert manifest["split_counts"] == {
        "train": 16,
        "validation": 2,
        "test": 2,
    }

    assert (output / "train.jsonl").exists()
    assert (output / "validation.jsonl").exists()
    assert (output / "test.jsonl").exists()
    assert (output / "manifest.json").exists()


def test_split_rejects_invalid_ratios():
    with pytest.raises(ValueError):
        split_dataset(make_records(10), 0.8, 0.1, 0.2, 42)
