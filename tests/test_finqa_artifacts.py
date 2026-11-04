import hashlib
import json
from pathlib import Path

import pytest

from finetunex.data import finqa_artifacts


class FakeTokenizer:
    eos_token = "<eos>"
    eos_token_id = 99


def create_source(tmp_path):
    input_dir = tmp_path / "source"
    input_dir.mkdir()

    records = {
        "train": [{"id": "train-1", "question": "Revenue?"}],
        "validation": [{"id": "validation-1", "question": "Profit?"}],
    }

    hashes = {}

    for split, split_records in records.items():
        path = input_dir / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in split_records),
            encoding="utf-8",
        )
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        "dataset": "FinQA",
        "repository": "bevaya/FinQA",
        "revision": "test-revision",
        "split_counts": {"train": 1, "validation": 1, "test": 1},
        "processed_sha256": hashes,
        "official_splits_preserved": True,
        "examples_removed": 0,
    }

    (input_dir / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    return input_dir


def fake_preprocess(record, tokenizer, *, split, **kwargs):
    return {
        "input_ids": [10, 99],
        "attention_mask": [1, 1],
        "labels": [-100, 99],
        "metadata": {
            "record_id": record["id"],
            "split": split,
            "table_strategy": "complete",
            "prompt_tokens": 1,
            "target_tokens_including_eos": 1,
            "evidence_tokens": 12,
            "sequence_length": 2,
        },
    }


def run_preprocessing(input_dir, output_dir, *, overwrite=False):
    return finqa_artifacts.preprocess_finqa_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        source_repository="bevaya/FinQA",
        source_revision="test-revision",
        expected_counts={"train": 1, "validation": 1},
        tokenizer=FakeTokenizer(),
        tokenizer_name="test-tokenizer",
        tokenizer_revision="test-revision",
        model_name="test-model",
        model_revision="test-revision",
        evidence_budget=1024,
        max_sequence_length=1536,
        config_hashes={"model.yaml": "abc123"},
        overwrite=overwrite,
    )


@pytest.fixture
def mocked_preprocessor(monkeypatch):
    monkeypatch.setattr(
        finqa_artifacts,
        "preprocess_record",
        fake_preprocess,
    )


def test_writes_train_validation_and_manifest(
    tmp_path,
    mocked_preprocessor,
):
    input_dir = create_source(tmp_path)
    output_dir = tmp_path / "artifacts" / "finqa"

    manifest = run_preprocessing(input_dir, output_dir)

    assert manifest["total_records"] == 2
    assert manifest["preprocessing"]["test_split_included"] is False
    assert manifest["splits"]["train"]["record_count"] == 1
    assert manifest["splits"]["validation"]["record_count"] == 1

    assert (output_dir / "train.jsonl").is_file()
    assert (output_dir / "validation.jsonl").is_file()
    assert (output_dir / "manifest.json").is_file()
    assert not (output_dir / "test.jsonl").exists()

    train_record = json.loads((output_dir / "train.jsonl").read_text(encoding="utf-8"))

    assert train_record["input_ids"] == [10, 99]
    assert train_record["labels"] == [-100, 99]


def test_serialization_is_deterministic(
    tmp_path,
    mocked_preprocessor,
):
    input_dir = create_source(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"

    run_preprocessing(input_dir, first)
    run_preprocessing(input_dir, second)

    for filename in ("train.jsonl", "validation.jsonl", "manifest.json"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_refuses_to_overwrite_existing_output(
    tmp_path,
    mocked_preprocessor,
):
    input_dir = create_source(tmp_path)
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    sentinel = output_dir / "keep.txt"
    sentinel.write_text("original", encoding="utf-8")

    with pytest.raises(FileExistsError):
        run_preprocessing(input_dir, output_dir)

    assert sentinel.read_text(encoding="utf-8") == "original"


def test_rejects_modified_source_file(
    tmp_path,
    mocked_preprocessor,
):
    input_dir = create_source(tmp_path)

    with (input_dir / "train.jsonl").open("a", encoding="utf-8") as file:
        file.write('{"id":"tampered"}\n')

    output_dir = tmp_path / "artifacts"

    with pytest.raises(ValueError, match="source manifest hash"):
        run_preprocessing(input_dir, output_dir)

    assert not output_dir.exists()


def test_rejects_wrong_dataset_revision(
    tmp_path,
    mocked_preprocessor,
):
    input_dir = create_source(tmp_path)
    manifest_path = input_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["revision"] = "wrong-revision"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="revision"):
        run_preprocessing(input_dir, tmp_path / "artifacts")


def test_restores_previous_output_if_publish_fails(
    tmp_path,
    mocked_preprocessor,
    monkeypatch,
):
    input_dir = create_source(tmp_path)
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    sentinel = output_dir / "keep.txt"
    sentinel.write_text("original", encoding="utf-8")

    original_replace = finqa_artifacts.os.replace

    def failing_replace(source, destination):
        source_path = Path(source)

        if ".staging-" in source_path.name:
            raise OSError("simulated publish failure")

        return original_replace(source, destination)

    monkeypatch.setattr(finqa_artifacts.os, "replace", failing_replace)

    with pytest.raises(OSError, match="simulated publish failure"):
        run_preprocessing(input_dir, output_dir, overwrite=True)

    assert sentinel.read_text(encoding="utf-8") == "original"
    assert not list(tmp_path.glob(".artifacts.backup-*"))


def test_rejects_boolean_record_counts(tmp_path, mocked_preprocessor):
    input_dir = create_source(tmp_path)

    with pytest.raises(ValueError, match="positive integer"):
        finqa_artifacts.preprocess_finqa_dataset(
            input_dir=input_dir,
            output_dir=tmp_path / "artifacts",
            source_repository="bevaya/FinQA",
            source_revision="test-revision",
            expected_counts={"train": True, "validation": 1},
            tokenizer=FakeTokenizer(),
            tokenizer_name="test-tokenizer",
            tokenizer_revision="test-revision",
            model_name="test-model",
            model_revision="test-revision",
            evidence_budget=1024,
            max_sequence_length=1536,
            config_hashes={},
        )


def test_rejects_output_inside_source_directory(
    tmp_path,
    mocked_preprocessor,
):
    input_dir = create_source(tmp_path)

    with pytest.raises(ValueError, match="must not overlap"):
        run_preprocessing(input_dir, input_dir / "generated")


def test_rejects_output_parent_of_source_directory(
    tmp_path,
    mocked_preprocessor,
):
    input_dir = create_source(tmp_path)

    with pytest.raises(ValueError, match="must not overlap"):
        run_preprocessing(input_dir, tmp_path)


def test_rejects_misaligned_attention_mask(tmp_path, monkeypatch):
    input_dir = create_source(tmp_path)

    def malformed_preprocess(record, tokenizer, *, split, **kwargs):
        result = fake_preprocess(record, tokenizer, split=split, **kwargs)
        result["attention_mask"] = [1]
        return result

    monkeypatch.setattr(
        finqa_artifacts,
        "preprocess_record",
        malformed_preprocess,
    )

    with pytest.raises(RuntimeError, match="invalid lengths"):
        run_preprocessing(input_dir, tmp_path / "artifacts")
