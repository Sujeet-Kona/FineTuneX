"""Tests for FinQA training entrypoint validation helpers."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_finqa import (  # noqa: E402
    processed_example_count,
    validate_dataset_provenance,
    validate_fresh_output_dir,
)


@pytest.fixture
def dataset_config():
    return {
        "name": "finqa",
        "source": {
            "repository": "bevaya/FinQA",
            "revision": "pinned-revision",
        },
    }


def test_accepts_matching_dataset_provenance(dataset_config):
    validate_dataset_provenance(
        {
            "name": "FinQA",
            "repository": "bevaya/FinQA",
            "revision": "pinned-revision",
        },
        dataset_config,
    )


def test_rejects_missing_dataset_provenance(dataset_config):
    with pytest.raises(ValueError, match="no dataset provenance"):
        validate_dataset_provenance(None, dataset_config)


def test_rejects_different_dataset(dataset_config):
    with pytest.raises(ValueError, match="different dataset"):
        validate_dataset_provenance(
            {
                "name": "OtherDataset",
                "repository": "bevaya/FinQA",
                "revision": "pinned-revision",
            },
            dataset_config,
        )


@pytest.mark.parametrize(
    "repository,revision",
    [
        ("another/repository", "pinned-revision"),
        ("bevaya/FinQA", "different-revision"),
    ],
)
def test_rejects_different_dataset_source(dataset_config, repository, revision):
    with pytest.raises(ValueError, match="different dataset revision"):
        validate_dataset_provenance(
            {
                "name": "FinQA",
                "repository": repository,
                "revision": revision,
            },
            dataset_config,
        )


def test_rejects_existing_output_directory(tmp_path):
    output_dir = tmp_path / "existing"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="starts a fresh training run"):
        validate_fresh_output_dir(output_dir)


def test_accepts_nonexistent_output_directory(tmp_path):
    output_dir = tmp_path / "new-checkpoint"

    validate_fresh_output_dir(output_dir)

    assert not output_dir.exists()


def test_rejects_output_directory_symlink(tmp_path):
    target = tmp_path / "target"
    target.mkdir()

    output_link = tmp_path / "checkpoint-link"
    output_link.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="symlinks"):
        validate_fresh_output_dir(output_link)


def test_rejects_symlinked_parent_directory(tmp_path):
    target = tmp_path / "real-parent"
    target.mkdir()

    parent_link = tmp_path / "linked-parent"
    parent_link.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="symlinks"):
        validate_fresh_output_dir(parent_link / "checkpoint")


@pytest.mark.parametrize(
    ("dataset_size", "microbatches", "batch_size", "expected"),
    [
        (10, 3, 4, 10),
        (10, 2, 4, 8),
        (10, 1, 4, 4),
        (8, 2, 4, 8),
        (1, 1, 4, 1),
    ],
)
def test_processed_example_count(
    dataset_size,
    microbatches,
    batch_size,
    expected,
):
    assert (
        processed_example_count(
            dataset_size,
            microbatches,
            batch_size,
        )
        == expected
    )
