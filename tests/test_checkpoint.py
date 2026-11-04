"""Tests for validation-driven, rollback-safe checkpoint publishing."""

import json
from pathlib import Path

import pytest

from finetunex.artifacts.checkpoint import save_best_checkpoint


class FakeModel:
    def __init__(self, content="adapter-v1", fail=False):
        self.content = content
        self.fail = fail

    def save_pretrained(self, directory, safe_serialization):
        assert safe_serialization is True

        if self.fail:
            raise RuntimeError("simulated model save failure")

        Path(directory, "adapter_model.safetensors").write_text(
            self.content,
            encoding="utf-8",
        )
        Path(directory, "adapter_config.json").write_text(
            '{"peft_type": "LORA"}',
            encoding="utf-8",
        )


class FakeTokenizer:
    def save_pretrained(self, directory):
        Path(directory, "tokenizer_config.json").write_text(
            '{"padding_side": "right"}',
            encoding="utf-8",
        )


def metadata(validation_loss, epoch=1):
    return {
        "validation_loss": validation_loss,
        "prompt_version": "finqa-program-v2-chat",
        "epoch": epoch,
        "global_step": 10,
        "model_revision": "pinned-revision",
    }


def read_metadata(checkpoint_dir):
    return json.loads(
        (checkpoint_dir / "checkpoint_metadata.json").read_text(encoding="utf-8")
    )


def test_saves_first_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"

    saved = save_best_checkpoint(
        checkpoint_dir,
        FakeModel(),
        FakeTokenizer(),
        metadata(1.25),
        [{"epoch": 1, "validation_loss": 1.25}],
    )

    assert saved is True
    assert (checkpoint_dir / "adapter_model.safetensors").read_text() == ("adapter-v1")
    assert (checkpoint_dir / "adapter_config.json").is_file()
    assert (checkpoint_dir / "tokenizer_config.json").is_file()
    assert read_metadata(checkpoint_dir)["validation_loss"] == 1.25
    assert read_metadata(checkpoint_dir)["checkpoint_schema_version"] == 2

    history = json.loads((checkpoint_dir / "training_history.json").read_text())
    assert len(history) == 1


@pytest.mark.parametrize("new_loss", [1.25, 1.5, 2.0])
def test_does_not_replace_checkpoint_without_improvement(tmp_path, new_loss):
    checkpoint_dir = tmp_path / "checkpoint"

    save_best_checkpoint(
        checkpoint_dir,
        FakeModel("best-adapter"),
        FakeTokenizer(),
        metadata(1.25),
        [],
    )

    saved = save_best_checkpoint(
        checkpoint_dir,
        FakeModel("worse-adapter"),
        FakeTokenizer(),
        metadata(new_loss, epoch=2),
        [{"epoch": 2, "validation_loss": new_loss}],
    )

    assert saved is False
    assert (checkpoint_dir / "adapter_model.safetensors").read_text() == (
        "best-adapter"
    )
    assert read_metadata(checkpoint_dir)["validation_loss"] == 1.25

    history = json.loads((checkpoint_dir / "training_history.json").read_text())
    assert history == [{"epoch": 2, "validation_loss": new_loss}]


def test_replaces_checkpoint_when_validation_improves(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"

    save_best_checkpoint(
        checkpoint_dir,
        FakeModel("old-adapter"),
        FakeTokenizer(),
        metadata(1.25),
        [],
    )

    saved = save_best_checkpoint(
        checkpoint_dir,
        FakeModel("new-best-adapter"),
        FakeTokenizer(),
        metadata(0.95, epoch=2),
        [{"epoch": 1, "validation_loss": 1.25}, {"epoch": 2, "validation_loss": 0.95}],
    )

    assert saved is True
    assert (checkpoint_dir / "adapter_model.safetensors").read_text() == (
        "new-best-adapter"
    )
    assert read_metadata(checkpoint_dir)["validation_loss"] == 0.95


def test_failed_model_save_preserves_previous_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"

    save_best_checkpoint(
        checkpoint_dir,
        FakeModel("known-good-adapter"),
        FakeTokenizer(),
        metadata(1.25),
        [],
    )

    with pytest.raises(RuntimeError, match="simulated model save failure"):
        save_best_checkpoint(
            checkpoint_dir,
            FakeModel("broken", fail=True),
            FakeTokenizer(),
            metadata(0.75, epoch=2),
            [],
        )

    assert (checkpoint_dir / "adapter_model.safetensors").read_text() == (
        "known-good-adapter"
    )
    assert read_metadata(checkpoint_dir)["validation_loss"] == 1.25


@pytest.mark.parametrize(
    "invalid_loss",
    [float("nan"), float("inf"), -1.0, True, "0.5", None],
)
def test_rejects_invalid_validation_loss(tmp_path, invalid_loss):
    with pytest.raises(ValueError, match="validation_loss"):
        save_best_checkpoint(
            tmp_path / "checkpoint",
            FakeModel(),
            FakeTokenizer(),
            metadata(invalid_loss),
            [],
        )


def test_rejects_invalid_history(tmp_path):
    with pytest.raises(TypeError, match="history"):
        save_best_checkpoint(
            tmp_path / "checkpoint",
            FakeModel(),
            FakeTokenizer(),
            metadata(1.0),
            [{"epoch": 1}, "invalid"],
        )


def test_rejects_checkpoint_symlink(tmp_path):
    actual_dir = tmp_path / "actual"
    actual_dir.mkdir()

    checkpoint_link = tmp_path / "checkpoint"
    checkpoint_link.symlink_to(actual_dir, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link"):
        save_best_checkpoint(
            checkpoint_link,
            FakeModel(),
            FakeTokenizer(),
            metadata(1.0),
            [],
        )


def test_rejects_existing_checkpoint_with_corrupt_metadata(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "checkpoint_metadata.json").write_text(
        "{broken",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="unreadable"):
        save_best_checkpoint(
            checkpoint_dir,
            FakeModel(),
            FakeTokenizer(),
            metadata(0.5),
            [],
        )

    assert (checkpoint_dir / "checkpoint_metadata.json").read_text() == ("{broken")


def test_failed_tokenizer_save_preserves_previous_checkpoint(tmp_path):
    class BrokenTokenizer:
        def save_pretrained(self, directory):
            raise RuntimeError("simulated tokenizer save failure")

    checkpoint_dir = tmp_path / "checkpoint"

    save_best_checkpoint(
        checkpoint_dir,
        FakeModel("known-good-adapter"),
        FakeTokenizer(),
        metadata(1.25),
        [],
    )

    with pytest.raises(RuntimeError, match="tokenizer save failure"):
        save_best_checkpoint(
            checkpoint_dir,
            FakeModel("new-adapter"),
            BrokenTokenizer(),
            metadata(0.75, epoch=2),
            [],
        )

    assert (checkpoint_dir / "adapter_model.safetensors").read_text() == (
        "known-good-adapter"
    )
    assert read_metadata(checkpoint_dir)["validation_loss"] == 1.25


def test_rejects_existing_non_directory_checkpoint_path(tmp_path):
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ValueError, match="not a directory"):
        save_best_checkpoint(
            checkpoint_path,
            FakeModel(),
            FakeTokenizer(),
            metadata(1.0),
            [],
        )


def test_preserves_backup_if_publication_and_rollback_both_fail(tmp_path, monkeypatch):
    import finetunex.artifacts.checkpoint as checkpoint_module

    checkpoint_dir = tmp_path / "checkpoint"

    save_best_checkpoint(
        checkpoint_dir,
        FakeModel("known-good-adapter"),
        FakeTokenizer(),
        metadata(1.25),
        [],
    )

    real_replace = checkpoint_module.os.replace
    calls = 0

    def failing_replace(source, destination):
        nonlocal calls
        calls += 1

        if calls == 2:
            raise OSError("simulated publication failure")

        if calls == 3:
            raise OSError("simulated rollback failure")

        return real_replace(source, destination)

    monkeypatch.setattr(checkpoint_module.os, "replace", failing_replace)

    with pytest.raises(RuntimeError, match="rollback also failed") as exc_info:
        save_best_checkpoint(
            checkpoint_dir,
            FakeModel("new-adapter"),
            FakeTokenizer(),
            metadata(0.75, epoch=2),
            [],
        )

    assert "simulated publication failure" in str(exc_info.value)
    assert "simulated rollback failure" in str(exc_info.value)

    backups = list(tmp_path.glob(".checkpoint.backup-*"))
    assert len(backups) == 1
    assert (
        backups[0] / "adapter_model.safetensors"
    ).read_text() == "known-good-adapter"


@pytest.mark.parametrize(
    "unsafe_path",
    [Path("."), Path(".."), Path("/")],
)
def test_rejects_protected_checkpoint_paths(tmp_path, unsafe_path):
    with pytest.raises(ValueError, match="dedicated|protected"):
        save_best_checkpoint(
            unsafe_path,
            FakeModel(),
            FakeTokenizer(),
            metadata(1.0),
            [],
        )


def test_rejects_home_directory_as_checkpoint_target():
    with pytest.raises(ValueError, match="protected"):
        save_best_checkpoint(
            Path.home(),
            FakeModel(),
            FakeTokenizer(),
            metadata(1.0),
            [],
        )
