"""Safe, validation-driven checkpoint publishing for LoRA fine-tuning."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

_METADATA_FILENAME = "checkpoint_metadata.json"
_HISTORY_FILENAME = "training_history.json"
_SCHEMA_VERSION = 2


def _validate_metadata(metadata: dict[str, Any]) -> float:
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary")

    prompt_version = metadata.get("prompt_version")
    if not isinstance(prompt_version, str) or not prompt_version.strip():
        raise ValueError("metadata.prompt_version must be a non-empty string")

    validation_loss = metadata.get("validation_loss")

    if (
        isinstance(validation_loss, bool)
        or not isinstance(validation_loss, (int, float))
        or not math.isfinite(validation_loss)
        or validation_loss < 0
    ):
        raise ValueError("metadata.validation_loss must be finite and non-negative")

    return float(validation_loss)


def _read_existing_validation_loss(checkpoint_dir: Path) -> float:
    metadata_path = checkpoint_dir / _METADATA_FILENAME

    if not metadata_path.is_file():
        raise RuntimeError(
            f"Existing checkpoint is missing {_METADATA_FILENAME}: {checkpoint_dir}"
        )

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Existing checkpoint metadata is unreadable") from exc

    try:
        return _validate_metadata(metadata)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Existing checkpoint metadata is invalid") from exc


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(
            value,
            file,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())


def _write_json_atomically(path: Path, value: Any) -> None:
    """Atomically replace a JSON file without disturbing its previous version."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)

    try:
        _write_json(temporary_path, value)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _publish_directory(staging_dir: Path, checkpoint_dir: Path) -> None:
    """Publish staging and restore the previous checkpoint if publication fails."""
    backup_dir = checkpoint_dir.with_name(
        f".{checkpoint_dir.name}.backup-{uuid.uuid4().hex}"
    )
    had_previous_checkpoint = checkpoint_dir.exists()

    if had_previous_checkpoint:
        os.replace(checkpoint_dir, backup_dir)

    try:
        os.replace(staging_dir, checkpoint_dir)
    except Exception as publish_error:
        if had_previous_checkpoint and backup_dir.exists():
            try:
                os.replace(backup_dir, checkpoint_dir)
            except Exception as rollback_error:
                raise RuntimeError(
                    "Checkpoint publication failed and rollback also failed. "
                    f"The previous checkpoint is preserved at: {backup_dir}. "
                    f"Publication error: {publish_error!r}. "
                    f"Rollback error: {rollback_error!r}."
                ) from rollback_error
        raise

    if had_previous_checkpoint:
        # Publication succeeded. Backup cleanup is best-effort.
        shutil.rmtree(backup_dir, ignore_errors=True)


def save_best_checkpoint(
    checkpoint_dir: str | Path,
    model: Any,
    tokenizer: Any,
    metadata: dict[str, Any],
    history: list[dict[str, Any]],
) -> bool:
    """Save a checkpoint only when its validation loss strictly improves.

    Returns True when a new checkpoint is published, otherwise False.
    The model is expected to be a PEFT model supporting save_pretrained().
    """
    checkpoint_dir = Path(checkpoint_dir)

    if checkpoint_dir.name in {"", ".", ".."}:
        raise ValueError("checkpoint_dir must identify a dedicated directory")

    resolved_target = checkpoint_dir.resolve()
    protected_paths = {
        Path(resolved_target.anchor),
        Path.home().resolve(),
        Path.cwd().resolve(),
    }

    if resolved_target in protected_paths:
        raise ValueError("checkpoint_dir must not target a protected directory")

    if checkpoint_dir.is_symlink():
        raise ValueError("checkpoint_dir must not be a symbolic link")

    if not isinstance(history, list) or any(
        not isinstance(item, dict) for item in history
    ):
        raise TypeError("history must be a list of dictionaries")

    validation_loss = _validate_metadata(metadata)

    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)

    if checkpoint_dir.exists():
        if not checkpoint_dir.is_dir():
            raise ValueError("checkpoint_dir exists and is not a directory")

        previous_loss = _read_existing_validation_loss(checkpoint_dir)

        if validation_loss >= previous_loss:
            _write_json_atomically(
                checkpoint_dir / _HISTORY_FILENAME,
                history,
            )
            return False

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{checkpoint_dir.name}.staging-",
            dir=checkpoint_dir.parent,
        )
    )

    try:
        model.save_pretrained(
            staging_dir,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(staging_dir)

        if not any(staging_dir.iterdir()):
            raise RuntimeError("Model and tokenizer produced an empty checkpoint")

        checkpoint_metadata = {
            **metadata,
            "checkpoint_schema_version": _SCHEMA_VERSION,
            "validation_loss": validation_loss,
        }

        _write_json(
            staging_dir / _METADATA_FILENAME,
            checkpoint_metadata,
        )
        _write_json(
            staging_dir / _HISTORY_FILENAME,
            history,
        )

        _publish_directory(staging_dir, checkpoint_dir)
        return True
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
