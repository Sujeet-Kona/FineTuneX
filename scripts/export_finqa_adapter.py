"""Export a verified FinQA LoRA adapter as a merged Transformers model.

The source checkpoint is never modified. The destination must be new.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finetunex.config import load_yaml_config  # noqa: E402
from finetunex.inference.generator import _validate_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _validate_destination(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        raise ValueError("Export destination must differ from the source checkpoint")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Export destination already exists: {destination}")
    if any(parent.is_symlink() for parent in (destination, *destination.parents)):
        raise ValueError("Export destination path cannot contain symlinks")

    protected_paths = {
        Path(destination.resolve().anchor),
        Path.home().resolve(),
        PROJECT_ROOT.resolve(),
    }
    if destination.resolve() in protected_paths:
        raise ValueError("Export destination cannot be a protected directory")


def main() -> int:
    args = parse_args()
    source = _resolve(args.checkpoint)
    destination = _resolve(
        args.output_dir or source.parent / f"{source.name}-merged"
    )
    _validate_destination(source, destination)

    model_config = load_yaml_config(PROJECT_ROOT / "configs/model.yaml")
    training_config = load_yaml_config(PROJECT_ROOT / "configs/training.yaml")
    metadata: dict[str, Any] = _validate_checkpoint(
        source,
        model_config,
        training_config,
    )

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    settings = model_config["model"]
    loading = settings["loading"]
    tokenizer_settings = settings["tokenizer"]
    dtype = {"float16": torch.float16, "float32": torch.float32}[
        loading["dtype"]
    ]

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_settings["name"],
        revision=tokenizer_settings["revision"],
        local_files_only=True,
        trust_remote_code=False,
    )
    base = AutoModelForCausalLM.from_pretrained(
        settings["name"],
        revision=settings["revision"],
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    adapter = PeftModel.from_pretrained(
        base,
        str(source),
        is_trainable=False,
        local_files_only=True,
    )
    merged = adapter.merge_and_unload(safe_merge=True)

    destination.mkdir(parents=True, exist_ok=False)
    try:
        merged.save_pretrained(destination, safe_serialization=True)
        tokenizer.save_pretrained(destination)
        export_metadata = {
            "export_schema_version": 1,
            "format": "merged-transformers",
            "source_checkpoint": str(source.resolve()),
            "model": metadata["model"],
            "model_revision": metadata["model_revision"],
            "tokenizer": metadata["tokenizer"],
            "tokenizer_revision": metadata["tokenizer_revision"],
            "prompt_version": metadata["prompt_version"],
            "validation_loss": metadata["validation_loss"],
        }
        (destination / "export_metadata.json").write_text(
            json.dumps(export_metadata, indent=2, sort_keys=True) + "\\n",
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise

    print(f"Merged model exported to: {destination}")
    print("Source adapter left unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
