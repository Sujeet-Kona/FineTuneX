"""Pinned base-model and LoRA adapter loading for FinQA inference."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from finetunex.data.finqa_preprocessing import PROMPT_VERSION


@dataclass
class InferenceComponents:
    """Model, tokenizer, device, and verified checkpoint provenance."""

    model: Any
    tokenizer: Any
    device: torch.device
    checkpoint_metadata: dict[str, Any]


def _select_inference_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    raise RuntimeError("FineTuneX inference requires CUDA or Apple MPS.")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON from {path}") from exc

    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")

    return value


def _validate_checkpoint(
    checkpoint_dir: Path,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(model_config, dict) or not isinstance(
        model_config.get("model"), dict
    ):
        raise ValueError("Invalid model configuration")

    if not isinstance(training_config, dict) or not isinstance(
        training_config.get("training"), dict
    ):
        raise ValueError("Invalid training configuration")

    model_settings = model_config["model"]
    training_settings = training_config["training"]
    loading = model_settings.get("loading")
    tokenizer_settings = model_settings.get("tokenizer")
    lora_settings = training_settings.get("lora")

    if not isinstance(loading, dict):
        raise ValueError("model.loading must be a mapping")

    if not isinstance(tokenizer_settings, dict):
        raise ValueError("model.tokenizer must be a mapping")

    if not isinstance(lora_settings, dict):
        raise ValueError("training.lora must be a mapping")

    if model_settings.get("task") != "text-generation":
        raise ValueError("Only text-generation models are supported")

    if model_settings.get("trust_remote_code") is not False:
        raise ValueError("trust_remote_code must be false")

    if loading.get("local_files_only") is not True:
        raise ValueError("Model loading must use local cached files only")

    if loading.get("quantization") != "none":
        raise ValueError("Quantized model loading is not supported")

    if loading.get("dtype") not in {"float16", "float32"}:
        raise ValueError("Inference supports float16 or float32")

    model_name = model_settings.get("name")
    model_revision = model_settings.get("revision")

    if not isinstance(model_name, str) or not model_name:
        raise ValueError("Model name must be non-empty")

    if not isinstance(model_revision, str) or not model_revision:
        raise ValueError("Model revision must be pinned")

    if tokenizer_settings.get("name") != model_name:
        raise ValueError("Tokenizer and model names must match")

    if tokenizer_settings.get("revision") != model_revision:
        raise ValueError("Tokenizer and model revisions must match")

    if tokenizer_settings.get("padding_side") != "right":
        raise ValueError("Only right padding is supported")

    if not checkpoint_dir.is_dir() or checkpoint_dir.is_symlink():
        raise ValueError("Checkpoint must be an existing non-symlink directory")

    required_files = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "checkpoint_metadata.json",
    )

    for filename in required_files:
        path = checkpoint_dir / filename
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Checkpoint is missing a regular {filename} file")

    metadata = _read_json(checkpoint_dir / "checkpoint_metadata.json")
    adapter_config = _read_json(checkpoint_dir / "adapter_config.json")

    if metadata.get("checkpoint_schema_version") != 2:
        raise ValueError("Unsupported checkpoint metadata schema")

    if metadata.get("prompt_version") != PROMPT_VERSION:
        raise ValueError(
            "Checkpoint prompt version does not match inference prompt version"
        )

    if metadata.get("model") != model_name:
        raise ValueError("Checkpoint base model name does not match configuration")

    if metadata.get("model_revision") != model_revision:
        raise ValueError("Checkpoint base model revision does not match configuration")

    if metadata.get("tokenizer") != tokenizer_settings["name"]:
        raise ValueError("Checkpoint tokenizer does not match configuration")

    if metadata.get("tokenizer_revision") != tokenizer_settings["revision"]:
        raise ValueError("Checkpoint tokenizer revision does not match configuration")

    if metadata.get("base_dtype") != loading["dtype"]:
        raise ValueError("Checkpoint base dtype does not match configuration")

    if adapter_config.get("base_model_name_or_path") != model_name:
        raise ValueError("Adapter base model name does not match configuration")

    if adapter_config.get("peft_type") != "LORA":
        raise ValueError("Checkpoint is not a LoRA adapter")

    if adapter_config.get("task_type") != "CAUSAL_LM":
        raise ValueError("Adapter task type must be CAUSAL_LM")

    if adapter_config.get("inference_mode") is not True:
        raise ValueError("Adapter must be saved in inference mode")

    if adapter_config.get("r") != lora_settings.get("rank"):
        raise ValueError("Adapter LoRA rank does not match training configuration")

    if adapter_config.get("lora_alpha") != lora_settings.get("alpha"):
        raise ValueError("Adapter LoRA alpha does not match training configuration")

    if not math.isclose(
        adapter_config.get("lora_dropout", -1),
        lora_settings.get("dropout", -2),
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise ValueError("Adapter LoRA dropout does not match training configuration")

    adapter_targets = adapter_config.get("target_modules")
    configured_targets = lora_settings.get("target_modules")

    if (
        not isinstance(adapter_targets, list)
        or not isinstance(configured_targets, list)
        or set(adapter_targets) != set(configured_targets)
    ):
        raise ValueError("Adapter target modules do not match training configuration")

    if metadata.get("dataset") != "finqa":
        raise ValueError("Checkpoint dataset must be FinQA")

    return metadata


def load_inference_components(
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    checkpoint_dir: str | Path,
) -> InferenceComponents:
    """Load a pinned base model and a verified, frozen LoRA adapter."""
    checkpoint_path = Path(checkpoint_dir).expanduser().absolute()

    metadata = _validate_checkpoint(
        checkpoint_path,
        model_config,
        training_config,
    )

    model_settings = model_config["model"]
    loading = model_settings["loading"]
    tokenizer_settings = model_settings["tokenizer"]

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
    }

    device = _select_inference_device()
    dtype = dtype_map[loading["dtype"]]

    print("Loading pinned tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_settings["name"],
        revision=tokenizer_settings["revision"],
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer.padding_side = tokenizer_settings["padding_side"]

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has neither a pad token nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define an EOS token ID")

    print(f"Loading pinned base model on {device}...")

    base_model = AutoModelForCausalLM.from_pretrained(
        model_settings["name"],
        revision=model_settings["revision"],
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
    )

    base_model.config.use_cache = True

    if hasattr(base_model, "gradient_checkpointing_disable"):
        base_model.gradient_checkpointing_disable()

    base_model.to(device)

    print("Loading verified LoRA adapter...")

    model = PeftModel.from_pretrained(
        base_model,
        str(checkpoint_path),
        is_trainable=False,
        local_files_only=True,
    )

    model.config.use_cache = True
    model.eval()

    trainable_parameters = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]

    if trainable_parameters:
        raise RuntimeError(
            "Inference adapter unexpectedly contains trainable parameters: "
            f"{trainable_parameters[:10]}"
        )

    if not model.config.use_cache:
        raise RuntimeError("KV caching must be enabled for inference")

    print("\n===== INFERENCE MODEL SUMMARY =====")
    print("Model:", model_settings["name"])
    print("Revision:", model_settings["revision"])
    print("Checkpoint:", checkpoint_path)
    print("Device:", device)
    print("Dtype:", dtype)
    print("Adapter trainable parameters: 0")
    print("KV cache: enabled")

    return InferenceComponents(
        model=model,
        tokenizer=tokenizer,
        device=device,
        checkpoint_metadata=metadata,
    )


def load_base_inference_components(
    model_config: dict[str, Any],
    training_config: dict[str, Any],
) -> InferenceComponents:
    """Load the pinned, unadapted base model for a controlled benchmark."""
    if not isinstance(model_config, dict) or not isinstance(
        model_config.get("model"), dict
    ):
        raise ValueError("Invalid model configuration")
    if not isinstance(training_config, dict) or not isinstance(
        training_config.get("training"), dict
    ):
        raise ValueError("Invalid training configuration")

    model_settings = model_config["model"]
    training_settings = training_config["training"]
    loading = model_settings.get("loading")
    tokenizer_settings = model_settings.get("tokenizer")

    if not isinstance(loading, dict):
        raise ValueError("model.loading must be a mapping")
    if not isinstance(tokenizer_settings, dict):
        raise ValueError("model.tokenizer must be a mapping")
    if model_settings.get("task") != "text-generation":
        raise ValueError("Only text-generation models are supported")
    if model_settings.get("trust_remote_code") is not False:
        raise ValueError("trust_remote_code must be false")
    if loading.get("local_files_only") is not True:
        raise ValueError("Model loading must use local cached files only")
    if loading.get("quantization") != "none":
        raise ValueError("Quantized model loading is not supported")
    if loading.get("dtype") not in {"float16", "float32"}:
        raise ValueError("Inference supports float16 or float32")

    model_name = model_settings.get("name")
    revision = model_settings.get("revision")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("Model name must be non-empty")
    if not isinstance(revision, str) or not revision:
        raise ValueError("Model revision must be pinned")
    if tokenizer_settings.get("name") != model_name:
        raise ValueError("Tokenizer and model names must match")
    if tokenizer_settings.get("revision") != revision:
        raise ValueError("Tokenizer and model revisions must match")
    if tokenizer_settings.get("padding_side") != "right":
        raise ValueError("Only right padding is supported")

    max_sequence_length = training_settings.get("max_sequence_length")
    if (
        isinstance(max_sequence_length, bool)
        or not isinstance(max_sequence_length, int)
        or max_sequence_length <= 0
    ):
        raise ValueError("training.max_sequence_length must be a positive integer")

    dtype_map = {"float16": torch.float16, "float32": torch.float32}
    device = _select_inference_device()
    dtype = dtype_map[loading["dtype"]]

    print("Loading pinned tokenizer for base-model benchmark...")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_settings["name"],
        revision=tokenizer_settings["revision"],
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer.padding_side = tokenizer_settings["padding_side"]
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has neither a pad token nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define an EOS token ID")

    print(f"Loading unadapted base model on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    model.config.use_cache = True
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.to(device)
    model.eval()
    # from_pretrained() leaves parameters trainable by default. Freeze explicitly:
    # this is an inference-only benchmark, not a training setup.
    model.requires_grad_(False)

    trainable_parameters = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if trainable_parameters:
        raise RuntimeError(
            "Base model unexpectedly contains trainable parameters: "
            f"{trainable_parameters[:10]}"
        )
    if not model.config.use_cache:
        raise RuntimeError("KV caching must be enabled for inference")

    metadata = {
        "model": model_name,
        "model_revision": revision,
        "tokenizer": tokenizer_settings["name"],
        "tokenizer_revision": tokenizer_settings["revision"],
        "base_dtype": loading["dtype"],
        "training_config": {"max_sequence_length": max_sequence_length},
    }
    print("\n===== BASE MODEL SUMMARY =====")
    print("Model:", model_name)
    print("Revision:", revision)
    print("Device:", device)
    print("Dtype:", dtype)
    print("Trainable parameters: 0")
    print("KV cache: enabled")

    return InferenceComponents(
        model=model,
        tokenizer=tokenizer,
        device=device,
        checkpoint_metadata=metadata,
    )


def generate_program(
    components: InferenceComponents,
    prompt: str,
    *,
    max_new_tokens: int = 96,
) -> dict[str, Any]:
    """Generate one program deterministically and preserve raw decoded output."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    if (
        not isinstance(max_new_tokens, int)
        or isinstance(max_new_tokens, bool)
        or max_new_tokens <= 0
    ):
        raise ValueError("max_new_tokens must be a positive integer")

    tokenizer = components.tokenizer

    if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define EOS and padding token IDs")

    input_ids_list = tokenizer.encode(prompt, add_special_tokens=False)

    if not input_ids_list:
        raise ValueError("Prompt produced no tokens")

    training_config = components.checkpoint_metadata.get("training_config", {})
    max_sequence_length = training_config.get("max_sequence_length")

    if (
        isinstance(max_sequence_length, int)
        and len(input_ids_list) > max_sequence_length
    ):
        raise ValueError(
            f"Prompt has {len(input_ids_list)} tokens, exceeding "
            f"the configured maximum of {max_sequence_length}"
        )

    input_ids = torch.tensor(
        [input_ids_list],
        dtype=torch.long,
        device=components.device,
    )
    attention_mask = torch.ones_like(input_ids)

    components.model.eval()

    with torch.inference_mode():
        generated = components.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )

    if not isinstance(generated, torch.Tensor) or generated.ndim != 2:
        raise RuntimeError("Generation returned an unexpected output shape")

    if generated.shape[0] != 1 or generated.shape[1] < input_ids.shape[1]:
        raise RuntimeError("Generation output does not contain the input prompt")

    generated_token_ids = generated[0, input_ids.shape[1] :].tolist()

    raw_output = tokenizer.decode(
        generated_token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    normalized_output = tokenizer.decode(
        generated_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return {
        "raw_output": raw_output,
        "normalized_output": normalized_output,
        "generated_token_ids": generated_token_ids,
        "prompt_tokens": len(input_ids_list),
        "generated_tokens": len(generated_token_ids),
        "ended_with_eos": (
            bool(generated_token_ids)
            and generated_token_ids[-1] == tokenizer.eos_token_id
        ),
    }
