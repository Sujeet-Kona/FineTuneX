"""Pinned model and LoRA initialization for FineTuneX."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class TrainingComponents:
    """Objects required by the training pipeline."""

    model: Any
    tokenizer: Any
    device: torch.device
    trainable_parameters: list[torch.nn.Parameter]
    trainable_parameter_count: int
    total_parameter_count: int


def _select_device() -> torch.device:
    """Select the best supported local training device."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    raise RuntimeError(
        "No supported accelerator found. FineTuneX currently requires "
        "CUDA or Apple MPS for its configured FP16 training."
    )


def _validate_configs(
    model_config: dict[str, Any],
    training_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate model and LoRA configuration before loading weights."""
    if not isinstance(model_config, dict) or "model" not in model_config:
        raise ValueError("Missing model configuration section")

    if not isinstance(training_config, dict) or "training" not in training_config:
        raise ValueError("Missing training configuration section")

    model = model_config["model"]
    training = training_config["training"]

    if not isinstance(model, dict) or not isinstance(training, dict):
        raise ValueError("Model and training sections must be mappings")

    loading = model.get("loading")
    tokenizer_config = model.get("tokenizer")
    lora = training.get("lora")

    if not isinstance(loading, dict):
        raise ValueError("model.loading must be a mapping")

    if not isinstance(tokenizer_config, dict):
        raise ValueError("model.tokenizer must be a mapping")

    if not isinstance(lora, dict):
        raise ValueError("training.lora must be a mapping")

    if model.get("task") != "text-generation":
        raise ValueError("Only text-generation models are supported")

    if model.get("trust_remote_code") is not False:
        raise ValueError("trust_remote_code must be false")

    if training.get("method") != "lora":
        raise ValueError("Only LoRA training is supported")

    if loading.get("quantization") != "none":
        raise ValueError("Quantized model loading is not supported")

    if loading.get("local_files_only") is not True:
        raise ValueError("Model loading must use local cached files only")

    if loading.get("dtype") not in {"float16", "float32", "bfloat16"}:
        raise ValueError("Unsupported model dtype")

    model_name = model.get("name")
    model_revision = model.get("revision")

    if not isinstance(model_name, str) or not model_name:
        raise ValueError("Invalid model name")

    if not isinstance(model_revision, str) or not model_revision:
        raise ValueError("Model revision must be pinned")

    if tokenizer_config.get("name") != model_name:
        raise ValueError("Tokenizer and model names must match")

    if tokenizer_config.get("revision") != model_revision:
        raise ValueError("Tokenizer and model revisions must match")

    if tokenizer_config.get("padding_side") != "right":
        raise ValueError("Only right padding is supported")

    if training.get("gradient_checkpointing") is not True:
        raise ValueError("Gradient checkpointing must be enabled")

    if not isinstance(
        training.get("gradient_checkpointing_use_reentrant"),
        bool,
    ):
        raise ValueError("Invalid gradient checkpointing configuration")

    target_modules = lora.get("target_modules")

    if (
        not isinstance(target_modules, list)
        or not target_modules
        or any(not isinstance(name, str) or not name for name in target_modules)
        or len(target_modules) != len(set(target_modules))
    ):
        raise ValueError("LoRA target_modules must be a unique non-empty list")

    rank = lora.get("rank")
    alpha = lora.get("alpha")
    dropout = lora.get("dropout")

    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("LoRA rank must be a positive integer")

    if isinstance(alpha, bool) or not isinstance(alpha, int) or alpha <= 0:
        raise ValueError("LoRA alpha must be a positive integer")

    if (
        isinstance(dropout, bool)
        or not isinstance(dropout, (int, float))
        or not 0 <= dropout < 1
    ):
        raise ValueError("LoRA dropout must be in the range [0, 1)")

    return model, training


def load_training_components(
    model_config: dict[str, Any],
    training_config: dict[str, Any],
) -> TrainingComponents:
    """Load the pinned model and tokenizer and attach LoRA adapters."""
    model_settings, training_settings = _validate_configs(
        model_config,
        training_config,
    )

    loading = model_settings["loading"]
    tokenizer_settings = model_settings["tokenizer"]
    lora_settings = training_settings["lora"]

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }

    device = _select_device()
    dtype = dtype_map[loading["dtype"]]

    if device.type == "mps" and dtype == torch.bfloat16:
        raise ValueError("bfloat16 is not selected for this MPS training configuration")

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

    print(f"Loading base model on {device}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_settings["name"],
        revision=model_settings["revision"],
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
    )

    model.config.use_cache = False

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": training_settings["gradient_checkpointing_use_reentrant"]
        }
    )

    model.enable_input_require_grads()
    model.to(device)

    print("Attaching LoRA adapters...")

    peft_config = LoraConfig(
        r=lora_settings["rank"],
        lora_alpha=lora_settings["alpha"],
        lora_dropout=lora_settings["dropout"],
        target_modules=lora_settings["target_modules"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, peft_config)
    model.train()

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]

    total_parameter_count = sum(parameter.numel() for parameter in model.parameters())

    trainable_parameter_count = sum(
        parameter.numel() for parameter in trainable_parameters
    )

    if not trainable_parameters:
        raise RuntimeError("LoRA setup produced no trainable parameters")

    if total_parameter_count <= 0:
        raise RuntimeError("Model contains no parameters")

    unexpected_trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    ]

    if unexpected_trainable:
        raise RuntimeError(
            f"Unexpected non-LoRA trainable parameters: {unexpected_trainable[:10]}"
        )

    print("\n===== MODEL SUMMARY =====")
    print("Model:", model_settings["name"])
    print("Revision:", model_settings["revision"])
    print("Device:", device)
    print("Base dtype:", dtype)
    print("Gradient checkpointing: enabled")
    print("LoRA target modules:", lora_settings["target_modules"])
    print("Trainable parameters:", trainable_parameter_count)
    print("Total parameters:", total_parameter_count)
    print(
        "Trainable percentage:",
        round(100 * trainable_parameter_count / total_parameter_count, 4),
    )
    print("Tokenizer pad token ID:", tokenizer.pad_token_id)

    return TrainingComponents(
        model=model,
        tokenizer=tokenizer,
        device=device,
        trainable_parameters=trainable_parameters,
        trainable_parameter_count=trainable_parameter_count,
        total_parameter_count=total_parameter_count,
    )
