"""Tests for model configuration validation and initialization."""

from unittest.mock import MagicMock

import pytest
import torch

from finetunex.training import model as model_module


@pytest.fixture
def configs():
    model_config = {
        "model": {
            "name": "Qwen/Qwen3-0.6B",
            "revision": "pinned-revision",
            "task": "text-generation",
            "trust_remote_code": False,
            "loading": {
                "dtype": "float16",
                "quantization": "none",
                "local_files_only": True,
            },
            "tokenizer": {
                "name": "Qwen/Qwen3-0.6B",
                "revision": "pinned-revision",
                "padding_side": "right",
            },
        }
    }

    training_config = {
        "training": {
            "method": "lora",
            "gradient_checkpointing": True,
            "gradient_checkpointing_use_reentrant": False,
            "lora": {
                "rank": 8,
                "alpha": 16,
                "dropout": 0.05,
                "target_modules": [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            },
        }
    }

    return model_config, training_config


def test_validates_model_and_training_configs(configs):
    model_config, training_config = configs

    model, training = model_module._validate_configs(
        model_config,
        training_config,
    )

    assert model["name"] == "Qwen/Qwen3-0.6B"
    assert training["method"] == "lora"


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("model", "revision"), "", "revision"),
        (("model", "loading", "quantization"), "4bit", "Quantized"),
        (("model", "loading", "local_files_only"), False, "cached files"),
        (("model", "tokenizer", "revision"), "different", "revisions"),
        (("training", "method"), "qlora", "Only LoRA"),
        (("training", "gradient_checkpointing"), False, "checkpointing"),
        (("training", "lora", "rank"), 0, "rank"),
        (("training", "lora", "dropout"), 1.0, "dropout"),
        (("training", "lora", "target_modules"), [], "target_modules"),
    ],
)
def test_rejects_invalid_configuration(configs, path, value, message):
    model_config, training_config = configs

    target = model_config if path[0] == "model" else training_config

    for key in path[:-1]:
        target = target[key]

    target[path[-1]] = value

    with pytest.raises(ValueError, match=message):
        model_module._validate_configs(model_config, training_config)


def test_loads_model_and_attaches_lora(monkeypatch, configs):
    model_config, training_config = configs

    fake_tokenizer = MagicMock()
    fake_tokenizer.pad_token_id = 1
    fake_tokenizer.eos_token = "<eos>"

    fake_model = MagicMock()
    fake_model.config = MagicMock()

    base_parameter = torch.nn.Parameter(
        torch.ones(4),
        requires_grad=False,
    )
    lora_parameter = torch.nn.Parameter(torch.ones(2))

    # Return a fresh iterator on every parameters() call.
    fake_model.parameters.side_effect = lambda: iter([base_parameter, lora_parameter])

    fake_model.named_parameters.return_value = [
        ("base.weight", base_parameter),
        ("lora_A.default.weight", lora_parameter),
    ]

    fake_model.to.return_value = fake_model
    fake_model.train.return_value = fake_model

    monkeypatch.setattr(
        model_module.AutoTokenizer,
        "from_pretrained",
        MagicMock(return_value=fake_tokenizer),
    )
    monkeypatch.setattr(
        model_module.AutoModelForCausalLM,
        "from_pretrained",
        MagicMock(return_value=fake_model),
    )
    monkeypatch.setattr(
        model_module,
        "get_peft_model",
        MagicMock(return_value=fake_model),
    )
    monkeypatch.setattr(
        model_module,
        "_select_device",
        lambda: torch.device("mps"),
    )

    result = model_module.load_training_components(
        model_config,
        training_config,
    )

    assert result.model is fake_model
    assert result.tokenizer is fake_tokenizer
    assert result.device == torch.device("mps")
    assert result.trainable_parameter_count == 2
    assert result.total_parameter_count == 6

    fake_model.gradient_checkpointing_enable.assert_called_once_with(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    fake_model.enable_input_require_grads.assert_called_once()
    fake_model.to.assert_called_once_with(torch.device("mps"))
    fake_model.train.assert_called_once()


def test_rejects_missing_model_section():
    with pytest.raises(ValueError, match="Missing model"):
        model_module._validate_configs({}, {"training": {}})


def test_rejects_unsupported_dtype(configs):
    model_config, training_config = configs
    model_config["model"]["loading"]["dtype"] = "int8"

    with pytest.raises(ValueError, match="dtype"):
        model_module._validate_configs(model_config, training_config)
