import json

import pytest
import torch

from finetunex.inference.generator import (
    InferenceComponents,
    _validate_checkpoint,
    generate_program,
)

REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def make_configs():
    return (
        {
            "model": {
                "name": "Qwen/Qwen3-0.6B",
                "revision": REVISION,
                "task": "text-generation",
                "trust_remote_code": False,
                "loading": {
                    "dtype": "float16",
                    "quantization": "none",
                    "local_files_only": True,
                },
                "tokenizer": {
                    "name": "Qwen/Qwen3-0.6B",
                    "revision": REVISION,
                    "padding_side": "right",
                },
            }
        },
        {
            "training": {
                "method": "lora",
                "max_sequence_length": 1536,
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
        },
    )


def make_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    metadata = {
        "checkpoint_schema_version": 2,
        "prompt_version": "finqa-program-v2-chat",
        "model": "Qwen/Qwen3-0.6B",
        "model_revision": REVISION,
        "tokenizer": "Qwen/Qwen3-0.6B",
        "tokenizer_revision": REVISION,
        "base_dtype": "float16",
        "dataset": "finqa",
        "training_config": {"max_sequence_length": 1536},
    }

    adapter = {
        "base_model_name_or_path": "Qwen/Qwen3-0.6B",
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "inference_mode": True,
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    }

    (checkpoint / "checkpoint_metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    (checkpoint / "adapter_config.json").write_text(
        json.dumps(adapter),
        encoding="utf-8",
    )
    (checkpoint / "adapter_model.safetensors").write_bytes(b"test")

    return checkpoint


def test_validates_checkpoint_provenance(tmp_path):
    checkpoint = make_checkpoint(tmp_path)
    model_config, training_config = make_configs()

    metadata = _validate_checkpoint(
        checkpoint,
        model_config,
        training_config,
    )

    assert metadata["model_revision"] == REVISION


def test_rejects_checkpoint_prompt_version_mismatch(tmp_path):
    checkpoint = make_checkpoint(tmp_path)
    metadata_path = checkpoint / "checkpoint_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["prompt_version"] = "finqa-program-v1"
    metadata_path.write_text(json.dumps(metadata))

    model_config, training_config = make_configs()

    with pytest.raises(ValueError, match="prompt version"):
        _validate_checkpoint(checkpoint, model_config, training_config)


def test_rejects_checkpoint_without_prompt_version(tmp_path):
    checkpoint = make_checkpoint(tmp_path)
    metadata_path = checkpoint / "checkpoint_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    del metadata["prompt_version"]
    metadata_path.write_text(json.dumps(metadata))

    model_config, training_config = make_configs()

    with pytest.raises(ValueError, match="prompt version"):
        _validate_checkpoint(checkpoint, model_config, training_config)


def test_rejects_base_model_revision_mismatch(tmp_path):
    checkpoint = make_checkpoint(tmp_path)
    model_config, training_config = make_configs()
    model_config["model"]["revision"] = "different-revision"
    model_config["model"]["tokenizer"]["revision"] = "different-revision"

    with pytest.raises(ValueError, match="revision"):
        _validate_checkpoint(checkpoint, model_config, training_config)


def test_rejects_adapter_base_model_mismatch(tmp_path):
    checkpoint = make_checkpoint(tmp_path)
    adapter_path = checkpoint / "adapter_config.json"
    adapter = json.loads(adapter_path.read_text())
    adapter["base_model_name_or_path"] = "different-model"
    adapter_path.write_text(json.dumps(adapter))

    model_config, training_config = make_configs()

    with pytest.raises(ValueError, match="Adapter base model"):
        _validate_checkpoint(checkpoint, model_config, training_config)


def test_rejects_adapter_rank_mismatch(tmp_path):
    checkpoint = make_checkpoint(tmp_path)
    adapter_path = checkpoint / "adapter_config.json"
    adapter = json.loads(adapter_path.read_text())
    adapter["r"] = 16
    adapter_path.write_text(json.dumps(adapter))

    model_config, training_config = make_configs()

    with pytest.raises(ValueError, match="rank"):
        _validate_checkpoint(checkpoint, model_config, training_config)


def test_rejects_missing_adapter_weights(tmp_path):
    checkpoint = make_checkpoint(tmp_path)
    (checkpoint / "adapter_model.safetensors").unlink()

    model_config, training_config = make_configs()

    with pytest.raises(ValueError, match="adapter_model.safetensors"):
        _validate_checkpoint(checkpoint, model_config, training_config)


def test_rejects_unsupported_checkpoint_schema(tmp_path):
    checkpoint = make_checkpoint(tmp_path)
    metadata_path = checkpoint / "checkpoint_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["checkpoint_schema_version"] = 99
    metadata_path.write_text(json.dumps(metadata))

    model_config, training_config = make_configs()

    with pytest.raises(ValueError, match="schema"):
        _validate_checkpoint(checkpoint, model_config, training_config)


class FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [10, 11, 12]

    def decode(
        self,
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        assert clean_up_tokenization_spaces is False

        if skip_special_tokens:
            token_ids = [
                token_id for token_id in token_ids if token_id != self.eos_token_id
            ]

        return ",".join(map(str, token_ids))


class FakeModel:
    def __init__(self, generated_ids):
        self.generated_ids = generated_ids
        self.arguments = None
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def generate(self, **kwargs):
        self.arguments = kwargs
        suffix = torch.tensor(
            [self.generated_ids],
            dtype=torch.long,
            device=kwargs["input_ids"].device,
        )
        return torch.cat([kwargs["input_ids"], suffix], dim=1)


def make_components(generated_ids):
    tokenizer = FakeTokenizer()
    model = FakeModel(generated_ids)
    components = InferenceComponents(
        model=model,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        checkpoint_metadata={"training_config": {"max_sequence_length": 1536}},
    )
    return components, model


def test_generates_deterministically_and_preserves_raw_output():
    components, model = make_components([42, 43, 2])

    result = generate_program(
        components,
        "FinQA program:\n",
        max_new_tokens=96,
    )

    assert result["raw_output"] == "42,43,2"
    assert result["normalized_output"] == "42,43"
    assert result["generated_token_ids"] == [42, 43, 2]
    assert result["prompt_tokens"] == 3
    assert result["generated_tokens"] == 3
    assert result["ended_with_eos"] is True
    assert model.eval_called is True
    assert model.arguments["do_sample"] is False
    assert model.arguments["num_beams"] == 1
    assert model.arguments["use_cache"] is True
    assert model.arguments["max_new_tokens"] == 96


def test_reports_generation_without_eos():
    components, _ = make_components([42, 43])

    result = generate_program(components, "FinQA program:\n")

    assert result["ended_with_eos"] is False


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_rejects_invalid_generation_limit(limit):
    components, _ = make_components([42])

    with pytest.raises(ValueError, match="max_new_tokens"):
        generate_program(
            components,
            "FinQA program:\n",
            max_new_tokens=limit,
        )


def test_rejects_prompt_exceeding_checkpoint_limit():
    components, _ = make_components([42])
    components.checkpoint_metadata["training_config"]["max_sequence_length"] = 2

    with pytest.raises(ValueError, match="exceeding"):
        generate_program(components, "FinQA program:\n")


def test_rejects_empty_prompt():
    components, _ = make_components([42])

    with pytest.raises(ValueError, match="prompt"):
        generate_program(components, "   ")


def test_loads_unadapted_base_model_with_pinned_local_revision(monkeypatch):
    from types import SimpleNamespace

    import finetunex.inference.generator as generator

    calls = {}

    class Tokenizer:
        padding_side = None
        pad_token_id = 0
        eos_token_id = 2
        eos_token = "</s>"
        pad_token = None

    class Model:
        def __init__(self):
            self.config = SimpleNamespace(use_cache=False)
            self.moved_to = None
            self.eval_called = False
            self.parameters_frozen = False

        def gradient_checkpointing_disable(self):
            pass

        def to(self, device):
            self.moved_to = device
            return self

        def eval(self):
            self.eval_called = True
            return self

        def requires_grad_(self, requires_grad=True):
            self.parameters_frozen = not requires_grad
            return self

        def named_parameters(self):
            return []

    tokenizer = Tokenizer()
    model = Model()

    def load_tokenizer(name, **kwargs):
        calls["tokenizer"] = (name, kwargs)
        return tokenizer

    def load_model(name, **kwargs):
        calls["model"] = (name, kwargs)
        return model

    monkeypatch.setattr(
        generator,
        "_select_inference_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(generator.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(generator.AutoModelForCausalLM, "from_pretrained", load_model)

    model_config, training_config = make_configs()
    components = generator.load_base_inference_components(model_config, training_config)

    assert components.model is model
    assert components.tokenizer is tokenizer
    assert components.checkpoint_metadata["model_revision"] == REVISION
    assert (
        components.checkpoint_metadata["training_config"]["max_sequence_length"] == 1536
    )
    assert calls["tokenizer"][1]["revision"] == REVISION
    assert calls["tokenizer"][1]["local_files_only"] is True
    assert calls["model"][1]["revision"] == REVISION
    assert calls["model"][1]["local_files_only"] is True
    assert model.config.use_cache is True
    assert model.eval_called is True
    assert model.parameters_frozen is True


def test_base_loader_rejects_unpinned_model_revision():
    from finetunex.inference.generator import load_base_inference_components

    model_config, training_config = make_configs()
    model_config["model"]["revision"] = ""
    model_config["model"]["tokenizer"]["revision"] = ""

    with pytest.raises(ValueError, match="revision must be pinned"):
        load_base_inference_components(model_config, training_config)


def test_base_loader_rejects_invalid_sequence_limit():
    from finetunex.inference.generator import load_base_inference_components

    model_config, training_config = make_configs()
    training_config["training"]["max_sequence_length"] = 0

    with pytest.raises(ValueError, match="max_sequence_length"):
        load_base_inference_components(model_config, training_config)
