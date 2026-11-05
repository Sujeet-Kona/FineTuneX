import pytest

from scripts.compare_finqa_evaluations import compare


def make_summary(source):
    return {
        "model_source": {
            "type": source,
            "model": "Qwen/Qwen3-0.6B",
            "model_revision": "revision-1",
        },
        "scope": {
            "examples_evaluated": 2,
            "sample_method": "complete_official_order",
            "sample_seed": None,
            "max_new_tokens": 128,
            "max_prompt_tokens": 1408,
            "evidence_budget": 1024,
            "generation": "greedy",
            "answer_reference": "exe_ans",
            "answer_comparison": "exact_after_five_decimal_rounding",
        },
        "inputs": {
            "validation_sha256": "validation",
            "preprocessed_validation_sha256": "tokenized-validation",
            "model_config_sha256": "model-config",
            "training_config_sha256": "training-config",
            "preprocessing_config_sha256": "preprocessing-config",
            "preprocessing_manifest_sha256": "manifest",
        },
        "metrics": {
            "parse_rate": 0.5,
            "execution_rate": 0.5,
            "answer_accuracy": 0.25 if source == "base_model" else 0.5,
        },
    }


def make_rows():
    return [
        {
            "record_id": "record-1",
            "prompt_sha256": "prompt-1",
            "expected_answer": "10",
        },
        {
            "record_id": "record-2",
            "prompt_sha256": "prompt-2",
            "expected_answer": "20",
        },
    ]


def test_compares_only_identical_prompted_examples():
    result = compare(
        make_summary("base_model"),
        make_rows(),
        make_summary("lora_adapter"),
        make_rows(),
    )

    assert result["identical_prompts_and_record_order"] is True
    assert result["examples"] == 2
    assert result["adapter_minus_base"]["answer_accuracy"] == 0.25

def test_rejects_prompt_mismatch():
    adapter_rows = make_rows()
    adapter_rows[1]["prompt_sha256"] = "different"

    with pytest.raises(ValueError, match="Prompt hash mismatch"):
        compare(
            make_summary("base_model"),
            make_rows(),
            make_summary("lora_adapter"),
            adapter_rows,
        )

def test_rejects_record_order_mismatch():
    adapter_rows = list(reversed(make_rows()))

    with pytest.raises(ValueError, match="Record order mismatch"):
        compare(
            make_summary("base_model"),
            make_rows(),
            make_summary("lora_adapter"),
            adapter_rows,
        )

def test_rejects_validation_provenance_mismatch():
    adapter_summary = make_summary("lora_adapter")
    adapter_summary["inputs"]["validation_sha256"] = "different"

    with pytest.raises(ValueError, match="Input provenance mismatch"):
        compare(
            make_summary("base_model"),
            make_rows(),
            adapter_summary,
            make_rows(),
        )

def test_rejects_preprocessed_validation_artifact_mismatch():
    adapter_summary = make_summary("lora_adapter")
    adapter_summary["inputs"]["preprocessed_validation_sha256"] = "different"

    with pytest.raises(ValueError, match="Input provenance mismatch"):
        compare(
            make_summary("base_model"),
            make_rows(),
            adapter_summary,
            make_rows(),
        )

def test_rejects_prompt_budget_mismatch():
    adapter_summary = make_summary("lora_adapter")
    adapter_summary["scope"]["max_prompt_tokens"] = 1200

    with pytest.raises(ValueError, match="Scope mismatch: max_prompt_tokens"):
        compare(
            make_summary("base_model"),
            make_rows(),
            adapter_summary,
            make_rows(),
        )

def test_rejects_validation_sample_method_mismatch():
    adapter_summary = make_summary("lora_adapter")
    adapter_summary["scope"]["sample_method"] = "sha256_record_id"
    adapter_summary["scope"]["sample_seed"] = 7

    with pytest.raises(ValueError, match="Scope mismatch: sample_method"):
        compare(
            make_summary("base_model"),
            make_rows(),
            adapter_summary,
            make_rows(),
        )



def test_rejects_validation_sample_seed_mismatch():
    adapter_summary = make_summary("lora_adapter")
    adapter_summary["scope"]["sample_seed"] = 7

    with pytest.raises(ValueError, match="Scope mismatch: sample_seed"):
        compare(
            make_summary("base_model"),
            make_rows(),
            adapter_summary,
            make_rows(),
        )
