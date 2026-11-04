from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import finetunex.api.app as api_module
from finetunex.api.app import create_app


class FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    ):
        assert tokenize is False
        assert enable_thinking is False
        rendered = "".join(
            f"<{message['role']}>{message['content']}" for message in messages
        )
        return rendered + ("<assistant>" if add_generation_prompt else "")

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(range(len(text[::5])))


@pytest.fixture
def components():
    return SimpleNamespace(
        tokenizer=FakeTokenizer(),
        checkpoint_metadata={
            "model": "Qwen/Qwen3-0.6B",
            "model_revision": "pinned-revision",
        },
    )


def test_health_endpoints_report_liveness_and_readiness(components):
    with TestClient(create_app(components)) as client:
        assert client.get("/health/live").json() == {"status": "alive"}
        assert client.get("/health/ready").json() == {"status": "ready"}


def test_score_returns_program_and_executed_value(monkeypatch, components):
    monkeypatch.setattr(
        api_module,
        "generate_program",
        lambda *_args, **_kwargs: {
            "raw_output": "divide(125, 100)",
            "normalized_output": "divide(125, 100)",
            "prompt_tokens": 120,
            "generated_tokens": 8,
            "ended_with_eos": True,
        },
    )

    payload = {
        "id": "request-1",
        "question": "What is revenue in 2023 divided by revenue in 2022?",
        "table": [
            ["Year", "Revenue"],
            ["2022", "100"],
            ["2023", "125"],
        ],
    }

    with TestClient(create_app(components)) as client:
        response = client.post("/v1/finqa/score", json=payload)

    assert response.status_code == 200
    result = response.json()
    assert result["request_id"] == "request-1"
    assert result["program"] == "divide(125, 100)"
    assert result["parse_status"] == "success"
    assert result["execution_status"] == "success"
    assert result["predicted_value"] == 1.25
    assert result["operations"] == ["divide"]
    assert result["model"] == "Qwen/Qwen3-0.6B"
    assert result["prompt_metadata"]["max_prompt_tokens"] == 1408
    assert result["prompt_metadata"]["prompt_tokens"] <= 1408


def test_score_reports_parse_failure_without_execution(monkeypatch, components):
    monkeypatch.setattr(
        api_module,
        "generate_program",
        lambda *_args, **_kwargs: {
            "raw_output": "The answer is 1.25.",
            "normalized_output": "The answer is 1.25.",
            "prompt_tokens": 120,
            "generated_tokens": 8,
            "ended_with_eos": True,
        },
    )

    with TestClient(create_app(components)) as client:
        response = client.post(
            "/v1/finqa/score",
            json={"question": "What is the ratio?", "table": []},
        )

    assert response.status_code == 200
    result = response.json()
    assert result["parse_status"] == "failed"
    assert result["execution_status"] == "not_attempted"
    assert result["predicted_value"] is None
    assert result["parse_error"]


def test_request_rejects_gold_fields(components):
    with TestClient(create_app(components)) as client:
        response = client.post(
            "/v1/finqa/score",
            json={
                "question": "What is the answer?",
                "program": "divide(1, 2)",
                "exe_ans": "0.5",
            },
        )

    assert response.status_code == 422


def test_request_rejects_inconsistent_table_width(components):
    with TestClient(create_app(components)) as client:
        response = client.post(
            "/v1/finqa/score",
            json={
                "question": "What is the answer?",
                "table": [["Year", "Revenue"], ["2023"]],
            },
        )

    assert response.status_code == 422


def test_readiness_is_unavailable_before_startup():
    application = create_app()
    response = TestClient(application, raise_server_exceptions=False).get(
        "/health/ready"
    )
    assert response.status_code == 503
