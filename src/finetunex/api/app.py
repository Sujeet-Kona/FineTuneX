"""FastAPI serving interface for the FineTuneX FinQA adapter."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator

from finetunex.config import load_yaml_config
from finetunex.evaluation.finqa_executor import (
    FinQAExecutionError,
    execute_program,
)
from finetunex.evaluation.finqa_parser import (
    FinQAProgramParseError,
    parse_program,
)
from finetunex.inference.generator import (
    InferenceComponents,
    generate_program,
    load_inference_components,
)
from finetunex.inference.prompt import build_inference_prompt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_CONFIG_PATH = PROJECT_ROOT / "configs/model.yaml"
TRAINING_CONFIG_PATH = PROJECT_ROOT / "configs/training.yaml"
PREPROCESSING_CONFIG_PATH = PROJECT_ROOT / "configs/preprocessing.yaml"


class FinQARequest(BaseModel):
    """Inference-only request schema; gold labels are not accepted."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default="api-request", min_length=1, max_length=200)
    question: str = Field(min_length=1, max_length=4000)
    pre_text: list[str] = Field(default_factory=list, max_length=40)
    post_text: list[str] = Field(default_factory=list, max_length=40)
    table: list[list[str]] = Field(default_factory=list, max_length=100)

    @field_validator("pre_text", "post_text")
    @classmethod
    def validate_passages(cls, passages: list[str]) -> list[str]:
        if any(not passage.strip() or len(passage) > 5000 for passage in passages):
            raise ValueError("Passages must be non-empty and at most 5000 characters")
        return passages

    @field_validator("table")
    @classmethod
    def validate_table(cls, table: list[list[str]]) -> list[list[str]]:
        if not table:
            return table

        width = len(table[0])
        if width == 0 or width > 30:
            raise ValueError("Table rows must contain between 1 and 30 cells")

        if any(len(row) != width for row in table):
            raise ValueError("Table rows must have a consistent width")

        if any(not cell.strip() or len(cell) > 1000 for row in table for cell in row):
            raise ValueError(
                "Table cells must be non-empty and at most 1000 characters"
            )

        return table


def create_app(
    components: InferenceComponents | None = None,
) -> FastAPI:
    """Create the API; model loading occurs once during application startup."""

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        preprocessing = load_yaml_config(PREPROCESSING_CONFIG_PATH)["preprocessing"]
        application.state.evidence_budget = preprocessing["evidence"]["token_budget"]
        application.state.max_prompt_tokens = preprocessing["sequence"][
            "max_prompt_length"
        ]
        application.state.max_new_tokens = preprocessing["generation"]["max_new_tokens"]
        max_sequence_length = load_yaml_config(TRAINING_CONFIG_PATH)["training"][
            "max_sequence_length"
        ]
        if (
            application.state.max_prompt_tokens + application.state.max_new_tokens
            > max_sequence_length
        ):
            raise RuntimeError(
                "Prompt budget and generation reserve exceed sequence limit"
            )

        if components is None:
            checkpoint = os.environ.get("FINETUNEX_CHECKPOINT")
            if not checkpoint:
                raise RuntimeError(
                    "FINETUNEX_CHECKPOINT must point to a v4 LoRA checkpoint"
                )

            model_config = load_yaml_config(MODEL_CONFIG_PATH)
            training_config = load_yaml_config(TRAINING_CONFIG_PATH)
            application.state.components = load_inference_components(
                model_config,
                training_config,
                checkpoint,
            )
        else:
            application.state.components = components

        # The local accelerator/model is shared; serialize generation requests.
        application.state.inference_lock = Lock()
        yield

    application = FastAPI(
        title="FineTuneX API",
        version="0.1.0",
        description="Inference API for the FineTuneX FinQA reasoning adapter.",
        lifespan=lifespan,
    )

    metrics_registry = CollectorRegistry()
    request_counter = Counter(
        "finetunex_http_requests_total",
        "Total HTTP requests handled by FineTuneX.",
        labelnames=("method", "route", "status_code"),
        registry=metrics_registry,
    )
    request_latency = Histogram(
        "finetunex_http_request_duration_seconds",
        "HTTP request latency in seconds.",
        labelnames=("method", "route"),
        registry=metrics_registry,
    )
    application.state.metrics_registry = metrics_registry

    @application.middleware("http")
    async def observe_http_requests(request: Request, call_next):
        if request.url.path == "/metrics":
            return await call_next(request)

        started_at = perf_counter()
        route = request.scope.get("route")
        route_label = getattr(route, "path", "unmatched")
        status_code = "500"
        try:
            response = await call_next(request)
            status_code = str(response.status_code)
            return response
        finally:
            elapsed = perf_counter() - started_at
            request_counter.labels(
                request.method, route_label, status_code
            ).inc()
            request_latency.labels(request.method, route_label).observe(elapsed)

    @application.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(
            content=generate_latest(metrics_registry),
            media_type=CONTENT_TYPE_LATEST,
        )

    @application.get("/health/live")
    def liveness() -> dict[str, str]:
        return {"status": "alive"}

    @application.get("/health/ready")
    def readiness() -> dict[str, str]:
        if not hasattr(application.state, "components"):
            raise HTTPException(status_code=503, detail="Model is not loaded")
        return {"status": "ready"}

    @application.post("/v1/finqa/score")
    def score(request: FinQARequest) -> dict[str, Any]:
        if not hasattr(application.state, "components"):
            raise HTTPException(status_code=503, detail="Model is not loaded")

        record = request.model_dump()
        inference_components = application.state.components

        try:
            prompt_result = build_inference_prompt(
                record,
                inference_components.tokenizer,
                evidence_budget=application.state.evidence_budget,
                max_prompt_tokens=application.state.max_prompt_tokens,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        try:
            with application.state.inference_lock:
                generation = generate_program(
                    inference_components,
                    prompt_result["prompt"],
                    max_new_tokens=application.state.max_new_tokens,
                )
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        normalized_output = generation["normalized_output"]
        result: dict[str, Any] = {
            "request_id": request.id,
            "program": normalized_output,
            "raw_output": generation["raw_output"],
            "parse_status": "failed",
            "execution_status": "not_attempted",
            "predicted_value": None,
            "operations": [],
            "parse_error": None,
            "execution_error": None,
            "generation": {
                "prompt_tokens": generation["prompt_tokens"],
                "generated_tokens": generation["generated_tokens"],
                "ended_with_eos": generation["ended_with_eos"],
            },
            "prompt_metadata": prompt_result["metadata"],
            "model": inference_components.checkpoint_metadata.get("model"),
            "model_revision": inference_components.checkpoint_metadata.get(
                "model_revision"
            ),
        }

        try:
            steps = parse_program(normalized_output)
        except FinQAProgramParseError as exc:
            result["parse_error"] = str(exc)
            return result

        result["parse_status"] = "success"
        result["operations"] = [step.operation for step in steps]

        try:
            result["predicted_value"] = execute_program(
                normalized_output,
                request.table,
            )
            result["execution_status"] = "success"
        except (FinQAExecutionError, ArithmeticError, ValueError) as exc:
            result["execution_status"] = "failed"
            result["execution_error"] = str(exc)

        return result

    return application


app = create_app()
