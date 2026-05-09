"""Phase 4.1: FastAPI orchestration for churn prediction and Gen AI explanations.

Run from the repository root (so ``churn_service`` imports resolve)::

    uv pip install -r requirements.txt
    uv run python -m uvicorn churn_service.main:app --host 0.0.0.0 --port 8000

Environment (LLM keys and models are read by LiteLLM via ``explainer``; never hardcode secrets):

- ``OPENAI_API_KEY`` / ``OPENROUTER_API_KEY`` as needed for ``LLM_MODEL`` / ``JUDGE_MODEL``
  (see ``churn_service/explainer.py``).

Service tuning (optional):

- ``HIGH_RISK_THRESHOLD``: minimum ``churn_probability`` to run the explainer when
  ``predicted_churn == 1`` (default ``0.5``).
- ``EXPLAIN_MAX_ATTEMPTS``: max generate/validate attempts (default ``3``).
- ``EXPLAIN_INITIAL_TEMPERATURE``: first LLM temperature for explanations (default ``0.3``).
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from churn_service.eda import FEATURE_COLUMNS
from churn_service.explainer import (
    generate_explanation,
    load_top_features,
    validate_explanation,
)
from churn_service.train import load_fitted_preprocessor

_LOGGER = logging.getLogger("churn_service.api")

MODEL_FEATURE_COLUMNS: list[str] = [c for c in FEATURE_COLUMNS if c != "customer_id"]


def _configure_api_logging() -> None:
    """Attach a JSON-per-line handler for structured API logs (idempotent)."""
    if getattr(_configure_api_logging, "_done", False):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonLineFormatter())
    _LOGGER.handlers.clear()
    _LOGGER.addHandler(handler)
    _LOGGER.setLevel(logging.INFO)
    _LOGGER.propagate = False
    _configure_api_logging._done = True  # type: ignore[attr-defined]


class _JsonLineFormatter(logging.Formatter):
    """Emit one JSON object per log line when ``record.api_json`` is set."""

    def format(self, record: logging.LogRecord) -> str:
        payload = getattr(record, "api_json", None)
        if isinstance(payload, dict):
            line = dict(payload)
            line.setdefault("level", record.levelname)
            line.setdefault("logger", record.name)
            return json.dumps(line, default=str)
        return json.dumps(
            {
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            },
            default=str,
        )


def _log_api_json(payload: dict[str, Any]) -> None:
    _LOGGER.info("", extra={"api_json": payload})


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _attempt_temperatures(initial_temperature: float, max_attempts: int) -> list[float]:
    """Same schedule as ``explainer._attempt_temperatures`` (kept local to avoid private imports)."""
    seq = [initial_temperature, initial_temperature * 0.5, 0.0]
    if max_attempts <= len(seq):
        return seq[:max_attempts]
    return seq + [0.0] * (max_attempts - len(seq))


class CustomerData(BaseModel):
    """Eleven input fields aligned with ``churn_service.eda.FEATURE_COLUMNS``."""

    customer_id: str | int = Field(..., description="Customer identifier")
    age: int = Field(..., ge=0)
    gender: str
    tenure: int = Field(..., ge=0)
    usage_frequency: int = Field(..., ge=0)
    support_calls: int = Field(..., ge=0)
    payment_delay: float = Field(..., ge=0)
    subscription_type: str
    contract_length: str
    total_spend: float = Field(..., ge=0)
    last_interaction: int = Field(..., ge=0)


class ValidationAttemptOut(BaseModel):
    attempt: int
    temperature: float
    faithfulness_score: int
    valid: bool
    judge_failed: bool
    reason: str | None = None


class PredictResponse(BaseModel):
    predicted_churn: int = Field(..., ge=0, le=1)
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    explanation: str | None = None
    faithfulness_score: int | None = None
    explanation_valid: bool | None = None
    judge_failed: bool | None = None
    validation_attempts: list[ValidationAttemptOut] | None = None


class HealthResponse(BaseModel):
    status: str
    preprocessor_loaded: bool
    model_loaded: bool
    top_features_loaded: bool


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _configure_api_logging()
    models_dir = Path(__file__).resolve().parent / "models"
    app.state.models_dir = models_dir
    app.state.preprocessor = load_fitted_preprocessor(models_dir)
    final_path = models_dir / "final_model.joblib"
    if not final_path.is_file():
        raise FileNotFoundError(
            f"Champion model not found: {final_path}. Run `python churn_service/tune.py` from the repo root."
        )
    app.state.model = joblib.load(final_path)
    top_features, importance_meta = load_top_features(models_dir)
    app.state.top_features = top_features
    app.state.feature_importance_meta = importance_meta
    _LOGGER.info(
        "",
        extra={
            "api_json": {
                "event": "startup_complete",
                "ts": datetime.now(timezone.utc).isoformat(),
                "models_dir": str(models_dir),
                "model_type": importance_meta.get("model_type"),
            }
        },
    )
    yield


app = FastAPI(
    title="Churn Predict & Explain",
    description="Hybrid ML + Gen AI churn scoring API.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    st = app.state
    pre = getattr(st, "preprocessor", None) is not None
    mod = getattr(st, "model", None) is not None
    top = getattr(st, "top_features", None) is not None
    ok = pre and mod and top
    return HealthResponse(
        status="ok" if ok else "degraded",
        preprocessor_loaded=pre,
        model_loaded=mod,
        top_features_loaded=top,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: Request, body: CustomerData) -> PredictResponse:
    t0 = time.perf_counter()
    preprocessor = app.state.preprocessor
    model = app.state.model
    top_features: list[dict[str, Any]] = app.state.top_features

    row = {col: getattr(body, col) for col in MODEL_FEATURE_COLUMNS}
    X_df = pd.DataFrame([row], columns=MODEL_FEATURE_COLUMNS)
    X_t = preprocessor.transform(X_df)
    proba = float(model.predict_proba(X_t)[0, 1])
    pred = int(model.predict(X_t)[0])

    prediction_data: dict[str, Any] = {
        "customer_id": body.customer_id,
        "churn_probability": proba,
        "predicted_churn": pred,
        "features": row,
    }

    high_risk_threshold = _float_env("HIGH_RISK_THRESHOLD", 0.5)
    explain_max = max(1, _int_env("EXPLAIN_MAX_ATTEMPTS", 3))
    initial_temp = _float_env("EXPLAIN_INITIAL_TEMPERATURE", 0.3)

    high_risk = pred == 1 and proba >= high_risk_threshold
    explanation: str | None = None
    faithfulness_score: int | None = None
    explanation_valid: bool | None = None
    judge_failed: bool | None = None
    validation_attempts: list[ValidationAttemptOut] | None = None

    if high_risk:
        temps = _attempt_temperatures(initial_temp, explain_max)
        attempts_out: list[ValidationAttemptOut] = []
        last_explanation = ""
        for i, temp in enumerate(temps):
            attempt_no = i + 1
            last_explanation = generate_explanation(
                prediction_data,
                top_features,
                temperature=temp,
            )
            vr = validate_explanation(last_explanation, prediction_data, attempt=attempt_no)
            attempts_out.append(
                ValidationAttemptOut(
                    attempt=attempt_no,
                    temperature=temp,
                    faithfulness_score=vr.faithfulness_score,
                    valid=vr.valid,
                    judge_failed=vr.judge_failed,
                    reason=vr.reason,
                )
            )
            faithfulness_score = vr.faithfulness_score
            explanation_valid = vr.valid
            judge_failed = vr.judge_failed
            if vr.judge_failed:
                explanation = last_explanation
                break
            if vr.valid:
                explanation = last_explanation
                break
            explanation = last_explanation
        validation_attempts = attempts_out

    duration_ms = (time.perf_counter() - t0) * 1000.0
    _log_api_json(
        {
            "event": "predict",
            "ts": datetime.now(timezone.utc).isoformat(),
            "method": request.method,
            "path": str(request.url.path),
            "customer_id": str(body.customer_id),
            "predicted_churn": pred,
            "churn_probability": round(proba, 6),
            "explanation_ran": high_risk,
            "high_risk_threshold": high_risk_threshold,
            "faithfulness_score": faithfulness_score,
            "explanation_valid": explanation_valid,
            "judge_failed": judge_failed,
            "validation_attempts": [a.model_dump() for a in validation_attempts]
            if validation_attempts
            else None,
            "duration_ms": round(duration_ms, 2),
        }
    )

    return PredictResponse(
        predicted_churn=pred,
        churn_probability=proba,
        explanation=explanation,
        faithfulness_score=faithfulness_score,
        explanation_valid=explanation_valid,
        judge_failed=judge_failed,
        validation_attempts=validation_attempts,
    )
