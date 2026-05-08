"""Tests for Phase 3 explainer and LLM-as-a-Judge validation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from churn_service.explainer import explain_with_validation, validate_explanation


def _resp(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    ch = MagicMock()
    ch.message = msg
    out = MagicMock()
    out.choices = [ch]
    return out


@pytest.fixture
def sample_prediction() -> dict:
    return {
        "customer_id": "c1",
        "churn_probability": 0.82,
        "predicted_churn": 1,
        "features": {"support_calls": 5, "tenure": 12},
    }


@pytest.fixture
def sample_top_features() -> list[dict]:
    return [{"rank": 1, "feature": "numeric_plain__support_calls", "importance": 0.3}]


def test_validate_explanation_parses_json(sample_prediction: dict) -> None:
    judge_json = '{"faithfulness_score": 4, "valid": true, "reason": "Grounded."}'
    with patch("churn_service.explainer.completion", return_value=_resp(judge_json)) as m:
        vr = validate_explanation("Some explanation.", sample_prediction)
    assert vr.valid is True
    assert vr.faithfulness_score == 4
    assert vr.judge_failed is False
    assert m.call_count == 1
    assert m.call_args.kwargs["temperature"] == 0.0


def test_explain_with_validation_stops_when_valid(
    sample_prediction: dict,
    sample_top_features: list,
) -> None:
    gen_text = "Retention strategy based on support_calls."
    judge_json = '{"faithfulness_score": 5, "valid": true, "reason": ""}'
    with patch(
        "churn_service.explainer.completion",
        side_effect=[_resp(gen_text), _resp(judge_json)],
    ) as m:
        text, audit = explain_with_validation(sample_prediction, sample_top_features)
    assert text == gen_text
    assert audit["final_valid"] is True
    assert audit["validation_skipped"] is False
    assert len(audit["attempts"]) == 1
    assert m.call_args_list[0].kwargs["temperature"] == 0.3
    assert m.call_args_list[1].kwargs["temperature"] == 0.0


def test_explain_with_validation_retries_lower_temperature(
    sample_prediction: dict,
    sample_top_features: list,
) -> None:
    gen1 = "This customer owes $99999 fabricated."
    gen2 = "Grounded explanation using support_calls."
    judge_bad = '{"faithfulness_score": 2, "valid": false, "reason": "Hallucination"}'
    judge_good = '{"faithfulness_score": 5, "valid": true, "reason": ""}'
    with patch(
        "churn_service.explainer.completion",
        side_effect=[
            _resp(gen1),
            _resp(judge_bad),
            _resp(gen2),
            _resp(judge_good),
        ],
    ) as m:
        text, audit = explain_with_validation(sample_prediction, sample_top_features)
    assert text == gen2
    assert audit["final_valid"] is True
    assert len(audit["attempts"]) == 2
    assert m.call_args_list[0].kwargs["temperature"] == pytest.approx(0.3)
    assert m.call_args_list[2].kwargs["temperature"] == pytest.approx(0.15)


def test_validate_explanation_judge_failed(sample_prediction: dict) -> None:
    with patch("churn_service.explainer.completion", side_effect=RuntimeError("API down")):
        vr = validate_explanation("x", sample_prediction)
    assert vr.judge_failed is True
    assert vr.valid is False


def test_explain_with_validation_skipped_when_judge_fails(
    sample_prediction: dict,
    sample_top_features: list,
) -> None:
    gen_text = "Some text."
    with patch(
        "churn_service.explainer.completion",
        side_effect=[_resp(gen_text), RuntimeError("down")],
    ):
        text, audit = explain_with_validation(sample_prediction, sample_top_features)
    assert text == gen_text
    assert audit["validation_skipped"] is True
    assert audit["final_valid"] is False
