"""Phase 3.1–3.3: reasoning layer — map ML outputs to LLM explanations via LiteLLM.

Loads champion predictions + Phase 2 top features, formats business-facing evidence from
raw feature values, and calls an OpenAI or OpenRouter model through LiteLLM (model-agnostic).
Phase 3.3 adds LLM-as-a-Judge faithfulness validation (RAGAS-style grounding check) and
temperature backoff when the judge flags unsupported claims.

Environment (typical):
  - ``OPENAI_API_KEY`` for OpenAI models (default ``LLM_MODEL=gpt-4o-mini``).
  - ``OPENROUTER_API_KEY`` and ``LLM_MODEL=openrouter/<provider>/<model>`` for OpenRouter.
  - ``JUDGE_MODEL`` (optional): model used only for validation; defaults to ``LLM_MODEL``.

From the repository root, using a virtual environment (recommended)::

    uv venv
    uv pip install -r requirements.txt
    .venv\\Scripts\\python.exe churn_service/explainer.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
from litellm import completion

_LOGGER = logging.getLogger(__name__)

_CHURN_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CHURN_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from churn_service.eda import CATEGORICAL_FEATURE_COLUMNS  # noqa: E402
from churn_service.preprocess import _load_train_frame, prepare_xy  # noqa: E402
from churn_service.train import load_fitted_preprocessor  # noqa: E402

# Business-facing labels (avoid pipeline / sklearn jargon in customer-facing text).
_FEATURE_LABELS: dict[str, str] = {
    "age": "Age",
    "support_calls": "Support calls",
    "payment_delay": "Payment delay (days)",
    "total_spend": "Total spend",
    "tenure": "Tenure (months)",
    "usage_frequency": "Usage frequency",
    "last_interaction": "Days since last interaction",
    "gender": "Gender",
    "subscription_type": "Subscription type",
    "contract_length": "Contract length",
}

_SYSTEM_PROMPT = """You are a Customer Success Expert. You will be provided with a Churn Prediction and the top factors contributing to that prediction. Your goal is to explain to a manager why this customer is at risk and suggest one retention strategy based on the data.

Rules:
- Write in clear, professional business language. Focus on actionable insight.
- Base your reasoning only on the structured data provided (prediction, probabilities, and feature facts).
- Propose exactly one concrete retention strategy aligned with the stated factors.
- Do not mention statistical testing, null hypotheses, or P-values.
- Do not name specific machine learning libraries or algorithms (for example: do not say XGBoost, Random Forest, gradient boosting, or similar). Refer to the output as a churn risk assessment or risk score if needed.
- Do not speculate beyond what the data supports."""


def _default_model() -> str:
    return os.environ.get("LLM_MODEL", "gpt-4o-mini")


def _default_judge_model() -> str:
    return os.environ.get("JUDGE_MODEL") or _default_model()


@dataclass
class ValidationResult:
    """Structured output from the LLM-as-a-Judge faithfulness check (Phase 3.3)."""

    faithfulness_score: int
    valid: bool
    reason: str | None = None
    raw_response: str | None = None
    judge_failed: bool = False


_JUDGE_SYSTEM_PROMPT = """You are a strict fact-checker for customer churn explanations.
Compare the explanation to the raw customer data (JSON). Determine whether the explanation
mentions any specific facts, numbers, categories, or attributes that are NOT present in that data.
General qualitative paraphrases of values that appear in the data are acceptable.
Answer with a Faithfulness Score from 1 (many fabricated or unsupported specifics) to 5 (fully grounded).
Set valid to true only if the explanation does not introduce fabricated metrics or attributes and is adequately grounded in the data (typically score 4 or 5).
Respond with JSON only, no markdown: {"faithfulness_score": <1-5 int>, "valid": <bool>, "reason": "<brief rationale>"}"""


def _json_safe_scalar(val: Any) -> Any:
    if isinstance(val, np.generic):
        return val.item()
    return val


def _grounding_payload_for_judge(prediction_data: dict[str, Any]) -> dict[str, Any]:
    features = prediction_data.get("features")
    if not isinstance(features, Mapping):
        raise ValueError("prediction_data['features'] must be a mapping for validation.")
    raw_features = {str(k): _json_safe_scalar(v) for k, v in features.items()}
    pred = prediction_data.get("predicted_churn", 0)
    pred_int = int(pred) if not isinstance(pred, bool) else int(pred)
    return {
        "customer_id": prediction_data.get("customer_id"),
        "churn_probability": float(prediction_data.get("churn_probability", 0.0)),
        "predicted_churn": pred_int,
        "features": raw_features,
    }


def _parse_judge_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        out = json.loads(text)
        if isinstance(out, dict):
            return out
        raise ValueError("expected JSON object")
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            out = json.loads(text[start : end + 1])
            if isinstance(out, dict):
                return out
        raise


def _attempt_temperatures(initial_temperature: float, max_attempts: int) -> list[float]:
    seq = [initial_temperature, initial_temperature * 0.5, 0.0]
    if max_attempts <= len(seq):
        return seq[:max_attempts]
    return seq + [0.0] * (max_attempts - len(seq))


def _parse_transformed_feature_name(engineered_name: str) -> tuple[str, str | None]:
    """Map a sklearn ``ColumnTransformer`` output name to a raw column and optional category.

    Examples: ``numeric_plain__support_calls`` → (``support_calls``, None);
    ``categorical__contract_length_Monthly`` → (``contract_length``, ``Monthly``).
    """
    if "__" not in engineered_name:
        return engineered_name, None
    prefix, rest = engineered_name.split("__", 1)
    if prefix in ("numeric_plain", "numeric_clip"):
        return rest, None
    if prefix == "categorical":
        for col in sorted(CATEGORICAL_FEATURE_COLUMNS, key=len, reverse=True):
            sep = col + "_"
            if rest.startswith(sep):
                return col, rest[len(sep) :]
        return rest, None
    return rest, None


def extract_top_feature_facts(
    features: Mapping[str, Any],
    top_features: list[dict[str, Any]],
    *,
    max_features: int = 5,
) -> list[str]:
    """Build short, factual lines from raw feature values for the top-ranked engineered names."""
    facts: list[str] = []
    seen_raw_cols: set[str] = set()

    for entry in top_features[:max_features]:
        engineered = str(entry.get("feature", ""))
        raw_col, _hot_category = _parse_transformed_feature_name(engineered)

        if raw_col not in features:
            _LOGGER.warning("Raw column %s not in features dict; skipping.", raw_col)
            continue

        if raw_col in seen_raw_cols:
            continue
        seen_raw_cols.add(raw_col)

        label = _FEATURE_LABELS.get(raw_col, raw_col.replace("_", " ").title())
        val = features.get(raw_col)
        if isinstance(val, (float, np.floating)):
            disp = float(val)
            disp_s = str(int(disp)) if disp.is_integer() else f"{disp:.2f}".rstrip("0").rstrip(".")
        else:
            disp_s = str(val)
        facts.append(f"{label}: {disp_s}")

    return facts


def generate_explanation(
    prediction_data: dict[str, Any],
    top_features: list[dict[str, Any]],
    *,
    model: str | None = None,
    temperature: float = 0.3,
) -> str:
    """Produce a natural-language explanation for one customer.

    ``prediction_data`` must include:
      - ``customer_id``: identifier
      - ``churn_probability``: float in [0, 1]
      - ``predicted_churn``: 0 or 1 (or bool)
      - ``features``: mapping of **raw** preprocessed column names (same as ``prepare_xy``)
        to values, used to ground the top factors in actual numbers.

    ``top_features`` matches Phase 2 JSON: list of dicts with ``rank``, ``feature``, ``importance``.
    """
    features = prediction_data.get("features")
    if not isinstance(features, Mapping):
        raise ValueError("prediction_data['features'] must be a mapping of raw column names to values.")

    facts = extract_top_feature_facts(features, top_features)
    cid = prediction_data.get("customer_id", "unknown")
    proba = float(prediction_data.get("churn_probability", 0.0))
    pred = prediction_data.get("predicted_churn", 0)
    pred_int = int(pred) if not isinstance(pred, bool) else int(pred)

    user_content = json.dumps(
        {
            "customer_id": cid,
            "churn_probability": round(proba, 4),
            "predicted_churn_label": pred_int,
            "top_predictive_factors_evidence": facts,
            "instruction": "Explain why this customer is at churn risk for a manager and suggest one retention strategy. Obey the system rules.",
        },
        indent=2,
    )

    use_model = model or _default_model()
    resp = completion(
        model=use_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=temperature,
    )
    choice = resp.choices[0].message
    text = getattr(choice, "content", None) or ""
    return str(text).strip()


def validate_explanation(
    explanation: str,
    prediction_data: dict[str, Any],
    *,
    model: str | None = None,
    attempt: int = 1,
) -> ValidationResult:
    """LLM-as-a-Judge: compare ``explanation`` to raw ``prediction_data`` for grounding.

    Uses ``JUDGE_MODEL`` when set, otherwise ``LLM_MODEL``. On judge API or parse failure,
    returns ``judge_failed=True`` so callers can fall back without blocking the pipeline.
    """
    use_model = model or _default_judge_model()
    grounding = _grounding_payload_for_judge(prediction_data)
    user_content = json.dumps(
        {
            "explanation_to_evaluate": explanation,
            "raw_customer_data": grounding,
            "instruction": (
                "Compare this explanation to the raw data provided. Does the explanation mention "
                "any facts or numbers NOT present in the data? Answer with a Faithfulness Score "
                "from 1-5 and a Boolean Valid flag in JSON: faithfulness_score, valid, reason."
            ),
        },
        indent=2,
    )
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    try:
        try:
            resp = completion(
                model=use_model,
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
        except Exception as first_err:
            _LOGGER.debug("Judge JSON mode failed (%s); retrying without response_format.", first_err)
            resp = completion(model=use_model, messages=messages, temperature=0.0)
    except Exception as e:
        _LOGGER.warning("Judge completion failed: %s", e)
        return ValidationResult(
            faithfulness_score=0,
            valid=False,
            reason=str(e),
            judge_failed=True,
        )

    choice = resp.choices[0].message
    raw = str(getattr(choice, "content", None) or "").strip()
    try:
        parsed = _parse_judge_json(raw)
    except (json.JSONDecodeError, ValueError) as e:
        _LOGGER.warning("Judge returned unparseable JSON: %s", e)
        return ValidationResult(
            faithfulness_score=0,
            valid=False,
            reason=f"judge_parse_error: {e}",
            raw_response=raw,
            judge_failed=True,
        )

    score_raw = parsed.get("faithfulness_score")
    try:
        score = int(score_raw) if score_raw is not None else 1
    except (TypeError, ValueError):
        score = 1
    score = max(1, min(5, score))
    valid = bool(parsed.get("valid", False))
    reason = parsed.get("reason")
    if reason is not None and not isinstance(reason, str):
        reason = str(reason)

    cid = prediction_data.get("customer_id", "unknown")
    _LOGGER.info(
        "explanation_validation customer_id=%s faithfulness_score=%s valid=%s attempt=%s judge_model=%s",
        cid,
        score,
        valid,
        attempt,
        use_model,
    )

    return ValidationResult(
        faithfulness_score=score,
        valid=valid,
        reason=reason,
        raw_response=raw,
        judge_failed=False,
    )


def explain_with_validation(
    prediction_data: dict[str, Any],
    top_features: list[dict[str, Any]],
    *,
    model: str | None = None,
    judge_model: str | None = None,
    initial_temperature: float = 0.3,
    max_attempts: int = 3,
) -> tuple[str, dict[str, Any]]:
    """Generate an explanation, validate with the judge, retry with lower temperature if invalid.

    Returns ``(explanation, audit)``. If the judge fails (API/parse), returns the latest
    explanation with ``audit["validation_skipped"]=True`` so upstream callers can surface
    a warning without failing the request.

    ``audit`` includes ``attempts`` (per-attempt temperature and validation summary),
    ``validation_skipped``, ``final_valid`` (whether the judge accepted an explanation), and
    ``final_attempt``.
    """
    temps = _attempt_temperatures(initial_temperature, max_attempts)
    audit_attempts: list[dict[str, Any]] = []
    last_explanation = ""
    validation_skipped_final = False
    judge_model_resolved = judge_model or _default_judge_model()

    for i, temp in enumerate(temps):
        attempt_no = i + 1
        last_explanation = generate_explanation(
            prediction_data,
            top_features,
            model=model,
            temperature=temp,
        )
        vr = validate_explanation(
            last_explanation,
            prediction_data,
            model=judge_model_resolved,
            attempt=attempt_no,
        )
        audit_attempts.append(
            {
                "attempt": attempt_no,
                "temperature": temp,
                "validation": {
                    "faithfulness_score": vr.faithfulness_score,
                    "valid": vr.valid,
                    "reason": vr.reason,
                    "judge_failed": vr.judge_failed,
                },
            }
        )

        if vr.judge_failed:
            validation_skipped_final = True
            _LOGGER.warning(
                "Judge unavailable or unparseable; returning last explanation without faithfulness gate (attempt %s).",
                attempt_no,
            )
            break

        if vr.valid:
            return last_explanation, {
                "attempts": audit_attempts,
                "validation_skipped": False,
                "final_valid": True,
                "final_attempt": attempt_no,
            }

    return last_explanation, {
        "attempts": audit_attempts,
        "validation_skipped": validation_skipped_final,
        "final_valid": False,
        "final_attempt": audit_attempts[-1]["attempt"] if audit_attempts else 0,
    }


def load_top_features(models_dir: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load ``feature_importance.json`` and return ``top_features`` plus full payload."""
    root = models_dir if models_dir is not None else _CHURN_DIR / "models"
    path = root / "feature_importance.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run `python churn_service/tune.py` first.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    top = payload.get("top_features", [])
    if not top:
        raise ValueError(f"No top_features in {path}")
    return top, payload


def pick_high_risk_customer_index(
    y_proba: np.ndarray,
    y_pred: np.ndarray,
) -> int:
    """Prefer a predicted churner with highest probability; else global argmax of proba."""
    churners = np.where(y_pred == 1)[0]
    if len(churners) > 0:
        local = churners[np.argmax(y_proba[churners])]
        return int(local)
    return int(np.argmax(y_proba))


def demo_console_explanation() -> None:
    """Score the test set with the saved champion, pick one high-risk row, print LLM text."""
    models_dir = _CHURN_DIR / "models"
    final_model_path = models_dir / "final_model.joblib"
    if not final_model_path.is_file():
        raise FileNotFoundError(
            f"Missing {final_model_path}. Run `python churn_service/tune.py` to train and save the champion."
        )

    preprocessor = load_fitted_preprocessor(models_dir)
    model = joblib.load(final_model_path)
    top_features, importance_meta = load_top_features(models_dir)

    test_path = _CHURN_DIR / "data" / "test.csv"
    test_df = _load_train_frame(test_path)
    X_test_df, y_test = prepare_xy(test_df)

    X_mat = preprocessor.transform(X_test_df)
    proba = model.predict_proba(X_mat)[:, 1]
    pred = model.predict(X_mat)

    idx = pick_high_risk_customer_index(proba, pred)
    row = X_test_df.iloc[idx]
    features_dict = row.to_dict()
    raw_customer_id = test_df.iloc[idx].get("customer_id")

    prediction_data = {
        "customer_id": raw_customer_id if raw_customer_id is not None else idx,
        "churn_probability": float(proba[idx]),
        "predicted_churn": int(pred[idx]),
        "actual_churn": int(y_test[idx]),
        "features": features_dict,
    }

    print("\n=== Phase 3 demo: one high-risk customer (test set) ===")
    print(f"Index in test split: {idx}")
    print(f"Customer ID: {prediction_data['customer_id']}")
    print(f"Churn probability: {prediction_data['churn_probability']:.4f}")
    print(f"Predicted churn: {prediction_data['predicted_churn']} | Actual churn: {prediction_data['actual_churn']}")
    print(
        f"Model context (for logging only): {importance_meta.get('model_type', 'unknown')} "
        "- not sent to the LLM."
    )
    print("\n--- Grounded top-factor facts sent to the LLM ---")
    for line in extract_top_feature_facts(features_dict, top_features):
        print(f"  - {line}")

    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")):
        print(
            "\n[Skip LLM call] Set OPENAI_API_KEY or OPENROUTER_API_KEY (and optionally LLM_MODEL) "
            "to generate the explanation."
        )
        return

    print(f"\n--- AI-generated explanation ({_default_model()}) + judge ({_default_judge_model()}) ---\n")
    explanation, audit = explain_with_validation(prediction_data, top_features)
    print(explanation)
    print()
    last_attempt = audit.get("attempts", [{}])[-1]
    val = last_attempt.get("validation", {})
    print(
        "--- Validation summary (faithfulness / rigor) ---\n"
        f"  final_valid={audit.get('final_valid')} | validation_skipped={audit.get('validation_skipped')} | "
        f"attempts={len(audit.get('attempts', []))}\n"
        f"  last faithfulness_score={val.get('faithfulness_score')} | valid={val.get('valid')}"
    )
    print()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    demo_console_explanation()


if __name__ == "__main__":
    main()
