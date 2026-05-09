"""Streamlit demo UI for the churn FastAPI service (Phase 4.2).

Reads ``API_BASE_URL`` (default ``http://127.0.0.1:8000``). Run the API first, then::

    uv run streamlit run app_ui.py

With Docker Compose, set ``API_BASE_URL=http://api:8000`` for the UI service.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


@st.cache_data(show_spinner=False)
def _load_train_bounds() -> dict[str, Any]:
    """Slider ranges and categorical levels from training data (read once)."""
    root = Path(__file__).resolve().parent
    path = root / "churn_service" / "data" / "train.csv"
    defaults = _default_bounds()
    if not path.is_file():
        return defaults
    raw = pd.read_csv(path)
    df = raw.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    num_cols = [
        "age",
        "tenure",
        "usage_frequency",
        "support_calls",
        "payment_delay",
        "total_spend",
        "last_interaction",
    ]
    bounds: dict[str, Any] = {}
    for c in num_cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            lo = int(s.min()) if s.notna().any() else 0
            hi = int(s.max()) + (1 if c in ("support_calls", "usage_frequency") else 0)
            bounds[c] = (max(0, lo), max(lo + 1, hi))
        else:
            bounds[c] = (0, 100)
    for cat, col in (
        ("genders", "gender"),
        ("subscriptions", "subscription_type"),
        ("contracts", "contract_length"),
    ):
        if col in df.columns:
            vals = sorted({str(x) for x in df[col].dropna().unique()})
            bounds[cat] = vals or _default_bounds()[cat]
        else:
            bounds[cat] = defaults[cat]
    for k, v in defaults.items():
        bounds.setdefault(k, v)
    return bounds


def _default_bounds() -> dict[str, Any]:
    return {
        "age": (18, 65),
        "tenure": (1, 60),
        "usage_frequency": (1, 30),
        "support_calls": (0, 10),
        "payment_delay": (0, 30),
        "total_spend": (100, 1000),
        "last_interaction": (1, 30),
        "genders": ["Female", "Male"],
        "subscriptions": ["Basic", "Standard", "Premium"],
        "contracts": ["Monthly", "Quarterly", "Annual"],
    }


def _split_narrative_and_strategy(explanation: str | None) -> tuple[str, str]:
    """Separate risk narrative from retention strategy when the model structures them."""
    if not explanation or not str(explanation).strip():
        return "", ""
    text = str(explanation).strip()
    lower = text.lower()
    markers = [
        "retention strategy",
        "recommended retention",
        "retention recommendation",
        "one retention",
        "retention:",
        "strategy:",
        "## retention",
        "### retention",
    ]
    cut = -1
    for m in markers:
        i = lower.find(m)
        if i >= 0 and (cut < 0 or i < cut):
            cut = i
    if cut > 0:
        narrative = text[:cut].strip().rstrip(":-— \n")
        strategy = text[cut:].strip()
        if len(narrative) < 30:
            return text, strategy
        return narrative, strategy or text
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(parts) >= 2:
        last = parts[-1].lower()
        if any(
            k in last
            for k in ("retention", "strategy", "recommend", "offer", "discount", "contact", "csm")
        ):
            return "\n\n".join(parts[:-1]), parts[-1]
    return text, ""


def _predict(payload: dict[str, Any]) -> dict[str, Any]:
    url = f"{API_BASE_URL}/predict"
    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    return r.json()


def main() -> None:
    st.set_page_config(
        page_title="CoE - Churn Insights",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .coe-header { font-size: 1.75rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0.25rem; }
        .coe-sub { color: #4a5568; font-size: 1rem; margin-bottom: 1.5rem; }
        .risk-gauge-wrap { background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
            border-radius: 12px; padding: 1.25rem 1.5rem; border: 1px solid #e2e8f0; }
        .risk-pct { font-size: 2.5rem; font-weight: 800; color: #2d3748; }
        .explanation-box { background: #fff; border: 1px solid #e2e8f0; border-radius: 10px;
            padding: 1.25rem; min-height: 120px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="coe-header">Center of Excellence - AI Predictive Insights</p>'
        '<p class="coe-sub">Telecom customer churn risk assessment with grounded Gen AI narrative</p>',
        unsafe_allow_html=True,
    )

    b = _load_train_bounds()

    with st.sidebar:
        st.header("Customer profile")
        customer_id = st.text_input("Customer ID", value="demo-001")
        age = st.slider("Age", b["age"][0], b["age"][1], 42)
        gender = st.selectbox("Gender", b["genders"])
        tenure = st.slider("Tenure (months)", b["tenure"][0], b["tenure"][1], 24)
        usage_frequency = st.slider(
            "Usage frequency",
            b["usage_frequency"][0],
            b["usage_frequency"][1],
            14,
        )
        support_calls = st.slider(
            "Support calls",
            b["support_calls"][0],
            b["support_calls"][1],
            3,
        )
        payment_delay = st.slider(
            "Payment delay (days)",
            float(b["payment_delay"][0]),
            float(b["payment_delay"][1]),
            5.0,
        )
        subscription_type = st.selectbox("Subscription type", b["subscriptions"])
        contract_length = st.selectbox("Contract length", b["contracts"])
        total_spend = st.slider(
            "Total spend",
            float(b["total_spend"][0]),
            float(b["total_spend"][1]),
            500.0,
        )
        last_interaction = st.slider(
            "Days since last interaction",
            b["last_interaction"][0],
            b["last_interaction"][1],
            14,
        )
        st.caption(f"API: `{API_BASE_URL}`")
        predict = st.button("Predict", type="primary", use_container_width=True)

    if not predict:
        st.info("Configure the customer in the sidebar and click **Predict**.")
        return

    payload = {
        "customer_id": customer_id,
        "age": int(age),
        "gender": gender,
        "tenure": int(tenure),
        "usage_frequency": int(usage_frequency),
        "support_calls": int(support_calls),
        "payment_delay": float(payment_delay),
        "subscription_type": subscription_type,
        "contract_length": contract_length,
        "total_spend": float(total_spend),
        "last_interaction": int(last_interaction),
    }

    with st.spinner("Scoring customer and generating insights…"):
        try:
            data = _predict(payload)
        except requests.RequestException as e:
            st.error(f"Could not reach the API at `{API_BASE_URL}`. Start the FastAPI service or check Docker networking. ({e})")
            return

    proba = float(data.get("churn_probability", 0.0))
    pred = int(data.get("predicted_churn", 0))
    explanation = data.get("explanation")
    narrative, strategy = _split_narrative_and_strategy(explanation)

    col_gauge, col_meta = st.columns([2, 1])
    with col_gauge:
        st.markdown('<div class="risk-gauge-wrap">', unsafe_allow_html=True)
        st.markdown("##### Churn risk score")
        pct = min(100.0, max(0.0, proba * 100.0))
        st.markdown(f'<p class="risk-pct">{pct:.1f}%</p>', unsafe_allow_html=True)
        st.progress(min(1.0, max(0.0, proba)))
        st.caption(f"Estimated probability of churn (predicted class: {'churn' if pred == 1 else 'no churn'}).")
        st.markdown("</div>", unsafe_allow_html=True)
    with col_meta:
        st.metric("Binary prediction", "Churn" if pred == 1 else "No churn")
        st.metric("Raw score", f"{proba:.4f}")
        if data.get("faithfulness_score") is not None:
            st.metric("Faithfulness (LLM judge)", data.get("faithfulness_score"))
        if data.get("judge_failed"):
            st.warning("Judge layer unavailable or failed; explanation may be unverified.")

    st.divider()
    st.subheader("AI explanation")
    with st.container():
        st.markdown('<div class="explanation-box">', unsafe_allow_html=True)
        if narrative:
            st.markdown(narrative)
        elif explanation:
            st.markdown(str(explanation))
        else:
            st.markdown(
                "_No Gen AI narrative for this profile (typically low risk). "
                "Raise risk factors or lower `HIGH_RISK_THRESHOLD` on the API to trigger the explainer._"
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Retention strategy")
    if strategy and strategy.strip() and strategy.strip() != (narrative or "").strip():
        st.info(strategy)
    elif explanation:
        st.info(
            "The narrative above is written to include **one concrete retention strategy** aligned "
            "with the customer’s data. Use marker phrases in the LLM prompt if you need a hard split "
            "into separate API fields."
        )
    else:
        st.caption("Retention suggestions appear when the API returns a Gen AI explanation (high-risk customers).")


if __name__ == "__main__":
    main()
