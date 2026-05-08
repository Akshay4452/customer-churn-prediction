"""Unit tests for Phase 2.2–2.3 tuning helpers (`churn_service/tune.py`)."""

from __future__ import annotations

import numpy as np
import pytest

from churn_service.tune import top_n_feature_importance


def test_top_n_feature_importance_order_and_rank() -> None:
    names = np.array(["a", "b", "c", "d"])
    imp = np.array([0.1, 0.5, 0.2, 0.05])
    top = top_n_feature_importance(names, imp, n=3)
    assert len(top) == 3
    assert top[0]["rank"] == 1 and top[0]["feature"] == "b" and top[0]["importance"] == 0.5
    assert top[1]["feature"] == "c" and top[1]["rank"] == 2
    assert top[2]["feature"] == "a" and top[2]["rank"] == 3


def test_top_n_feature_importance_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        top_n_feature_importance(np.array(["x"]), np.array([0.1, 0.2]))
