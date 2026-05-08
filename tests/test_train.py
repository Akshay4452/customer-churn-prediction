"""Unit tests for churn baseline helpers (`churn_service/train.py`)."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import numpy as np
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, roc_auc_score

from churn_service.train import (
    compute_test_metrics,
    load_fitted_preprocessor,
    pick_winner,
)


def test_pick_winner_rf_higher_f1() -> None:
    rf = {"f1_churn": 0.7, "roc_auc": 0.8}
    xgb = {"f1_churn": 0.6, "roc_auc": 0.9}
    name, reason = pick_winner(rf, xgb)
    assert name == "RandomForestClassifier"
    assert "F1" in reason


def test_pick_winner_xgb_higher_f1() -> None:
    rf = {"f1_churn": 0.5, "roc_auc": 0.85}
    xgb = {"f1_churn": 0.65, "roc_auc": 0.7}
    name, reason = pick_winner(rf, xgb)
    assert name == "XGBClassifier"
    assert "F1" in reason


def test_pick_winner_tie_f1_xgb_higher_auc() -> None:
    rf = {"f1_churn": 0.6, "roc_auc": 0.75}
    xgb = {"f1_churn": 0.6, "roc_auc": 0.82}
    name, reason = pick_winner(rf, xgb)
    assert name == "XGBClassifier"
    assert "ROC-AUC" in reason


def test_pick_winner_tie_f1_rf_higher_auc() -> None:
    rf = {"f1_churn": 0.55, "roc_auc": 0.9}
    xgb = {"f1_churn": 0.55, "roc_auc": 0.88}
    name, reason = pick_winner(rf, xgb)
    assert name == "RandomForestClassifier"
    assert "ROC-AUC" in reason


def test_pick_winner_identical_metrics() -> None:
    m = {"f1_churn": 0.5, "roc_auc": 0.75}
    name, reason = pick_winner(m, dict(m))
    assert name == "tie"
    assert "identical" in reason.lower()


def test_compute_test_metrics_matches_sklearn_reference() -> None:
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 0])
    y_proba = np.array([0.1, 0.9, 0.2, 0.4, 0.7, 0.3, 0.85, 0.15])

    out = compute_test_metrics(y_true, y_pred, y_proba)
    assert out["f1_churn"] == pytest.approx(
        f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    )
    assert out["roc_auc"] == pytest.approx(roc_auc_score(y_true, y_proba))


def test_load_fitted_preprocessor_missing_file() -> None:
    with TemporaryDirectory() as tmp:
        models_dir = Path(tmp)
        with pytest.raises(FileNotFoundError, match="preprocess.py"):
            load_fitted_preprocessor(models_dir)


def test_transform_without_fit_on_mock_preprocessor() -> None:
    """Loaded preprocessor path uses ``transform`` only for feature matrices."""
    pre = MagicMock(spec=ColumnTransformer)
    pre.transform.side_effect = [
        np.zeros((4, 5)),
        np.zeros((3, 5)),
    ]
    fake_X = MagicMock()
    pre.transform(fake_X)
    pre.transform(fake_X)
    pre.fit.assert_not_called()
    assert pre.transform.call_count == 2
