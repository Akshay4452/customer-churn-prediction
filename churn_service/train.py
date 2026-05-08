"""Phase 2.1: baseline bake-off — Random Forest vs XGBoost on preprocessed churn features.

Loads Phase 1 ``preprocessor.joblib``, transforms ``train.csv`` / ``test.csv`` without
refitting the pipeline, trains default classifiers, prints classification reports and
ROC-AUC, and selects the better model by churn (class 1) F1 on the test set.

Run from the repository root (``python churn_service/train.py``) so ``churn_service``
imports resolve. Internal validation splits should use ``stratify=y``; this script uses
the fixed train/test CSV partition from Phase 1.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

_LOGGER = logging.getLogger(__name__)

_CHURN_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CHURN_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from churn_service.preprocess import (  # noqa: E402
    _load_train_frame,
    prepare_xy,
)


def load_fitted_preprocessor(models_dir: Path) -> ColumnTransformer:
    """Load the Phase 1 fitted ColumnTransformer; fail fast if missing."""
    path = models_dir / "preprocessor.joblib"
    if not path.is_file():
        raise FileNotFoundError(
            f"Preprocessor artifact not found: {path}. "
            "Run `python churn_service/preprocess.py` from the repo root first."
        )
    preprocessor = joblib.load(path)
    if not isinstance(preprocessor, ColumnTransformer):
        raise TypeError(f"Expected ColumnTransformer at {path}, got {type(preprocessor)}")
    return preprocessor


def compute_test_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba_positive: np.ndarray,
) -> dict[str, Any]:
    """Classification report + ROC-AUC; churn F1 is ``metrics['f1_churn']``."""
    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )
    roc = float(roc_auc_score(y_true, y_proba_positive))
    f1_churn = float(report["1"]["f1-score"])
    precision_churn = float(report["1"]["precision"])
    recall_churn = float(report["1"]["recall"])
    return {
        "classification_report": report,
        "roc_auc": roc,
        "f1_churn": f1_churn,
        "precision_churn": precision_churn,
        "recall_churn": recall_churn,
    }


def pick_winner(
    rf_metrics: dict[str, Any],
    xgb_metrics: dict[str, Any],
) -> tuple[str, str]:
    """Compare models by churn F1, then ROC-AUC. Returns (winner_label, reason)."""
    rf_f1 = rf_metrics["f1_churn"]
    xgb_f1 = xgb_metrics["f1_churn"]
    if rf_f1 > xgb_f1:
        return "RandomForestClassifier", "higher churn F1 on test"
    if xgb_f1 > rf_f1:
        return "XGBClassifier", "higher churn F1 on test"
    rf_auc = rf_metrics["roc_auc"]
    xgb_auc = xgb_metrics["roc_auc"]
    if rf_auc > xgb_auc:
        return "RandomForestClassifier", "tied churn F1; higher ROC-AUC on test"
    if xgb_auc > rf_auc:
        return "XGBClassifier", "tied churn F1; higher ROC-AUC on test"
    return "tie", "identical churn F1 and ROC-AUC on test"


def _format_comparison_row(name: str, m: dict[str, Any]) -> str:
    return (
        f"{name:26}  P1={m['precision_churn']:.4f}  R1={m['recall_churn']:.4f}  "
        f"F1(churn)={m['f1_churn']:.4f}  ROC-AUC={m['roc_auc']:.4f}"
    )


def run_baseline(
    churn_root: Path | None = None,
    *,
    random_state: int = 42,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    tuple[str, str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Load data, transform, train RF + XGB, evaluate on test.

    Returns metrics dicts, winner info, ``y_test``, ``rf_pred``, ``xgb_pred``.
    """
    root = churn_root if churn_root is not None else _CHURN_DIR
    train_path = root / "data" / "train.csv"
    test_path = root / "data" / "test.csv"
    models_dir = root / "models"

    preprocessor = load_fitted_preprocessor(models_dir)

    train_df = _load_train_frame(train_path)
    test_df = _load_train_frame(test_path)

    X_train_df, y_train = prepare_xy(train_df)
    X_test_df, y_test = prepare_xy(test_df)

    X_train = preprocessor.transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)

    _LOGGER.info(
        "Transformed shapes - train %s, test %s",
        X_train.shape,
        X_test.shape,
    )

    rf = RandomForestClassifier(random_state=random_state)
    xgb = XGBClassifier(random_state=random_state)

    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]

    xgb_pred = xgb.predict(X_test)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]

    rf_metrics = compute_test_metrics(y_test, rf_pred, rf_proba)
    xgb_metrics = compute_test_metrics(y_test, xgb_pred, xgb_proba)

    winner_label, reason = pick_winner(rf_metrics, xgb_metrics)

    return rf_metrics, xgb_metrics, (winner_label, reason), y_test, rf_pred, xgb_pred


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    rf_metrics, xgb_metrics, (winner_label, reason), y_test, rf_pred, xgb_pred = (
        run_baseline()
    )

    target_names = ["no_churn", "churn"]
    print("\n=== RandomForestClassifier - classification_report (test) ===")
    print(
        classification_report(
            y_test,
            rf_pred,
            labels=[0, 1],
            target_names=target_names,
            zero_division=0,
        )
    )

    print("\n=== XGBClassifier - classification_report (test) ===")
    print(
        classification_report(
            y_test,
            xgb_pred,
            labels=[0, 1],
            target_names=target_names,
            zero_division=0,
        )
    )

    print("\n=== Comparison (test set; primary metric: churn F1) ===")
    print(_format_comparison_row("RandomForestClassifier", rf_metrics))
    print(_format_comparison_row("XGBClassifier", xgb_metrics))
    print(f"\nBetter model on test ({reason}): {winner_label}")


if __name__ == "__main__":
    main()
