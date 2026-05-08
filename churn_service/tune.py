"""Phase 2.2–2.3: hyperparameter tuning, feature importance, and production artifacts.

Runs 5-fold ``RandomizedSearchCV`` for ``RandomForestClassifier`` and ``XGBClassifier``,
compares tuned models on the held-out test set (same churn-F1 / ROC-AUC rules as
Phase 2.1), saves the champion to ``models/final_model.joblib``, top-5 importances to
``models/feature_importance.json``, and a test-set confusion matrix to
``assets/plots/confusion_matrix.png``.

Run from the repository root::

    python churn_service/tune.py

``n_iter`` per search is ``TUNE_N_ITER`` (default 30); adjust at module top for speed.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
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
from churn_service.train import (  # noqa: E402
    compute_test_metrics,
    load_fitted_preprocessor,
    pick_winner,
)

# Randomized search iterations per estimator (total work ≈ 2 * n_iter * cv folds).
TUNE_N_ITER = 30
CV_SPLITS = 5
RANDOM_STATE = 42


def _display_params(params: dict[str, Any]) -> dict[str, Any]:
    """Convert numpy search dtypes to native Python for readable logs."""
    out: dict[str, Any] = {}
    for key, val in params.items():
        if hasattr(val, "item"):
            out[key] = val.item()
        else:
            out[key] = val
    return out


def top_n_feature_importance(
    feature_names: np.ndarray,
    importances: np.ndarray,
    n: int = 5,
) -> list[dict[str, Any]]:
    """Return the top ``n`` features by importance with 1-based ranks."""
    if len(feature_names) != len(importances):
        raise ValueError("feature_names and importances must have the same length")
    order = np.argsort(importances)[::-1][:n]
    out: list[dict[str, Any]] = []
    for rank, idx in enumerate(order, start=1):
        out.append(
            {
                "rank": rank,
                "feature": str(feature_names[idx]),
                "importance": float(importances[idx]),
            }
        )
    return out


def run_tuning(
    churn_root: Path | None = None,
    *,
    random_state: int = RANDOM_STATE,
    n_iter: int = TUNE_N_ITER,
) -> dict[str, Any]:
    """Tune RF and XGB with RandomizedSearchCV; evaluate on test; write artifacts.

    Returns a summary dict with paths, metrics, and champion metadata.
    """
    root = churn_root if churn_root is not None else _CHURN_DIR
    train_path = root / "data" / "train.csv"
    test_path = root / "data" / "test.csv"
    models_dir = root / "models"
    plots_dir = root / "assets" / "plots"

    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = load_fitted_preprocessor(models_dir)

    train_df = _load_train_frame(train_path)
    test_df = _load_train_frame(test_path)
    X_train_df, y_train = prepare_xy(train_df)
    X_test_df, y_test = prepare_xy(test_df)

    X_train = preprocessor.transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)

    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=random_state)
    scorer = make_scorer(f1_score, pos_label=1)

    rf_base = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    rf_params = {
        "n_estimators": randint(50, 501),
        "max_depth": [None, 8, 12, 16, 24, 32],
        "min_samples_split": randint(2, 21),
    }
    rf_search = RandomizedSearchCV(
        rf_base,
        param_distributions=rf_params,
        n_iter=n_iter,
        scoring=scorer,
        cv=cv,
        refit=True,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )
    _LOGGER.info("Starting RandomizedSearchCV for RandomForestClassifier...")
    rf_search.fit(X_train, y_train)
    _LOGGER.info(
        "RandomForestClassifier — best CV F1 (churn): %.6f | best_params=%s",
        rf_search.best_score_,
        _display_params(dict(rf_search.best_params_)),
    )
    print(
        f"\n[RandomForestClassifier] Best CV F1 (churn): {rf_search.best_score_:.6f}\n"
        f"Best parameters: {_display_params(dict(rf_search.best_params_))}\n"
    )

    xgb_base = XGBClassifier(random_state=random_state)
    xgb_params = {
        "learning_rate": loguniform(1e-3, 0.3),
        "max_depth": randint(3, 12),
        "n_estimators": randint(50, 501),
        "subsample": uniform(0.6, 0.4),
    }
    xgb_search = RandomizedSearchCV(
        xgb_base,
        param_distributions=xgb_params,
        n_iter=n_iter,
        scoring=scorer,
        cv=cv,
        refit=True,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )
    _LOGGER.info("Starting RandomizedSearchCV for XGBClassifier...")
    xgb_search.fit(X_train, y_train)
    _LOGGER.info(
        "XGBClassifier — best CV F1 (churn): %.6f | best_params=%s",
        xgb_search.best_score_,
        _display_params(dict(xgb_search.best_params_)),
    )
    print(
        f"\n[XGBClassifier] Best CV F1 (churn): {xgb_search.best_score_:.6f}\n"
        f"Best parameters: {_display_params(dict(xgb_search.best_params_))}\n"
    )

    rf_est = rf_search.best_estimator_
    xgb_est = xgb_search.best_estimator_

    rf_pred = rf_est.predict(X_test)
    rf_proba = rf_est.predict_proba(X_test)[:, 1]
    xgb_pred = xgb_est.predict(X_test)
    xgb_proba = xgb_est.predict_proba(X_test)[:, 1]

    rf_metrics = compute_test_metrics(y_test, rf_pred, rf_proba)
    xgb_metrics = compute_test_metrics(y_test, xgb_pred, xgb_proba)

    winner_label, reason = pick_winner(rf_metrics, xgb_metrics)
    tie_break_note = ""
    if winner_label == "tie":
        champion_est = xgb_est
        champion_name = "XGBClassifier"
        champion_metrics = xgb_metrics
        champion_best_params = _display_params(dict(xgb_search.best_params_))
        champion_cv_score = float(xgb_search.best_score_)
        tie_break_note = (
            "Test metrics tied; champion set to XGBClassifier per tie-break policy."
        )
        _LOGGER.info("%s | %s", tie_break_note, reason)
        print(f"\n{tie_break_note}\n")
    elif winner_label == "RandomForestClassifier":
        champion_est = rf_est
        champion_name = "RandomForestClassifier"
        champion_metrics = rf_metrics
        champion_best_params = _display_params(dict(rf_search.best_params_))
        champion_cv_score = float(rf_search.best_score_)
    else:
        champion_est = xgb_est
        champion_name = "XGBClassifier"
        champion_metrics = xgb_metrics
        champion_best_params = _display_params(dict(xgb_search.best_params_))
        champion_cv_score = float(xgb_search.best_score_)

    _LOGGER.info(
        "Selection reason: %s",
        reason if winner_label != "tie" else f"{reason}; {tie_break_note.strip()}",
    )
    _LOGGER.info(
        "Champion: %s | test F1(churn)=%.6f ROC-AUC=%.6f | best_params=%s",
        champion_name,
        champion_metrics["f1_churn"],
        champion_metrics["roc_auc"],
        champion_best_params,
    )
    print(
        f"\n=== Champion (test set) ===\n"
        f"Model: {champion_name}\n"
        f"Selection: {reason}\n"
        f"Best parameters: {champion_best_params}\n"
        f"Test F1 (churn)={champion_metrics['f1_churn']:.6f}  "
        f"ROC-AUC={champion_metrics['roc_auc']:.6f}\n"
    )

    final_model_path = models_dir / "final_model.joblib"
    joblib.dump(champion_est, final_model_path)
    _LOGGER.info("Saved champion estimator to %s", final_model_path)

    feature_names = preprocessor.get_feature_names_out()
    top_features = top_n_feature_importance(
        feature_names,
        champion_est.feature_importances_,
        n=5,
    )
    importance_payload = {
        "model_type": champion_name,
        "metric": "f1_churn_pos_label_1",
        "best_cv_score": champion_cv_score,
        "top_features": top_features,
    }
    importance_path = models_dir / "feature_importance.json"
    with importance_path.open("w", encoding="utf-8") as f:
        json.dump(importance_payload, f, indent=2)
    _LOGGER.info("Wrote top-5 feature importances to %s", importance_path)

    y_champion_pred = champion_est.predict(X_test)
    cm_path = plots_dir / "confusion_matrix.png"
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_champion_pred,
        labels=[0, 1],
        display_labels=["no_churn", "churn"],
    )
    disp.plot()
    disp.figure_.savefig(cm_path, bbox_inches="tight")
    plt.close(disp.figure_)
    _LOGGER.info("Saved confusion matrix plot to %s", cm_path)

    return {
        "champion": champion_name,
        "reason": reason,
        "champion_metrics": champion_metrics,
        "best_params": champion_best_params,
        "final_model_path": str(final_model_path),
        "feature_importance_path": str(importance_path),
        "confusion_matrix_path": str(cm_path),
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run_tuning()


if __name__ == "__main__":
    main()
