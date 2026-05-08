"""Phase 1.3–1.4: sanitization and transformation for telecom churn modeling.

Loads normalized training features, drops identifier noise, imputes missing
values, caps extreme values on spend/delay, one-hot encodes low-cardinality
categoricals, scales numerics, and persists the fitted preprocessor for
inference-time parity (PoC to prod).

Run from the repository root (e.g. ``python churn_service/preprocess.py``) so
``churn_service`` is importable and ``preprocessor.joblib`` references stable
pickle paths for ``PercentileClipper``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

_LOGGER = logging.getLogger(__name__)

_CHURN_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CHURN_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from churn_service.eda import (  # noqa: E402
    CATEGORICAL_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    _normalize_columns,
)
from churn_service.preprocessing_transformers import PercentileClipper  # noqa: E402

CLIP_NUMERIC_COLUMNS: list[str] = ["total_spend", "payment_delay"]
NUMERIC_NO_CLIP_COLUMNS: list[str] = [
    c for c in NUMERIC_FEATURE_COLUMNS if c not in CLIP_NUMERIC_COLUMNS
]


def _load_train_frame(train_path: Path) -> pd.DataFrame:
    if not train_path.is_file():
        raise FileNotFoundError(f"Training CSV not found: {train_path}")
    raw = pd.read_csv(train_path)
    df = _normalize_columns(raw)
    required = FEATURE_COLUMNS + ["churn"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Train data missing columns after normalize: {missing}")
    return df


def build_preprocessor() -> ColumnTransformer:
    """Bundle imputation, clipping (subset), one-hot, and scaling."""
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    numeric_clip_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clipper", PercentileClipper(lower_percentile=1.0, upper_percentile=99.0)),
            ("scaler", StandardScaler()),
        ]
    )

    numeric_plain_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    transformers: list[tuple[str, Pipeline, list[str]]] = [
        ("categorical", categorical_pipe, CATEGORICAL_FEATURE_COLUMNS),
        ("numeric_clip", numeric_clip_pipe, CLIP_NUMERIC_COLUMNS),
        ("numeric_plain", numeric_plain_pipe, NUMERIC_NO_CLIP_COLUMNS),
    ]

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )


def fit_preprocessors(
    X: pd.DataFrame,
    preprocessor: ColumnTransformer,
) -> tuple[np.ndarray, ColumnTransformer]:
    """Fit the column transformer on features only; returns dense ``X_train``."""
    X_train = preprocessor.fit_transform(X)
    return X_train, preprocessor


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Drop ID column, split features/target, coerce ``churn`` to 0/1 int."""
    n0 = len(df)
    df = df.loc[df["churn"].notna()].copy()
    if len(df) < n0:
        _LOGGER.warning("Dropped %d row(s) with missing churn target.", n0 - len(df))
    feature_cols = [c for c in FEATURE_COLUMNS if c != "customer_id"]
    X = df[feature_cols].copy()
    y = df["churn"].astype(int).to_numpy()
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Target ``churn`` must be binary 0/1 after load.")
    return X, y


def main() -> tuple[np.ndarray, np.ndarray]:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    churn_root = _CHURN_DIR
    train_path = churn_root / "data" / "train.csv"
    models_dir = churn_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = models_dir / "preprocessor.joblib"

    train_df = _load_train_frame(train_path)
    X_df, y = prepare_xy(train_df)

    preprocessor = build_preprocessor()
    X_train, fitted = fit_preprocessors(X_df, preprocessor)

    joblib.dump(fitted, out_path)
    _LOGGER.info("Saved fitted preprocessor to %s", out_path)
    _LOGGER.info("X_train shape: %s | y_train shape: %s", X_train.shape, y.shape)
    names = fitted.get_feature_names_out()
    _LOGGER.info("Transformed feature count: %d", len(names))

    # Optional sanity: stratified holdout is not required for this script;
    # full matrix is returned for downstream training notebooks/services.
    print(f"X_train (dense) shape: {X_train.shape}")
    print(f"y_train shape: {y.shape}")
    print(f"Preprocessor saved to: {out_path}")
    return X_train, y


if __name__ == "__main__":
    X_train, y_train = main()
