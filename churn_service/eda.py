"""Exploratory data analysis and structural audit for telecom churn datasets.

This module supports Phase 1.1–1.2: ingestion, profiling, and visual EDA for
``train.csv`` / ``test.csv`` under ``churn_service/data/``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

LOGGER = logging.getLogger(__name__)

# Eleven business features (post-normalization snake_case).
FEATURE_COLUMNS: list[str] = [
    "customer_id",
    "age",
    "gender",
    "tenure",
    "usage_frequency",
    "support_calls",
    "payment_delay",
    "subscription_type",
    "contract_length",
    "total_spend",
    "last_interaction",
]

NUMERIC_FEATURE_COLUMNS: list[str] = [
    "age",
    "tenure",
    "usage_frequency",
    "support_calls",
    "payment_delay",
    "total_spend",
    "last_interaction",
]

CATEGORICAL_FEATURE_COLUMNS: list[str] = [
    "gender",
    "subscription_type",
    "contract_length",
]

# Map stripped lowercase source header -> canonical column name.
_COLUMN_ALIASES: dict[str, str] = {
    "customerid": "customer_id",
    "customer_id": "customer_id",
    "age": "age",
    "gender": "gender",
    "tenure": "tenure",
    "usage frequency": "usage_frequency",
    "usage_frequency": "usage_frequency",
    "support calls": "support_calls",
    "support_calls": "support_calls",
    "payment delay": "payment_delay",
    "payment_delay": "payment_delay",
    "subscription type": "subscription_type",
    "subscription_type": "subscription_type",
    "contract length": "contract_length",
    "contract_length": "contract_length",
    "total spend": "total_spend",
    "total_spend": "total_spend",
    "last interaction": "last_interaction",
    "last_interaction": "last_interaction",
    "churn": "churn",
}


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging for CLI and library-style imports.

    Args:
        level: Logging level (default ``logging.INFO``).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical snake_case names using alias lookup.

    Args:
        df: Raw dataframe as read from CSV.

    Returns:
        Copy of ``df`` with recognized columns renamed; unrecognized columns
        are left unchanged.
    """
    out = df.copy()
    new_names: dict[str, str] = {}
    for col in out.columns:
        key = str(col).strip().lower().replace("_", " ")
        key = key.replace(" ", " ")  # normalize spaces
        # Try direct match on lowered original (handles snake_case keys)
        lowered = str(col).strip().lower()
        if lowered in _COLUMN_ALIASES:
            new_names[col] = _COLUMN_ALIASES[lowered]
        elif key in _COLUMN_ALIASES:
            new_names[col] = _COLUMN_ALIASES[key]
        else:
            # e.g. "Usage Frequency" -> "usage frequency"
            spaced = lowered.replace("_", " ")
            if spaced in _COLUMN_ALIASES:
                new_names[col] = _COLUMN_ALIASES[spaced]
    return out.rename(columns=new_names)


def _require_paths(train_path: Path, test_path: Path) -> None:
    if not train_path.is_file():
        raise FileNotFoundError(f"Training CSV not found: {train_path}")
    if not test_path.is_file():
        raise FileNotFoundError(f"Test CSV not found: {test_path}")


def load_and_audit_data(
    train_path: str | Path,
    test_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load train/test CSVs, normalize headers, and run structural audit.

    Prints head(5), dtypes, and missing-value summaries for the eleven
    business features. Logs whether ``customer_id`` is a unique identifier
    per split (recommended exclusion from modeling features).

    Args:
        train_path: Path to ``train.csv``.
        test_path: Path to ``test.csv``.

    Returns:
        Tuple of ``(train_df, test_df, audit_meta)`` where ``audit_meta``
        includes ``customer_id_is_unique`` per split and column presence flags.

    Raises:
        FileNotFoundError: If either CSV path does not exist.
        ValueError: If required columns are missing after normalization.
    """
    train_p = Path(train_path)
    test_p = Path(test_path)
    _require_paths(train_p, test_p)

    train_raw = pd.read_csv(train_p)
    test_raw = pd.read_csv(test_p)
    train_df = _normalize_columns(train_raw)
    test_df = _normalize_columns(test_raw)

    required = FEATURE_COLUMNS + ["churn"]
    for name, df in ("train", train_df), ("test", test_df):
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{name} split missing columns after normalize: {missing}")

    audit: dict[str, Any] = {}

    def _audit_split(split_name: str, df: pd.DataFrame) -> None:
        print(f"\n{'=' * 60}\nFirst 5 rows - {split_name}\n{'=' * 60}")
        print(df.head().to_string())
        print(f"\n{'=' * 60}\ndtypes - {split_name} (11 features + target)\n{'=' * 60}")
        cols_show = FEATURE_COLUMNS + ["churn"]
        print(df[cols_show].dtypes.to_string())

        print(f"\n{'=' * 60}\nMissing values - {split_name}\n{'=' * 60}")
        miss = df[FEATURE_COLUMNS].isna()
        miss_count = miss.sum()
        miss_pct = (miss.mean() * 100).round(2)
        summary = pd.DataFrame({"missing_count": miss_count, "missing_pct": miss_pct})
        print(summary.to_string())

        cid = df["customer_id"]
        n_rows = len(df)
        n_unique = cid.nunique()
        dupes = int(cid.duplicated().sum())
        is_unique = n_unique == n_rows and dupes == 0
        audit[f"{split_name}_customer_id_is_unique"] = is_unique
        LOGGER.info(
            "%s customer_id: n_rows=%d nunique=%d duplicated_rows=%d unique=%s",
            split_name,
            n_rows,
            n_unique,
            dupes,
            is_unique,
        )
        rec = (
            "customer_id appears unique per row - exclude from feature matrix for modeling."
            if is_unique
            else "customer_id is NOT unique - investigate duplicates before modeling."
        )
        LOGGER.info("Modeling note (%s): %s", split_name, rec)
        print(f"\n--- customer_id audit ({split_name}) ---\n{rec}\n")

        for c in CATEGORICAL_FEATURE_COLUMNS:
            df[c] = df[c].astype("string")

    _audit_split("train", train_df)
    _audit_split("test", test_df)

    audit["customer_id_is_unique"] = {
        "train": audit.get("train_customer_id_is_unique", False),
        "test": audit.get("test_customer_id_is_unique", False),
    }

    return train_df, test_df, audit


def generate_eda_plots(
    train_df: pd.DataFrame,
    output_dir: str | Path,
    target_col: str = "churn",
) -> list[Path]:
    """Write class-balance and numeric correlation plots to ``output_dir``.

    Args:
        train_df: Training dataframe (normalized column names).
        output_dir: Directory for PNG outputs (created if missing).
        target_col: Binary or categorical churn column name.

    Returns:
        List of paths to saved PNG files.

    Raises:
        KeyError: If ``target_col`` is not present.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    if target_col not in train_df.columns:
        raise KeyError(f"Target column {target_col!r} not in dataframe.")

    target_series = train_df[target_col]
    plot_df = train_df.copy()
    x_plot = target_col
    num_target = pd.to_numeric(target_series, errors="coerce")
    if num_target.notna().all() and set(num_target.astype(int).unique()).issubset({0, 1}):
        plot_df = plot_df.copy()
        x_plot = "_churn_label"
        plot_df[x_plot] = num_target.astype(int).map({0: "No Churn", 1: "Churn"})
    elif target_series.dtype == object or isinstance(target_series.dtype, pd.StringDtype):
        lowered = target_series.astype(str).str.strip().str.lower()
        if set(lowered.dropna().unique()).issubset({"yes", "no", "true", "false"}):
            plot_df = plot_df.copy()
            x_plot = "_churn_label"
            plot_df[x_plot] = lowered.map(
                {"yes": "Churn", "true": "Churn", "no": "No Churn", "false": "No Churn"}
            )

    # --- Target distribution ---
    plt.figure(figsize=(8, 5))
    order = plot_df[x_plot].value_counts().index.tolist()
    ax = sns.countplot(data=plot_df, x=x_plot, order=order)
    ax.set_title("Churn class distribution (train)")
    ax.set_xlabel("Churn status")
    ax.set_ylabel("Count")
    plt.tight_layout()
    p1 = out / "target_distribution.png"
    plt.savefig(p1, dpi=300, bbox_inches="tight")
    plt.close()
    saved.append(p1)
    LOGGER.info("Saved plot: %s", p1)

    # --- Numeric correlation heatmap (features only; exclude ID and categoricals) ---
    num_cols = [c for c in NUMERIC_FEATURE_COLUMNS if c in train_df.columns]
    num_df = train_df[num_cols].apply(pd.to_numeric, errors="coerce")
    # Optionally include binary numeric target for correlation with features
    target_numeric = pd.to_numeric(train_df[target_col], errors="coerce")
    if target_numeric.notna().all() and set(
        int(x) for x in target_numeric.dropna().unique()
    ).issubset({0, 1}):
        num_df = num_df.copy()
        num_df[target_col] = target_numeric
        LOGGER.info("Including binary %r in correlation matrix.", target_col)
    elif target_numeric.notna().sum() > 0:
        LOGGER.info(
            "Target %r is not strictly 0/1 numeric; correlation heatmap uses numeric features only.",
            target_col,
        )

    plt.figure(figsize=(10, 8))
    corr = num_df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title("Pearson correlation - numeric features (train)")
    plt.tight_layout()
    p2 = out / "numeric_correlation_heatmap.png"
    plt.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close()
    saved.append(p2)
    LOGGER.info("Saved plot: %s", p2)

    return saved


def main() -> None:
    """CLI entry: load data from ``churn_service/data/`` and write EDA plots."""
    configure_logging()
    base = Path(__file__).resolve().parent
    train_path = base / "data" / "train.csv"
    test_path = base / "data" / "test.csv"
    plots_dir = base / "assets" / "plots"

    train_df, test_df, audit = load_and_audit_data(train_path, test_path)
    _ = test_df  # audited; plots use train only per plan
    _ = audit
    generate_eda_plots(train_df, plots_dir, target_col="churn")
    LOGGER.info("EDA complete. Plots directory: %s", plots_dir)


if __name__ == "__main__":
    main()
