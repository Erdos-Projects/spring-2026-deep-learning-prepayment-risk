"""Data preparation and chronological train/test split pipeline.

This script builds a monthly modeling dataset by combining loan-level LendingClub
records with monthly macroeconomic series, then saves a chronological train/test
split. It is designed to be clear, reproducible, and easy to adapt for similar
projects.

Default workflow
----------------
1. Load the source CSV files.
2. Clean loan-level data and remove duplicate loan IDs.
3. Aggregate the loan data to monthly frequency.
4. Standardize the macroeconomic series to monthly frequency.
5. Build a continuous macroeconomic month index with a configurable lookback.
6. Merge loan and macroeconomic data on month.
7. Validate chronological continuity.
8. Split the final dataset into train and test sets without shuffling.
9. Save train/test CSV files (and optionally the full merged dataset).

Example
-------
python train_test_pipeline.py --lendingclub-file ../Data/LendingClub.csv --fedfunds-file ../Data/FEDFUNDS.csv --tbill-file ../Data/tbill.csv --output-dir ../outputs
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


LOGGER = logging.getLogger(__name__)



@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the end-to-end data preparation pipeline."""

    lendingclub_file: Path
    fedfunds_file: Path
    tbill_file: Path
    output_dir: Path = Path(".")
    train_filename: str = "train.csv"
    test_filename: str = "test.csv"
    full_filename: str = "lendingclub_fed_tbill.csv"
    save_full_dataset: bool = False
    split_ratio: float = 0.75
    macro_lookback_months: int = 12

    # Loan-level schema
    loan_id_col: str = "id"
    loan_issue_date_col: str = "issue_d"
    loan_amount_col: str = "loan_amnt"
    loan_term_col: str = "term"
    loan_interest_rate_col: str = "int_rate"
    loan_issue_date_format: str = "%b-%Y"

    # Macroeconomic schema
    macro_date_col: str = "observation_date"
    tbill_value_col: str = "TB3MS"
    fedfunds_value_col: str = "FEDFUNDS"


class PipelineError(RuntimeError):
    """Raised when the input data violates a pipeline assumption."""


def parse_args() -> PipelineConfig:
    """Parse command-line arguments into a validated pipeline configuration."""
    parser = argparse.ArgumentParser(
        description="Prepare a monthly modeling dataset and save a chronological train/test split."
    )
    parser.add_argument("--lendingclub-file", required=True, type=Path, help="Path to the loan-level CSV file.")
    parser.add_argument("--fedfunds-file", required=True, type=Path, help="Path to the FED funds rate CSV file.")
    parser.add_argument("--tbill-file", required=True, type=Path, help="Path to the Treasury bill CSV file.")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Directory where outputs will be saved.")
    parser.add_argument("--train-filename", default="train.csv", help="Filename for the training split.")
    parser.add_argument("--test-filename", default="test.csv", help="Filename for the test split.")
    parser.add_argument(
        "--full-filename",
        default="lendingclub_fed_tbill.csv",
        help="Filename for the full merged dataset (used only when --save-full-dataset is set).",
    )
    parser.add_argument(
        "--save-full-dataset",
        action="store_true",
        help="Also save the complete merged monthly dataset.",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.75,
        help="Fraction of rows assigned to the training split. Must satisfy 0 < split_ratio < 1.",
    )
    parser.add_argument(
        "--macro-lookback-months",
        type=int,
        default=12,
        help="Number of months to include before the first loan month when building the macro date index.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )

    args = parser.parse_args()
    configure_logging(args.log_level)

    config = PipelineConfig(
        lendingclub_file=args.lendingclub_file,
        fedfunds_file=args.fedfunds_file,
        tbill_file=args.tbill_file,
        output_dir=args.output_dir,
        train_filename=args.train_filename,
        test_filename=args.test_filename,
        full_filename=args.full_filename,
        save_full_dataset=args.save_full_dataset,
        split_ratio=args.split_ratio,
        macro_lookback_months=args.macro_lookback_months,
    )
    validate_config(config)
    return config


def configure_logging(level: str) -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def validate_config(config: PipelineConfig) -> None:
    """Validate user-supplied configuration before the pipeline starts."""
    if not 0 < config.split_ratio < 1:
        raise ValueError("split_ratio must satisfy 0 < split_ratio < 1.")
    if config.macro_lookback_months < 0:
        raise ValueError("macro_lookback_months must be non-negative.")


def ensure_columns_exist(df: pd.DataFrame, required_columns: Iterable[str], df_name: str) -> None:
    """Fail fast when an input table does not contain the expected schema."""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise PipelineError(f"{df_name} is missing required columns: {missing}")


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Convert a column to numeric while tolerating strings like '13.56%'."""
    if pd.api.types.is_numeric_dtype(series):
        return series
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def load_lendingclub_data(config: PipelineConfig) -> pd.DataFrame:
    """Load only the loan columns needed by the pipeline."""
    loan_columns = [
        config.loan_id_col,
        config.loan_issue_date_col,
        config.loan_amount_col,
        config.loan_term_col,
        config.loan_interest_rate_col,
    ]

    df = pd.read_csv(config.lendingclub_file, usecols=loan_columns)
    ensure_columns_exist(df, loan_columns, "LendingClub data")
    LOGGER.info("Loaded LendingClub data with shape %s", df.shape)
    return df


def load_macro_data(file_path: Path, expected_date_col: str, expected_value_col: str, name: str) -> pd.DataFrame:
    """Load a macroeconomic CSV and validate its required columns."""
    df = pd.read_csv(file_path)
    ensure_columns_exist(df, [expected_date_col, expected_value_col], name)
    LOGGER.info("Loaded %s with shape %s", name, df.shape)
    return df


def clean_lendingclub_data(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Clean loan-level records and standardize them for monthly aggregation.

    Steps
    -----
    - Remove duplicate loan IDs, keeping the first record.
    - Parse the issue date.
    - Standardize numeric fields used downstream.
    - Sort chronologically and drop incomplete rows in required modeling fields.
    - Create a monthly period column for time-based aggregation.
    """
    cleaned = df.copy()

    duplicate_mask = cleaned.duplicated(subset=config.loan_id_col, keep="first")
    duplicate_count = int(duplicate_mask.sum())
    cleaned = cleaned.loc[~duplicate_mask].copy()

    cleaned[config.loan_issue_date_col] = pd.to_datetime(
        cleaned[config.loan_issue_date_col],
        format=config.loan_issue_date_format,
        errors="coerce",
    )
    cleaned[config.loan_amount_col] = coerce_numeric(cleaned[config.loan_amount_col])
    cleaned[config.loan_interest_rate_col] = coerce_numeric(cleaned[config.loan_interest_rate_col])

    required_fields = [
        config.loan_issue_date_col,
        config.loan_amount_col,
        config.loan_term_col,
        config.loan_interest_rate_col,
    ]
    cleaned = (
        cleaned[
            [
                config.loan_issue_date_col,
                config.loan_amount_col,
                config.loan_term_col,
                config.loan_interest_rate_col,
            ]
        ]
        .dropna(subset=required_fields)
        .sort_values(config.loan_issue_date_col)
        .reset_index(drop=True)
    )

    cleaned["Month"] = cleaned[config.loan_issue_date_col].dt.to_period("M")

    LOGGER.info("Removed %d duplicate loan records", duplicate_count)
    LOGGER.info("Cleaned LendingClub data shape: %s", cleaned.shape)
    return cleaned


def aggregate_lendingclub_monthly(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Aggregate cleaned loan data to monthly frequency.

    The output matches the notebook logic while making the aggregation logic
    explicit and easy to reuse elsewhere.
    """
    monthly = (
        df.groupby("Month", as_index=False)
        .agg(
            int_rate_mean=(config.loan_interest_rate_col, "mean"),
            int_rate_median=(config.loan_interest_rate_col, "median"),
            int_rate_std=(config.loan_interest_rate_col, "std"),
            loan_amnt_sum=(config.loan_amount_col, "sum"),
            loan_amnt_mean=(config.loan_amount_col, "mean"),
            loan_amnt_count=(config.loan_amount_col, "count"),
        )
        .sort_values("Month")
        .reset_index(drop=True)
    )

    LOGGER.info("Monthly LendingClub dataset shape: %s", monthly.shape)
    return monthly


def prepare_monthly_macro_series(
    df: pd.DataFrame,
    *,
    date_col: str,
    value_col: str,
    output_value_name: str,
) -> pd.DataFrame:
    """Convert a dated macroeconomic series to a monthly two-column table.

    The function is intentionally generic so it can be reused for additional
    macroeconomic inputs beyond the two default series.
    """
    prepared = df[[date_col, value_col]].copy()
    prepared[date_col] = pd.to_datetime(prepared[date_col], errors="coerce")
    prepared[value_col] = coerce_numeric(prepared[value_col])
    prepared = prepared.dropna(subset=[date_col, value_col])
    prepared["Month"] = prepared[date_col].dt.to_period("M")

    # Group by month to make the function safe even when multiple observations
    # exist within the same calendar month.
    monthly = (
        prepared.groupby("Month", as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: output_value_name})
        .sort_values("Month")
        .reset_index(drop=True)
    )
    return monthly


def build_macro_dataset(
    lendingclub_monthly: pd.DataFrame,
    tbill: pd.DataFrame,
    fedfunds: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    """Build a continuous monthly macroeconomic table aligned to the loan horizon."""
    tbill_monthly = prepare_monthly_macro_series(
        tbill,
        date_col=config.macro_date_col,
        value_col=config.tbill_value_col,
        output_value_name="Treasury_data",
    )
    fed_monthly = prepare_monthly_macro_series(
        fedfunds,
        date_col=config.macro_date_col,
        value_col=config.fedfunds_value_col,
        output_value_name="fed_rate",
    )

    first_loan_month = lendingclub_monthly["Month"].min()
    last_loan_month = lendingclub_monthly["Month"].max()

    if pd.isna(first_loan_month) or pd.isna(last_loan_month):
        raise PipelineError("Loan data is empty after cleaning; cannot build macro dataset.")

    macro_index = pd.DataFrame(
        {
            "Month": pd.period_range(
                start=first_loan_month - config.macro_lookback_months,
                end=last_loan_month,
                freq="M",
            )
        }
    )

    combined_macro = (
        macro_index.merge(tbill_monthly, on="Month", how="left")
        .merge(fed_monthly, on="Month", how="left")
        .sort_values("Month")
        .reset_index(drop=True)
    )

    LOGGER.info(
        "Macro dataset shape: %s | Missing Treasury months: %d | Missing Fed months: %d",
        combined_macro.shape,
        int(combined_macro["Treasury_data"].isna().sum()),
        int(combined_macro["fed_rate"].isna().sum()),
    )
    return combined_macro


def merge_modeling_dataset(lendingclub_monthly: pd.DataFrame, combined_macro: pd.DataFrame) -> pd.DataFrame:
    """Merge the monthly loan features with the monthly macroeconomic features."""
    merged = (
        lendingclub_monthly.merge(combined_macro, on="Month", how="left")
        .sort_values("Month")
        .reset_index(drop=True)
    )
    LOGGER.info("Final merged dataset shape: %s", merged.shape)
    return merged


def validate_monthly_continuity(df: pd.DataFrame) -> pd.PeriodIndex:
    """Check whether the final monthly dataset contains gaps in time.

    Returns the missing months so callers can decide whether gaps are acceptable.
    The original workflow only reported them, so this function does the same.
    """
    if df.empty:
        raise PipelineError("The merged dataset is empty; cannot validate continuity.")

    observed_months = df["Month"]
    full_range = pd.period_range(start=observed_months.min(), end=observed_months.max(), freq="M")
    missing_months = full_range[~full_range.isin(observed_months)]

    if len(missing_months) == 0:
        LOGGER.info("No missing months detected in the final dataset.")
    else:
        LOGGER.warning("Detected %d missing month(s): %s", len(missing_months), list(missing_months))

    return missing_months


def chronological_train_test_split(df: pd.DataFrame, split_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time-ordered dataset into train and test partitions without shuffling."""
    if df.empty:
        raise PipelineError("Cannot split an empty dataset.")

    split_idx = int(len(df) * split_ratio)
    if split_idx <= 0 or split_idx >= len(df):
        raise PipelineError(
            "The chosen split_ratio yields an empty train or test split. "
            "Adjust split_ratio or provide more data."
        )

    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    LOGGER.info(
        "Train: %s -> %s (%d rows) | Test: %s -> %s (%d rows)",
        train["Month"].min(),
        train["Month"].max(),
        len(train),
        test["Month"].min(),
        test["Month"].max(),
        len(test),
    )

    if train["Month"].max() >= test["Month"].min():
        LOGGER.warning("Train/test boundary overlaps or touches in time ordering.")

    return train, test


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame to CSV, creating parent directories when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    LOGGER.info("Saved %s", path)


def run_pipeline(config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute the full data preparation pipeline and return all major outputs."""
    lendingclub_raw = load_lendingclub_data(config)
    fedfunds_raw = load_macro_data(
        config.fedfunds_file,
        expected_date_col=config.macro_date_col,
        expected_value_col=config.fedfunds_value_col,
        name="FED funds data",
    )
    tbill_raw = load_macro_data(
        config.tbill_file,
        expected_date_col=config.macro_date_col,
        expected_value_col=config.tbill_value_col,
        name="Treasury bill data",
    )

    lendingclub_clean = clean_lendingclub_data(lendingclub_raw, config)
    lendingclub_monthly = aggregate_lendingclub_monthly(lendingclub_clean, config)
    macro_monthly = build_macro_dataset(lendingclub_monthly, tbill_raw, fedfunds_raw, config)
    final_dataset = merge_modeling_dataset(lendingclub_monthly, macro_monthly)

    validate_monthly_continuity(final_dataset)
    train, test = chronological_train_test_split(final_dataset, config.split_ratio)

    save_dataframe(train, config.output_dir / config.train_filename)
    save_dataframe(test, config.output_dir / config.test_filename)
    if config.save_full_dataset:
        save_dataframe(final_dataset, config.output_dir / config.full_filename)

    return final_dataset, train, test


def main() -> None:
    """CLI entry point."""
    config = parse_args()
    final_dataset, train, test = run_pipeline(config)

    LOGGER.info("Pipeline completed successfully.")
    LOGGER.info(
        "Output summary | Final rows: %d | Train rows: %d | Test rows: %d",
        len(final_dataset),
        len(train),
        len(test),
    )


if __name__ == "__main__":
    main()
