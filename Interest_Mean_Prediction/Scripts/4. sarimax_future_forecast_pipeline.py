#!/usr/bin/env python3
"""
General SARIMAX future forecasting pipeline.

This script reproduces the notebook workflow in a reusable, executable form:
1. Load train and future T-Bill datasets
2. Apply the same train-time transformations
   - Box-Cox transform on the target
   - first and second differencing of the transformed target
   - log transform on the T-Bill series after a positivity shift
   - first differencing of the transformed T-Bill series
   - lagged exogenous feature construction
3. Fit SARIMAX on the transformed target using the lagged T-Bill feature
4. Forecast future transformed values
5. Recursively reconstruct forecasts back to the original target scale
6. Optionally evaluate RMSE and MAE when future actuals are provided
7. Save outputs to disk

Example:
    python sarimax_future_forecast_pipeline.py --train-file ../Data/train.csv --future-file ../Data/tbill.csv --output-dir outputs/sarimax_future --date-col-train Month
--date-col-future observation_date --target-col int_rate_mean --tbill-col-train Treasury_data --tbill-col-future TB3MS
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
import importlib
import subprocess
import sys
from dataclasses import asdict, dataclass


REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "statsmodels": "statsmodels",
}

for module_name, package_name in REQUIRED_PACKAGES.items():
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError:
        print(f"Installing missing package: {package_name}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import inv_boxcox
from scipy.stats import boxcox
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TransformState:
    target_boxcox_lambda: float
    target_boxcox_shift: float
    tbill_log_shift: float


@dataclass(frozen=True)
class ForecastSummary:
    n_train_rows: int
    n_model_rows: int
    n_future_rows: int
    n_forecasts: int
    sarimax_order: tuple[int, int, int]
    tbill_col_train: str
    tbill_col_future: str
    tbill_lag: int
    rmse: float | None
    mae: float | None


@dataclass(frozen=True)
class PipelineConfig:
    train_file: Path
    future_file: Path
    output_dir: Path
    date_col_train: str
    date_col_future: str
    target_col: str
    tbill_col_train: str
    tbill_col_future: str
    sarimax_order: tuple[int, int, int]
    tbill_lag: int
    filter_future_after_train: bool
    save_plot: bool
    show_plot: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a SARIMAX model on transformed training data and forecast future mean interest rates."
    )
    parser.add_argument("--train-file", type=Path, required=True, help="CSV containing training data.")
    parser.add_argument(
        "--future-file",
        type=Path,
        required=True,
        help="CSV containing future T-Bill values and optionally the actual target values.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/sarimax_future"),
        help="Directory for generated outputs.",
    )

    parser.add_argument("--date-col-train", default="Month", help="Date column in the train file.")
    parser.add_argument(
        "--date-col-future",
        default="observation_date",
        help="Date column in the future file.",
    )
    parser.add_argument(
        "--target-col",
        default="int_rate_mean",
        help="Target column name. If also present in the future file, RMSE and MAE are computed.",
    )
    parser.add_argument(
        "--tbill-col-train",
        default="Treasury_data",
        help="T-Bill column in the training file.",
    )
    parser.add_argument(
        "--tbill-col-future",
        default="TB3MS",
        help="T-Bill column in the future file.",
    )
    parser.add_argument(
        "--sarimax-order",
        default="2,0,1",
        help="SARIMAX order as p,d,q. Example: 2,0,1",
    )
    parser.add_argument(
        "--tbill-lag",
        type=int,
        default=1,
        help="Lag to apply to the transformed T-Bill exogenous feature.",
    )
    parser.add_argument(
        "--keep-overlapping-future",
        action="store_true",
        help="Do not filter the future file to dates strictly after the latest training date.",
    )
    parser.add_argument(
        "--save-plot",
        action="store_true",
        help="Save a forecast plot to the output directory.",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display the forecast plot interactively.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_order(order_text: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in order_text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"--sarimax-order must have exactly three integers, got: {order_text!r}")
    try:
        order = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"Invalid --sarimax-order value: {order_text!r}") from exc
    return order  # type: ignore[return-value]


def compute_positive_shift(series: pd.Series, buffer: float = 0.0) -> float:
    minimum = float(series.min())
    return max(0.0, -minimum + buffer)


def load_dataset(path: Path, date_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"Column {date_col!r} not found in {path}.")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().any():
        bad_rows = df[df[date_col].isna()]
        raise ValueError(
            f"Some values in date column {date_col!r} could not be parsed in {path}. "
            f"Bad row count: {len(bad_rows)}"
        )

    df = df.sort_values(date_col).reset_index(drop=True)
    return df


def validate_required_columns(df: pd.DataFrame, required_columns: Iterable[str], dataset_name: str) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {dataset_name}: {missing}")


def apply_train_transformations(
    train_df: pd.DataFrame,
    date_col: str,
    target_col: str,
    tbill_col: str,
) -> tuple[pd.DataFrame, TransformState]:
    train_ts = train_df[[date_col, target_col, tbill_col]].copy()
    train_ts = train_ts.set_index(date_col).sort_index()

    if train_ts[target_col].isna().any():
        raise ValueError(f"Training target column {target_col!r} contains missing values.")
    if train_ts[tbill_col].isna().any():
        raise ValueError(f"Training T-Bill column {tbill_col!r} contains missing values.")

    target_boxcox_shift = compute_positive_shift(train_ts[target_col])
    target_input = train_ts[target_col] + target_boxcox_shift

    if (target_input <= 0).any():
        raise ValueError(
            "Target values are not strictly positive after shift, so Box-Cox cannot be applied."
        )

    train_ts["target_boxcox"], target_boxcox_lambda = boxcox(target_input)
    train_ts["target_diff1"] = train_ts["target_boxcox"].diff(1)
    train_ts["target_diff2"] = train_ts["target_boxcox"].diff(1).diff(1)

    tbill_log_shift = compute_positive_shift(train_ts[tbill_col], buffer=0.1)
    train_ts["tbill_shifted"] = train_ts[tbill_col] + tbill_log_shift
    if (train_ts["tbill_shifted"] <= 0).any():
        raise ValueError(
            "Training T-Bill values are not strictly positive after shift, so log cannot be applied."
        )

    train_ts["tbill_log"] = np.log(train_ts["tbill_shifted"])
    train_ts["tbill_log_diff"] = train_ts["tbill_log"].diff(1)

    state = TransformState(
        target_boxcox_lambda=float(target_boxcox_lambda),
        target_boxcox_shift=float(target_boxcox_shift),
        tbill_log_shift=float(tbill_log_shift),
    )
    return train_ts, state


def build_train_model_frame(train_ts: pd.DataFrame, tbill_lag: int) -> pd.DataFrame:
    if tbill_lag < 0:
        raise ValueError("tbill_lag must be non-negative.")

    frame = pd.DataFrame(index=train_ts.index)
    frame["target_diff2"] = train_ts["target_diff2"]
    frame["tbill_lag_1"] = train_ts["tbill_log_diff"].shift(tbill_lag)
    frame["prev_diff1"] = train_ts["target_diff1"].shift(1)
    frame["prev_boxcox"] = train_ts["target_boxcox"].shift(1)
    frame = frame.dropna()

    if frame.empty:
        raise ValueError(
            "The transformed training frame is empty after differencing and lagging. "
            "Check that the training dataset has enough rows."
        )

    return frame


def build_future_exog(
    train_ts: pd.DataFrame,
    future_df: pd.DataFrame,
    date_col_future: str,
    tbill_col_future: str,
    tbill_col_train: str,
    tbill_lag: int,
    state: TransformState,
) -> pd.DataFrame:
    future_ts = future_df[[date_col_future, tbill_col_future]].copy()
    future_ts = future_ts.rename(
        columns={date_col_future: "date", tbill_col_future: tbill_col_train}
    ).set_index("date").sort_index()

    combined_tbill = pd.concat([train_ts[[tbill_col_train]], future_ts[[tbill_col_train]]], axis=0)
    combined_tbill["tbill_shifted"] = combined_tbill[tbill_col_train] + state.tbill_log_shift

    if (combined_tbill["tbill_shifted"] <= 0).any():
        raise ValueError(
            "The training-derived log shift is not sufficient for the supplied future T-Bill values."
        )

    combined_tbill["tbill_log"] = np.log(combined_tbill["tbill_shifted"])
    combined_tbill["tbill_log_diff"] = combined_tbill["tbill_log"].diff(1)
    combined_tbill["tbill_lag_1"] = combined_tbill["tbill_log_diff"].shift(tbill_lag)

    future_exog = combined_tbill.loc[future_ts.index, ["tbill_lag_1"]].copy()
    if future_exog["tbill_lag_1"].isna().any():
        missing_dates = future_exog.index[future_exog["tbill_lag_1"].isna()].strftime("%Y-%m-%d").tolist()
        raise ValueError(
            f"Missing exogenous lag values for future dates: {missing_dates}. "
            "Make sure the future file starts immediately after the train period, "
            "or includes enough historical context."
        )

    return future_exog


def fit_sarimax(train_frame: pd.DataFrame, order: tuple[int, int, int]) -> SARIMAXResultsWrapper:
    model = SARIMAX(
        endog=train_frame["target_diff2"],
        exog=train_frame[["tbill_lag_1"]],
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    return fitted


def forecast_transformed(
    fitted: SARIMAXResultsWrapper,
    future_exog: pd.DataFrame,
) -> pd.Series:
    forecasts = fitted.forecast(steps=len(future_exog), exog=future_exog[["tbill_lag_1"]])
    forecasts.index = future_exog.index
    return forecasts


def invert_boxcox_value(value: float, state: TransformState) -> float:
    return float(inv_boxcox(value, state.target_boxcox_lambda) - state.target_boxcox_shift)


def reconstruct_recursive_predictions(
    pred_diff2: pd.Series,
    last_train_boxcox: float,
    last_train_diff1: float,
    state: TransformState,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    prev_boxcox = float(last_train_boxcox)
    prev_diff1 = float(last_train_diff1)

    for date, diff2_hat in pred_diff2.items():
        diff2_hat = float(diff2_hat)
        boxcox_hat = diff2_hat + prev_diff1 + prev_boxcox
        diff1_hat = boxcox_hat - prev_boxcox
        original_hat = invert_boxcox_value(boxcox_hat, state)

        rows.append(
            {
                "date": pd.to_datetime(date),
                "pred_target_diff2": diff2_hat,
                "pred_target_boxcox": boxcox_hat,
                "pred_target_diff1": diff1_hat,
                "forecast_mean_interest_rate": original_hat,
            }
        )

        prev_boxcox = boxcox_hat
        prev_diff1 = diff1_hat

    return pd.DataFrame(rows)


def attach_actuals_and_metrics(
    pred_df: pd.DataFrame,
    future_df: pd.DataFrame,
    date_col_future: str,
    target_col: str,
) -> tuple[pd.DataFrame, float | None, float | None]:
    pred_df = pred_df.copy()
    rmse: float | None = None
    mae: float | None = None

    has_actuals = target_col in future_df.columns
    merge_columns = [date_col_future] + ([target_col] if has_actuals else [])
    merged = pred_df.merge(
        future_df[merge_columns],
        left_on="date",
        right_on=date_col_future,
        how="left",
    ).drop(columns=[date_col_future])

    if has_actuals:
        merged["actual"] = merged[target_col].astype(float)
        merged["abs_error"] = (merged["actual"] - merged["forecast_mean_interest_rate"]).abs()
        merged["squared_error"] = (merged["actual"] - merged["forecast_mean_interest_rate"]) ** 2

        valid = merged[["actual", "forecast_mean_interest_rate"]].dropna()
        if not valid.empty:
            rmse = float(np.sqrt(mean_squared_error(valid["actual"], valid["forecast_mean_interest_rate"])))
            mae = float(mean_absolute_error(valid["actual"], valid["forecast_mean_interest_rate"]))
        else:
            LOGGER.warning(
                "The future file contains the target column, but no rows had non-null actual values after merging."
            )
    else:
        merged["actual"] = np.nan
        merged["abs_error"] = np.nan
        merged["squared_error"] = np.nan

    return merged, rmse, mae


def save_plot(pred_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(pd.to_datetime(pred_df["date"]), pred_df["forecast_mean_interest_rate"], label="Forecast")

    actual_non_null = pred_df["actual"].notna().any() if "actual" in pred_df.columns else False
    if actual_non_null:
        plt.plot(pd.to_datetime(pred_df["date"]), pred_df["actual"], label="Actual")

    plt.xlabel("Date")
    plt.ylabel("Mean interest rate")
    plt.title("SARIMAX Future Forecast")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


class SarimaxFutureForecastPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def run(self) -> dict[str, Any]:
        cfg = self.config
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        train_df = load_dataset(cfg.train_file, cfg.date_col_train)
        future_df = load_dataset(cfg.future_file, cfg.date_col_future)

        validate_required_columns(
            train_df,
            [cfg.date_col_train, cfg.target_col, cfg.tbill_col_train],
            dataset_name="train file",
        )
        validate_required_columns(
            future_df,
            [cfg.date_col_future, cfg.tbill_col_future],
            dataset_name="future file",
        )

        latest_train_date = train_df[cfg.date_col_train].max()
        if cfg.filter_future_after_train:
            original_count = len(future_df)
            future_df = future_df[future_df[cfg.date_col_future] > latest_train_date].copy()
            LOGGER.info(
                "Filtered future rows to dates after latest training date (%s): %d -> %d rows",
                latest_train_date.date(),
                original_count,
                len(future_df),
            )

        if future_df.empty:
            raise ValueError("No future rows remain after filtering. Check your future file and date columns.")

        LOGGER.info("Train shape: %s", train_df.shape)
        LOGGER.info("Future shape: %s", future_df.shape)

        train_ts, state = apply_train_transformations(
            train_df=train_df,
            date_col=cfg.date_col_train,
            target_col=cfg.target_col,
            tbill_col=cfg.tbill_col_train,
        )
        train_frame = build_train_model_frame(train_ts=train_ts, tbill_lag=cfg.tbill_lag)

        fitted = fit_sarimax(train_frame=train_frame, order=cfg.sarimax_order)
        future_exog = build_future_exog(
            train_ts=train_ts,
            future_df=future_df,
            date_col_future=cfg.date_col_future,
            tbill_col_future=cfg.tbill_col_future,
            tbill_col_train=cfg.tbill_col_train,
            tbill_lag=cfg.tbill_lag,
            state=state,
        )
        pred_diff2 = forecast_transformed(fitted=fitted, future_exog=future_exog)

        pred_df = reconstruct_recursive_predictions(
            pred_diff2=pred_diff2,
            last_train_boxcox=float(train_ts["target_boxcox"].iloc[-1]),
            last_train_diff1=float(train_ts["target_diff1"].iloc[-1]),
            state=state,
        )

        pred_df, rmse, mae = attach_actuals_and_metrics(
            pred_df=pred_df,
            future_df=future_df,
            date_col_future=cfg.date_col_future,
            target_col=cfg.target_col,
        )

        summary = ForecastSummary(
            n_train_rows=int(len(train_df)),
            n_model_rows=int(len(train_frame)),
            n_future_rows=int(len(future_df)),
            n_forecasts=int(len(pred_df)),
            sarimax_order=cfg.sarimax_order,
            tbill_col_train=cfg.tbill_col_train,
            tbill_col_future=cfg.tbill_col_future,
            tbill_lag=cfg.tbill_lag,
            rmse=rmse,
            mae=mae,
        )

        artifacts = self._save_outputs(
            pred_df=pred_df,
            summary=summary,
            state=state,
            fitted=fitted,
        )

        if rmse is not None and mae is not None:
            LOGGER.info("RMSE: %.6f", rmse)
            LOGGER.info("MAE:  %.6f", mae)
        else:
            LOGGER.info("No future actual target values were available, so RMSE and MAE were not computed.")

        if cfg.show_plot:
            self._show_plot(pred_df)

        return {
            "train_df": train_df,
            "future_df": future_df,
            "train_ts": train_ts,
            "train_frame": train_frame,
            "future_exog": future_exog,
            "pred_df": pred_df,
            "transform_state": state,
            "summary": summary,
            "artifacts": artifacts,
            "fitted_model": fitted,
        }

    def _save_outputs(
        self,
        pred_df: pd.DataFrame,
        summary: ForecastSummary,
        state: TransformState,
        fitted: SARIMAXResultsWrapper,
    ) -> dict[str, Path]:
        cfg = self.config

        pred_path = cfg.output_dir / "sarimax_future_forecasts.csv"
        summary_path = cfg.output_dir / "sarimax_future_forecast_summary.json"
        transform_path = cfg.output_dir / "transform_state.json"
        params_path = cfg.output_dir / "fitted_sarimax_params.json"
        model_summary_path = cfg.output_dir / "sarimax_model_summary.txt"

        pred_to_save = pred_df.copy()
        pred_to_save["date"] = pd.to_datetime(pred_to_save["date"]).dt.strftime("%Y-%m-%d")
        pred_to_save.to_csv(pred_path, index=False)

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(asdict(summary), f, indent=2)

        with open(transform_path, "w", encoding="utf-8") as f:
            json.dump(asdict(state), f, indent=2)

        fitted_params = {str(key): float(value) for key, value in fitted.params.items()}
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(fitted_params, f, indent=2)

        with open(model_summary_path, "w", encoding="utf-8") as f:
            f.write(str(fitted.summary()))

        artifacts: dict[str, Path] = {
            "forecast_csv": pred_path,
            "summary_json": summary_path,
            "transform_state_json": transform_path,
            "params_json": params_path,
            "model_summary_txt": model_summary_path,
        }

        if cfg.save_plot:
            plot_path = cfg.output_dir / "sarimax_future_forecast.png"
            save_plot(pred_df=pred_df, output_path=plot_path)
            artifacts["forecast_plot"] = plot_path

        return artifacts

    @staticmethod
    def _show_plot(pred_df: pd.DataFrame) -> None:
        plt.figure(figsize=(10, 5))
        plt.plot(pd.to_datetime(pred_df["date"]), pred_df["forecast_mean_interest_rate"], label="Forecast")
        actual_non_null = pred_df["actual"].notna().any() if "actual" in pred_df.columns else False
        if actual_non_null:
            plt.plot(pd.to_datetime(pred_df["date"]), pred_df["actual"], label="Actual")
        plt.xlabel("Date")
        plt.ylabel("Mean interest rate")
        plt.title("SARIMAX Future Forecast")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        train_file=args.train_file,
        future_file=args.future_file,
        output_dir=args.output_dir,
        date_col_train=args.date_col_train,
        date_col_future=args.date_col_future,
        target_col=args.target_col,
        tbill_col_train=args.tbill_col_train,
        tbill_col_future=args.tbill_col_future,
        sarimax_order=parse_order(args.sarimax_order),
        tbill_lag=args.tbill_lag,
        filter_future_after_train=not args.keep_overlapping_future,
        save_plot=args.save_plot,
        show_plot=args.show_plot,
    )


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    config = build_config(args)
    pipeline = SarimaxFutureForecastPipeline(config)
    results = pipeline.run()

    print("\nRun completed successfully.")
    print(f"Forecast rows: {len(results['pred_df'])}")
    print(f"Output directory: {config.output_dir}")

    summary: ForecastSummary = results["summary"]
    if summary.rmse is not None:
        print(f"RMSE: {summary.rmse:.6f}")
    if summary.mae is not None:
        print(f"MAE:  {summary.mae:.6f}")

    print("\nSaved artifacts:")
    for name, path in results["artifacts"].items():
        print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()
