#!/usr/bin/env python3

#python sarimax_test_evaluator.py --train-file train.csv --test-file test.csv --output-dir outputs/sarimax_test_eval

from __future__ import annotations

import argparse
import importlib
import json
import logging
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


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

import numpy as np
import pandas as pd
from scipy.special import inv_boxcox
from scipy.stats import boxcox
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX


@dataclass(frozen=True)
class Config:
    train_file: Path
    test_file: Path
    output_dir: Path
    date_col: str = "Month"
    target_col: str = "int_rate_mean"
    treasury_col: str = "Treasury_data"
    sarimax_order: tuple[int, int, int] = (2, 0, 1)
    treasury_lag: int = 1
    log_level: str = "INFO"


@dataclass(frozen=True)
class TransformState:
    target_boxcox_lambda: float
    target_boxcox_shift: float
    treasury_log_shift: float


@dataclass(frozen=True)
class EvaluationResult:
    n_test_rows: int
    n_forecasts: int
    sarimax_order: tuple[int, int, int]
    rmse: float | None
    mae: float | None


def compute_positive_shift(series: pd.Series, buffer: float = 0.0) -> float:
    minimum = float(series.min())
    return max(0.0, -minimum + buffer)


def load_dataset(path: Path, date_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in {path}.")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    return df


def validate_columns(df: pd.DataFrame, required_cols: list[str], df_name: str) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def apply_train_transformations(train_df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, TransformState]:
    train_ts = train_df[[cfg.date_col, cfg.target_col, cfg.treasury_col]].copy()
    train_ts = train_ts.set_index(cfg.date_col).sort_index()

    target_boxcox_shift = compute_positive_shift(train_ts[cfg.target_col])
    target_input = train_ts[cfg.target_col] + target_boxcox_shift
    train_ts["target_boxcox"], target_boxcox_lambda = boxcox(target_input)
    train_ts["target_diff1"] = train_ts["target_boxcox"].diff(1)
    train_ts["target_diff2"] = train_ts["target_boxcox"].diff(1).diff(1)

    treasury_log_shift = compute_positive_shift(train_ts[cfg.treasury_col], buffer=0.1)
    train_ts["treasury_shifted"] = train_ts[cfg.treasury_col] + treasury_log_shift
    train_ts["treasury_log"] = np.log(train_ts["treasury_shifted"])
    train_ts["treasury_log_diff"] = train_ts["treasury_log"].diff(1)

    state = TransformState(
        target_boxcox_lambda=float(target_boxcox_lambda),
        target_boxcox_shift=float(target_boxcox_shift),
        treasury_log_shift=float(treasury_log_shift),
    )
    return train_ts, state


def build_train_model_frame(train_ts: pd.DataFrame, treasury_lag: int) -> pd.DataFrame:
    frame = pd.DataFrame(index=train_ts.index)
    frame["target_diff2"] = train_ts["target_diff2"]
    frame["treasury_lag_1"] = train_ts["treasury_log_diff"].shift(treasury_lag)
    frame["prev_diff1"] = train_ts["target_diff1"].shift(1)
    frame["prev_boxcox"] = train_ts["target_boxcox"].shift(1)
    return frame.dropna()


def build_test_exog(train_ts: pd.DataFrame, test_df: pd.DataFrame, cfg: Config, state: TransformState) -> pd.DataFrame:
    test_ts = test_df[[cfg.date_col, cfg.treasury_col]].copy().set_index(cfg.date_col).sort_index()

    combined_treasury = pd.concat(
        [
            train_ts[[cfg.treasury_col]],
            test_ts[[cfg.treasury_col]],
        ],
        axis=0,
    )

    combined_treasury["treasury_shifted"] = combined_treasury[cfg.treasury_col] + state.treasury_log_shift
    if (combined_treasury["treasury_shifted"] <= 0).any():
        raise ValueError(
            "Treasury transformation produced non-positive values. "
            "The training-derived log shift is not sufficient for the test data."
        )

    combined_treasury["treasury_log"] = np.log(combined_treasury["treasury_shifted"])
    combined_treasury["treasury_log_diff"] = combined_treasury["treasury_log"].diff(1)
    combined_treasury["treasury_lag_1"] = combined_treasury["treasury_log_diff"].shift(cfg.treasury_lag)

    test_exog = combined_treasury.loc[test_ts.index, ["treasury_lag_1"]].copy()
    if test_exog["treasury_lag_1"].isna().any():
        missing_dates = test_exog.index[test_exog["treasury_lag_1"].isna()].strftime("%Y-%m-%d").tolist()
        raise ValueError(f"Missing exogenous values for test dates: {missing_dates}")
    return test_exog


def invert_boxcox(value: float, state: TransformState) -> float:
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
        original_hat = invert_boxcox(boxcox_hat, state)
        diff1_hat = boxcox_hat - prev_boxcox

        rows.append(
            {
                "date": date,
                "pred_target_diff2": diff2_hat,
                "pred_target_boxcox": boxcox_hat,
                "pred_target_diff1": diff1_hat,
                "prediction": original_hat,
            }
        )

        prev_boxcox = boxcox_hat
        prev_diff1 = diff1_hat

    return pd.DataFrame(rows)


def fit_and_forecast(train_frame: pd.DataFrame, test_exog: pd.DataFrame, cfg: Config) -> pd.Series:
    model = SARIMAX(
        endog=train_frame["target_diff2"],
        exog=train_frame[["treasury_lag_1"]],
        order=cfg.sarimax_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    forecasts = fitted.forecast(steps=len(test_exog), exog=test_exog[["treasury_lag_1"]])
    forecasts.index = test_exog.index
    return forecasts


def evaluate_predictions(pred_df: pd.DataFrame, test_df: pd.DataFrame, cfg: Config) -> EvaluationResult:
    has_actuals = cfg.target_col in test_df.columns

    merged = pred_df.merge(
        test_df[[cfg.date_col] + ([cfg.target_col] if has_actuals else [])],
        left_on="date",
        right_on=cfg.date_col,
        how="left",
    ).drop(columns=[cfg.date_col])

    rmse = None
    mae = None
    if has_actuals:
        merged["actual"] = merged[cfg.target_col].astype(float)
        merged["abs_error"] = (merged["actual"] - merged["prediction"]).abs()
        merged["squared_error"] = (merged["actual"] - merged["prediction"]) ** 2
        rmse = float(np.sqrt(mean_squared_error(merged["actual"], merged["prediction"])))
        mae = float(mean_absolute_error(merged["actual"], merged["prediction"]))
    else:
        merged["actual"] = np.nan
        merged["abs_error"] = np.nan
        merged["squared_error"] = np.nan

    pred_df.drop(pred_df.index, inplace=True)
    for col in merged.columns:
        pred_df[col] = merged[col]

    return EvaluationResult(
        n_test_rows=int(len(test_df)),
        n_forecasts=int(len(merged)),
        sarimax_order=cfg.sarimax_order,
        rmse=rmse,
        mae=mae,
    )


def save_outputs(
    pred_df: pd.DataFrame,
    result: EvaluationResult,
    cfg: Config,
    state: TransformState,
) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    pred_out = pred_df.copy()
    pred_out["date"] = pd.to_datetime(pred_out["date"]).dt.strftime("%Y-%m-%d")
    pred_out.to_csv(cfg.output_dir / "sarimax_test_predictions.csv", index=False)

    with open(cfg.output_dir / "sarimax_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2)

    with open(cfg.output_dir / "transform_state.json", "w", encoding="utf-8") as f:
        json.dump(asdict(state), f, indent=2)


def run(cfg: Config) -> EvaluationResult:
    logging.info("Loading datasets")
    train_df = load_dataset(cfg.train_file, cfg.date_col)
    test_df = load_dataset(cfg.test_file, cfg.date_col)

    validate_columns(train_df, [cfg.date_col, cfg.target_col, cfg.treasury_col], "Train data")
    validate_columns(test_df, [cfg.date_col, cfg.treasury_col], "Test data")

    logging.info("Applying training-set transformations")
    train_ts, state = apply_train_transformations(train_df, cfg)
    train_frame = build_train_model_frame(train_ts, cfg.treasury_lag)
    if train_frame.empty:
        raise ValueError("Training modeling frame is empty after transformations. Check data length.")

    logging.info("Preparing test-set exogenous regressor")
    test_exog = build_test_exog(train_ts, test_df, cfg, state)

    logging.info("Fitting SARIMAX%s on transformed training data", cfg.sarimax_order)
    pred_diff2 = fit_and_forecast(train_frame, test_exog, cfg)

    logging.info("Reconstructing predictions back to the original target scale")
    last_train_boxcox = float(train_ts["target_boxcox"].iloc[-1])
    last_train_diff1 = float(train_ts["target_diff1"].iloc[-1])
    pred_df = reconstruct_recursive_predictions(pred_diff2, last_train_boxcox, last_train_diff1, state)

    result = evaluate_predictions(pred_df, test_df, cfg)
    save_outputs(pred_df, result, cfg, state)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the fixed SARIMAX model chosen on the training set and evaluate it on a test set."
        )
    )
    parser.add_argument("--train-file", type=Path, required=True, help="CSV file for the training set.")
    parser.add_argument("--test-file", type=Path, required=True, help="CSV file for the test set.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/sarimax_test_eval"),
        help="Directory where predictions and metrics will be saved.",
    )
    parser.add_argument("--date-col", type=str, default="Month", help="Date column name.")
    parser.add_argument("--target-col", type=str, default="int_rate_mean", help="Target column name.")
    parser.add_argument("--treasury-col", type=str, default="Treasury_data", help="Treasury/exogenous column name.")
    parser.add_argument(
        "--sarimax-order",
        type=int,
        nargs=3,
        default=(2, 0, 1),
        metavar=("P", "D", "Q"),
        help="Fixed SARIMAX order. Default is the selected train-set winner: 2 0 1.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging verbosity.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    cfg = Config(
        train_file=args.train_file,
        test_file=args.test_file,
        output_dir=args.output_dir,
        date_col=args.date_col,
        target_col=args.target_col,
        treasury_col=args.treasury_col,
        sarimax_order=tuple(args.sarimax_order),
        log_level=args.log_level,
    )

    result = run(cfg)

    print("\nFixed SARIMAX evaluation complete")
    print(f"Order: {result.sarimax_order}")
    print(f"Number of forecasts: {result.n_forecasts}")

    if result.rmse is not None and result.mae is not None:
        print(f"RMSE: {result.rmse:.6f}")
        print(f"MAE:  {result.mae:.6f}")
    else:
        print("RMSE: unavailable (test file has no target column)")
        print("MAE:  unavailable (test file has no target column)")

    print(f"\nSaved outputs to: {cfg.output_dir.resolve()}")


if __name__ == "__main__":
    main()
