
#!/usr/bin/env python3
"""

This script does the following. It:
1. loads the monthly training data,
2. performs time-series diagnostics,
3. builds a transformed modeling frame,
4. evaluates baseline, classical ML, and deep learning models under a
   consistent rolling-origin validation design.


Example
-------
python model_implementations_pipeline.py --data-file ../Data/train.csv --output-dir ../outputs/model_implementations
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import random
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import sys
import subprocess

import sys
import subprocess
import importlib

import sys
import subprocess
import importlib

# Map import names to pip package names where they differ.
REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "sklearn": "scikit-learn",
    "statsmodels": "statsmodels",
    "scipy": "scipy",
    "optuna": "optuna",
    "torch": "torch",
}

for module_name, package_name in REQUIRED_PACKAGES.items():
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError:
        print(f"Installing missing package: {package_name}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

import optuna
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.special import inv_boxcox
from scipy.stats import boxcox
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, ccf

sns.set_theme(style="darkgrid")
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass(frozen=True)
class ExperimentConfig:
    """Central configuration for the forecasting experiment."""

    data_file: Path
    output_dir: Path

    date_col: str = "Month"
    target_col: str = "int_rate_mean"
    treasury_col: str = "Treasury_data"
    optional_driver_col: str = "fed_rate"

    seasonal_period: int = 12
    max_diagnostic_lags: int = 48
    target_diff2_col: str = "target_diff2"

    target_lags: tuple[int, ...] = (1, 3, 4)
    treasury_lag: int = 1

    cv_splits: int = 20
    cv_test_size: int = 1

    ridge_alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    rf_n_estimators: tuple[int, ...] = (50, 100)
    rf_max_depths: tuple[Optional[int], ...] = (3, 5, None)
    sarimax_orders: tuple[tuple[int, int, int], ...] = field(
        default_factory=lambda: tuple(itertools.product((2, 3, 4), (0,), (0, 1)))
    )

    mlp_trials: int = 50
    rnn_trials: int = 30
    cnn_trials: int = 30

    mlp_max_epochs: int = 100
    sequence_max_epochs: int = 150
    early_stopping_patience: int = 15
    validation_fraction: float = 0.85

    seed: int = 112
    save_plots: bool = True
    create_diagnostics: bool = True


@dataclass(frozen=True)
class TransformState:
    """Parameters required to invert the target transformation."""

    target_boxcox_lambda: float
    target_boxcox_shift: float
    treasury_log_shift: float


@dataclass
class PreparedData:
    """All data artifacts needed for modeling."""

    raw: pd.DataFrame
    train_ts: pd.DataFrame
    model_frame: pd.DataFrame
    feature_cols: list[str]
    sequence_feature_cols: list[str]
    splitter: TimeSeriesSplit
    transform_state: TransformState


def set_global_seed(seed: int) -> None:
    """Fix random seeds across libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_output_dir(path: Path) -> Path:
    """Create the output directory tree if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def as_jsonable(value: Any) -> Any:
    """Convert nested experiment outputs to JSON-friendly objects."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [as_jsonable(item) for item in value]
    if isinstance(value, list):
        return [as_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): as_jsonable(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Create a detached copy of a PyTorch model state."""
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def create_sequences(
    X_data: np.ndarray,
    y_data: np.ndarray,
    seq_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert arrays into sliding windows for sequence models."""
    X_seq, y_seq = [], []
    for start_idx in range(len(X_data) - seq_length):
        end_idx = start_idx + seq_length
        X_seq.append(X_data[start_idx:end_idx])
        y_seq.append(y_data[end_idx])
    return np.array(X_seq), np.array(y_seq)


def compute_positive_shift(series: pd.Series, buffer: float = 0.0) -> float:
    """Return the smallest nonnegative shift that makes a series positive."""
    minimum = float(series.min())
    return max(0.0, -minimum + buffer)


class DynamicMLP(nn.Module):
    """Compact multilayer perceptron for lag-based tabular features."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout_prob: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TinyRNN(nn.Module):
    """Single-layer GRU or LSTM with a linear output head."""

    def __init__(self, input_dim: int, hidden_dim: int, rnn_type: str) -> None:
        super().__init__()
        if rnn_type == "GRU":
            self.recurrent = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)
            self.is_lstm = False
        else:
            self.recurrent = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
            self.is_lstm = True
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.recurrent(x)
        if self.is_lstm:
            hidden = hidden[0]
        last_hidden = hidden[-1]
        return self.output(last_hidden)


class Tiny1DCNN(nn.Module):
    """Small 1D convolutional network for short temporal windows."""

    def __init__(self, input_dim: int, num_filters: int, kernel_size: int, seq_length: int) -> None:
        super().__init__()
        conv_output_length = seq_length - kernel_size + 1
        self.conv = nn.Conv1d(input_dim, num_filters, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(num_filters * conv_output_length, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.relu(self.conv(x))
        x = x.reshape(x.size(0), -1)
        return self.output(x)


class ForecastingExperiment:
    """End-to-end forecasting experiment with shared preprocessing and evaluation."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.output_dir = ensure_output_dir(config.output_dir)
        self.plots_dir = ensure_output_dir(self.output_dir / "plots")
        self.tables_dir = ensure_output_dir(self.output_dir / "tables")
        self.prepared: Optional[PreparedData] = None

    @property
    def cfg(self) -> ExperimentConfig:
        return self.config

    def save_figure(self, name: str) -> None:
        """Persist the current Matplotlib figure if plotting is enabled."""
        if self.cfg.save_plots:
            plt.savefig(self.plots_dir / f"{name}.png", dpi=200, bbox_inches="tight")
        plt.close()

    def save_table(self, df: pd.DataFrame, name: str) -> None:
        """Write a dataframe to the tables directory."""
        df.to_csv(self.tables_dir / f"{name}.csv", index=False)

    def save_json(self, payload: dict[str, Any], name: str) -> None:
        """Write structured experiment metadata to disk."""
        with open(self.output_dir / f"{name}.json", "w", encoding="utf-8") as file:
            json.dump(as_jsonable(payload), file, indent=2)

    def load_training_data(self) -> pd.DataFrame:
        """Load the dataset, parse the date column, and sort chronologically."""
        df = pd.read_csv(self.cfg.data_file)
        df[self.cfg.date_col] = pd.to_datetime(df[self.cfg.date_col])
        df = df.sort_values(self.cfg.date_col).reset_index(drop=True)
        return df

    def summarize_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Summarize missing values and dtypes for the raw dataset."""
        return (
            df.isna()
            .sum()
            .rename("missing_values")
            .to_frame()
            .assign(dtype=df.dtypes.astype(str).values)
            .reset_index(names="column")
        )

    def plot_series_diagnostics(self, series: pd.Series, title: str, filename: str) -> None:
        """Plot ACF and PACF for a single time series."""
        clean_series = series.dropna()
        lags = min(self.cfg.max_diagnostic_lags, max(1, len(clean_series) // 2 - 1))

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        plot_acf(clean_series, lags=lags, ax=axes[0], alpha=0.05)
        plot_pacf(clean_series, lags=lags, ax=axes[1], method="ywm", alpha=0.05)

        axes[0].set_title(f"ACF: {title}")
        axes[1].set_title(f"PACF: {title}")
        for ax in axes:
            ax.set_xlabel("Lag")
            ax.set_ylabel("Correlation")

        plt.tight_layout()
        self.save_figure(filename)

    def plot_cross_correlation(
        self,
        target: pd.Series,
        driver: pd.Series,
        title: str,
        filename: str,
        max_lag: int = 10,
    ) -> None:
        """Plot the leading cross-correlations between two aligned series."""
        aligned = pd.concat([target, driver], axis=1).dropna()
        cross_corr = ccf(aligned.iloc[:, 0], aligned.iloc[:, 1])

        plt.figure(figsize=(8, 4))
        plt.bar(range(max_lag), cross_corr[:max_lag])
        plt.title(title)
        plt.xlabel("Lag")
        plt.ylabel("Cross-correlation")
        plt.tight_layout()
        self.save_figure(filename)

    def plot_correlation_heatmap(self, df: pd.DataFrame, filename: str) -> None:
        """Plot the numeric correlation matrix."""
        corr = df.corr(numeric_only=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            linewidths=0.5,
            square=True,
        )
        plt.title("Correlation Matrix")
        plt.tight_layout()
        self.save_figure(filename)

    def run_adf_test(self, series: pd.Series, name: str) -> dict[str, Any]:
        """Compute Augmented Dickey-Fuller statistics for a series."""
        clean_series = series.dropna()
        statistic, p_value, _, _, critical_values, _ = adfuller(clean_series, autolag="AIC")
        return {
            "series": name,
            "statistic": statistic,
            "p_value": p_value,
            "critical_1pct": critical_values["1%"],
            "critical_5pct": critical_values["5%"],
            "critical_10pct": critical_values["10%"],
            "conclusion": "stationary" if p_value <= 0.05 else "non-stationary",
        }

    def plot_stl_decomposition(self, series: pd.Series, title: str, filename: str) -> dict[str, float]:
        """Run STL decomposition, save the plot, and report seasonal range."""
        decomposition = STL(series.dropna(), period=self.cfg.seasonal_period, robust=True).fit()
        fig = decomposition.plot()
        fig.set_size_inches(10, 8)
        plt.suptitle(title, y=1.02)
        plt.tight_layout()
        self.save_figure(filename)

        return {
            "title": title,
            "seasonal_min": float(decomposition.seasonal.min()),
            "seasonal_max": float(decomposition.seasonal.max()),
        }

    def create_time_series_frame(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Construct the date-indexed frame used throughout modeling."""
        return (
            raw[[self.cfg.date_col, self.cfg.target_col, self.cfg.treasury_col]]
            .copy()
            .set_index(self.cfg.date_col)
            .sort_index()
        )

    def apply_transformations(self, train_ts: pd.DataFrame) -> tuple[pd.DataFrame, TransformState]:
        """Apply the notebook transformation pipeline in a reusable way."""
        train_ts = train_ts.copy()

        target_boxcox_shift = compute_positive_shift(train_ts[self.cfg.target_col])
        target_for_boxcox = train_ts[self.cfg.target_col] + target_boxcox_shift
        train_ts["target_boxcox"], target_boxcox_lambda = boxcox(target_for_boxcox)
        train_ts["target_diff1"] = train_ts["target_boxcox"].diff(1)
        train_ts[self.cfg.target_diff2_col] = train_ts["target_boxcox"].diff(1).diff(1)

        treasury_log_shift = compute_positive_shift(train_ts[self.cfg.treasury_col], buffer=0.1)
        train_ts["treasury_shifted"] = train_ts[self.cfg.treasury_col] + treasury_log_shift
        train_ts["treasury_log"] = np.log(train_ts["treasury_shifted"])
        train_ts["treasury_log_diff"] = train_ts["treasury_log"].diff(1)

        transform_state = TransformState(
            target_boxcox_lambda=target_boxcox_lambda,
            target_boxcox_shift=target_boxcox_shift,
            treasury_log_shift=treasury_log_shift,
        )
        return train_ts, transform_state

    def create_modeling_frame(self, train_ts: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
        """Build the supervised feature table shared by all forecasting models."""
        frame = pd.DataFrame(index=train_ts.index)
        frame[self.cfg.target_diff2_col] = train_ts[self.cfg.target_diff2_col]

        for lag in self.cfg.target_lags:
            frame[f"target_lag_{lag}"] = frame[self.cfg.target_diff2_col].shift(lag)

        frame["treasury_lag_1"] = train_ts["treasury_log_diff"].shift(self.cfg.treasury_lag)
        frame["prev_diff1"] = train_ts["target_diff1"].shift(1)
        frame["prev_boxcox"] = train_ts["target_boxcox"].shift(1)
        frame = frame.dropna()

        feature_cols = [f"target_lag_{lag}" for lag in self.cfg.target_lags] + ["treasury_lag_1"]
        sequence_feature_cols = [self.cfg.target_diff2_col, "treasury_lag_1"]
        return frame, feature_cols, sequence_feature_cols

    def prepare_data(self) -> PreparedData:
        """Load raw data, create diagnostics, and build modeling artifacts."""
        logging.info("Loading data from %s", self.cfg.data_file)
        raw = self.load_training_data()
        self.save_table(self.summarize_missing_values(raw), "missing_summary")

        if self.cfg.create_diagnostics:
            logging.info("Creating diagnostic plots and decomposition summaries")
            self.plot_series_diagnostics(raw[self.cfg.target_col], self.cfg.target_col, "acf_pacf_target_raw")
            if self.cfg.optional_driver_col in raw.columns:
                self.plot_cross_correlation(
                    raw[self.cfg.target_col],
                    raw[self.cfg.optional_driver_col],
                    f"Cross-correlation: {self.cfg.optional_driver_col} vs. {self.cfg.target_col}",
                    "cross_correlation_optional_driver",
                )
            self.plot_cross_correlation(
                raw[self.cfg.target_col],
                raw[self.cfg.treasury_col],
                f"Cross-correlation: {self.cfg.treasury_col} vs. {self.cfg.target_col}",
                "cross_correlation_treasury",
            )
            self.plot_correlation_heatmap(raw, "correlation_heatmap")

        train_ts = self.create_time_series_frame(raw)

        if self.cfg.create_diagnostics:
            stl_records = [
                self.plot_stl_decomposition(train_ts[self.cfg.target_col], f"STL decomposition: {self.cfg.target_col}", "stl_target"),
                self.plot_stl_decomposition(train_ts[self.cfg.treasury_col], f"STL decomposition: {self.cfg.treasury_col}", "stl_treasury"),
            ]
            self.save_table(pd.DataFrame(stl_records), "stl_summary")

        train_ts, transform_state = self.apply_transformations(train_ts)

        adf_records = [
            self.run_adf_test(train_ts[self.cfg.target_col], "Raw target series"),
            self.run_adf_test(train_ts["target_diff1"], "Box-Cox target, first difference"),
            self.run_adf_test(train_ts[self.cfg.target_diff2_col], "Box-Cox target, second difference"),
            self.run_adf_test(train_ts[self.cfg.treasury_col], "Raw treasury series"),
            self.run_adf_test(train_ts["treasury_log_diff"], "Log-transformed treasury series, first difference"),
        ]
        self.save_table(pd.DataFrame(adf_records), "adf_summary")

        if self.cfg.create_diagnostics:
            plt.figure(figsize=(12, 4))
            plt.plot(train_ts.index, train_ts[self.cfg.target_diff2_col], label="Target second difference")
            plt.axhline(0, color="black", linestyle="--", linewidth=1)
            plt.title("Stationary target series used for modeling")
            plt.legend()
            plt.tight_layout()
            self.save_figure("stationary_target")

            plt.figure(figsize=(12, 4))
            plt.plot(train_ts.index, train_ts["treasury_log_diff"], label="Treasury log difference")
            plt.axhline(0, color="black", linestyle="--", linewidth=1)
            plt.title("Stationary treasury feature used for modeling")
            plt.legend()
            plt.tight_layout()
            self.save_figure("stationary_treasury")

            self.plot_series_diagnostics(train_ts[self.cfg.target_diff2_col], "Transformed target", "acf_pacf_target_transformed")

        model_frame, feature_cols, sequence_feature_cols = self.create_modeling_frame(train_ts)
        splitter = TimeSeriesSplit(n_splits=self.cfg.cv_splits, test_size=self.cfg.cv_test_size)

        fold_schedule = []
        for fold, (train_index, test_index) in enumerate(splitter.split(model_frame), start=1):
            fold_schedule.append(
                {
                    "fold": fold,
                    "train_end": model_frame.index[train_index[-1]].date().isoformat(),
                    "test_date": model_frame.index[test_index[0]].date().isoformat(),
                    "train_size": len(train_index),
                }
            )
        self.save_table(pd.DataFrame(fold_schedule), "fold_schedule")

        prepared = PreparedData(
            raw=raw,
            train_ts=train_ts,
            model_frame=model_frame,
            feature_cols=feature_cols,
            sequence_feature_cols=sequence_feature_cols,
            splitter=splitter,
            transform_state=transform_state,
        )
        self.prepared = prepared
        return prepared

    def inverse_boxcox_to_original(self, boxcox_value: float) -> float:
        """Invert the Box-Cox transform and remove the stored positive shift."""
        assert self.prepared is not None
        state = self.prepared.transform_state
        return float(inv_boxcox(boxcox_value, state.target_boxcox_lambda) - state.target_boxcox_shift)

    def reconstruct_prediction(self, pred_diff2: float, anchor_row: pd.Series) -> float:
        """Map a prediction on the second-difference scale back to the original target."""
        pred_boxcox = pred_diff2 + anchor_row["prev_diff1"] + anchor_row["prev_boxcox"]
        return self.inverse_boxcox_to_original(pred_boxcox)

    def summarize_forecasts(self, results: pd.DataFrame) -> dict[str, float]:
        """Compute standard forecast metrics on the original target scale."""
        return {
            "RMSE": float(np.sqrt(mean_squared_error(results["actual"], results["prediction"]))),
            "MAE": float(mean_absolute_error(results["actual"], results["prediction"])),
            "n_forecasts": int(len(results)),
        }

    def run_one_step_cv(
        self,
        predict_diff2_fn: Callable[[pd.DataFrame, pd.DataFrame], float],
    ) -> pd.DataFrame:
        """Evaluate a model that predicts the transformed target one step ahead."""
        assert self.prepared is not None
        records = []
        frame = self.prepared.model_frame

        for fold, (train_index, test_index) in enumerate(self.prepared.splitter.split(frame), start=1):
            train_df = frame.iloc[train_index]
            test_df = frame.iloc[test_index]

            pred_diff2 = predict_diff2_fn(train_df, test_df)
            anchor_row = test_df.iloc[0]
            prediction = self.reconstruct_prediction(pred_diff2, anchor_row)
            actual = float(self.prepared.train_ts.loc[test_df.index[0], self.cfg.target_col])

            records.append(
                {
                    "fold": fold,
                    "date": test_df.index[0],
                    "actual": actual,
                    "prediction": prediction,
                    "abs_error": abs(actual - prediction),
                }
            )

        return pd.DataFrame(records)

    def evaluate_baselines(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Evaluate simple benchmark forecasts under the same rolling schedule."""
        assert self.prepared is not None
        frame = self.prepared.model_frame
        ts = self.prepared.train_ts
        records = []

        for fold, (train_index, test_index) in enumerate(self.prepared.splitter.split(frame), start=1):
            last_train_date = frame.index[train_index[-1]]
            test_date = frame.index[test_index[0]]

            actual = float(ts.loc[test_date, self.cfg.target_col])
            naive_pred = float(ts.loc[last_train_date, self.cfg.target_col])
            rolling_avg4_pred = float(ts.loc[:last_train_date, self.cfg.target_col].tail(4).mean())

            records.extend(
                [
                    {"model": "Naive", "fold": fold, "date": test_date, "actual": actual, "prediction": naive_pred},
                    {
                        "model": "RollingAvg4",
                        "fold": fold,
                        "date": test_date,
                        "actual": actual,
                        "prediction": rolling_avg4_pred,
                    },
                ]
            )

        predictions = pd.DataFrame(records)
        summary = (
            predictions.groupby("model")
            .apply(lambda df: pd.Series(self.summarize_forecasts(df)))
            .reset_index()
            .sort_values("RMSE")
            .reset_index(drop=True)
        )
        self.save_table(predictions, "baseline_predictions")
        self.save_table(summary, "baseline_summary")
        return predictions, summary

    def evaluate_ridge(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Tune ridge regression over the notebook search grid."""
        assert self.prepared is not None

        def ridge_predict_diff2(train_df: pd.DataFrame, test_df: pd.DataFrame, alpha: float) -> float:
            model = Ridge(alpha=alpha)
            model.fit(train_df[self.prepared.feature_cols], train_df[self.cfg.target_diff2_col])
            return float(model.predict(test_df[self.prepared.feature_cols])[0])

        search_results = []
        for alpha in self.cfg.ridge_alphas:
            cv_results = self.run_one_step_cv(
                lambda train_df, test_df, alpha=alpha: ridge_predict_diff2(train_df, test_df, alpha)
            )
            metrics = self.summarize_forecasts(cv_results)
            search_results.append({"alpha": alpha, **metrics})

        search_df = pd.DataFrame(search_results).sort_values("RMSE").reset_index(drop=True)
        best = search_df.iloc[0].to_dict()
        self.save_table(search_df, "ridge_search_results")
        return search_df, best

    def evaluate_sarimax(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Tune SARIMAX across the notebook order grid."""
        assert self.prepared is not None

        def sarimax_predict_diff2(train_df: pd.DataFrame, test_df: pd.DataFrame, order: tuple[int, int, int]) -> float:
            model = SARIMAX(
                endog=train_df[self.cfg.target_diff2_col],
                exog=train_df[["treasury_lag_1"]],
                order=order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False)
            return float(fitted.forecast(steps=1, exog=test_df[["treasury_lag_1"]]).iloc[0])

        search_results = []
        for order in self.cfg.sarimax_orders:
            try:
                cv_results = self.run_one_step_cv(
                    lambda train_df, test_df, order=order: sarimax_predict_diff2(train_df, test_df, order)
                )
                metrics = self.summarize_forecasts(cv_results)
                search_results.append({"order": str(order), **metrics})
            except Exception as exc:
                logging.warning("Skipping SARIMAX order %s due to fitting error: %s", order, exc)
                continue

        if not search_results:
            raise RuntimeError("All SARIMAX configurations failed during fitting.")

        search_df = pd.DataFrame(search_results).sort_values("RMSE").reset_index(drop=True)
        best = search_df.iloc[0].to_dict()
        self.save_table(search_df, "sarimax_search_results")
        return search_df, best

    def evaluate_random_forest(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Tune random forest over the notebook search grid."""
        assert self.prepared is not None

        def random_forest_predict_diff2(
            train_df: pd.DataFrame,
            test_df: pd.DataFrame,
            n_estimators: int,
            max_depth: Optional[int],
        ) -> float:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=self.cfg.seed,
            )
            model.fit(train_df[self.prepared.feature_cols], train_df[self.cfg.target_diff2_col])
            return float(model.predict(test_df[self.prepared.feature_cols])[0])

        search_results = []
        for n_estimators, max_depth in itertools.product(self.cfg.rf_n_estimators, self.cfg.rf_max_depths):
            cv_results = self.run_one_step_cv(
                lambda train_df, test_df, n_estimators=n_estimators, max_depth=max_depth:
                    random_forest_predict_diff2(train_df, test_df, n_estimators, max_depth)
            )
            metrics = self.summarize_forecasts(cv_results)
            search_results.append(
                {"n_estimators": n_estimators, "max_depth": max_depth, **metrics}
            )

        search_df = pd.DataFrame(search_results).sort_values("RMSE").reset_index(drop=True)
        best = search_df.iloc[0].to_dict()
        self.save_table(search_df, "random_forest_search_results")
        return search_df, best

    def train_torch_model(
        self,
        model: nn.Module,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        optimizer: optim.Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        max_epochs: int,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        patience: Optional[int] = None,
    ) -> nn.Module:
        """Train a PyTorch model with optional validation-based early stopping."""
        best_state = None
        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for _ in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            train_output = model(X_train)
            train_loss = loss_fn(train_output, y_train)
            train_loss.backward()
            optimizer.step()

            if X_val is None or y_val is None:
                continue

            model.eval()
            with torch.no_grad():
                val_output = model(X_val)
                val_loss = loss_fn(val_output, y_val).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_state = clone_state_dict(model)
            else:
                epochs_without_improvement += 1
                if patience is not None and epochs_without_improvement >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        return model


    def tune_mlp(self) -> dict[str, Any]:
        """Tune the MLP and report rolling validation MAE and RMSE."""
        assert self.prepared is not None

        def objective(trial: optuna.Trial) -> float:
            hidden_dim = trial.suggest_categorical("hidden_dim", [4, 8, 16, 24])
            dropout_prob = trial.suggest_float("dropout", 0.1, 0.6)
            learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

            fold_records = []

            for fold, (train_index, test_index) in enumerate(
                self.prepared.splitter.split(self.prepared.model_frame),
                start=1,
            ):
                train_df = self.prepared.model_frame.iloc[train_index]
                test_df = self.prepared.model_frame.iloc[test_index]

                scaler = StandardScaler()
                X_train = scaler.fit_transform(train_df[self.prepared.feature_cols])
                X_test = scaler.transform(test_df[self.prepared.feature_cols])

                X_train_t = torch.tensor(X_train, dtype=torch.float32)
                y_train_t = torch.tensor(
                    train_df[self.cfg.target_diff2_col].to_numpy().reshape(-1, 1),
                    dtype=torch.float32,
                )
                X_test_t = torch.tensor(X_test, dtype=torch.float32)

                model = DynamicMLP(
                    input_dim=X_train_t.shape[1],
                    hidden_dim=hidden_dim,
                    dropout_prob=dropout_prob,
                )
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                loss_fn = nn.MSELoss()

                model = self.train_torch_model(
                    model=model,
                    X_train=X_train_t,
                    y_train=y_train_t,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    max_epochs=self.cfg.mlp_max_epochs,
                )

                model.eval()
                with torch.no_grad():
                    pred_diff2 = float(model(X_test_t).item())

                prediction = self.reconstruct_prediction(pred_diff2, test_df.iloc[0])
                actual = float(self.prepared.train_ts.loc[test_df.index[0], self.cfg.target_col])

                fold_records.append(
                    {
                        "fold": fold,
                        "date": test_df.index[0],
                        "actual": actual,
                        "prediction": prediction,
                    }
                )

            metrics = self.summarize_forecasts(pd.DataFrame(fold_records))
            trial.set_user_attr("rolling_val_rmse", float(metrics["RMSE"]))
            trial.set_user_attr("rolling_val_mae", float(metrics["MAE"]))
            trial.set_user_attr("n_forecasts", int(metrics["n_forecasts"]))

            return float(metrics["RMSE"])

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.cfg.seed),
        )
        study.optimize(objective, n_trials=self.cfg.mlp_trials)

        result = {
            "model": "MLP",
            "best_params": study.best_params,
            "rolling_val_rmse": float(study.best_trial.user_attrs.get("rolling_val_rmse", float("nan"))),
            "rolling_val_mae": float(study.best_trial.user_attrs.get("rolling_val_mae", study.best_value)),
            "n_forecasts": int(study.best_trial.user_attrs.get("n_forecasts", 0)),
            "n_trials": self.cfg.mlp_trials,
        }
        return result

    def split_train_validation(self, train_full: pd.DataFrame, seq_length: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split a rolling training window into train and validation segments."""
        split_idx = int(len(train_full) * self.cfg.validation_fraction)
        train_df = train_full.iloc[:split_idx]
        val_df = train_full.iloc[split_idx:]

        if len(train_df) <= seq_length or len(val_df) <= seq_length:
            raise ValueError("Insufficient observations for sequence train/validation split.")
        return train_df, val_df

    def prepare_sequence_fold_data(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        seq_length: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
        """Scale sequence features/targets and create sliding windows."""
        assert self.prepared is not None

        X_scaler = StandardScaler()
        y_scaler = StandardScaler()

        X_train = X_scaler.fit_transform(train_df[self.prepared.sequence_feature_cols])
        y_train = y_scaler.fit_transform(train_df[[self.cfg.target_diff2_col]])
        X_val = X_scaler.transform(val_df[self.prepared.sequence_feature_cols])
        y_val = y_scaler.transform(val_df[[self.cfg.target_diff2_col]])

        X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
        X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_length)

        if len(X_train_seq) == 0 or len(X_val_seq) == 0:
            raise ValueError("Empty sequence arrays after window construction.")

        return X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_scaler, y_scaler

    def forecast_from_recent_sequence(
        self,
        model: nn.Module,
        train_full: pd.DataFrame,
        seq_length: int,
        X_scaler: StandardScaler,
        y_scaler: StandardScaler,
    ) -> float:
        """Forecast the next transformed target from the most recent sequence window."""
        assert self.prepared is not None
        last_window = train_full[self.prepared.sequence_feature_cols].iloc[-seq_length:]
        X_test = X_scaler.transform(last_window)
        X_test_t = torch.tensor(X_test.reshape(1, seq_length, -1), dtype=torch.float32)

        model.eval()
        with torch.no_grad():
            pred_scaled = model(X_test_t).numpy().reshape(-1, 1)

        return float(y_scaler.inverse_transform(pred_scaled)[0, 0])

    def tune_rnn(self) -> dict[str, Any]:
        """Tune the recurrent model and report rolling validation MAE and RMSE."""
        assert self.prepared is not None

        def objective(trial: optuna.Trial) -> float:
            rnn_type = trial.suggest_categorical("rnn_type", ["GRU", "LSTM"])
            seq_length = trial.suggest_int("seq_length", 3, 5)
            hidden_dim = trial.suggest_int("hidden_dim", 4, 12, step=4)
            learning_rate = trial.suggest_float("lr", 1e-3, 5e-2, log=True)

            fold_records = []

            for fold, (train_index, test_index) in enumerate(
                self.prepared.splitter.split(self.prepared.model_frame),
                start=1,
            ):
                train_full = self.prepared.model_frame.iloc[train_index]
                test_df = self.prepared.model_frame.iloc[test_index]

                if len(train_full) <= seq_length + 2:
                    return float("inf")

                try:
                    train_df, val_df = self.split_train_validation(train_full, seq_length)
                    X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_scaler, y_scaler = self.prepare_sequence_fold_data(
                        train_df, val_df, seq_length
                    )
                except ValueError:
                    return float("inf")

                X_train_t = torch.tensor(X_train_seq, dtype=torch.float32)
                y_train_t = torch.tensor(y_train_seq.reshape(-1, 1), dtype=torch.float32)
                X_val_t = torch.tensor(X_val_seq, dtype=torch.float32)
                y_val_t = torch.tensor(y_val_seq.reshape(-1, 1), dtype=torch.float32)

                model = TinyRNN(
                    input_dim=X_train_t.shape[2],
                    hidden_dim=hidden_dim,
                    rnn_type=rnn_type,
                )
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                loss_fn = nn.HuberLoss()

                model = self.train_torch_model(
                    model=model,
                    X_train=X_train_t,
                    y_train=y_train_t,
                    X_val=X_val_t,
                    y_val=y_val_t,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    max_epochs=self.cfg.sequence_max_epochs,
                    patience=self.cfg.early_stopping_patience,
                )

                pred_diff2 = self.forecast_from_recent_sequence(model, train_full, seq_length, X_scaler, y_scaler)
                prediction = self.reconstruct_prediction(pred_diff2, test_df.iloc[0])
                actual = float(self.prepared.train_ts.loc[test_df.index[0], self.cfg.target_col])

                fold_records.append(
                    {
                        "fold": fold,
                        "date": test_df.index[0],
                        "actual": actual,
                        "prediction": prediction,
                    }
                )

            metrics = self.summarize_forecasts(pd.DataFrame(fold_records))
            trial.set_user_attr("rolling_val_rmse", float(metrics["RMSE"]))
            trial.set_user_attr("rolling_val_mae", float(metrics["MAE"]))
            trial.set_user_attr("n_forecasts", int(metrics["n_forecasts"]))

            return float(metrics["RMSE"])

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.cfg.seed),
        )
        study.optimize(objective, n_trials=self.cfg.rnn_trials)

        result = {
            "model": "RNN",
            "best_params": study.best_params,
            "rolling_val_rmse": float(study.best_trial.user_attrs.get("rolling_val_rmse", float("nan"))),
            "rolling_val_mae": float(study.best_trial.user_attrs.get("rolling_val_mae", study.best_value)),
            "n_forecasts": int(study.best_trial.user_attrs.get("n_forecasts", 0)),
            "n_trials": self.cfg.rnn_trials,
        }
        return result
    def tune_cnn(self) -> dict[str, Any]:
        
        """Tune the 1D CNN and report rolling validation MAE and RMSE."""
        assert self.prepared is not None

        def objective(trial: optuna.Trial) -> float:
            seq_length = trial.suggest_int("seq_length", 3, 4)
            kernel_size = trial.suggest_int("kernel_size", 2, 3)
            num_filters = trial.suggest_categorical("num_filters", [4, 8, 12])
            learning_rate = trial.suggest_float("lr", 1e-3, 5e-2, log=True)

            if kernel_size > seq_length:
                return float("inf")

            fold_records = []

            for fold, (train_index, test_index) in enumerate(
                self.prepared.splitter.split(self.prepared.model_frame),
                start=1,
            ):
                train_full = self.prepared.model_frame.iloc[train_index]
                test_df = self.prepared.model_frame.iloc[test_index]

                if len(train_full) <= seq_length + 2:
                    return float("inf")

                try:
                    train_df, val_df = self.split_train_validation(train_full, seq_length)
                    X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_scaler, y_scaler = self.prepare_sequence_fold_data(
                        train_df, val_df, seq_length
                    )
                except ValueError:
                    return float("inf")

                X_train_t = torch.tensor(X_train_seq, dtype=torch.float32)
                y_train_t = torch.tensor(y_train_seq.reshape(-1, 1), dtype=torch.float32)
                X_val_t = torch.tensor(X_val_seq, dtype=torch.float32)
                y_val_t = torch.tensor(y_val_seq.reshape(-1, 1), dtype=torch.float32)

                model = Tiny1DCNN(
                    input_dim=X_train_t.shape[2],
                    num_filters=num_filters,
                    kernel_size=kernel_size,
                    seq_length=seq_length,
                )
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                loss_fn = nn.HuberLoss()

                model = self.train_torch_model(
                    model=model,
                    X_train=X_train_t,
                    y_train=y_train_t,
                    X_val=X_val_t,
                    y_val=y_val_t,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    max_epochs=self.cfg.sequence_max_epochs,
                    patience=self.cfg.early_stopping_patience,
                )

                pred_diff2 = self.forecast_from_recent_sequence(model, train_full, seq_length, X_scaler, y_scaler)
                prediction = self.reconstruct_prediction(pred_diff2, test_df.iloc[0])
                actual = float(self.prepared.train_ts.loc[test_df.index[0], self.cfg.target_col])

                fold_records.append(
                    {
                        "fold": fold,
                        "date": test_df.index[0],
                        "actual": actual,
                        "prediction": prediction,
                    }
                )

            metrics = self.summarize_forecasts(pd.DataFrame(fold_records))
            trial.set_user_attr("rolling_val_rmse", float(metrics["RMSE"]))
            trial.set_user_attr("rolling_val_mae", float(metrics["MAE"]))
            trial.set_user_attr("n_forecasts", int(metrics["n_forecasts"]))

            return float(metrics["RMSE"])

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.cfg.seed),
        )
        study.optimize(objective, n_trials=self.cfg.cnn_trials)

        result = {
            "model": "CNN",
            "best_params": study.best_params,
            "rolling_val_rmse": float(study.best_trial.user_attrs.get("rolling_val_rmse", float("nan"))),
            "rolling_val_mae": float(study.best_trial.user_attrs.get("rolling_val_mae", study.best_value)),
            "n_forecasts": int(study.best_trial.user_attrs.get("n_forecasts", 0)),
            "n_trials": self.cfg.cnn_trials,
        }
        return result

    def run(self) -> dict[str, Any]:
        """Execute the full experiment and save all intermediate outputs."""
        prepared = self.prepare_data()
        logging.info("Prepared modeling frame with shape %s", prepared.model_frame.shape)

        baseline_predictions, baseline_summary = self.evaluate_baselines()
        logging.info("Best baseline RMSE: %.4f", baseline_summary["RMSE"].min())

        ridge_search_df, best_ridge = self.evaluate_ridge()
        sarimax_search_df, best_sarimax = self.evaluate_sarimax()
        rf_search_df, best_rf = self.evaluate_random_forest()

        logging.info("Best Ridge config: %s", best_ridge)
        logging.info("Best SARIMAX config: %s", best_sarimax)
        logging.info("Best RandomForest config: %s", best_rf)

        comparable_models = pd.DataFrame(
            [
                {
                    "model": "Naive",
                    **baseline_summary[baseline_summary["model"] == "Naive"].iloc[0].drop("model").to_dict(),
                },
                {
                    "model": "RollingAvg4",
                    **baseline_summary[baseline_summary["model"] == "RollingAvg4"].iloc[0].drop("model").to_dict(),
                },
                {"model": "Ridge", **{key: best_ridge[key] for key in ("RMSE", "MAE", "n_forecasts")}},
                {"model": "SARIMAX", **{key: best_sarimax[key] for key in ("RMSE", "MAE", "n_forecasts")}},
                {"model": "RandomForest", **{key: best_rf[key] for key in ("RMSE", "MAE", "n_forecasts")}},
            ]
        ).sort_values("RMSE").reset_index(drop=True)
        self.save_table(comparable_models, "comparable_models")



        mlp_result = self.tune_mlp()
        rnn_result = self.tune_rnn()
        cnn_result = self.tune_cnn()

        deep_learning_summary = (
            pd.DataFrame([mlp_result, rnn_result, cnn_result])
            .sort_values("rolling_val_mae")
            .reset_index(drop=True)
        )
        self.save_table(deep_learning_summary, "deep_learning_summary")

        experiment_summary = {
            "config": {
                "data_file": self.cfg.data_file,
                "output_dir": self.cfg.output_dir,
                "seed": self.cfg.seed,
                "cv_splits": self.cfg.cv_splits,
                "cv_test_size": self.cfg.cv_test_size,
            },
            "data_overview": {
                "n_rows": int(prepared.raw.shape[0]),
                "n_columns": int(prepared.raw.shape[1]),
                "date_min": str(prepared.raw[self.cfg.date_col].min().date()),
                "date_max": str(prepared.raw[self.cfg.date_col].max().date()),
                "model_frame_rows": int(prepared.model_frame.shape[0]),
            },
            "transform_state": prepared.transform_state.__dict__,
            "best_models": {
                "ridge": best_ridge,
                "sarimax": best_sarimax,
                "random_forest": best_rf,
                "mlp": mlp_result,
                "rnn": rnn_result,
                "cnn": cnn_result,
            },
}

        self.save_json(experiment_summary, "experiment_summary")

        results = {
            "baseline_predictions": baseline_predictions,
            "baseline_summary": baseline_summary,
            "ridge_search_results": ridge_search_df,
            "sarimax_search_results": sarimax_search_df,
            "random_forest_search_results": rf_search_df,
            "comparable_models": comparable_models,
            "deep_learning_summary": deep_learning_summary,
            "experiment_summary": experiment_summary,
        }
        return results


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface for the standalone script."""
    parser = argparse.ArgumentParser(
        description="Run the project forecasting pipeline as a standalone Python script."
    )
    parser.add_argument("--data-file", type=Path, required=True, help="Path to the input CSV file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/model_implementations"),
        help="Directory where tables, plots, and summaries will be saved.",
    )
    parser.add_argument(
        "--mlp-trials",
        type=int,
        default=50,
        help="Number of Optuna trials for the MLP search.",
    )
    parser.add_argument(
        "--rnn-trials",
        type=int,
        default=30,
        help="Number of Optuna trials for the recurrent model search.",
    )
    parser.add_argument(
        "--cnn-trials",
        type=int,
        default=30,
        help="Number of Optuna trials for the CNN search.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Disable saving plots and other graphical diagnostics.",
    )
    parser.add_argument(
        "--skip-diagnostics",
        action="store_true",
        help="Skip all diagnostics, including plots and decomposition summaries.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging verbosity.",
    )
    return parser


def main() -> None:
    """Parse arguments and run the experiment."""
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    set_global_seed(42)

    config = ExperimentConfig(
        data_file=args.data_file,
        output_dir=args.output_dir,
        mlp_trials=args.mlp_trials,
        rnn_trials=args.rnn_trials,
        cnn_trials=args.cnn_trials,
        save_plots=not args.skip_plots,
        create_diagnostics=not args.skip_diagnostics,
    )

    experiment = ForecastingExperiment(config)
    results = experiment.run()

    comparable = results["comparable_models"]
    deep_summary = results["deep_learning_summary"]

    print("\nComparable models (RMSE/MAE on original target scale):")
    print(comparable.to_string(index=False))

    print("\nDeep learning tuning summary:")
    print(deep_summary.to_string(index=False))

    print(f"\nSaved outputs to: {config.output_dir.resolve()}")


if __name__ == "__main__":
    main()
