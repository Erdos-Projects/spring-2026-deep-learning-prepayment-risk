


# python test_set_one_step_forecasts.py train-file path/to/train.csv test-file path/to/test.csv output-dir ./test_set_forecast_outputs save-plots

from __future__ import annotations

import argparse
import json
import logging
import random
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")


def _import_dependencies() -> dict[str, Any]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from scipy.special import boxcox as boxcox_with_lambda, inv_boxcox
        from scipy.stats import boxcox
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        from sklearn.preprocessing import StandardScaler
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency detected. Install the required packages with:\n"
            "pip install numpy pandas matplotlib scipy scikit-learn statsmodels torch"
        ) from exc

    return {
        "plt": plt,
        "np": np,
        "pd": pd,
        "torch": torch,
        "nn": nn,
        "optim": optim,
        "boxcox": boxcox,
        "boxcox_with_lambda": boxcox_with_lambda,
        "inv_boxcox": inv_boxcox,
        "mean_absolute_error": mean_absolute_error,
        "mean_squared_error": mean_squared_error,
        "StandardScaler": StandardScaler,
        "SARIMAX": SARIMAX,
    }


D = _import_dependencies()
plt = D["plt"]
np = D["np"]
pd = D["pd"]
torch = D["torch"]
nn = D["nn"]
optim = D["optim"]
boxcox = D["boxcox"]
boxcox_with_lambda = D["boxcox_with_lambda"]
inv_boxcox = D["inv_boxcox"]
mean_absolute_error = D["mean_absolute_error"]
mean_squared_error = D["mean_squared_error"]
StandardScaler = D["StandardScaler"]
SARIMAX = D["SARIMAX"]

SEED = 112
SARIMAX_ORDER = (2, 0, 1)
MLP_PARAMS = {
    "hidden_dim": 24,
    "dropout": 0.378016,
    "lr": 0.007305,
    "max_epochs": 150,
}
RNN_PARAMS = {
    "rnn_type": "LSTM",
    "seq_length": 5,
    "hidden_dim": 4,
    "lr": 0.021783,
    "max_epochs": 150,
    "patience": 15,
}
CNN_PARAMS = {
    "seq_length": 4,
    "kernel_size": 2,
    "num_filters": 12,
    "lr": 0.014647,
    "max_epochs": 150,
    "patience": 15,
}


def set_reproducibility(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DynamicMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout_prob: float):
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
    def __init__(self, input_dim: int, hidden_dim: int, rnn_type: str):
        super().__init__()
        if rnn_type.upper() == "GRU":
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
    def __init__(self, input_dim: int, num_filters: int, kernel_size: int, seq_length: int):
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the same one-step-ahead test-set forecasting workflow as the notebook, "
            "including train-fitted transformations and the SARIMAX / MLP / RNN / CNN models."
        )
    )
    parser.add_argument("--train-file", required=True, type=Path, help="Path to the train CSV file.")
    parser.add_argument("--test-file", required=True, type=Path, help="Path to the test CSV file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./test_set_forecast_outputs"),
        help="Directory where prediction files, summaries, and plots will be saved.",
    )
    parser.add_argument("--date-col", default="Month", help="Name of the date column.")
    parser.add_argument("--target-col", default="int_rate_mean", help="Name of the target column.")
    parser.add_argument("--treasury-col", default="Treasury_data", help="Name of the treasury / exogenous column.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["sarimax", "mlp", "rnn", "cnn"],
        default=["sarimax", "mlp", "rnn", "cnn"],
        help="Subset of models to run.",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save per-model plots and a combined overlay plot as PNG files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed used for NumPy / Python / PyTorch reproducibility.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging verbosity.",
    )
    return parser.parse_args()


def load_dataset(path: Path, date_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' was not found in {path}.")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    return df


def compute_positive_shift(series: pd.Series, buffer: float = 0.0) -> float:
    minimum = series.min()
    return max(0.0, -minimum + buffer)


def prepare_transformed_timeseries(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    date_col: str,
    target_col: str,
    treasury_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    train_ts = (
        train_df[[date_col, target_col, treasury_col]].copy().set_index(date_col).sort_index()
    )
    test_ts = (
        test_df[[date_col, target_col, treasury_col]].copy().set_index(date_col).sort_index()
    )
    combined_ts = pd.concat([train_ts, test_ts], axis=0).sort_index()

    target_shift = compute_positive_shift(train_ts[target_col])
    treasury_shift = compute_positive_shift(train_ts[treasury_col], buffer=0.1)

    if ((combined_ts[target_col] + target_shift) <= 0).any():
        raise ValueError(
            "The train-fitted Box-Cox shift is not large enough for some test values."
        )
    if ((combined_ts[treasury_col] + treasury_shift) <= 0).any():
        raise ValueError(
            "The train-fitted treasury log shift is not large enough for some test values."
        )

    train_target_positive = train_ts[target_col] + target_shift
    _, target_lambda = boxcox(train_target_positive)

    combined_ts["target_boxcox"] = boxcox_with_lambda(
        combined_ts[target_col] + target_shift,
        target_lambda,
    )
    combined_ts["target_diff1"] = combined_ts["target_boxcox"].diff(1)
    combined_ts["target_diff2"] = combined_ts["target_boxcox"].diff(1).diff(1)

    combined_ts["treasury_shifted"] = combined_ts[treasury_col] + treasury_shift
    combined_ts["treasury_log"] = np.log(combined_ts["treasury_shifted"])
    combined_ts["treasury_log_diff"] = combined_ts["treasury_log"].diff(1)

    transform_info = {
        "target_shift": float(target_shift),
        "target_lambda": float(target_lambda),
        "treasury_shift": float(treasury_shift),
    }
    return train_ts, test_ts, combined_ts, transform_info


def create_modeling_frame(
    ts_frame: pd.DataFrame,
    target_lags: list[int],
    treasury_lag: int,
    target_diff2_col: str,
) -> pd.DataFrame:
    frame = pd.DataFrame(index=ts_frame.index)
    frame[target_diff2_col] = ts_frame["target_diff2"]

    for lag in target_lags:
        frame[f"target_lag_{lag}"] = frame[target_diff2_col].shift(lag)

    frame["treasury_lag_1"] = ts_frame["treasury_log_diff"].shift(treasury_lag)
    frame["prev_diff1"] = ts_frame["target_diff1"].shift(1)
    frame["prev_boxcox"] = ts_frame["target_boxcox"].shift(1)
    return frame.dropna()


def inverse_boxcox_to_original(
    boxcox_value: float,
    boxcox_lambda: float,
    boxcox_shift: float = 0.0,
) -> float:
    return float(inv_boxcox(boxcox_value, boxcox_lambda) - boxcox_shift)


def reconstruct_prediction(pred_diff2: float, anchor_row: pd.Series, transform_info: dict[str, float]) -> float:
    pred_boxcox = pred_diff2 + anchor_row["prev_diff1"] + anchor_row["prev_boxcox"]
    return inverse_boxcox_to_original(
        pred_boxcox,
        transform_info["target_lambda"],
        transform_info["target_shift"],
    )


def summarize_forecasts(results: pd.DataFrame) -> dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(results["actual"], results["prediction"]))),
        "MAE": float(mean_absolute_error(results["actual"], results["prediction"])),
        "n_forecasts": int(len(results)),
    }


def make_forecast_df(dates, actuals, preds, model_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "actual": np.asarray(actuals, dtype=float),
            "prediction": np.asarray(preds, dtype=float),
            "model": model_name,
        }
    )


def plot_predictions(results: pd.DataFrame, title: str, y_label: str, output_path: Path) -> None:
    plot_df = results.sort_values("date").copy()
    plt.figure(figsize=(12, 4))
    plt.plot(plot_df["date"], plot_df["actual"], marker="o", label="Actual")
    plt.plot(plot_df["date"], plot_df["prediction"], marker="o", label="Prediction")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_overlay(all_results: pd.DataFrame, y_label: str, output_path: Path) -> None:
    plt.figure(figsize=(14, 5))
    actual_series = all_results.sort_values("date").drop_duplicates(subset=["date"], keep="first")
    plt.plot(actual_series["date"], actual_series["actual"], marker="o", linewidth=2, label="Actual")

    for model_name, model_df in all_results.groupby("model"):
        plot_df = model_df.sort_values("date")
        plt.plot(plot_df["date"], plot_df["prediction"], marker="o", label=model_name)

    plt.title("One-step-ahead test-set predictions")
    plt.xlabel("Date")
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _extract_scalar_forecast(forecast_output) -> float:
    if isinstance(forecast_output, pd.DataFrame):
        return float(forecast_output.iloc[0, 0])
    if isinstance(forecast_output, pd.Series):
        return float(forecast_output.iloc[0])
    arr = np.asarray(forecast_output).reshape(-1)
    return float(arr[0])


def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def create_sequences(X_data: np.ndarray, y_data: np.ndarray, seq_length: int) -> tuple[np.ndarray, np.ndarray]:
    X_seq, y_seq = [], []
    for start_idx in range(len(X_data) - seq_length):
        end_idx = start_idx + seq_length
        X_seq.append(X_data[start_idx:end_idx])
        y_seq.append(y_data[end_idx])
    return np.array(X_seq), np.array(y_seq)


def train_torch_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    optimizer: optim.Optimizer,
    loss_fn,
    max_epochs: int,
    X_val: torch.Tensor | None = None,
    y_val: torch.Tensor | None = None,
    patience: int | None = None,
) -> nn.Module:
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
            best_state = clone_state_dict(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if patience is not None and epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def sarimax_one_step_forecast(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    combined_ts: pd.DataFrame,
    order: tuple[int, int, int],
    transform_info: dict[str, float],
    target_diff2_col: str,
    target_col: str,
) -> pd.DataFrame:
    exog_cols = ["treasury_lag_1"]

    model = SARIMAX(
        endog=train_frame[[target_diff2_col]],
        exog=train_frame[exog_cols],
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    records = []

    for date in test_frame.index:
        exog_row = pd.DataFrame(
            test_frame.loc[date, exog_cols].values.reshape(1, -1),
            index=pd.Index([date], name=test_frame.index.name),
            columns=exog_cols,
        )
        forecast_output = fitted.forecast(steps=1, exog=exog_row)
        pred_diff2 = _extract_scalar_forecast(forecast_output)

        anchor_row = test_frame.loc[date]
        prediction = reconstruct_prediction(pred_diff2, anchor_row, transform_info)
        actual = float(combined_ts.loc[date, target_col])

        records.append(
            {
                "date": date,
                "actual": actual,
                "prediction": float(prediction),
                "model": f"SARIMAX{order}",
            }
        )

        observed_endog = pd.DataFrame(
            {target_diff2_col: [float(test_frame.loc[date, target_diff2_col])]},
            index=pd.Index([date], name=test_frame.index.name),
        )
        fitted = fitted.append(endog=observed_endog, exog=exog_row, refit=False)

    return pd.DataFrame(records)


def fit_mlp(train_frame: pd.DataFrame, feature_cols: list[str], target_diff2_col: str, params: dict[str, Any]):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_frame[feature_cols])
    y_train = train_frame[target_diff2_col].to_numpy().reshape(-1, 1)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)

    model = DynamicMLP(
        input_dim=X_train_t.shape[1],
        hidden_dim=params["hidden_dim"],
        dropout_prob=params["dropout"],
    )
    optimizer = optim.Adam(model.parameters(), lr=params["lr"])
    loss_fn = nn.MSELoss()

    model = train_torch_model(
        model=model,
        X_train=X_train_t,
        y_train=y_train_t,
        optimizer=optimizer,
        loss_fn=loss_fn,
        max_epochs=params["max_epochs"],
    )
    return model, scaler


def mlp_one_step_forecast(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    combined_ts: pd.DataFrame,
    transform_info: dict[str, float],
    feature_cols: list[str],
    target_diff2_col: str,
    target_col: str,
    params: dict[str, Any],
) -> pd.DataFrame:
    model, scaler = fit_mlp(train_frame, feature_cols, target_diff2_col, params)

    X_test = scaler.transform(test_frame[feature_cols])
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        pred_diff2 = model(X_test_t).cpu().numpy().reshape(-1)

    preds = []
    actuals = []
    for date, pred_d2 in zip(test_frame.index, pred_diff2):
        anchor_row = test_frame.loc[date]
        prediction = reconstruct_prediction(float(pred_d2), anchor_row, transform_info)
        preds.append(float(prediction))
        actuals.append(float(combined_ts.loc[date, target_col]))

    return make_forecast_df(test_frame.index, actuals, preds, "MLP")


def fit_rnn(
    train_frame: pd.DataFrame,
    sequence_feature_cols: list[str],
    target_diff2_col: str,
    params: dict[str, Any],
):
    seq_length = params["seq_length"]

    if len(train_frame) <= seq_length + 2:
        raise ValueError("The train modeling frame is too short for the requested RNN sequence length.")

    split_idx = int(len(train_frame) * 0.85)
    train_split = train_frame.iloc[:split_idx].copy()
    val_split = train_frame.iloc[split_idx:].copy()

    if len(train_split) <= seq_length or len(val_split) <= seq_length:
        raise ValueError("Not enough rows in the train split / validation split for RNN sequence creation.")

    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train = X_scaler.fit_transform(train_split[sequence_feature_cols])
    y_train = y_scaler.fit_transform(train_split[[target_diff2_col]])
    X_val = X_scaler.transform(val_split[sequence_feature_cols])
    y_val = y_scaler.transform(val_split[[target_diff2_col]])

    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_length)

    X_train_t = torch.tensor(X_train_seq, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_seq.reshape(-1, 1), dtype=torch.float32)
    X_val_t = torch.tensor(X_val_seq, dtype=torch.float32)
    y_val_t = torch.tensor(y_val_seq.reshape(-1, 1), dtype=torch.float32)

    model = TinyRNN(
        input_dim=X_train_t.shape[2],
        hidden_dim=params["hidden_dim"],
        rnn_type=params["rnn_type"],
    )
    optimizer = optim.Adam(model.parameters(), lr=params["lr"])
    loss_fn = nn.HuberLoss()

    model = train_torch_model(
        model=model,
        X_train=X_train_t,
        y_train=y_train_t,
        X_val=X_val_t,
        y_val=y_val_t,
        optimizer=optimizer,
        loss_fn=loss_fn,
        max_epochs=params["max_epochs"],
        patience=params["patience"],
    )
    return model, X_scaler, y_scaler


def rnn_one_step_forecast(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    combined_model: pd.DataFrame,
    combined_ts: pd.DataFrame,
    transform_info: dict[str, float],
    sequence_feature_cols: list[str],
    target_col: str,
    target_diff2_col: str,
    params: dict[str, Any],
) -> pd.DataFrame:
    model, X_scaler, y_scaler = fit_rnn(train_frame, sequence_feature_cols, target_diff2_col, params)
    seq_length = params["seq_length"]

    preds = []
    actuals = []

    for date in test_frame.index:
        current_pos = combined_model.index.get_loc(date)
        history_window = combined_model.iloc[current_pos - seq_length : current_pos][sequence_feature_cols]

        if len(history_window) != seq_length:
            raise ValueError(f"Insufficient history to create an RNN sequence for {date}.")

        X_test = X_scaler.transform(history_window)
        X_test_t = torch.tensor(X_test.reshape(1, seq_length, -1), dtype=torch.float32)

        model.eval()
        with torch.no_grad():
            pred_scaled = model(X_test_t).cpu().numpy().reshape(-1, 1)

        pred_diff2 = float(y_scaler.inverse_transform(pred_scaled)[0, 0])
        anchor_row = test_frame.loc[date]
        prediction = reconstruct_prediction(pred_diff2, anchor_row, transform_info)

        preds.append(float(prediction))
        actuals.append(float(combined_ts.loc[date, target_col]))

    return make_forecast_df(test_frame.index, actuals, preds, "RNN_GRU")


def fit_cnn(
    train_frame: pd.DataFrame,
    sequence_feature_cols: list[str],
    target_diff2_col: str,
    params: dict[str, Any],
):
    seq_length = params["seq_length"]
    kernel_size = params["kernel_size"]

    if kernel_size > seq_length:
        raise ValueError("CNN kernel_size cannot be larger than seq_length.")
    if len(train_frame) <= seq_length + 2:
        raise ValueError("The train modeling frame is too short for the requested CNN sequence length.")

    split_idx = int(len(train_frame) * 0.85)
    train_split = train_frame.iloc[:split_idx].copy()
    val_split = train_frame.iloc[split_idx:].copy()

    if len(train_split) <= seq_length or len(val_split) <= seq_length:
        raise ValueError("Not enough rows in the train split / validation split for CNN sequence creation.")

    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train = X_scaler.fit_transform(train_split[sequence_feature_cols])
    y_train = y_scaler.fit_transform(train_split[[target_diff2_col]])
    X_val = X_scaler.transform(val_split[sequence_feature_cols])
    y_val = y_scaler.transform(val_split[[target_diff2_col]])

    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_length)

    X_train_t = torch.tensor(X_train_seq, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_seq.reshape(-1, 1), dtype=torch.float32)
    X_val_t = torch.tensor(X_val_seq, dtype=torch.float32)
    y_val_t = torch.tensor(y_val_seq.reshape(-1, 1), dtype=torch.float32)

    model = Tiny1DCNN(
        input_dim=X_train_t.shape[2],
        num_filters=params["num_filters"],
        kernel_size=params["kernel_size"],
        seq_length=params["seq_length"],
    )
    optimizer = optim.Adam(model.parameters(), lr=params["lr"])
    loss_fn = nn.HuberLoss()

    model = train_torch_model(
        model=model,
        X_train=X_train_t,
        y_train=y_train_t,
        X_val=X_val_t,
        y_val=y_val_t,
        optimizer=optimizer,
        loss_fn=loss_fn,
        max_epochs=params["max_epochs"],
        patience=params["patience"],
    )
    return model, X_scaler, y_scaler


def cnn_one_step_forecast(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    combined_model: pd.DataFrame,
    combined_ts: pd.DataFrame,
    transform_info: dict[str, float],
    sequence_feature_cols: list[str],
    target_col: str,
    target_diff2_col: str,
    params: dict[str, Any],
) -> pd.DataFrame:
    model, X_scaler, y_scaler = fit_cnn(train_frame, sequence_feature_cols, target_diff2_col, params)
    seq_length = params["seq_length"]

    preds = []
    actuals = []

    for date in test_frame.index:
        current_pos = combined_model.index.get_loc(date)
        history_window = combined_model.iloc[current_pos - seq_length : current_pos][sequence_feature_cols]

        if len(history_window) != seq_length:
            raise ValueError(f"Insufficient history to create a CNN sequence for {date}.")

        X_test = X_scaler.transform(history_window)
        X_test_t = torch.tensor(X_test.reshape(1, seq_length, -1), dtype=torch.float32)

        model.eval()
        with torch.no_grad():
            pred_scaled = model(X_test_t).cpu().numpy().reshape(-1, 1)

        pred_diff2 = float(y_scaler.inverse_transform(pred_scaled)[0, 0])
        anchor_row = test_frame.loc[date]
        prediction = reconstruct_prediction(pred_diff2, anchor_row, transform_info)

        preds.append(float(prediction))
        actuals.append(float(combined_ts.loc[date, target_col]))

    return make_forecast_df(test_frame.index, actuals, preds, "CNN_1D")


def validate_columns(df: pd.DataFrame, required_cols: list[str], dataset_name: str) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")


def save_json(data: dict[str, Any], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    set_reproducibility(args.seed)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading train and test data.")
    train_raw = load_dataset(args.train_file, args.date_col)
    test_raw = load_dataset(args.test_file, args.date_col)

    required_cols = [args.date_col, args.target_col, args.treasury_col]
    validate_columns(train_raw, required_cols, "Train file")
    validate_columns(test_raw, required_cols, "Test file")

    logging.info("Applying train-fitted transformations to the combined time series.")
    train_ts, test_ts, combined_ts, transform_info = prepare_transformed_timeseries(
        train_df=train_raw,
        test_df=test_raw,
        date_col=args.date_col,
        target_col=args.target_col,
        treasury_col=args.treasury_col,
    )

    target_lags = [1, 3, 4]
    treasury_lag = 1
    target_diff2_col = "target_diff2"
    feature_cols = [f"target_lag_{lag}" for lag in target_lags] + ["treasury_lag_1"]
    sequence_feature_cols = [target_diff2_col, "treasury_lag_1"]

    combined_model = create_modeling_frame(
        ts_frame=combined_ts,
        target_lags=target_lags,
        treasury_lag=treasury_lag,
        target_diff2_col=target_diff2_col,
    )

    train_end_date = train_ts.index.max()
    train_model = combined_model.loc[combined_model.index <= train_end_date].copy()
    test_model = combined_model.loc[combined_model.index > train_end_date].copy()

    if train_model.empty or test_model.empty:
        raise ValueError("The modeling frame is empty for the train or test split after transformation and lag creation.")

    logging.info("Train modeling frame: %s", train_model.shape)
    logging.info("Test modeling frame: %s", test_model.shape)

    model_results: list[pd.DataFrame] = []

    if "sarimax" in args.models:
        logging.info("Running SARIMAX%s.", SARIMAX_ORDER)
        model_results.append(
            sarimax_one_step_forecast(
                train_frame=train_model,
                test_frame=test_model,
                combined_ts=combined_ts,
                order=SARIMAX_ORDER,
                transform_info=transform_info,
                target_diff2_col=target_diff2_col,
                target_col=args.target_col,
            )
        )

    if "mlp" in args.models:
        logging.info("Running MLP.")
        model_results.append(
            mlp_one_step_forecast(
                train_frame=train_model,
                test_frame=test_model,
                combined_ts=combined_ts,
                transform_info=transform_info,
                feature_cols=feature_cols,
                target_diff2_col=target_diff2_col,
                target_col=args.target_col,
                params=MLP_PARAMS,
            )
        )

    if "rnn" in args.models:
        logging.info("Running RNN / GRU model.")
        model_results.append(
            rnn_one_step_forecast(
                train_frame=train_model,
                test_frame=test_model,
                combined_model=combined_model,
                combined_ts=combined_ts,
                transform_info=transform_info,
                sequence_feature_cols=sequence_feature_cols,
                target_col=args.target_col,
                target_diff2_col=target_diff2_col,
                params=RNN_PARAMS,
            )
        )

    if "cnn" in args.models:
        logging.info("Running 1D CNN.")
        model_results.append(
            cnn_one_step_forecast(
                train_frame=train_model,
                test_frame=test_model,
                combined_model=combined_model,
                combined_ts=combined_ts,
                transform_info=transform_info,
                sequence_feature_cols=sequence_feature_cols,
                target_col=args.target_col,
                target_diff2_col=target_diff2_col,
                params=CNN_PARAMS,
            )
        )

    if not model_results:
        raise ValueError("No models were selected. Choose at least one model with --models.")

    all_results = pd.concat(model_results, ignore_index=True)
    summary = (
        all_results.groupby("model")
        .apply(lambda df: pd.Series(summarize_forecasts(df)))
        .reset_index()
        .sort_values(["RMSE", "MAE"])
        .reset_index(drop=True)
    )

    all_results_path = output_dir / "all_test_predictions.csv"
    summary_path = output_dir / "test_summary.csv"
    transform_info_path = output_dir / "transform_info.json"

    all_results.to_csv(all_results_path, index=False)
    summary.to_csv(summary_path, index=False)
    save_json(transform_info, transform_info_path)

    for model_name, model_df in all_results.groupby("model"):
        safe_model_name = model_name.replace("(", "").replace(")", "").replace(",", "_").replace("/", "_")
        model_df.to_csv(output_dir / f"predictions_{safe_model_name}.csv", index=False)
        if args.save_plots:
            plot_predictions(
                results=model_df,
                title=f"{model_name} one-step-ahead predictions on the test set",
                y_label=args.target_col,
                output_path=output_dir / f"plot_{safe_model_name}.png",
            )

    if args.save_plots:
        plot_overlay(all_results, args.target_col, output_dir / "plot_all_models.png")

    logging.info("Saved predictions to %s", all_results_path)
    logging.info("Saved summary to %s", summary_path)
    logging.info("Saved transform parameters to %s", transform_info_path)
    print("\nForecast summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
