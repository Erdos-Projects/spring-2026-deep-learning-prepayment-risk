# Forecasting Pipelines README

This repository contains three folder Data, Jupyter Notebooks and Scripts.

Scripts conntain four Python scripts that cover data preparation, model comparison on the training set, SARIMAX ome step test evaluation, and future forecasting.

## Recommended execution order

1. `train_test_pipeline.py`
2. `model_implementations_pipeline.py`
3. `sarimax_test_evaluator.py`
4. `sarimax_future_forecast_pipeline.py`

## General requirements

Use Python 3.10+.

Typical packages used across the scripts:
- `pandas`
- `numpy`
- `scipy`
- `scikit-learn`
- `statsmodels`
- `matplotlib`
- `optuna`
- `torch`
- `seaborn`

Notes:
- `model_implementations_pipeline.py`, `sarimax_test_evaluator.py` and `sarimax_future_forecast_pipeline.py` attempt to install missing packages automatically.
- `train_test_pipeline.py` only requires pandas.

---

## 1. `train_test_pipeline.py`

### What it does
Builds a monthly modeling dataset by:
- loading LendingClub loan-level data,
- loading Treasury Bill and Fed Funds macro data,
- cleaning and aggregating the loan data to monthly level,
- merging the macro series,
- creating a chronological train/test split,
- saving the resulting CSV files.

### Main inputs
Required files:
- LendingClub CSV (`--lendingclub-file`)
- Fed Funds CSV (`--fedfunds-file`)
- T-Bill CSV (`--tbill-file`)

Expected columns:
- LendingClub file: `id`, `issue_d`, `loan_amnt`, `term`, `int_rate`
- Fed Funds file: `observation_date`, `FEDFUNDS`
- T-Bill file: `observation_date`, `TB3MS`

### Main arguments
- `--lendingclub-file` *(required)*
- `--fedfunds-file` *(required)*
- `--tbill-file` *(required)*
- `--output-dir` *(default: current directory)*
- `--train-filename` *(default: `train.csv`)*
- `--test-filename` *(default: `test.csv`)*
- `--save-full-dataset` *(optional flag)*
- `--full-filename` *(default: `lendingclub_fed_tbill.csv`)*
- `--split-ratio` *(default: `0.75`)*
- `--macro-lookback-months` *(default: `12`)*

### How to run
```bash
python train_test_pipeline.py \
  --lendingclub-file ../Data/LendingClub.csv \
  --fedfunds-file ../Data/FEDFUNDS.csv \
  --tbill-file ../Data/tbill.csv \
  --output-dir ../outputs
```

### Outputs
- `train.csv`
- `test.csv`
- optionally the full merged dataset, e.g. `lendingclub_fed_tbill.csv`

---

## 2. `model_implementations_pipeline.py`

### What it does
Runs the training-set model comparison pipeline. It:
- loads the monthly training dataset,
- performs diagnostics such as ACF/PACF, STL decomposition, ADF tests, and correlations,
- applies the transformation pipeline used for forecasting,
- creates lagged modeling features,
- evaluates baseline, classical ML, and deep learning models using rolling-origin validation,
- saves comparison tables, summaries, and optionally plots.

### Main input
Required file:
- Monthly training CSV (`--data-file`), usually the `train.csv` produced by `train_test_pipeline.py`

Expected columns:
- required: `Month`, `int_rate_mean`, `Treasury_data`
- optional but used in diagnostics if present: `fed_rate`

### Main arguments
- `--data-file` *(required)*
- `--output-dir` *(default: `outputs/model_implementations`)*
- `--mlp-trials` *(default: `50`)*
- `--rnn-trials` *(default: `30`)*
- `--cnn-trials` *(default: `30`)*
- `--skip-plots` *(optional flag)*
- `--skip-diagnostics` *(optional flag)*

### How to run
```bash
python model_implementations_pipeline.py \
  --data-file ../Data/train.csv \
  --output-dir ../outputs/model_implementations
```

### Outputs
Saved under the output directory, including:
- `tables/missing_summary.csv`
- `tables/adf_summary.csv`
- `tables/fold_schedule.csv`
- `tables/baseline_predictions.csv`
- `tables/baseline_summary.csv`
- `tables/ridge_search_results.csv`
- `tables/sarimax_search_results.csv`
- `tables/random_forest_search_results.csv`
- `tables/comparable_models.csv`
- `tables/deep_learning_summary.csv`
- `experiment_summary.json`
- plot files under `plots/` unless plotting is skipped

### Purpose in the workflow
Use this script to determine which model configuration performs best on the training set under rolling validation.

---

## 3. `sarimax_test_evaluator.py`

### What it does
Fits the fixed SARIMAX model selected on the training set and evaluates it on a held-out test set. It:
- applies the same train-time transformations,
- builds the lagged Treasury exogenous feature,
- fits SARIMAX on transformed training data,
- forecasts across the full test horizon,
- reconstructs predictions back to the original target scale,
- computes RMSE and MAE if the test file contains actual target values.

### Main inputs
Required files:
- training CSV (`--train-file`)
- test CSV (`--test-file`)

Expected columns:
- training file: `Month`, `int_rate_mean`, `Treasury_data`
- test file: `Month`, `Treasury_data`
- optional in test file for scoring: `int_rate_mean`

### Main arguments
- `--train-file` *(required)*
- `--test-file` *(required)*
- `--output-dir` *(default: `outputs/sarimax_test_eval`)*
- `--date-col` *(default: `Month`)*
- `--target-col` *(default: `int_rate_mean`)*
- `--treasury-col` *(default: `Treasury_data`)*
- `--sarimax-order` *(default: `2 0 1`)*

### How to run
```bash
python sarimax_test_evaluator.py \
  --train-file train.csv \
  --test-file test.csv \
  --output-dir outputs/sarimax_test_eval
```

Or explicitly set the fixed order:
```bash
python sarimax_test_evaluator.py \
  --train-file train.csv \
  --test-file test.csv \
  --output-dir outputs/sarimax_test_eval \
  --sarimax-order 2 0 1
```

### Outputs
- `sarimax_test_predictions.csv`
- `sarimax_test_metrics.json`
- `transform_state.json`

### Purpose in the workflow
Use this after the best SARIMAX order has already been chosen from the training-set experiments.

---

## 4. `sarimax_future_forecast_pipeline.py`

### What it does
Uses the trained SARIMAX workflow to forecast future mean interest rates from future T-Bill data. It:
- loads the train dataset and a future T-Bill dataset,
- applies the same training transformations,
- fits SARIMAX on transformed training data,
- forecasts future transformed values using T-Bill data and previous outputs
- reconstructs the forecasts back to the original target scale,
- optionally computes RMSE and MAE if the future file also contains actual target values,
- saves forecast artifacts and optionally a forecast plot.

### Main inputs
Required files:
- training CSV (`--train-file`)
- future CSV (`--future-file`)

Expected columns:
- training file: `Month`, `int_rate_mean`, `Treasury_data`
- future file: by default `observation_date`, `TB3MS`
- optional in future file for scoring: `int_rate_mean`

### Main arguments
- `--train-file` *(required)*
- `--future-file` *(required)*
- `--output-dir` *(default: `outputs/sarimax_future`)*
- `--date-col-train` *(default: `Month`)*
- `--date-col-future` *(default: `observation_date`)*
- `--target-col` *(default: `int_rate_mean`)*
- `--tbill-col-train` *(default: `Treasury_data`)*
- `--tbill-col-future` *(default: `TB3MS`)*
- `--sarimax-order` *(default: `2,0,1`)*
- `--tbill-lag` *(default: `1`)*
- `--keep-overlapping-future` *(optional flag)*
- `--save-plot` *(optional flag)*
- `--show-plot` *(optional flag)*

### How to run
```bash
python sarimax_future_forecast_pipeline.py \
  --train-file ../Data/train.csv \
  --future-file ../Data/tbill.csv \
  --output-dir outputs/sarimax_future \
  --date-col-train Month \
  --date-col-future observation_date \
  --target-col int_rate_mean \
  --tbill-col-train Treasury_data \
  --tbill-col-future TB3MS \
  --sarimax-order 2,0,1 \
  --save-plot
```

### Outputs
- `sarimax_future_forecasts.csv`
- `sarimax_future_forecast_summary.json`
- `transform_state.json`
- `fitted_sarimax_params.json`
- `sarimax_model_summary.txt`
- optionally `sarimax_future_forecast.png`

### Purpose in the workflow
Use this when you want forward forecasts beyond the train/test split, driven by future T-Bill values.

---

