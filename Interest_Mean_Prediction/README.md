# Loan Interest Rate Forecasting Notebook Pipeline

This folder contains a four-notebook workflow for forecasting the **monthly average LendingClub loan interest rate** using LendingClub loan-level data together with macroeconomic variables, especially the **3-Month Treasury Bill rate**.

The notebooks are written as a staged pipeline:

1. prepare the monthly dataset and create the train/test split,
2. run diagnostics and model selection on the training set,
3. evaluate the selected models on the held-out test set using one-step-ahead forecasts,
4. use the selected SARIMAX specification to forecast future monthly mean interest rates from future T-Bill inputs.

---

## Folder contents

### `1.Train_Test_Split.ipynb`
Builds the monthly modeling dataset from raw files.
Download the data from https://www.kaggle.com/datasets/wordsforthewise/lending-club  and save it as LendingClub.csv in the Data folder.


What it does:
- loads raw LendingClub loan-level data from Kaggle,
- cleans and sorts the loan data chronologically,
- aggregates loans to a **monthly** level,
- loads monthly macro series:
  - `FEDFUNDS.csv`
  - `tbill.csv`
- merges the macro data with the LendingClub monthly aggregates,
- checks for missing months,
- creates a **chronological 75/25 train-test split**,
- saves:
  - `../Data/train.csv`
  - `../Data/test.csv`

Observed output in the notebook:
- final merged monthly dataset: **139 rows**
- train period: **2007-06 to 2016-01** (**104 rows**)
- test period: **2016-02 to 2018-12** (**35 rows**)

---

### `2.Model_Implementations_Annotated.ipynb`
Performs exploratory diagnostics, transformations, feature construction, rolling validation, and model comparison on the training set.

What it does:
- loads `../Data/train.csv`,
- checks correlations and diagnostics,
- performs STL decomposition and ADF stationarity checks,
- applies transformations:
  - **Box-Cox** transform to the target `int_rate_mean`,
  - first and second differencing of the transformed target,
  - **log + first difference** transform to `Treasury_data`,
- uses **PACF-inspired target lags**:
  - `target_lag_1`
  - `target_lag_3`
  - `target_lag_4`
- uses `treasury_lag_1` as the exogenous macro feature,
- evaluates models with **rolling one-step cross-validation** using:
  - `TimeSeriesSplit(n_splits=20, test_size=1)`

Models evaluated:
- Naive baseline
- Rolling average baseline
- Ridge regression
- SARIMAX
- Random forest
- MLP
- RNN / LSTM / GRU family
- 1D CNN

Search setup used in the notebook:
- Ridge alphas: `0.01, 0.1, 1.0, 10.0, 100.0`
- Random forest:
  - `n_estimators in {50, 100}`
  - `max_depth in {3, 5, None}`
- SARIMAX orders searched:
  - `(p,0,q)` with `p in {2,3,4}`, `q in {0,1}`
- Optuna trial budgets:
  - MLP: `50`
  - RNN: `30`
  - CNN: `30`

Key training-set rolling-validation result from the notebook:
- **Best comparable classical model:** `SARIMAX(2,0,1)`
- Validation metrics:
  - **SARIMAX RMSE:** `0.190637`
  - **SARIMAX MAE:** `0.138230`

Comparable model table shown in the notebook:
- SARIMAX: RMSE `0.190637`, MAE `0.138230`
- Naive: RMSE `0.210555`, MAE `0.151338`
- Ridge: RMSE `0.271028`, MAE `0.214878`
- RandomForest: RMSE `0.285610`, MAE `0.213652`
- RollingAvg4: RMSE `0.318288`, MAE `0.245548`

Best deep-learning settings found in the notebook:
- **MLP**
  - `hidden_dim=24`
  - `dropout=0.378016`
  - `lr=0.007305`
  - rolling validation RMSE `0.261729`
- **RNN**
  - `rnn_type=LSTM`
  - `seq_length=5`
  - `hidden_dim=4`
  - `lr=0.021783`
  - rolling validation RMSE `0.225552`
- **CNN**
  - `seq_length=4`
  - `kernel_size=2`
  - `num_filters=12`
  - `lr=0.014647`
  - rolling validation RMSE `0.259748`

---

### `3.Test_Set_One_Step_Forecasts_Annotated.ipynb`
Evaluates the selected models on the held-out test set.

What it does:
- loads:
  - `../Data/train.csv`
  - `../Data/test.csv`
- refits the same train-time transformations on the training portion,
- builds the combined transformed series,
- evaluates the selected fixed models on the **35-month test horizon**,
- reconstructs predictions back to the original interest-rate scale,
- generates comparison plots,
- saves:
  - `./test_set_forecast_outputs/all_test_predictions.csv`
  - `./test_set_forecast_outputs/test_summary.csv`

Fixed model choices used here:
- `SARIMAX_ORDER = (2, 0, 1)`
- MLP parameters from notebook 2
- RNN parameters from notebook 2
- CNN parameters from notebook 2

Important implementation detail:
- the SARIMAX test evaluation is **true one-step-ahead forecasting** on the test horizon:
  after each prediction, the model state is updated with the **observed transformed test value** using `append(..., refit=False)`.

Test-set summary shown in the notebook:
- **SARIMAX(2,0,1)**: RMSE `0.313541`, MAE `0.220349`
- **MLP**: RMSE `0.359538`, MAE `0.273506`
- **RNN_GRU**: RMSE `0.382019`, MAE `0.296814`
- **CNN_1D**: RMSE `0.395246`, MAE `0.301591`

So the test notebook again identifies **SARIMAX(2,0,1)** as the best-performing model among the evaluated choices.

---

### `4.sarimax_future_forecast_tbill_notebook.ipynb`
Uses the fixed SARIMAX model to forecast future average monthly interest rates from future T-Bill values.

What it does:
- loads:
  - `../Data/train.csv`
  - `../Data/tbill.csv`
- applies the same train-time transformations used earlier,
- fits the fixed `SARIMAX(2,0,1)` model on the transformed training target,
- builds the future exogenous series from the T-Bill file,
- recursively forecasts future monthly mean interest rates,
- optionally computes RMSE and MAE **if the future file also contains actual target values**.

Default settings in the notebook:
- SARIMAX order: `(2, 0, 1)`
- treasury lag: `1`

Observed behavior in the saved notebook output:
- the notebook produced future forecasts beginning immediately after the training period,
- since the provided future T-Bill file did **not** include actual target values, RMSE and MAE were **not computed** in that run.

Note:
- this notebook defines `OUTPUT_DIR = "../outputs/sarimax_future"`, but in the uploaded version there is **no explicit save step** that writes the forecast dataframe to disk. The forecast dataframe is displayed in the notebook output.

---

## Recommended execution order

Run the notebooks in this order:

1. `1.Train_Test_Split.ipynb`
2. `2.Model_Implementations_Annotated.ipynb`
3. `3.Test_Set_One_Step_Forecasts_Annotated.ipynb`
4. `4.sarimax_future_forecast_tbill_notebook.ipynb`

This preserves the intended pipeline and the relative file dependencies.

---

## Expected data files

These notebooks expect the following CSV files to exist:

### Raw input files
- `../Data/LendingClub.csv`
- `../Data/FEDFUNDS.csv`
- `../Data/tbill.csv`

### Intermediate/generated files
- `../Data/train.csv`
- `../Data/test.csv`

Important columns expected by the notebooks include:

### `LendingClub.csv`
- `id`
- `issue_d`
- `loan_amnt`
- `term`
- `int_rate`

### `FEDFUNDS.csv`
- `observation_date`
- `FEDFUNDS`

### `tbill.csv`
- `observation_date`
- `TB3MS`

### generated training/test files
- `Month`
- `int_rate_mean`
- `Treasury_data`
- `fed_rate`
- plus the monthly aggregate columns created in notebook 1

---

## Expected project layout

The notebooks use relative paths like `../Data/...`, so they are designed for a structure similar to:

```text
project_root/
├── Data/
│   ├── LendingClub.csv
│   ├── FEDFUNDS.csv
│   ├── tbill.csv
│   ├── train.csv
│   └── test.csv
└── notebooks/
    ├── 1.Train_Test_Split.ipynb
    ├── 2.Model_Implementations_Annotated.ipynb
    ├── 3.Test_Set_One_Step_Forecasts_Annotated.ipynb
    └── 4.sarimax_future_forecast_tbill_notebook.ipynb
```

If your folder structure is different, update the path variables at the top of each notebook.

---

## Main modeling choices used throughout the folder

### Target
- `int_rate_mean`  
  Monthly average LendingClub loan interest rate.

### Exogenous macro variable
- `Treasury_data`
  Built from the 3-Month Treasury Bill series and used with a one-month lag.

### Transformations
- target:
  - Box-Cox
  - first difference
  - second difference
- treasury:
  - positive shift if needed
  - log transform
  - first difference

### Main lag features
- target lags: `1, 3, 4`
- treasury lag: `1`

### Main selected model
- `SARIMAX(2,0,1)` with lagged treasury input

---

## Python dependencies

The notebooks use packages such as:

- `pandas`
- `numpy`
- `matplotlib`
- `scipy`
- `scikit-learn`
- `statsmodels`
- `optuna`
- `torch`
- `seaborn`
- `IPython`

A minimal install command is:

```bash
pip install pandas numpy matplotlib scipy scikit-learn statsmodels optuna torch seaborn ipython
```

---

## Outputs produced by the folder

### From notebook 1
- `../Data/train.csv`
- `../Data/test.csv`

### From notebook 3
- `./test_set_forecast_outputs/all_test_predictions.csv`
- `./test_set_forecast_outputs/test_summary.csv`

### From notebook 4
- displayed forecast dataframe in the notebook
- optional RMSE / MAE if actual future targets are supplied

---

## Summary

This folder implements a complete small-sample monthly forecasting workflow for LendingClub average loan interest rates. The notebooks consistently use the same transformation pipeline, compare classical and deep-learning models fairly, and show that:

- **SARIMAX(2,0,1)** is the best model in rolling validation,
- it is also the strongest performer on the held-out test set,
- deep learning models are competitive but do **not** beat the selected SARIMAX model on this dataset.

