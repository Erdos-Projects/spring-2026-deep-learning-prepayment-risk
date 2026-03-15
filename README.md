# Prepayment & Default Risk — Deep Learning Survival Analysis

## Summary

DeepHit competing-risks survival model for predicting loan default and prepayment. Trained on 113K Prosper loans (2005–2014) with 9 macro/loan features — no borrower-specific features, enabling use in Monte Carlo simulation.

**Key result:** DeepHit beats the RF baseline on default (2.68x vs 1.2x top-decile lift) but trails on prepayment (KS 0.141 vs 0.67) due to the RF's structural advantage with `month_since_orig` at loan-month granularity.

## Models

### Static DeepHit (`DeepHit.ipynb`) — Best Model

128→64→32 MLP (14K params) predicting monthly default/prepay probability over 36 months.

| Metric | DeepHit | Morgan RF |
|---|---|---|
| Default top-decile lift | **2.68x** | 1.2x |
| Prepay top-decile lift | 1.42x | **2.5x** |
| Default KS | **0.287** | 0.051 |
| Prepay KS | 0.141 | **0.67** |

**9 features:** BorrowerRate, FED rate, T-bill rate, unemployment rate, APR-FED spread, APR-T-bill spread, loan amount (log), Term, monthly payment.

### Single-Risk DeepHit (`DeepHit_SingleRisk.ipynb`)

Two separate 128→64→32 MLPs — one for default, one for prepay — each treating the other event as censored. Improves prepay discrimination (KS 0.141→0.244) at the cost of default ranking (KS 0.287→0.204).

### Dynamic-DeepHit (`DynamicDeepHit.ipynb`)

LSTM(8→48) + MLP encoder for time-varying features. Higher default C-index (0.665 vs 0.579) but overfits — non-monotonic lift curves, worse KS and PR-AUC than static version.

### Monte Carlo Simulation (`MonteCarloSimulation.ipynb`)

Chains all 3 project models: FED scenario → T-bill → borrower APR → DeepHit CIF → cash flows → portfolio IRR. Key finding: prepay rates swing 55% to 4% with FED rates; prepayment is the main driver of returns.

## Split

Train ≤2010 / Val 2011 / Test 2012-2014 (matching Morgan's RF, 1-year gap to prevent leakage).

## Setup

Requires **Python 3.10 or 3.11** (`pycox`/`torchtuples` incompatible with 3.12+).

```bash
conda create -n prepayment python=3.10 -y && conda activate prepayment
pip install -r requirements.txt
```

## Files

```
├── DeepHit.ipynb               # Static DeepHit competing-risks (best model)
├── DeepHit_SingleRisk.ipynb    # Separate default & prepay models
├── DynamicDeepHit.ipynb         # Dynamic-DeepHit with LSTM
├── MonteCarloSimulation.ipynb   # Portfolio simulation under FED scenarios
├── data/
│   ├── prosperLoanData.csv      # ~113K Prosper loans, 81 columns
│   ├── FEDFUNDS.csv             # Monthly Federal Funds Rate
│   ├── TB3MS.csv                # 3-Month Treasury Bill rate
│   ├── UNRATE.csv               # Monthly Unemployment Rate
│   └── MORTGAGE30US.csv         # 30-Year Fixed Mortgage Rate
└── requirements.txt
```
