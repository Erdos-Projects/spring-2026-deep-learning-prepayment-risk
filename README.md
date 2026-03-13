# Prepayment & Default Risk — Deep Learning Survival Analysis

## What's in this branch

### 1. Static DeepHit (`DeepHit.ipynb`)

Competing risks survival model using `pycox`. Predicts monthly probability of default vs prepayment for each loan over 36 months.

- **Why survival instead of classification?** The existing NCR approach failed due to extreme class imbalance (~0.4% monthly default). Survival framing eliminates this — instead of "will this loan default this month?", it asks "given survival to month t, what's the hazard?"
- **17 features**: BorrowerRate, FED rate at origination, APR-FED spread, credit score, ProsperScore, ProsperRating, DTI, income, etc.
- **Architecture**: 256→128→64 MLP (51K params) → [batch, 2, 36] output (2 risks × 36 months)
- **Train/test**: ≤2012 / 2013-2014 (aligned with RF baseline holdout), 34K/43K loans

**Results:**
| Metric | DeepHit | RF Baseline |
|---|---|---|
| Default top-decile lift | **2.58x** | ~2.7x |
| Prepay top-decile lift | **2.25x** | — (see note) |
| Default C-index | 0.639 | — |
| Prepay C-index | 0.626 | — |
| Default KS | **0.159** | 0.051 |
| Prepay KS | 0.180 | **0.674** |
| Default PR-AUC (t=36) | 0.009 | 0.010 |
| Prepay PR-AUC (t=36) | **0.037** | 0.007 |

Note: PR-AUC is not directly comparable across models — DeepHit uses loan-level scores (base rate ~0.7% default, ~4.9% prepay) vs RF monthly loan-month rows (base rate ~0.8% default, ~0.13% prepay). KS is more comparable: DeepHit has 3x better default separation (0.159 vs 0.051), while RF dominates prepay separation (0.674 vs 0.180).

### 2. Dynamic-DeepHit (`DynamicDeepHit.ipynb`)

Adds an LSTM encoder for time-varying features (FED rate path, spread momentum) on top of the static DeepHit.

**Result: failed.** The LSTM overfit to training-set rate movements that don't exist in the 2013-2014 test set (ZIRP — FED rate stuck at 0.1% with zero variance). Prepay C-index dropped to 0.498 (worse than random). Static DeepHit remains the best model.

### 3. Monte Carlo Portfolio Simulation (`MonteCarloSimulation.ipynb`)

Chains all 3 project models to simulate portfolio performance under different FED rate scenarios:

**FED scenario → Model 2 (FED→T-bill) → Model 1 (FED+T-bill→APR) → Model 3 (DeepHit CIF) → cash flows → portfolio IRR**

Models 1 & 2 use simple placeholder functions (linear regression, median spread) marked with `# TODO` for teammates to swap in their trained models.

CIFs are calibrated to historical Kaplan-Meier base rates before cash flow computation (raw DeepHit CIFs are well-ranked but poorly calibrated in absolute terms).

**Portfolio metrics (43K loans, $480M):**
| Scenario | FED | Mean IRR | Default | Prepay |
|---|---|---|---|---|
| ZIRP | 0.1% | **+9.0%** | 25% | 55% |
| Crisis cut | 0.25% | +1.7% | 28% | 15% |
| Low | 1.0% | −1.5% | 24% | 4% |
| Moderate | 2.5% | −4.5% | 23% | 5% |
| High | 5.0% | **−10.7%** | 30% | 4% |

Key finding: prepay rates swing wildly with FED rates (55% vs 4%), while default rates are stable (23-30%). Prepayment is the main driver of portfolio returns.

Monte Carlo (K=500): IRR std is ±0.1-0.2% — idiosyncratic risk diversifies away with 43K loans, portfolio risk is purely systematic.

## Environment Setup

Requires **Python 3.10 or 3.11**. `pycox` and `torchtuples` are not compatible with Python 3.12+.

```bash
conda create -n prepayment python=3.10 -y
conda activate prepayment
pip install -r requirements.txt
```

In VS Code: open any `.ipynb` → kernel picker → **prepayment**.

## Files

```
├── DeepHit.ipynb               # Static DeepHit model (best Model 3)
├── DynamicDeepHit.ipynb         # Dynamic-DeepHit with LSTM (failed — ZIRP)
├── MonteCarloSimulation.ipynb   # Goal 3: portfolio simulation under FED scenarios
├── data/
│   ├── prosperLoanData.csv      # ~113K Prosper loans, 81 columns
│   ├── FEDFUNDS.csv             # Monthly Federal Funds Rate
│   └── TB3MS.csv                # 3-Month Treasury Bill rate
└── requirements.txt
```
