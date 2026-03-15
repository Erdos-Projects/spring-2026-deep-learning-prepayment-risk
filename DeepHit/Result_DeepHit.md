# DeepHit Results — Prepayment & Default Risk

## Why DeepHit?

We ranked 10 candidate deep learning models for competing-risks survival analysis. DeepHit scored highest because:

- **No distributional assumptions** — unlike Cox-based models, DeepHit directly estimates the probability mass function over discrete time steps, making it flexible for non-proportional hazards.
- **Native competing-risks support** — a single model outputs P(default), P(prepay), and P(survival) that sum to 1 at each time step, which is exactly what the Monte Carlo simulation pipeline needs.
- **Survival framing solves class imbalance** — monthly default rate is ~0.4%, but cumulative 36-month default rate is 27% in training. No oversampling or class weighting needed.
- **Works with static features only** — important because Monte Carlo simulation generates loan-level scenarios at origination, not month-by-month.

## Variants Tried

### Static DeepHit (Best Model) — `DeepHit.ipynb`
128->64->32 MLP, 14K params, 9 features. This is our final model. It produces clean monotonic lift curves and internally consistent competing-risks probabilities.

### Dynamic-DeepHit (Overfit) — `DynamicDeepHit.ipynb`
Added an LSTM encoder (8->48 hidden) over 8 time-varying features (monthly macro rates, spread momentum, months remaining, month since origination). 31K params. Higher default C-index (0.665 vs 0.579) but **overfits**: the lift curve is non-monotonic (decile 2 = 3.22x > decile 1 = 2.31x), and both KS stats are worse than static DeepHit. Root cause: the LSTM memorized pre-2012 rate movements that don't generalize to the near-zero-rate (ZIRP) test period (2013-2014).

### Single-Risk DeepHit (Trade-off) — `DeepHit_SingleRisk.ipynb`
Trained two independent models — one for default (treating prepay as censored), one for prepay (treating default as censored). Prepay KS improved (0.141->0.244) but default KS degraded (0.287->0.204), and the default lift curve became non-monotonic. The competing-risks model is preferred because it produces joint probabilities needed for Monte Carlo and has better default discrimination.

### Feature Engineering Iterations (Failed)
- **17 borrower-level features** (credit score, income, debt-to-income, etc.): performed well but these features aren't available at Monte Carlo simulation time, so we dropped them.
- **6 features** (no UNRATE, no TB3MS): model died — C-index ~0.50 (random). Adding unemployment rate and T-bill rate back (6->9 features) resurrected the model.
- **22-feature expansion + 60-combo HP grid search**: everything got worse. More features != better. Reverted to 9 features.

## Features (9)

BorrowerRate, FED rate, T-bill rate, unemployment rate, APR-FED spread, APR-T-bill spread, loan amount (log), Term, monthly payment.

Split: Train <= 2010 (36,626 loans) | Val 2011 (11,228) | Test >= 2012 (66,070).

## Results: DeepHit Variants

| Metric | Static DeepHit | Dynamic-DeepHit | Single-Risk (Default) | Single-Risk (Prepay) |
|---|---|---|---|---|
| **Default C-index** | 0.579 | 0.665 | 0.607 | — |
| **Prepay C-index** | 0.518 | 0.574 | — | 0.541 |
| **Default KS** | **0.287** | 0.209 | 0.204 | — |
| **Prepay KS** | 0.141 | 0.093 | — | **0.244** |
| **Default PR-AUC (t=36)** | **0.075** | 0.059 | 0.061 | — |
| **Prepay PR-AUC (t=36)** | 0.135 | 0.133 | — | **0.169** |
| **Default top-decile lift** | **2.68x** | 2.31x | 1.06x | — |
| **Prepay top-decile lift** | **1.42x** | 1.23x | — | 1.38x |
| **Default IBS** | 0.125 | 0.040 | 0.199 | — |
| **Prepay IBS** | 0.402 | 0.399 | — | 0.178 |
| **Parameters** | 14,440 | 31,720 | 13,252 | 13,252 |

**Winner: Static DeepHit** — best default discrimination (KS 0.287, lift 2.68x), monotonic lift curves, and joint competing-risks output for Monte Carlo.

## Results: DeepHit vs Random Forest Baseline

Comparison with Morgan's Random Forest (loan-month panel, includes `month_since_orig`).

| Metric | Static DeepHit | Morgan RF |
|---|---|---|
| **Default top-decile lift** | **2.68x** | 1.2x |
| **Prepay top-decile lift** | 1.42x | **2.5x** |
| **Default KS** | **0.287** | 0.051 |
| **Prepay KS** | 0.141 | **0.67** |

**Key takeaway:** DeepHit strongly outperforms RF on default risk (2.68x vs 1.2x lift, 5.6x higher KS). RF dominates prepay — largely because it uses `month_since_orig` at loan-month granularity, which directly captures the seasoning curve. DeepHit uses only origination-time features (by design, for Monte Carlo compatibility), so it cannot leverage that signal.
