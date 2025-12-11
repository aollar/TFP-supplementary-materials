# Data Dictionary

This document defines all variables in the result CSV files.

---

## Core Benchmark Results

### `results/synthetic_results_mae.csv`

| Column | Type | Description |
|--------|------|-------------|
| `family` | string | Synthetic data family: `trend`, `seasonal`, `logistic_growth`, `logistic_saturated` |
| `model` | string | Forecasting model: `TFP`, `SimpleTheta`, `Naive` |
| `horizon` | int | Forecast horizon in time steps: 1, 4, 8, or 12 |
| `mean_mae` | float | Mean Absolute Error averaged over all series and rolling origins |
| `std_mae` | float | Standard deviation of MAE across series |
| `n_series` | int | Number of series evaluated (200) |

### `results/synthetic_results_wis.csv`

| Column | Type | Description |
|--------|------|-------------|
| `family` | string | Synthetic data family |
| `model` | string | Forecasting model |
| `horizon` | int | Forecast horizon |
| `mean_wis` | float | Mean Weighted Interval Score (using IntervalLawV2) |
| `std_wis` | float | Standard deviation of WIS across series |
| `coverage_90` | float | Empirical coverage of 90% prediction intervals (0-1) |
| `n_series` | int | Number of series evaluated |

### `results/synthetic_bootstrap_mae_ratios.csv`

| Column | Type | Description |
|--------|------|-------------|
| `family` | string | Synthetic data family |
| `comparison` | string | Model comparison: `TFP/SimpleTheta` or `TFP/Naive` |
| `horizon` | int | Forecast horizon |
| `mae_ratio` | float | Point estimate of MAE ratio (TFP / baseline) |
| `mae_ci_lower` | float | 95% CI lower bound (bootstrap percentile) |
| `mae_ci_upper` | float | 95% CI upper bound (bootstrap percentile) |
| `wis_ratio` | float | Point estimate of WIS ratio |
| `wis_ci_lower` | float | 95% CI lower bound for WIS ratio |
| `wis_ci_upper` | float | 95% CI upper bound for WIS ratio |

**Interpretation of ratios:**
- Ratio < 1.0: TFP outperforms baseline
- Ratio > 1.0: Baseline outperforms TFP
- CI excludes 1.0: Statistically significant difference

---

## Hybrid Benchmark Results

### `hybrid_results/synthetic_hybrid_results_mae.csv`

| Column | Type | Description |
|--------|------|-------------|
| `family` | string | Hybrid family: `trend_logistic_mix` or `seasonal_logistic_mix` |
| `w_logistic` | float | S-curve mixing weight: 0.0, 0.1, 0.25, 0.5, 0.75, 1.0 |
| `model` | string | Forecasting model: `TFP`, `SimpleTheta`, `Naive` |
| `horizon` | int | Forecast horizon: 1, 4, 8, or 12 |
| `mean_mae` | float | Mean Absolute Error |
| `std_mae` | float | Standard deviation of MAE |
| `n_series` | int | Number of series (200) |

### `hybrid_results/synthetic_hybrid_results_wis.csv`

Same structure as MAE file, with `mean_wis`, `std_wis`, and `coverage_90` columns.

### `hybrid_results/synthetic_hybrid_bootstrap_ratios.csv`

| Column | Type | Description |
|--------|------|-------------|
| `family` | string | Hybrid family |
| `w_logistic` | float | S-curve mixing weight |
| `comparison` | string | Model comparison |
| `horizon` | int | Forecast horizon |
| `mae_ratio` | float | MAE ratio point estimate |
| `mae_ci_lower` | float | 95% CI lower bound |
| `mae_ci_upper` | float | 95% CI upper bound |
| `wis_ratio` | float | WIS ratio point estimate |
| `wis_ci_lower` | float | 95% CI lower bound |
| `wis_ci_upper` | float | 95% CI upper bound |

---

## Key Metrics Definitions

### Mean Absolute Error (MAE)

```
MAE = (1/n) * Σ|y_t - ŷ_t|
```

Where `y_t` is the actual value and `ŷ_t` is the point forecast.

### Weighted Interval Score (WIS)

The WIS is a proper scoring rule for probabilistic forecasts:

```
WIS = (1/K) * Σ_k [ (upper_k - lower_k) + (2/α_k) * (lower_k - y) * I(y < lower_k)
                                        + (2/α_k) * (y - upper_k) * I(y > upper_k) ]
```

We use K=9 quantile levels: 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45 (and symmetric upper quantiles).

### IntervalLawV2 Configuration

All interval forecasts use IntervalLawV2 with:
- `lookback = 104` (equivalent to 2 years of weekly data)
- `horizon_scale = 0.1` (linear width scaling: 1 + 0.1 * h)
- Level-based residuals (not differenced)

### Bootstrap Procedure

- **Method:** Block bootstrap over series (not time points)
- **Resamples:** 1,000
- **CI method:** Percentile (2.5th and 97.5th percentiles)
- **Seed:** 42 (deterministic)

---

## Synthetic Data Specifications

### Trend + Noise (`trend`)
- Length: 200 time steps
- Piecewise linear with potential turning point at t=100
- Slope: Uniform[-0.5, +0.5]
- 50% probability of sign flip at turning point
- Noise: Gaussian, σ = 5-15% of signal range

### Seasonal AR(1) (`seasonal`)
- Length: 240 time steps
- Period: 12
- AR(1) coefficient: Uniform[0.3, 0.9]
- Seasonal amplitude: Uniform[10, 40]
- Phase: Uniform[0, 2π]

### Logistic Growth (`logistic_growth`)
- Length: 160 time steps
- Asymptote K: Uniform[80, 120]
- Growth rate r: Uniform[0.03, 0.15]
- Midpoint t0: Uniform[70, 130]
- Evaluation window: 20-80% of K

### Logistic Saturated (`logistic_saturated`)
- Same parameters but t0: Uniform[30, 70]
- Evaluation window: 90-100% of K

### Hybrid Mixtures
- `trend_logistic_mix`: Linear trend + w_logistic × rescaled logistic
- `seasonal_logistic_mix`: Seasonal AR(1) + w_logistic × rescaled logistic
- Logistic component rescaled to match base signal amplitude
