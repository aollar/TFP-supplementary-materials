# Supplementary Materials

**Paper:** A Simple Three-Parameter Method for Cross-Domain Probabilistic Forecasting

---

## S1. Complete Coverage Results by Domain

### Table S1: Coverage at Multiple Nominal Levels

| Domain | 80% Nominal | 90% Nominal | 95% Nominal | n |
|--------|-------------|-------------|-------------|---|
| Flu US | 100.0% | 100.0% | 100.0% | 1,590 |
| Covid Hosp | 100.0% | 100.0% | 100.0% | 1,500 |
| Bass Tech | 100.0% | 100.0% | 100.0% | 123 |
| Finance | 96.7% | 99.0% | 99.7% | 1,050 |
| NYISO | 100.0% | 100.0% | 100.0% | 330 |
| M4 Weekly | 95.5% | 97.0% | 97.8% | 6,000 |
| M4 Daily | 95.9% | 97.1% | 97.6% | 3,000 |
| M4 Monthly | 92.7% | 97.1% | 98.4% | 3,000 |
| Kaggle Store | 92.9% | 96.2% | 98.1% | 3,000 |
| Wikipedia | 81.9% | 90.3% | 93.2% | 3,000 |
| M5 Retail | 78.0% | 82.4% | 85.5% | 3,000 |
| **Average** | **91.9%** | **94.9%** | **96.3%** | **25,593** |

### Table S2: Coverage by Horizon (Flu US Domain)

| Horizon | 80% Coverage | 90% Coverage | 95% Coverage | n |
|---------|--------------|--------------|--------------|---|
| 1 | 100.0% | 100.0% | 100.0% | 530 |
| 2 | 100.0% | 100.0% | 100.0% | 530 |
| 4 | 100.0% | 100.0% | 100.0% | 530 |

---

## S2. Grid Search Results

### Table S3: Top 10 Parameter Combinations by Coverage Error

| Rank | Lookback | Growth Mode | Strength | Cov Error | WIS | Safety |
|------|----------|-------------|----------|-----------|-----|--------|
| 1 | 104 | linear | 0.10 | 0.191 | 12,532 | Pass |
| 2 | 104 | linear | 0.08 | 0.193 | 12,891 | Pass |
| 3 | 78 | linear | 0.10 | 0.197 | 12,845 | Pass |
| 4 | 156 | linear | 0.10 | 0.198 | 12,412 | Pass |
| 5 | 104 | linear | 0.12 | 0.199 | 12,198 | Pass |
| 6 | 78 | linear | 0.08 | 0.201 | 13,201 | Pass |
| 7 | 78 | linear | 0.12 | 0.204 | 12,654 | Pass |
| 8 | 156 | linear | 0.08 | 0.206 | 12,987 | Pass |
| 9 | 156 | linear | 0.12 | 0.207 | 12,089 | Pass |
| 10 | 104 | sqrt | 0.10 | 0.215 | 13,421 | Pass |

### Table S4: Linear vs Square-Root Scaling Comparison

| Domain | Linear (0.1) Cov Err | Sqrt (0.1) Cov Err | Winner |
|--------|---------------------|-------------------|--------|
| M4 Monthly | 0.178 | 0.195 | Linear |
| M4 Weekly | 0.197 | 0.221 | Linear |
| Finance | 0.204 | 0.198 | Sqrt |
| Flu US | 0.121 | 0.145 | Linear |
| M4 Daily | 0.204 | 0.228 | Linear |
| NYISO | 0.099 | 0.112 | Linear |
| **Average** | **0.167** | **0.183** | **Linear** |

---

## S3. Rejected Alternative Methods

### S3.1 Gaussian Quantiles (V3)

Method: Use normal distribution quantiles with MAD-estimated scale.

```
scale = 1.4826 * MAD(residuals)
q(alpha) = point + z(alpha) * scale * (1 + 0.1*h)
```

Results:
- Coverage at 80%: 94.2% (acceptable)
- Coverage at 90%: 97.8% (acceptable)
- WIS: 31,250 (2.5x worse than empirical)

Rejection reason: Excessive interval width due to light tails assumption.

### S3.2 Laplace Quantiles (V4)

Method: Use Laplace distribution quantiles.

```
b = MAD(residuals) / log(2)
q(alpha) = point + Laplace_quantile(alpha, b) * (1 + 0.1*h)
```

Results:
- Coverage at 80%: 67.3% (severe under-coverage)
- Coverage at 90%: 78.1% (severe under-coverage)
- WIS: 11,892 (best WIS)

Rejection reason: Unacceptable under-coverage despite best WIS.

### S3.3 Width Factor Method with 1.645 Multiplier (V2)

Method: Pre-trained width factors multiplied by 1.645.

Results:
- Coverage at 80%: 99.1% (excessive over-coverage)
- Coverage at 90%: 99.8% (excessive over-coverage)
- Intervals 65% wider than necessary

Rejection reason: Implementation bug caused systematic over-widening.

### S3.4 Log-Space for Count Data (V2_Count)

Method: Transform to log space, apply intervals, transform back.

```
log_y = log(y + 1)
log_intervals = empirical_quantiles(log_y)
intervals = exp(log_intervals) - 1
```

Results on M5 Retail:
- Coverage at 90%: 49.3% (severe under-coverage)

Rejection reason: Log transformation inappropriate for zero-inflated data.

---

## S4. Domain-Specific Diagnostic Analyses

### S4.1 Wikipedia Traffic

**Marginal Coverage (90.3% at 90% nominal):** High volatility and trend changes

Wikipedia achieves only marginal coverage, just barely meeting the 90% threshold. At 80% nominal, it drops to 81.9%—the second-worst domain after M5 Retail.

Wikipedia page view series exhibit:
- Sudden viral spikes (news events, trending topics)
- Rapid decay after peak interest
- Non-stationary patterns that differ from recent history

Level-based residuals underestimate uncertainty when recent history is calm but future holds volatility spikes. At 95% nominal, Wikipedia under-covers (93.2% vs 95% target).

### S4.2 M5 Retail

**Root Cause of Under-Coverage (82.4% at 90% nominal):** Zero-inflated count data

Characteristics of M5 retail series:
- 47% of observations are zero
- High coefficient of variation (>5.0)
- Intermittent demand pattern

Level-based residuals are inappropriate because:
1. Mean is pulled down by zeros
2. Positive observations appear as large positive residuals
3. Intervals are asymmetric around zero

Conformal prediction achieves 96.3% coverage on this domain, demonstrating that error-based calibration handles zero-inflation better than residual-based approaches.

### S4.3 High-Coverage Domains

Several domains achieve 100% coverage at 90% nominal (Flu US, Covid Hosp, Bass Tech, NYISO), indicating over-coverage. This conservative behavior arises from:
- Heavy-tailed residual distributions
- High variability in historical data relative to forecast horizons
- Persistence point forecast naturally captures mean-reverting behavior

While over-coverage wastes interval width (reduced sharpness), it ensures safety for risk-sensitive applications.

---

## S5. Weighted Interval Score Formula

The Weighted Interval Score (WIS) used in CDC FluSight combines sharpness (interval width) and calibration (coverage):

$$\text{WIS} = \frac{1}{K+0.5} \left[ \sum_{k=1}^{K} \frac{\alpha_k}{2} \text{IS}_{\alpha_k} + 0.5 |y - q_{0.5}| \right]$$

where for each interval level $\alpha_k$:

$$\text{IS}_{\alpha_k} = (u_k - l_k) + \frac{2}{\alpha_k}(l_k - y)\mathbf{1}_{y < l_k} + \frac{2}{\alpha_k}(y - u_k)\mathbf{1}_{y > u_k}$$

Components:
- $(u_k - l_k)$: Sharpness penalty (wider = worse)
- Under-prediction penalty: Activated when $y < l_k$
- Over-prediction penalty: Activated when $y > u_k$

The 0.5 weight on median absolute error ensures point forecast quality contributes to the score.

---

## S6. Data Sources and Availability

### Table S5: Dataset Sources

| Domain | Source | Access | License |
|--------|--------|--------|---------|
| Flu US | CDC FluSight | Public | Public Domain |
| Covid Hosp | HHS | Public | Public Domain |
| Bass Tech | Various | Derived | Academic |
| Finance | Yahoo Finance | Public | Fair Use |
| M4 Weekly | M4 Competition | Public | CC BY |
| M4 Daily | M4 Competition | Public | CC BY |
| M4 Monthly | M4 Competition | Public | CC BY |
| Kaggle Store | Kaggle | Public | CC BY-NC |
| NYISO Load | NYISO | Public | Public |
| Wikipedia | Wikimedia | Public | CC BY-SA |
| M5 Retail | M5 Competition | Public | Academic |

### M4 Series Selection

The M4 Competition contains thousands of series per frequency. We use a deterministic subset for computational tractability:

| Dataset | Total in M4 | Used | Selection Criteria |
|---------|-------------|------|-------------------|
| M4 Weekly | 359 | 150 | First 150 series with ≥50 observations |
| M4 Daily | 4,227 | 100 | First 100 series with ≥50 observations |
| M4 Monthly | 48,000 | 100 | First 100 series with ≥30 observations |

Selection is deterministic (no random seed) based on file ordering. Series are read in order from the original M4 Competition CSV files and filtered by minimum length. This ensures reproducibility without random sampling.

### Data Processing

1. **Missing values:** Forward-filled then backward-filled
2. **Outliers:** No removal (empirical quantiles are robust)
3. **Seasonality:** Not explicitly modeled (captured in residuals)
4. **Normalization:** None required

---

## S7. Computational Requirements

| Operation | Complexity | Time (100 series) |
|-----------|------------|-------------------|
| Residual computation | O(L) | <1ms |
| Quantile estimation | O(L log L) | <1ms |
| Single forecast | O(L log L) | <1ms |
| Full evaluation (11 domains) | O(N × H × L log L) | ~30 seconds |

Where:
- L = lookback window (104)
- N = number of series
- H = forecast horizons

Memory: <10MB for full evaluation pipeline.

---

## S8. Code Availability

The complete implementation is available at:

```
Interval-Research/
├── code/
│   └── empirical_quantile_intervals.py  # Canonical implementation
├── manuscript/
│   └── PAPER_DRAFT.md                   # Full paper
└── supplementary/
    └── supplementary_materials.md       # This document
```

Reproducibility requirements:
- Python 3.8+
- NumPy 1.20+
- Pandas 1.3+ (for evaluation scripts)
