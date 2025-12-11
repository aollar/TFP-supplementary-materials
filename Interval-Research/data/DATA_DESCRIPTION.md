# Data Description

**Paper:** A Simple Three-Parameter Method for Cross-Domain Probabilistic Forecasting

---

## Overview

The evaluation uses 11 datasets spanning diverse forecasting domains. This document describes each dataset, its source, and preprocessing applied.

---

## Dataset Summaries

### 1. Flu US (CDC FluSight)

| Property | Value |
|----------|-------|
| Series | 53 (US national + 52 jurisdictions) |
| Frequency | Weekly |
| Observations | ~500 per series |
| Metric | Weighted ILI percentage |
| Source | CDC FluSight Challenge |
| Years | 2010-2024 |

**Characteristics:**
- Strong seasonal pattern (winter peaks)
- High coefficient of variation (7.24)
- 9.2% zero observations
- Right-skewed during epidemics

### 2. Covid Hospitalizations (HHS)

| Property | Value |
|----------|-------|
| Series | 50 (US states) |
| Frequency | Weekly |
| Observations | ~150 per series |
| Metric | Hospital admissions |
| Source | HHS Protect |
| Years | 2020-2023 |

**Characteristics:**
- Multiple waves with different magnitudes
- Non-stationary mean
- High volatility during surges

### 3. Bass Technology Adoption

| Property | Value |
|----------|-------|
| Series | 5 (technology products) |
| Frequency | Annual |
| Observations | 30-50 per series |
| Metric | Adoption percentage |
| Source | US Census, industry reports |
| Years | 1920-2020 |

**Characteristics:**
- S-curve adoption patterns
- Monotonically increasing (until saturation)
- Smooth trajectories

Technologies included:
- Automobile, Electricity (Electric Power)
- Air Conditioning, Refrigerator, Landline (Telephone)

### 4. Finance (Dow 30)

| Property | Value |
|----------|-------|
| Series | 35 (stocks + indices) |
| Frequency | Daily |
| Observations | ~2,500 per series |
| Metric | Adjusted close price |
| Source | Yahoo Finance |
| Years | 2014-2024 |

**Characteristics:**
- Random walk behavior
- Heteroskedastic volatility
- Fat tails

### 5. M4 Weekly

| Property | Value |
|----------|-------|
| Series | 150 (sampled) |
| Frequency | Weekly |
| Observations | 80-2,500 per series |
| Metric | Various |
| Source | M4 Competition |
| Years | Various |

**Characteristics:**
- Mixed domains (macro, micro, demographic)
- Varying seasonality patterns
- Different scales

### 6. M4 Daily

| Property | Value |
|----------|-------|
| Series | 100 (sampled) |
| Frequency | Daily |
| Observations | 100-9,900 per series |
| Metric | Various |
| Source | M4 Competition |
| Years | Various |

### 7. M4 Monthly

| Property | Value |
|----------|-------|
| Series | 100 (sampled) |
| Frequency | Monthly |
| Observations | 48-2,800 per series |
| Metric | Various |
| Source | M4 Competition |
| Years | Various |

### 8. Kaggle Store Sales

| Property | Value |
|----------|-------|
| Series | 100 (sampled store-items) |
| Frequency | Daily |
| Observations | ~1,800 per series |
| Metric | Unit sales |
| Source | Kaggle Competition |
| Years | 2013-2017 |

**Characteristics:**
- Day-of-week seasonality
- Promotional effects
- Some zero sales days

### 9. NYISO Energy Load

| Property | Value |
|----------|-------|
| Series | 11 (load zones) |
| Frequency | Weekly (aggregated from hourly) |
| Observations | ~500 per series |
| Metric | MWh demand |
| Source | NYISO |
| Years | 2014-2024 |

**Characteristics:**
- Strong seasonal pattern (summer/winter peaks)
- **Negative lag-13 autocorrelation** (-0.501 median)
- Weather-driven volatility

### 10. Wikipedia Traffic

| Property | Value |
|----------|-------|
| Series | 100 (sampled pages) |
| Frequency | Daily |
| Observations | ~550 per series |
| Metric | Page views |
| Source | Wikimedia |
| Years | 2015-2017 |

**Characteristics:**
- Bursty traffic patterns
- Day-of-week effects
- Occasional viral spikes

### 11. M5 Retail

| Property | Value |
|----------|-------|
| Series | 100 (sampled store-items) |
| Frequency | Daily |
| Observations | ~1,900 per series |
| Metric | Unit sales |
| Source | M5 Competition (Walmart) |
| Years | 2011-2016 |

**Characteristics:**
- **47% zero observations** (intermittent demand)
- High coefficient of variation
- Promotional effects

---

## Preprocessing

### Standard Pipeline

1. **Missing values:**
   - Forward-fill from last valid observation
   - Backward-fill for leading missing values
   - No interpolation

2. **Outliers:**
   - No removal (empirical quantiles are robust)
   - Extreme values retained for realistic evaluation

3. **Transformations:**
   - None applied (method operates on raw scale)
   - Bounds applied post-hoc for non-negative series

### Domain-Specific Notes

**Flu US:** ILI percentage can be zero during summer; retained as-is.

**M5 Retail:** Zero sales retained; no pseudo-count added.

**Finance:** Adjusted for splits and dividends via Yahoo Finance.

---

## Evaluation Windows

| Domain | Train Min | Forecast Horizons | Windows |
|--------|-----------|-------------------|---------|
| Flu US | 104 | 1, 2, 4 | 10 per series |
| Covid Hosp | 52 | 1, 2, 4 | 10 per series |
| Bass Tech | 20 | 1, 2, 4 | ~8 per series |
| Finance | 252 | 1, 5, 10 | 10 per series |
| M4 Weekly | 104 | 1, 2, 4, 8 | 10 per series |
| M4 Daily | 104 | 1, 7, 14 | 10 per series |
| M4 Monthly | 48 | 1, 3, 6 | 10 per series |
| Kaggle Store | 365 | 1, 7, 14 | 10 per series |
| NYISO Load | 104 | 1, 2, 4 | 10 per series |
| Wikipedia | 365 | 1, 7, 14 | 10 per series |
| M5 Retail | 365 | 1, 7, 14 | 10 per series |

*Note: Each domain uses a sparse subset of horizons appropriate to its frequency. Maximum horizon evaluated is h=14.*

---

## Data Availability

| Dataset | Access | URL |
|---------|--------|-----|
| Flu US | Public | https://github.com/cdcepi/Flusight-forecast-data |
| Covid Hosp | Public | https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Capa/g62h-syeh |
| M4 | Public | https://github.com/Mcompetitions/M4-methods |
| M5 | Public | https://www.kaggle.com/c/m5-forecasting-accuracy |
| NYISO | Public | https://www.nyiso.com/load-data |
| Wikipedia | Public | https://www.kaggle.com/c/web-traffic-time-series-forecasting |
| Kaggle Store | Public | https://www.kaggle.com/c/store-sales-time-series-forecasting |

---

## Citation

If using these datasets, please cite the original sources:

```bibtex
@article{makridakis2020m4,
  title={The M4 Competition: 100,000 time series and 61 forecasting methods},
  author={Makridakis, Spyros and Spiliotis, Evangelos and Assimakopoulos, Vassilios},
  journal={International Journal of Forecasting},
  volume={36},
  number={1},
  pages={54--74},
  year={2020}
}

@article{reich2019collaborative,
  title={A collaborative multiyear, multimodel assessment of seasonal influenza forecasting in the United States},
  author={Reich, Nicholas G and others},
  journal={Proceedings of the National Academy of Sciences},
  volume={116},
  number={8},
  pages={3146--3154},
  year={2019}
}
```
