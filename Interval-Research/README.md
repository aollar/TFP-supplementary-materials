# Empirical Residual Scaling for Cross-Domain Probabilistic Forecasting

**Austin Ollar** | Independent Researcher | austin@austinollar.com

**Research Package for Peer Review**

---

## Quick Summary

This paper presents the Empirical Residual Scaling (ERS) rule—a simple, non-parametric method for generating prediction intervals—and compares it against split conformal prediction across **11 forecasting domains** with **25,593 forecast instances**.

**Key Results:**
- ERS achieves **94.9% coverage** at 90% nominal [95% CI: 94.7–95.2%]
- Conformal achieves **95.6% coverage** with **32% lower WIS** (p<0.001)
- 3 frozen parameters work across all domains (no per-domain tuning)
- ERS has calibration asymmetry (3.0% at 5% nominal); conformal better at middle quantiles

**Main Finding:** Across 25,593 forecast instances from 11 datasets, a single linear scaling rule for residual quantiles, tuned once on a dev set, achieves 94.9% empirical coverage at the 90% nominal level, with no per-domain or per-series adjustment.

---

## Repository Structure

```
Interval-Research/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── manuscript/
│   └── PAPER_DRAFT.md                  # Full paper manuscript
├── code/
│   ├── empirical_quantile_intervals.py # ERS implementation (FROZEN)
│   ├── conformal_baseline.py           # Split conformal implementation
│   └── run_full_11_domain_evaluation.py # Main evaluation script
├── supplementary/
│   ├── supplementary_materials.md      # Extended tables and analyses
│   ├── grid_search_results.csv         # Parameter search results
│   └── method_selection_rationale.md   # Method selection documentation
└── data/
    └── DATA_DESCRIPTION.md             # Dataset sources and preprocessing
```

---

## The ERS Rule (3 Lines of Code)

```python
def empirical_quantile_intervals(y, point, horizon, lookback=104, scale=0.1):
    residuals = y[-lookback:] - np.mean(y[-lookback:])
    factor = 1.0 + scale * horizon
    return {q: point + np.quantile(residuals, q) * factor for q in QUANTILES}
```

**Frozen Parameters:**
- `lookback = 104` (~2 years of weekly data)
- `scale = 0.1` (10% uncertainty growth per horizon step)
- Point forecast = persistence (`y_T`)

---

## Results Summary

### Coverage Comparison (90% Nominal)

| Domain | ERS | Conformal | n |
|--------|-----|-----------|---|
| Flu US | 100.0% | 100.0% | 1,590 |
| Covid Hosp | 100.0% | 100.0% | 1,500 |
| Bass Tech | 100.0% | 72.4% | 123 |
| Finance | 99.0% | 94.4% | 1,050 |
| NYISO | 100.0% | 96.1% | 330 |
| M4 Weekly | 97.0% | 94.5% | 6,000 |
| M4 Daily | 97.1% | 92.2% | 3,000 |
| M4 Monthly | 97.1% | 95.7% | 3,000 |
| Kaggle Store | 96.2% | 99.0% | 3,000 |
| Wikipedia | 90.3% | 93.8% | 3,000 |
| M5 Retail | 82.4% | 96.3% | 3,000 |
| **Average** | **94.9%** | **95.6%** | **25,593** |

### Key Findings

1. **ERS provides robust coverage** across diverse domains with zero per-domain tuning
2. **Conformal is better calibrated** at middle quantiles and produces sharper intervals
3. **ERS fails on zero-inflated data** (M5 Retail: 82.4% vs conformal's 96.3%)
4. **Conformal fails on short series** (Bass Tech: 72.4% vs ERS's 100%)

---

## Recommendations

| Use Case | Recommended Method |
|----------|-------------------|
| Simple coverage sufficient | ERS (simpler implementation) |
| Distributional calibration required | Split conformal |
| Zero-inflated count data | Split conformal |
| Very short series (<30 points) | ERS |

---

## Reproducibility

### Requirements
```
numpy>=1.20.0
pandas>=1.3.0
```

### Running the Evaluation
```bash
cd Interval-Research/code
python run_full_11_domain_evaluation.py
```

---

## File Descriptions

### Code
1. **`code/empirical_quantile_intervals.py`** - Frozen ERS implementation
2. **`code/conformal_baseline.py`** - Split conformal implementation
3. **`code/run_full_11_domain_evaluation.py`** - Main evaluation script

### Results
4. **`code/full_11domain_raw_forecasts.csv`** - All 25,593 forecasts (×2 methods)
5. **`code/full_11domain_reliability.csv`** - Calibration analysis

### Documentation
6. **`manuscript/PAPER_DRAFT.md`** - Complete paper
7. **`supplementary/supplementary_materials.md`** - Extended results
8. **`data/DATA_DESCRIPTION.md`** - Dataset specifications

---

## Citation

[To be added upon publication]

---

## License

Academic use permitted. See individual dataset licenses in `data/DATA_DESCRIPTION.md`.
