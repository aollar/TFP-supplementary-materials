# TFP v2.2 vs Bass Diffusion Model on Technology Adoption

**Austin Ollar** | Independent Researcher | austin@austinollar.com

**Research Package for Peer Review**

---

## Quick Summary

This paper evaluates TFP v2.2—a theory-driven generalist forecaster—against classical technology adoption models (Bass, Gompertz, Logistic) and statistical baselines (SimpleTheta, Naive) on **21 US household technology adoption curves** across **840 evaluation windows**.

**Key Results:**
- TFP reduces MAE by **34%** vs Bass [95% CI: 0.514–0.901]
- TFP reduces MAE by **35%** vs SimpleTheta [95% CI: 0.540–0.801]
- TFP wins on **16 of 21 technologies** (76%)
- Same frozen configuration works across 11 diverse domains (no adoption-specific tuning)

**Main Finding:** A single, theory-driven generalist forecaster achieves one-third lower MAE than the Bass diffusion model on US technology adoption curves, using a fixed configuration with no adoption-specific tuning.

---

## Repository Structure

```
Bass-Research/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── manuscript/
│   └── PAPER_DRAFT.md                  # Full paper manuscript
├── code/
│   ├── tfp_v2_2_lawlike_standalone.py  # TFP forecaster (~3,800 lines)
│   ├── tfp_v2_2_theory_aligned.py      # IEC-explicit refactor (~4,800 lines)
│   ├── bass_revalidation_v22.py        # Main evaluation script
│   ├── interval_law_v2.py              # Interval construction
│   ├── strong_simple_theta.py          # SimpleTheta baseline
│   ├── generate_figure.py              # Figure generation script
│   └── generate_bootstrap_ci.py        # Bootstrap CI generation script
├── data/
│   └── technology-adoption-by-households-in-the-united-states.csv
├── figures/
│   └── figure1_mae_comparison.png      # Figure 1: MAE comparison
└── results/
    ├── bass_revalidation_raw.csv       # All 840 evaluation windows
    ├── bass_revalidation_tech_summary.csv  # Per-technology summary
    ├── bass_bootstrap_ci_10k.csv       # Bootstrap CIs (10,000 resamples)
    ├── bass_regime_bootstrap_ci_10k.csv # Regime-level bootstrap CIs
    └── bass_matched_window_summary.csv # Matched-window MAE comparison
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the evaluation
python code/bass_revalidation_v22.py

# Generate bootstrap CIs
python code/generate_bootstrap_ci.py

# Generate Figure 1
python code/generate_figure.py
```

---

## Key Results

### Overall Performance (840 Windows)

| Model | MAE (%) | N Windows |
|-------|---------|-----------|
| **TFP v2.2** | **2.81** | 840 |
| Naive | 3.13 | 840 |
| SimpleTheta | 4.35 | 840 |
| Logistic | 5.43 | 513 |
| Gompertz | 5.66 | 513 |
| Bass | 6.36 | 513 |

### Bootstrap Confidence Intervals (10,000 Resamples)

| Comparison | MAE Ratio | 95% CI | Improvement |
|------------|-----------|--------|-------------|
| TFP vs Bass | 0.658 | [0.514, 0.901] | **+34.2%** |
| TFP vs SimpleTheta | 0.646 | [0.540, 0.801] | **+35.4%** |
| TFP vs Naive | 0.897 | [0.835, 0.960] | **+10.3%** |

All confidence intervals exclude 1.0, confirming statistical significance.

---

## Technologies Where TFP Excels vs Bass Wins

**TFP wins (16):** Vacuum (+80%), Household refrigerator (+77%), Stove (+75%), Automobile (+71%), Electric power (+68%), Washing machine (+60%), Refrigerator (+54%), Power steering (+46%), Dishwasher (+38%), Home air conditioning (+35%), Radio (+32%), RTGS adoption (+30%), Automatic transmission (+7%), Social media usage (+1%), Microcomputer (+1%), Cable TV (+0.5%)

**Bass wins (5):** Colour TV (-81%), Internet (-47%), Cellular phone (-36%), Shipping container (-8%), NOx pollution controls (-8%)

---

## Cross-Domain Provenance

TFP v2.2 uses the **same frozen configuration** validated across 11 diverse forecasting domains:
- Epidemiological forecasting (US flu, COVID hospitalizations)
- Technology adoption (Bass diffusion curves)
- Energy load (NYISO electricity demand)
- Retail demand (M4 competition subsets, M5 retail, Kaggle retail)
- Web traffic (Wikipedia page views)
- Financial time series

No Bass-specific or diffusion-specific tuning occurred.

---

## Citation

If you use this work, please cite:

> Ollar, A. (2025). A Theory-Driven Generalist Forecaster Cuts MAE by One Third Versus Bass and Classical Diffusion Models on 21 Technology Adoption Curves.

---

## License

Code and data are available from the author upon reasonable request.

## Contact

Austin Ollar - austin@austinollar.com
