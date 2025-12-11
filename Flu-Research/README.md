# TFP v2.2 vs FluSight Ensembles on Influenza Forecasting

**Austin Ollar** | Independent Researcher | austin@austinollar.com

**Research Package for Peer Review**

---

## Quick Summary

This paper evaluates TFP v2.2—a theory-driven generalist forecaster—against CDC FluSight Ensemble and UMass-Flusion ensemble methods on **US state-level influenza hospitalization data** from two post-COVID seasons (2023-2024 and 2024-2025).

**Key Results:**
- TFP reduces H1 (one-week-ahead) MAE by **37%** vs both FluSight and UMass ensembles (p < 0.001)
- TFP reduces H1 MAE by **69%** vs Seasonal Naive
- At H2, TFP maintains **8-12%** MAE advantage
- At H3, performance is **statistically comparable** to ensembles
- Same frozen configuration works across 11 diverse domains (no flu-specific tuning)

**Main Finding:** A single, theory-driven generalist forecaster achieves 37% lower one-week-ahead MAE than operational ensemble methods on US flu data, without requiring multi-model infrastructure.

---

## Repository Structure

```
Flu-Research/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── manuscript/
│   └── PAPER_DRAFT.md                  # Full paper manuscript
├── code/
│   ├── flu_states_publication_eval.py  # Main evaluation script
│   └── verify_umass_data.py            # Data verification script
├── data/
│   ├── FINAL - Flusight-ensemble-2023-2024-2025_combined.csv
│   ├── UMass.fix.csv
│   └── target-hospital-admissions-NEW.csv
├── raw_data/
│   └── [UMass weekly submission files]
└── results/
    ├── FLU_STATES_PUBLICATION_RESULTS_intervalLawAll.csv
    ├── FLU_STATES_BOOTSTRAP_CI_intervalLawAll.csv
    ├── FLU_STATES_PER_STATE_RATIOS_intervalLawAll.csv
    └── FLU_STATES_PUBLICATION_SUMMARY.md
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the evaluation
python code/flu_states_publication_eval.py
```

---

## Key Results

### H1 Performance (One-Week-Ahead)

| Comparison | MAE Ratio | Improvement | p-value |
|------------|-----------|-------------|---------|
| TFP vs FluSight | 0.63 | **+37%** | < 0.001 |
| TFP vs UMass | 0.63 | **+37%** | < 0.001 |
| TFP vs Seasonal Naive | 0.31 | **+69%** | < 0.001 |

### Performance by Horizon

| Horizon | TFP vs FluSight | TFP vs UMass |
|---------|-----------------|--------------|
| H1 (1 week) | **+37%** | **+37%** |
| H2 (2 weeks) | **+8-12%** | **+8-12%** |
| H3 (3 weeks) | ~0% (neutral) | -4% |

---

## Data Sources

- **FluSight Ensemble**: CDC FluSight collaborative forecasting initiative
- **UMass-Flusion**: University of Massachusetts Amherst ensemble model (Gibson et al., 2024)
- **Target Data**: US state-level influenza hospitalizations from HHS Protect

---

## Cross-Domain Provenance

TFP v2.2 uses the **same frozen configuration** validated across 11 diverse forecasting domains:
- Epidemiological forecasting (US flu, COVID hospitalizations)
- Technology adoption (Bass diffusion curves)
- Energy load (NYISO electricity demand)
- Retail demand (M4 competition subsets, M5 retail, Kaggle retail)
- Web traffic (Wikipedia page views)
- Financial time series

Flu was part of the cross-domain tuning pool, but no flu-specific optimization occurred after v2.2 was frozen.

---

## Citation

If you use this work, please cite:

> Ollar, A. (2025). A Theory-Driven Generalist Forecaster for Influenza: TFP Cuts One-Week-Ahead Flu MAE by 37% Compared With FluSight and UMass Ensembles.

---

## License

Code and data are available from the author upon reasonable request.

## Contact

Austin Ollar - austin@austinollar.com
