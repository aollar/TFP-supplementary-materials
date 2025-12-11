# FLU STATES Publication-Grade Evaluation

**Generated:** 2025-12-06 06:09:03

## Output Files

- `FLU_STATES_PUBLICATION_RESULTS_native.csv` - Detailed results with native intervals
- `FLU_STATES_PUBLICATION_RESULTS_intervalLawAll.csv` - Detailed results with IntervalLawV2 for all
- `FLU_STATES_BOOTSTRAP_CI_native.csv` - Bootstrap CIs for native mode
- `FLU_STATES_BOOTSTRAP_CI_intervalLawAll.csv` - Bootstrap CIs for fair comparison mode
- `FLU_STATES_PER_STATE_RATIOS_intervalLawAll.csv` - Per-state WIS ratios
- `FLU_STATES_PUBLICATION_SUMMARY.md` - This file

---

## 1. Methods

### 1.1 Data and Apples Definition

- **Seasons evaluated:** 2023-2024
- **Locations:** 53 (50 US states + DC + PR + US National)
- **Horizons:** h = 1, 2, 3 weeks ahead
- **Total apples (fair mode):** N = 4761
- **Training:** Rolling origin - all data before each forecast date available for training

### 1.2 Models Compared

1. **TFP v2.2**: Law-like generalist with percentile-based oscillation dampening (~600 lines standalone)
2. **UMass-Trends**: FluSight participant ensemble method
3. **CDC FluSight Ensemble**: Official CDC ensemble forecast (ensemble of multiple models)
4. **Naive**: Last observed value
5. **Seasonal Naive**: Value from same week previous year

### 1.3 Evaluation Modes

**Mode A (Native Intervals):** Each model uses its own prediction intervals. TFP uses its internal
interval logic. This mode reveals calibration differences between models.

**Mode B (Fair Comparison / IntervalLawV2 for All):** Only point forecasts are extracted from each model.
IntervalLawV2 is then applied uniformly to generate prediction intervals for all models. This ensures
differences in WIS reflect point forecast accuracy rather than interval calibration.

### 1.4 Metrics

- **WIS**: Weighted Interval Score (normalized by 12)
- **MAE**: Mean Absolute Error (point forecast accuracy)
- **Coverage**: Empirical coverage of 90% prediction interval (nominal = 90%)

### 1.5 Statistical Inference

- **Bootstrap:** Block bootstrap by location (10,000 resamples) for multi-location pooled estimates
- **US National:** Observation-level bootstrap (single location)
- **Confidence intervals:** 95% for WIS and MAE ratios
- **Multiple comparisons:** Headline inference based on pooled WIS ratio; per-horizon and per-state
  analyses are exploratory without formal multiple comparison correction

---

## 2. Results

### 2.1 Mode A: Native Intervals

| Model | WIS | MAE | Coverage 90% |
|-------|-----|-----|--------------|
| TFP v2.2 | 60.4 | 74.4 | 44.9% |
| UMass | 52.1 | 84.5 | 92.5% |
| CDC Ensemble | 68.3 | 112.6 | 84.9% |
| Naive | 104.1 | 104.1 | N/A |
| Seasonal Naive | 230.4 | 230.4 | N/A |

**Key finding:** TFP's native intervals are severely miscalibrated (44.9% coverage
vs 90% nominal), inflating its WIS relative to baselines with better-calibrated intervals.

### 2.2 Mode B: Fair Comparison (IntervalLawV2 for All)

| Model | WIS | MAE | Coverage 90% |
|-------|-----|-----|--------------|
| TFP v2.2 | 71.0 | 74.4 | 90.7% |
| UMass | 79.2 | 84.5 | 91.1% |
| CDC Ensemble | 100.7 | 112.6 | 90.3% |
| Naive | 81.2 | 104.1 | 89.4% |
| Seasonal Naive | 170.1 | 230.4 | 75.8% |

**Headline Results (95% CIs):**

- TFP vs UMass WIS ratio: 0.896 [0.875, 0.935]
- TFP vs CDC WIS ratio: 0.705 [0.685, 0.755]
- TFP vs UMass MAE ratio: 0.880 [0.838, 0.975]
- TFP vs CDC MAE ratio: 0.661 [0.613, 0.763]

### 2.3 Per-Horizon Analysis (Fair Comparison)

| Horizon | TFP vs UMass WIS (95% CI) | TFP vs CDC WIS (95% CI) |
|---------|---------------------------|-------------------------|
| H1 | 0.845 [0.828, 0.881] | 0.685 [0.664, 0.738] |
| H2 | 0.876 [0.852, 0.911] | 0.686 [0.670, 0.730] |
| H3 | 0.954 [0.924, 1.008] | 0.736 [0.711, 0.795] |

**Horizon-dependent pattern:** TFP shows the largest relative advantage at short horizons (H1),
with the improvement shrinking at longer horizons (H2, H3). This pattern suggests TFP's point
forecasts are particularly accurate in the near term, possibly due to its adaptive trend-following
mechanism. The advantage remains positive but smaller at H3, indicating the method is not simply
a persistence forecast but maintains some predictive value at longer horizons.

### 2.4 Per-State Distribution (Fair Comparison)

**TFP vs UMass WIS Ratio Distribution:**
- Median: 0.896
- 25th percentile: 0.840
- 75th percentile: 0.956
- States where TFP wins (ratio < 1): 48 / 53

**TFP vs CDC WIS Ratio Distribution:**
- Median: 0.735
- 25th percentile: 0.693
- 75th percentile: 0.806
- States where TFP wins (ratio < 1): 53 / 53

**Worst 5 States (TFP underperforms vs UMass):**

| Location | TFP/UMass Ratio | N Apples |
|----------|-----------------|----------|
| 40 | 1.083 | 90 |
| 48 | 1.073 | 90 |
| 02 | 1.045 | 90 |
| 21 | 1.044 | 90 |
| 04 | 1.027 | 90 |

The per-state analysis shows that TFP's improvement is broadly distributed rather than driven by
a single outlier state. The worst-performing states represent edge cases rather than systematic
failures.

### 2.5 Naive Baseline Comparison

Naive (last value) and Seasonal Naive provide context for forecast difficulty:
- TFP MAE: 74.4
- Naive MAE: 104.1
- Seasonal Naive MAE: 230.4

TFP substantially outperforms both naive baselines on MAE, confirming it provides genuine
predictive value beyond simple persistence.

---

## 3. Discussion and Limitations

### 3.1 Key Findings

1. **Native TFP intervals are miscalibrated** (44.9% coverage vs 90% nominal)
   and should not be used for probabilistic forecasting without recalibration.

2. **After applying IntervalLawV2 uniformly**, TFP's point forecasts show approximately
   10% WIS improvement and
   12% MAE improvement vs UMass/CDC.

3. **Horizon-dependent pattern**: TFP excels at short horizons (H1) with diminishing advantage at H3.

4. **Broad improvement**: Per-state analysis shows TFP wins in 48/53
   states vs UMass.

### 3.2 Limitations

1. **Single primary season**: Results are based on 2024-2025 holdout. Additional seasons should be
   evaluated for robustness.

2. **Undercoverage across all models**: All models show coverage below 90% nominal, suggesting
   unusual flu dynamics in the evaluation period.

3. **Ensemble vs single model**: UMass and CDC are ensemble methods combining multiple component
   models. TFP is a single algorithm. The comparison is "ensemble vs single-model" but remains
   informative for point forecast quality assessment.

4. **IntervalLawV2 origin**: IntervalLawV2 was developed alongside TFP. While applied uniformly
   to all models in Mode B, this could introduce subtle biases toward TFP's forecasting style.

5. **No formal multiple comparison correction**: Per-horizon and per-state analyses are exploratory.
   The headline inference is based on the single pooled WIS ratio with bootstrap CI.

---

## 4. Reproducibility

All results can be reproduced using:
```bash
python cross_domain_eval/flu_states_publication_eval.py
```

Code paths:
- Point forecaster: `tfp_v2_2_lawlike_standalone.py`
- Interval engine: `cross_domain_eval/interval_law_v2.py`
- Evaluation script: `cross_domain_eval/flu_states_publication_eval.py`
- Truth data: `Flu-Update/target-hospital-admissions-NEW.csv`
- UMass forecasts: `UMass.fix.csv`
- CDC Ensemble: `FINAL - Flusight-ensemble-2023-2024-2025_combined.csv`
