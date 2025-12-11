# FLU STATES Publication-Grade Evaluation

**Generated:** 2025-12-06 06:18:24

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

- **Seasons evaluated:** 2024-2025
- **Locations:** 53 (50 US states + DC + PR + US National)
- **Horizons:** h = 1, 2, 3 weeks ahead
- **Total apples (fair mode):** N = 4119
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
| TFP v2.2 | 211.7 | 245.7 | 49.1% |
| UMass | 195.8 | 274.4 | 76.4% |
| CDC Ensemble | 198.3 | 282.1 | 76.1% |
| Naive | 306.2 | 306.2 | N/A |
| Seasonal Naive | 414.4 | 414.4 | N/A |

**Key finding:** TFP's native intervals are severely miscalibrated (49.1% coverage
vs 90% nominal), inflating its WIS relative to baselines with better-calibrated intervals.

### 2.2 Mode B: Fair Comparison (IntervalLawV2 for All)

| Model | WIS | MAE | Coverage 90% |
|-------|-----|-----|--------------|
| TFP v2.2 | 183.4 | 245.7 | 77.3% |
| UMass | 225.3 | 274.4 | 76.6% |
| CDC Ensemble | 229.8 | 282.1 | 76.2% |
| Naive | 214.5 | 306.2 | 69.2% |
| Seasonal Naive | 333.8 | 414.4 | 75.7% |

**Headline Results (95% CIs):**

- TFP vs UMass WIS ratio: 0.814 [0.779, 0.831]
- TFP vs CDC WIS ratio: 0.798 [0.771, 0.813]
- TFP vs UMass MAE ratio: 0.896 [0.821, 0.930]
- TFP vs CDC MAE ratio: 0.871 [0.812, 0.901]

### 2.3 Per-Horizon Analysis (Fair Comparison)

| Horizon | TFP vs UMass WIS (95% CI) | TFP vs CDC WIS (95% CI) |
|---------|---------------------------|-------------------------|
| H1 | 0.645 [0.606, 0.663] | 0.660 [0.624, 0.676] |
| H2 | 0.792 [0.764, 0.815] | 0.772 [0.752, 0.801] |
| H3 | 0.950 [0.903, 0.978] | 0.910 [0.875, 0.933] |

**Horizon-dependent pattern:** TFP shows the largest relative advantage at short horizons (H1),
with the improvement shrinking at longer horizons (H2, H3). This pattern suggests TFP's point
forecasts are particularly accurate in the near term, possibly due to its adaptive trend-following
mechanism. The advantage remains positive but smaller at H3, indicating the method is not simply
a persistence forecast but maintains some predictive value at longer horizons.

### 2.4 Per-State Distribution (Fair Comparison)

**TFP vs UMass WIS Ratio Distribution:**
- Median: 0.814
- 25th percentile: 0.760
- 75th percentile: 0.877
- States where TFP wins (ratio < 1): 50 / 53

**TFP vs CDC WIS Ratio Distribution:**
- Median: 0.826
- 25th percentile: 0.777
- 75th percentile: 0.860
- States where TFP wins (ratio < 1): 52 / 53

**Worst 5 States (TFP underperforms vs UMass):**

| Location | TFP/UMass Ratio | N Apples |
|----------|-----------------|----------|
| 25 | 1.039 | 78 |
| 11 | 1.012 | 78 |
| 40 | 1.004 | 78 |
| 19 | 0.999 | 78 |
| 55 | 0.968 | 78 |

The per-state analysis shows that TFP's improvement is broadly distributed rather than driven by
a single outlier state. The worst-performing states represent edge cases rather than systematic
failures.

### 2.5 Naive Baseline Comparison

Naive (last value) and Seasonal Naive provide context for forecast difficulty:
- TFP MAE: 245.7
- Naive MAE: 306.2
- Seasonal Naive MAE: 414.4

TFP substantially outperforms both naive baselines on MAE, confirming it provides genuine
predictive value beyond simple persistence.

---

## 3. Discussion and Limitations

### 3.1 Key Findings

1. **Native TFP intervals are miscalibrated** (49.1% coverage vs 90% nominal)
   and should not be used for probabilistic forecasting without recalibration.

2. **After applying IntervalLawV2 uniformly**, TFP's point forecasts show approximately
   19% WIS improvement and
   10% MAE improvement vs UMass/CDC.

3. **Horizon-dependent pattern**: TFP excels at short horizons (H1) with diminishing advantage at H3.

4. **Broad improvement**: Per-state analysis shows TFP wins in 50/53
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
