## FluSight State-Level Forecast Evaluation

### Evaluation Setup

**Interval Mode:** Native intervals (each model uses its own quantiles)

**Seasons Evaluated:** 2024-2025

**Apples Definition:**
- Locations: 53 (50 states + DC + PR + US national)
- Horizons: h = 1, 2, 3 weeks ahead
- Forecast dates: 26 weekly dates (2024-11-23 to 2025-05-31)
- Total N = 4119 apples

**Methodology:**
- Training: All available data before each forecast date (rolling origin)
- Metrics: WIS (normalized by 12), MAE, 90% coverage
- Bootstrap: Block bootstrap by location (10k resamples)

### Models Compared

1. **TFP v2.2**: Law-like generalist with percentile-based oscillation dampening
2. **UMass-Trends**: FluSight participant ensemble method
3. **CDC FluSight Ensemble**: Official CDC ensemble forecast

### Results: All Locations Pooled (N = 4119)

| Model | WIS | MAE | Coverage 90% |
|-------|-----|-----|--------------|
| TFP v2.2 | 211.7 | 245.7 | 49.1% |
| UMass | 195.8 | 274.4 | 76.4% |
| CDC Ensemble | 198.3 | 282.1 | 76.1% |

**TFP vs UMass:** WIS ratio = 1.081 (-8.1%)
**TFP vs CDC:** WIS ratio = 1.067 (-6.7%)

### Results: US National Only (N = 78)

| Model | WIS | MAE | Coverage 90% |
|-------|-----|-----|--------------|
| TFP v2.2 | 5239.2 | 6045.4 | 44.9% |
| UMass | 4675.7 | 6351.0 | 75.6% |
| CDC Ensemble | 4687.3 | 6580.3 | 75.6% |

**TFP vs UMass:** WIS ratio = 1.121 (-12.1%)
**TFP vs CDC:** WIS ratio = 1.118 (-11.8%)

### Coverage Analysis by Horizon

| Horizon | TFP v2.2 | UMass | CDC Ensemble |
|---------|----------|-------|--------------|
| H1 | 56.7% | 76.3% | 78.4% |
| H2 | 47.8% | 77.1% | 76.8% |
| H3 | 42.8% | 75.8% | 73.0% |
| **Overall** | **49.1%** | **76.4%** | **76.1%** |

### Calibration

The 90% prediction interval should contain the true value 90% of the time. Values below 90% indicate **undercoverage**
(intervals too narrow), while values above 90% indicate **overcoverage** (intervals too wide).

**Overall Calibration Status:**
- **TFP v2.2**: 49.1% coverage (severely undercovered, 40.9pp below nominal)
- **UMass**: 76.4% coverage (substantially undercovered, 13.6pp below nominal)
- **CDC Ensemble**: 76.1% coverage (substantially undercovered, 13.9pp below nominal)

All three models exhibit undercoverage in this evaluation period, which is common during atypical flu seasons
(e.g., pandemic-affected years or unusual epidemic timing). This suggests prediction intervals from all models
are too narrow to fully capture the uncertainty in influenza hospitalizations.

**Note on Interval Modes:**
This evaluation uses each model's native prediction intervals. When running with `--interval-mode interval_law_all`,
IntervalLawV2 is applied to all models' point forecasts to generate calibrated intervals, enabling fair comparison
of point forecast accuracy while using identical interval generation methodology.
