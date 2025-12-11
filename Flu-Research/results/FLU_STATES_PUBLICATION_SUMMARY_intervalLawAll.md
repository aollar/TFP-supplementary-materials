## FluSight State-Level Forecast Evaluation

### Evaluation Setup

**Interval Mode:** IntervalLawV2 applied to all models (fair comparison)

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
| TFP v2.2 | 183.4 | 245.7 | 77.3% |
| UMass | 225.3 | 274.4 | 76.6% |
| CDC Ensemble | 229.8 | 282.1 | 76.2% |

**TFP vs UMass:** WIS ratio = 0.814 (+18.6%)
**TFP vs CDC:** WIS ratio = 0.798 (+20.2%)

### Results: US National Only (N = 78)

| Model | WIS | MAE | Coverage 90% |
|-------|-----|-----|--------------|
| TFP v2.2 | 4382.9 | 6045.4 | 78.2% |
| UMass | 5270.4 | 6351.0 | 79.5% |
| CDC Ensemble | 5409.7 | 6580.3 | 80.8% |

**TFP vs UMass:** WIS ratio = 0.832 (+16.8%)
**TFP vs CDC:** WIS ratio = 0.810 (+19.0%)

### Coverage Analysis by Horizon

| Horizon | TFP v2.2 | UMass | CDC Ensemble |
|---------|----------|-------|--------------|
| H1 | 89.2% | 80.6% | 80.5% |
| H2 | 76.0% | 76.3% | 76.0% |
| H3 | 66.8% | 73.1% | 72.1% |
| **Overall** | **77.3%** | **76.6%** | **76.2%** |

### Calibration

The 90% prediction interval should contain the true value 90% of the time. Values below 90% indicate **undercoverage**
(intervals too narrow), while values above 90% indicate **overcoverage** (intervals too wide).

**Overall Calibration Status:**
- **TFP v2.2**: 77.3% coverage (substantially undercovered, 12.7pp below nominal)
- **UMass**: 76.6% coverage (substantially undercovered, 13.4pp below nominal)
- **CDC Ensemble**: 76.2% coverage (substantially undercovered, 13.8pp below nominal)

All three models exhibit undercoverage in this evaluation period, which is common during atypical flu seasons
(e.g., pandemic-affected years or unusual epidemic timing). This suggests prediction intervals from all models
are too narrow to fully capture the uncertainty in influenza hospitalizations.

**Note on Interval Modes:**
This evaluation applies IntervalLawV2 to all models, generating calibrated intervals around each model's point forecast.
This ensures a fair comparison where differences in WIS reflect point forecast accuracy rather than interval calibration differences.
