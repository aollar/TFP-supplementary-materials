# Results File Dictionary

## synthesis_paper_master_table.csv

Per-domain evaluation results with bootstrap 95% confidence intervals.

| Column | Description |
|--------|-------------|
| domain | Human-readable domain name |
| domain_key | Machine key for domain |
| n_series | Number of time series evaluated |
| n_windows | Total forecast windows (series × origins) |
| has_owa | Whether M4-style OWA was computed |
| tfp_mae | TFP mean absolute error |
| tfp_smape | TFP symmetric MAPE (%) |
| tfp_wis | TFP weighted interval score |
| tfp_cov | TFP 90% interval coverage (%) |
| theta_mae | StrongSimpleTheta MAE |
| theta_smape | StrongSimpleTheta sMAPE (%) |
| theta_wis | StrongSimpleTheta WIS |
| theta_cov | StrongSimpleTheta coverage (%) |
| naive2_mae | Naive2 MAE |
| naive2_smape | Naive2 sMAPE (%) |
| naive2_wis | Naive2 WIS |
| naive2_cov | Naive2 coverage (%) |
| tfp_vs_theta_*_ratio | Ratio of TFP to Theta (< 1 = TFP wins) |
| tfp_vs_theta_*_ci_lo | Bootstrap 95% CI lower bound |
| tfp_vs_theta_*_ci_hi | Bootstrap 95% CI upper bound |
| tfp_vs_naive2_* | Same for Naive2 comparison |
| tfp_mase, theta_mase, naive2_mase | M4-style MASE (M4 domains only) |
| tfp_owa, theta_owa | Overall Weighted Average (M4 domains only) |

## synthesis_paper_summary.json

Machine-readable summary for automated extraction.

```json
{
  "n_domains_headline": 11,
  "n_total_windows": 8097,
  "geometric_mean_ratios": {
    "tfp_vs_theta": { "mae": 0.453, "smape": 0.478, "wis": 0.638 },
    "tfp_vs_naive2": { "mae": 0.719, "smape": 0.788, "wis": 0.857 }
  },
  "win_tie_loss": {
    "tfp_vs_theta": { "smape": { "wins": 11, "ties": 0, "losses": 0 } }
  },
  "coverage_by_domain": { ... },
  "seed_robustness": { ... },
  "statistical_significance": {
    "binomial_sign_test": { "p_value": 0.000488 },
    "wilcoxon_per_domain": { ... }
  }
}
```

## synthesis_paper_methods.txt

Complete methods section with:
- Evaluation design
- Baseline descriptions
- Interval law explanation
- Coverage behavior
- Domain selection rationale
- Horizon heterogeneity
- Retail exceptions
- Reproducibility details
- Statistical significance results

## synthesis_paper_raw_results.pkl

Python pickle containing:
- `summaries`: List of per-domain summary dicts
- `raw`: List of 8,097 window-level forecast results

Each raw item contains:
- `domain`: Domain key
- `series`: Series identifier
- `tfp`: { `mae`, `smape`, `wis`, `coverage` }
- `theta`: { `mae`, `smape`, `wis`, `coverage` }
- `naive2`: { `mae`, `smape`, `wis`, `coverage` }

Use for custom analysis or additional statistical tests.
