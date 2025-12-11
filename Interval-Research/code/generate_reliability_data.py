"""
Generate Raw Forecast Data for Reliability Diagrams
====================================================

This script generates forecast-actual pairs across multiple domains
for computing reliability diagrams and statistical comparisons.

Outputs:
- Raw forecasts with all quantiles
- Coverage at each quantile level
- WIS scores
- Data for reliability diagrams
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from empirical_quantile_intervals import generate_intervals, compute_wis, QGRID_23
from conformal_baseline import conformal_full_quantiles


def generate_synthetic_domains(seed: int = 42) -> Dict[str, List[np.ndarray]]:
    """
    Generate synthetic time series representing different domain characteristics.

    Returns dict mapping domain name to list of series.
    """
    np.random.seed(seed)
    domains = {}

    # Domain 1: Random walks (like finance)
    domains['random_walk'] = [
        np.cumsum(np.random.randn(200)) + 100 for _ in range(50)
    ]

    # Domain 2: AR(1) stationary (like detrended macro)
    domains['ar1_stationary'] = []
    for _ in range(50):
        y = np.zeros(200)
        y[0] = 100
        for t in range(1, 200):
            y[t] = 0.8 * y[t-1] + 20 + np.random.randn() * 5
        domains['ar1_stationary'].append(y)

    # Domain 3: Trending (like technology adoption)
    domains['trending'] = []
    for _ in range(30):
        t = np.arange(150)
        trend = 10 + 0.5 * t + 0.002 * t**2
        noise = np.random.randn(150) * (2 + 0.02 * t)
        domains['trending'].append(trend + noise)

    # Domain 4: Seasonal (like energy/flu)
    domains['seasonal'] = []
    for _ in range(40):
        t = np.arange(200)
        seasonal = 50 + 20 * np.sin(2 * np.pi * t / 52)
        noise = np.random.randn(200) * 8
        domains['seasonal'].append(seasonal + noise)

    # Domain 5: Heavy-tailed (like retail)
    domains['heavy_tailed'] = []
    for _ in range(40):
        y = 100 + np.random.standard_t(3, 200) * 10
        domains['heavy_tailed'].append(np.cumsum(y - y.mean()) + 100)

    return domains


def evaluate_method(
    y: np.ndarray,
    method: str,
    horizons: List[int] = [1, 2, 4, 8]
) -> List[Dict]:
    """
    Evaluate a single method on a single series.

    Returns list of forecast records with all quantiles and actuals.
    """
    results = []
    min_train = 104 + max(horizons) + 10

    if len(y) < min_train:
        return results

    # Evaluation windows
    for t in range(min_train, len(y) - max(horizons)):
        y_train = y[:t]
        point = y_train[-1]  # Persistence

        for h in horizons:
            if t + h >= len(y):
                continue

            actual = y[t + h]

            # Generate intervals based on method
            if method == 'empirical':
                quantiles = generate_intervals(y_train, point, h)
            elif method == 'conformal':
                quantiles = conformal_full_quantiles(y_train, point, h)
            else:
                raise ValueError(f"Unknown method: {method}")

            # Record result
            record = {
                'method': method,
                'origin': t,
                'horizon': h,
                'actual': actual,
                'point': point,
            }

            # Add all quantiles
            for q in QGRID_23:
                q_name = f"q{q:.3f}".replace(".", "_")
                if q_name in quantiles:
                    record[q_name] = quantiles[q_name]

            # Compute coverage at each level
            for level in [0.50, 0.80, 0.90, 0.95]:
                alpha = (1 - level) / 2
                low_q = f"q{alpha:.3f}".replace(".", "_")
                high_q = f"q{1-alpha:.3f}".replace(".", "_")
                if low_q in quantiles and high_q in quantiles:
                    covered = quantiles[low_q] <= actual <= quantiles[high_q]
                    record[f'cov_{int(level*100)}'] = 1 if covered else 0

            # WIS
            record['wis'] = compute_wis(actual, quantiles)

            results.append(record)

    return results


def run_full_evaluation(
    domains: Dict[str, List[np.ndarray]],
    methods: List[str] = ['empirical', 'conformal'],
    horizons: List[int] = [1, 2, 4, 8],
    max_series_per_domain: int = 20
) -> pd.DataFrame:
    """
    Run full evaluation across all domains and methods.
    """
    all_results = []

    for domain_name, series_list in domains.items():
        print(f"Evaluating {domain_name}...")

        for i, y in enumerate(series_list[:max_series_per_domain]):
            for method in methods:
                results = evaluate_method(y, method, horizons)
                for r in results:
                    r['domain'] = domain_name
                    r['series_id'] = i
                all_results.extend(results)

    return pd.DataFrame(all_results)


def compute_reliability_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute reliability diagram data from raw forecasts.

    For each quantile level, compute:
    - Nominal coverage (the quantile level)
    - Observed coverage (actual proportion below quantile)
    """
    reliability = []

    for method in df['method'].unique():
        method_df = df[df['method'] == method]

        for q in QGRID_23:
            q_name = f"q{q:.3f}".replace(".", "_")
            if q_name not in method_df.columns:
                continue

            # Proportion of actuals below this quantile
            below = (method_df['actual'] <= method_df[q_name]).mean()

            reliability.append({
                'method': method,
                'nominal': q,
                'observed': below,
                'n': len(method_df)
            })

    return pd.DataFrame(reliability)


def compute_coverage_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute coverage summary by method and domain.
    """
    summary = df.groupby(['method', 'domain']).agg({
        'cov_50': 'mean',
        'cov_80': 'mean',
        'cov_90': 'mean',
        'cov_95': 'mean',
        'wis': 'mean',
        'actual': 'count'
    }).reset_index()

    summary.columns = ['method', 'domain', 'cov_50', 'cov_80', 'cov_90', 'cov_95', 'mean_wis', 'n_forecasts']

    # Convert to percentages
    for col in ['cov_50', 'cov_80', 'cov_90', 'cov_95']:
        summary[col] = summary[col] * 100

    return summary


if __name__ == "__main__":
    print("=" * 70)
    print("GENERATING RAW FORECAST DATA FOR RELIABILITY ANALYSIS")
    print("=" * 70)

    # Generate synthetic domains
    print("\nGenerating synthetic domains...")
    domains = generate_synthetic_domains()

    for name, series_list in domains.items():
        print(f"  {name}: {len(series_list)} series")

    # Run evaluation
    print("\nRunning evaluation...")
    df = run_full_evaluation(domains, max_series_per_domain=15)

    print(f"\nTotal forecasts: {len(df)}")

    # Coverage summary
    print("\n" + "=" * 70)
    print("COVERAGE SUMMARY BY METHOD AND DOMAIN")
    print("=" * 70)

    summary = compute_coverage_summary(df)
    print(summary.to_string(index=False))

    # Overall by method
    print("\n" + "=" * 70)
    print("OVERALL BY METHOD")
    print("=" * 70)

    overall = df.groupby('method').agg({
        'cov_90': 'mean',
        'wis': 'mean',
        'actual': 'count'
    }).reset_index()

    overall['cov_90'] = overall['cov_90'] * 100
    print(f"\n{'Method':<12} {'90% Cov':>10} {'Mean WIS':>12} {'N':>8}")
    print("-" * 45)
    for _, row in overall.iterrows():
        print(f"{row['method']:<12} {row['cov_90']:>9.1f}% {row['wis']:>12.2f} {row['actual']:>8}")

    # Reliability data
    print("\n" + "=" * 70)
    print("RELIABILITY DATA (for diagrams)")
    print("=" * 70)

    reliability = compute_reliability_data(df)

    print("\nEmpirical Method:")
    emp_rel = reliability[reliability['method'] == 'empirical']
    print(f"{'Nominal':>8} {'Observed':>10}")
    for _, row in emp_rel.iterrows():
        print(f"{row['nominal']:>8.2f} {row['observed']:>10.3f}")

    # Save outputs
    df.to_csv('raw_forecasts.csv', index=False)
    summary.to_csv('coverage_summary.csv', index=False)
    reliability.to_csv('reliability_data.csv', index=False)

    print("\n" + "=" * 70)
    print("Saved: raw_forecasts.csv, coverage_summary.csv, reliability_data.csv")
    print("=" * 70)
