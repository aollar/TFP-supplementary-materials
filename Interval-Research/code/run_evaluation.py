"""
Reproducibility Script - Cross-Domain Evaluation
=================================================

This script demonstrates how to evaluate the empirical quantile prediction
interval method on a sample dataset. For full reproduction of paper results,
see the data sources in DATA_DESCRIPTION.md.

Usage:
    python run_evaluation.py

Requirements:
    - Python 3.8+
    - NumPy 1.20+
    - Pandas 1.3+
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from empirical_quantile_intervals import (
    generate_intervals,
    compute_wis,
    compute_coverage,
    compute_interval_width,
    LOOKBACK,
    HORIZON_SCALE
)


def generate_sample_series(n_series: int = 10, length: int = 200) -> List[np.ndarray]:
    """Generate sample time series for demonstration."""
    np.random.seed(42)
    series_list = []

    for i in range(n_series):
        # Mix of random walks and AR(1) processes
        if i % 2 == 0:
            # Random walk
            y = np.cumsum(np.random.randn(length)) + 100
        else:
            # AR(1) with drift
            y = np.zeros(length)
            y[0] = 100
            for t in range(1, length):
                y[t] = 0.95 * y[t-1] + 0.5 + np.random.randn()

        series_list.append(y)

    return series_list


def evaluate_series(
    y: np.ndarray,
    horizons: List[int] = [1, 2, 3, 4, 5],
    n_windows: int = 10
) -> pd.DataFrame:
    """
    Evaluate prediction intervals on a single series using expanding windows.

    Args:
        y: Full time series
        horizons: Forecast horizons to evaluate
        n_windows: Number of evaluation windows

    Returns:
        DataFrame with evaluation results
    """
    results = []
    max_h = max(horizons)

    # Determine evaluation points
    min_train = LOOKBACK + 10  # Ensure enough training data
    eval_start = max(min_train, len(y) - n_windows - max_h)

    for t in range(eval_start, len(y) - max_h):
        y_train = y[:t]
        point = y_train[-1]  # Persistence forecast

        for h in horizons:
            if t + h >= len(y):
                continue

            actual = y[t + h]
            intervals = generate_intervals(y_train, point, h)

            results.append({
                'origin': t,
                'horizon': h,
                'actual': actual,
                'point': point,
                'q05': intervals['q0_05'],
                'q10': intervals['q0_10'],
                'q50': intervals['q0_50'],
                'q90': intervals['q0_90'],
                'q95': intervals['q0_95'],
                'wis': compute_wis(actual, intervals),
                'cov_80': compute_coverage(actual, intervals, 0.80),
                'cov_90': compute_coverage(actual, intervals, 0.90),
                'cov_95': compute_coverage(actual, intervals, 0.95),
                'width_90': compute_interval_width(intervals, 0.90)
            })

    return pd.DataFrame(results)


def summarize_results(results_df: pd.DataFrame) -> Dict:
    """Compute summary statistics from evaluation results."""
    return {
        'n_forecasts': len(results_df),
        'mean_wis': results_df['wis'].mean(),
        'median_wis': results_df['wis'].median(),
        'coverage_80': results_df['cov_80'].mean() * 100,
        'coverage_90': results_df['cov_90'].mean() * 100,
        'coverage_95': results_df['cov_95'].mean() * 100,
        'mean_width_90': results_df['width_90'].mean(),
        'mae': (results_df['actual'] - results_df['point']).abs().mean()
    }


def main():
    print("=" * 70)
    print("EMPIRICAL QUANTILE INTERVALS - REPRODUCIBILITY DEMONSTRATION")
    print("=" * 70)

    # Method parameters
    print(f"\nMethod Parameters (Frozen):")
    print(f"  Lookback window:  {LOOKBACK}")
    print(f"  Horizon scale:    {HORIZON_SCALE}")
    print(f"  Point forecast:   Persistence (y[-1])")

    # Generate sample data
    print("\n" + "-" * 70)
    print("Generating sample time series...")
    series_list = generate_sample_series(n_series=10, length=200)
    print(f"Generated {len(series_list)} series, {len(series_list[0])} observations each")

    # Evaluate each series
    print("\n" + "-" * 70)
    print("Evaluating prediction intervals...")

    all_results = []
    for i, y in enumerate(series_list):
        results = evaluate_series(y, horizons=[1, 2, 3, 4, 5], n_windows=20)
        results['series_id'] = i
        all_results.append(results)

    combined = pd.concat(all_results, ignore_index=True)
    print(f"Total forecasts: {len(combined)}")

    # Summary statistics
    print("\n" + "-" * 70)
    print("RESULTS SUMMARY")
    print("-" * 70)

    summary = summarize_results(combined)

    print(f"\nForecasts:        {summary['n_forecasts']}")
    print(f"\nPoint Forecast (Persistence):")
    print(f"  MAE:            {summary['mae']:.3f}")
    print(f"\nInterval Calibration:")
    print(f"  80% Coverage:   {summary['coverage_80']:.1f}% (target: 80%)")
    print(f"  90% Coverage:   {summary['coverage_90']:.1f}% (target: 90%)")
    print(f"  95% Coverage:   {summary['coverage_95']:.1f}% (target: 95%)")
    print(f"\nInterval Quality:")
    print(f"  Mean WIS:       {summary['mean_wis']:.3f}")
    print(f"  Median WIS:     {summary['median_wis']:.3f}")
    print(f"  Mean 90% Width: {summary['mean_width_90']:.3f}")

    # Coverage by horizon
    print("\n" + "-" * 70)
    print("COVERAGE BY HORIZON")
    print("-" * 70)

    horizon_summary = combined.groupby('horizon').agg({
        'cov_90': 'mean',
        'wis': 'mean',
        'width_90': 'mean'
    }).reset_index()

    print(f"\n{'Horizon':>8} {'90% Cov':>10} {'Mean WIS':>12} {'90% Width':>12}")
    print("-" * 45)
    for _, row in horizon_summary.iterrows():
        print(f"{row['horizon']:>8} {row['cov_90']*100:>9.1f}% {row['wis']:>12.2f} {row['width_90']:>12.2f}")

    print("\n" + "=" * 70)
    print("Evaluation complete. See supplementary materials for full results.")
    print("=" * 70)

    return combined, summary


if __name__ == "__main__":
    results, summary = main()
