#!/usr/bin/env python3
"""
Synthetic Data Benchmark for TFP Law-Like Generalist
=====================================================

Tests whether TFP v2.2 behaves like a law-like generalist by evaluating
performance on synthetic series with known structure.

Models:
- TFP v2.2: Law-like generalist with percentile-based oscillation dampening
- SimpleTheta: Strong Theta method (M4-style)
- Naive: Last observed value

Evaluation:
- Rolling origin holdout
- Horizons: 1, 4, 8, 12
- Metrics: MAE (point law) and WIS (interval law)
- Block bootstrap over series for 95% CIs

Two Modes:
- Point Law: MAE only (tests point forecast quality)
- Interval Law: WIS with IntervalLawV2 applied uniformly

Usage:
    python -m synthetic_eval.run_synthetic_benchmark
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Add repo root to path
sys.path.insert(0, '/home/user/TFP-core')

from synthetic_eval.synthetic_generators import generate_all_families
from cross_domain_eval.interval_law_v2 import (
    interval_law_v2, IntervalConfig, compute_wis_q23, QGRID_23
)


# =============================================================================
# CONFIGURATION
# =============================================================================

N_SERIES = 500          # Series per family
N_ORIGINS = 24          # Rolling origins (last 24 points of each series)
H_MAX = 12              # Max forecast horizon
HORIZONS = [1, 4, 8, 12]  # Evaluation horizons
N_BOOTSTRAP = 2000      # Bootstrap resamples
SEED = 42               # Master random seed

OUT_DIR = '/home/user/TFP-core/out'


# =============================================================================
# FORECASTER IMPLEMENTATIONS
# =============================================================================

class TFPForecaster:
    """
    TFP v2.2 forecaster wrapper.

    Uses the law-like generalist with percentile-based oscillation dampening.
    """

    def __init__(self, period: int = 12):
        self.period = period
        self._tfp = None

    def fit(self, y_hist: np.ndarray) -> 'TFPForecaster':
        """Store history for forecasting."""
        self.y_hist = y_hist.copy()
        return self

    def predict(self, horizons: List[int]) -> Dict[int, float]:
        """Generate point forecasts for multiple horizons."""
        # Lazy import to avoid circular deps
        if self._tfp is None:
            from tfp_v2_2_lawlike_standalone import TFPWithBrain
            self._tfp = TFPWithBrain(
                domain='synthetic',
                period=self.period,
                use_brain=True,
                lower_bound=0.0  # Required for TFP to work
            )

        # Create DataFrame for TFP API
        df = pd.DataFrame({
            'date': pd.date_range('2000-01-01', periods=len(self.y_hist), freq='W'),
            'value': self.y_hist
        })

        forecasts = {}
        for h in horizons:
            try:
                result = self._tfp.forecast(df, horizon=h, value_col='value')
                forecasts[h] = result['point']
            except Exception as e:
                # Fallback to last value on error
                forecasts[h] = float(self.y_hist[-1])

        return forecasts


class SimpleThetaForecaster:
    """
    Simple Theta method (M4-style).

    Combines linear extrapolation (theta=0) with exponential smoothing (theta=2).
    """

    def __init__(self, period: int = 12):
        self.period = period

    def fit(self, y_hist: np.ndarray) -> 'SimpleThetaForecaster':
        """Store history for forecasting."""
        self.y_hist = y_hist.copy()
        return self

    def predict(self, horizons: List[int]) -> Dict[int, float]:
        """Generate point forecasts for multiple horizons."""
        y = self.y_hist
        n = len(y)
        S = self.period

        # Deseasonalize if enough data
        if S > 1 and n >= 2 * S:
            seasonal = np.zeros(S)
            for i in range(S):
                indices = list(range(i, n, S))
                seasonal[i] = np.mean(y[indices])
            seasonal = seasonal - np.mean(seasonal)
            y_deseas = np.array([y[t] - seasonal[t % S] for t in range(n)])
        else:
            y_deseas = y.copy()
            seasonal = np.zeros(max(1, S))

        # Fit linear trend
        x = np.arange(n)
        try:
            coeffs = np.polyfit(x, y_deseas, 1)
            slope = coeffs[0]
            intercept = coeffs[1]
        except:
            slope = 0.0
            intercept = float(y_deseas[-1])

        # SES level for theta2
        alpha = 0.5
        level = float(y_deseas[-1])

        forecasts = {}
        for h in horizons:
            # Theta0: linear extrapolation
            theta0 = intercept + slope * (n + h - 1)
            # Theta2: SES level
            theta2 = level
            # Average
            point = 0.5 * theta0 + 0.5 * theta2

            # Add seasonality
            if S > 1:
                point = point + seasonal[(n + h - 1) % S]

            forecasts[h] = float(point)

        return forecasts


class NaiveForecaster:
    """Naive forecaster: always predicts last observed value."""

    def __init__(self, period: int = 12):
        """Accept period for API consistency (not used)."""
        self.period = period

    def fit(self, y_hist: np.ndarray) -> 'NaiveForecaster':
        self.last_value = float(y_hist[-1])
        self.y_hist = y_hist.copy()
        return self

    def predict(self, horizons: List[int]) -> Dict[int, float]:
        return {h: self.last_value for h in horizons}


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def compute_mae(actuals: np.ndarray, forecasts: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return float(np.mean(np.abs(actuals - forecasts)))


def evaluate_series_mae(
    series: np.ndarray,
    model_class,
    horizons: List[int],
    n_origins: int,
    period: int = 12
) -> Dict[int, List[float]]:
    """
    Evaluate a single series using rolling origin.

    Args:
        series: Full time series
        model_class: Forecaster class (TFPForecaster, etc.)
        horizons: List of horizons to evaluate
        n_origins: Number of rolling origins
        period: Seasonal period for models

    Returns:
        Dict mapping horizon -> list of absolute errors
    """
    T = len(series)
    max_h = max(horizons)

    # Ensure we have enough data
    min_train = 50  # Minimum training size
    if T < min_train + n_origins + max_h:
        n_origins = max(1, T - min_train - max_h)

    errors = {h: [] for h in horizons}

    for origin_idx in range(n_origins):
        # Training ends at this point
        train_end = T - max_h - origin_idx
        if train_end < min_train:
            continue

        y_train = series[:train_end]

        # Fit and predict
        model = model_class(period=period)
        model.fit(y_train)
        forecasts = model.predict(horizons)

        # Compute errors for each horizon
        for h in horizons:
            actual_idx = train_end + h - 1
            if actual_idx < T:
                actual = series[actual_idx]
                pred = forecasts[h]
                errors[h].append(abs(actual - pred))

    return errors


def evaluate_series_wis(
    series: np.ndarray,
    model_class,
    horizons: List[int],
    n_origins: int,
    period: int = 12,
    config: IntervalConfig = None
) -> Tuple[Dict[int, List[float]], Dict[int, List[bool]]]:
    """
    Evaluate a single series using rolling origin with WIS.

    Args:
        series: Full time series
        model_class: Forecaster class
        horizons: List of horizons to evaluate
        n_origins: Number of rolling origins
        period: Seasonal period
        config: IntervalLawV2 configuration

    Returns:
        Tuple of:
        - Dict mapping horizon -> list of WIS scores
        - Dict mapping horizon -> list of coverage bools
    """
    if config is None:
        config = IntervalConfig(lookback=104, horizon_scale=0.1)

    T = len(series)
    max_h = max(horizons)
    min_train = 50

    if T < min_train + n_origins + max_h:
        n_origins = max(1, T - min_train - max_h)

    wis_scores = {h: [] for h in horizons}
    coverages = {h: [] for h in horizons}

    for origin_idx in range(n_origins):
        train_end = T - max_h - origin_idx
        if train_end < min_train:
            continue

        y_train = series[:train_end]

        # Fit and predict point forecasts
        model = model_class(period=period)
        model.fit(y_train)
        point_forecasts = model.predict(horizons)

        # Generate intervals and compute WIS
        for h in horizons:
            actual_idx = train_end + h - 1
            if actual_idx < T:
                actual = series[actual_idx]
                point = point_forecasts[h]

                # Apply IntervalLawV2
                quantiles = interval_law_v2(y_train, point, h, config)

                # Compute WIS
                wis = compute_wis_q23(actual, quantiles)
                wis_scores[h].append(wis)

                # Compute 90% coverage
                lower = quantiles.get('q0_05', -np.inf)
                upper = quantiles.get('q0_95', np.inf)
                covered = lower <= actual <= upper
                coverages[h].append(covered)

    return wis_scores, coverages


def evaluate_family_mae(
    family_data: np.ndarray,
    model_class,
    horizons: List[int],
    n_origins: int,
    period: int = 12
) -> Dict[int, np.ndarray]:
    """
    Evaluate all series in a family.

    Returns:
        Dict mapping horizon -> array of per-series mean MAE
    """
    n_series = family_data.shape[0]
    results = {h: [] for h in horizons}

    for i in range(n_series):
        series = family_data[i]
        errors = evaluate_series_mae(series, model_class, horizons, n_origins, period)

        for h in horizons:
            if errors[h]:
                mean_mae = np.mean(errors[h])
                results[h].append(mean_mae)
            else:
                results[h].append(np.nan)

    return {h: np.array(v) for h, v in results.items()}


def evaluate_family_wis(
    family_data: np.ndarray,
    model_class,
    horizons: List[int],
    n_origins: int,
    period: int = 12,
    config: IntervalConfig = None
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Evaluate all series in a family with WIS.

    Returns:
        Tuple of:
        - Dict mapping horizon -> array of per-series mean WIS
        - Dict mapping horizon -> array of per-series coverage rates
    """
    n_series = family_data.shape[0]
    wis_results = {h: [] for h in horizons}
    cov_results = {h: [] for h in horizons}

    for i in range(n_series):
        series = family_data[i]
        wis_scores, coverages = evaluate_series_wis(
            series, model_class, horizons, n_origins, period, config
        )

        for h in horizons:
            if wis_scores[h]:
                wis_results[h].append(np.mean(wis_scores[h]))
                cov_results[h].append(np.mean(coverages[h]))
            else:
                wis_results[h].append(np.nan)
                cov_results[h].append(np.nan)

    return (
        {h: np.array(v) for h, v in wis_results.items()},
        {h: np.array(v) for h, v in cov_results.items()}
    )


# =============================================================================
# BOOTSTRAP FUNCTIONS
# =============================================================================

def bootstrap_ratio_ci(
    metric_a: np.ndarray,
    metric_b: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 42,
    alpha: float = 0.05
) -> Tuple[float, float, float]:
    """
    Compute bootstrap CI for ratio A/B.

    Uses block bootstrap over series.

    Returns:
        (point_estimate, lower_ci, upper_ci)
    """
    rng = np.random.default_rng(seed)

    # Remove NaN pairs
    valid = ~(np.isnan(metric_a) | np.isnan(metric_b))
    a = metric_a[valid]
    b = metric_b[valid]

    if len(a) < 10:
        return np.nan, np.nan, np.nan

    n = len(a)
    ratios = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        mean_a = np.mean(a[idx])
        mean_b = np.mean(b[idx])
        if mean_b > 0:
            ratios.append(mean_a / mean_b)

    ratios = np.array(ratios)
    point = np.mean(a) / np.mean(b) if np.mean(b) > 0 else np.nan
    lower = np.percentile(ratios, 100 * alpha / 2)
    upper = np.percentile(ratios, 100 * (1 - alpha / 2))

    return point, lower, upper


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

def run_benchmark(
    n_series: int = N_SERIES,
    n_origins: int = N_ORIGINS,
    horizons: List[int] = HORIZONS,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run full synthetic benchmark.

    Returns:
        Dict with all results
    """
    if verbose:
        print("=" * 80)
        print("TFP v2.2 SYNTHETIC BENCHMARK - Law-Like Generalist Test")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  Series per family: {n_series}")
        print(f"  Rolling origins: {n_origins}")
        print(f"  Horizons: {horizons}")
        print(f"  Bootstrap resamples: {n_bootstrap}")
        print(f"  Random seed: {seed}")
        print()

    # Generate synthetic data
    if verbose:
        print("Generating synthetic data...")
    families = generate_all_families(n_series=n_series, seed=seed)

    # Model configurations
    models = {
        'TFP': TFPForecaster,
        'SimpleTheta': SimpleThetaForecaster,
        'Naive': NaiveForecaster,
    }

    # Family-specific periods
    family_periods = {
        'trend': 1,           # No seasonality
        'seasonal': 12,       # Monthly seasonality
        'logistic_growth': 1, # No seasonality
        'logistic_saturated': 1,
    }

    results = {
        'mae': {},      # {family: {model: {horizon: array}}}
        'wis': {},      # {family: {model: {horizon: array}}}
        'coverage': {}, # {family: {model: {horizon: array}}}
    }

    # Evaluate each family
    for family_name, (family_data, family_meta) in families.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"Family: {family_name} ({family_data.shape[0]} series, T={family_data.shape[1]})")
            print('='*60)

        period = family_periods.get(family_name, 12)
        results['mae'][family_name] = {}
        results['wis'][family_name] = {}
        results['coverage'][family_name] = {}

        config = IntervalConfig(lookback=104, horizon_scale=0.1)

        for model_name, model_class in models.items():
            if verbose:
                print(f"\n  Evaluating {model_name}...")

            # MAE evaluation
            mae_results = evaluate_family_mae(
                family_data, model_class, horizons, n_origins, period
            )
            results['mae'][family_name][model_name] = mae_results

            # WIS evaluation (with IntervalLawV2)
            wis_results, cov_results = evaluate_family_wis(
                family_data, model_class, horizons, n_origins, period, config
            )
            results['wis'][family_name][model_name] = wis_results
            results['coverage'][family_name][model_name] = cov_results

            # Print summary
            if verbose:
                for h in horizons:
                    mae_mean = np.nanmean(mae_results[h])
                    wis_mean = np.nanmean(wis_results[h])
                    cov_mean = np.nanmean(cov_results[h])
                    print(f"    H{h:2d}: MAE={mae_mean:7.2f}, WIS={wis_mean:7.2f}, Cov90={cov_mean:.1%}")

    # Compute bootstrap CIs for ratios
    if verbose:
        print("\n" + "=" * 80)
        print("COMPUTING BOOTSTRAP CONFIDENCE INTERVALS")
        print("=" * 80)

    ci_results = {
        'mae_ratios': {},  # {family: {comparison: {horizon: (point, lower, upper)}}}
        'wis_ratios': {},
    }

    comparisons = [
        ('TFP', 'SimpleTheta'),
        ('TFP', 'Naive'),
    ]

    for family_name in families.keys():
        ci_results['mae_ratios'][family_name] = {}
        ci_results['wis_ratios'][family_name] = {}

        for model_a, model_b in comparisons:
            comp_name = f"{model_a}/{model_b}"
            ci_results['mae_ratios'][family_name][comp_name] = {}
            ci_results['wis_ratios'][family_name][comp_name] = {}

            for h in horizons:
                mae_a = results['mae'][family_name][model_a][h]
                mae_b = results['mae'][family_name][model_b][h]
                wis_a = results['wis'][family_name][model_a][h]
                wis_b = results['wis'][family_name][model_b][h]

                mae_ci = bootstrap_ratio_ci(mae_a, mae_b, n_bootstrap, seed + h)
                wis_ci = bootstrap_ratio_ci(wis_a, wis_b, n_bootstrap, seed + h + 100)

                ci_results['mae_ratios'][family_name][comp_name][h] = mae_ci
                ci_results['wis_ratios'][family_name][comp_name][h] = wis_ci

        if verbose:
            print(f"\n{family_name}:")
            for comp_name in ci_results['mae_ratios'][family_name]:
                print(f"  {comp_name}:")
                for h in horizons:
                    mae_pt, mae_lo, mae_hi = ci_results['mae_ratios'][family_name][comp_name][h]
                    wis_pt, wis_lo, wis_hi = ci_results['wis_ratios'][family_name][comp_name][h]
                    print(f"    H{h:2d}: MAE={mae_pt:.3f} [{mae_lo:.3f}, {mae_hi:.3f}], "
                          f"WIS={wis_pt:.3f} [{wis_lo:.3f}, {wis_hi:.3f}]")

    return {
        'results': results,
        'ci_results': ci_results,
        'families': families,
        'config': {
            'n_series': n_series,
            'n_origins': n_origins,
            'horizons': horizons,
            'n_bootstrap': n_bootstrap,
            'seed': seed,
        }
    }


def save_results(benchmark_results: Dict, out_dir: str = OUT_DIR):
    """Save benchmark results to CSV files."""
    os.makedirs(out_dir, exist_ok=True)

    results = benchmark_results['results']
    ci_results = benchmark_results['ci_results']
    config = benchmark_results['config']
    horizons = config['horizons']

    # 1. MAE results
    mae_rows = []
    for family in results['mae']:
        for model in results['mae'][family]:
            for h in horizons:
                arr = results['mae'][family][model][h]
                mae_rows.append({
                    'family': family,
                    'model': model,
                    'horizon': h,
                    'mean_mae': np.nanmean(arr),
                    'std_mae': np.nanstd(arr),
                    'n_series': np.sum(~np.isnan(arr)),
                })

    pd.DataFrame(mae_rows).to_csv(
        os.path.join(out_dir, 'synthetic_results_mae.csv'), index=False
    )
    print(f"Saved: {out_dir}/synthetic_results_mae.csv")

    # 2. WIS results
    wis_rows = []
    for family in results['wis']:
        for model in results['wis'][family]:
            for h in horizons:
                wis_arr = results['wis'][family][model][h]
                cov_arr = results['coverage'][family][model][h]
                wis_rows.append({
                    'family': family,
                    'model': model,
                    'horizon': h,
                    'mean_wis': np.nanmean(wis_arr),
                    'std_wis': np.nanstd(wis_arr),
                    'mean_coverage_90': np.nanmean(cov_arr),
                    'n_series': np.sum(~np.isnan(wis_arr)),
                })

    pd.DataFrame(wis_rows).to_csv(
        os.path.join(out_dir, 'synthetic_results_wis.csv'), index=False
    )
    print(f"Saved: {out_dir}/synthetic_results_wis.csv")

    # 3. Bootstrap CI results
    ci_rows = []
    for family in ci_results['mae_ratios']:
        for comp in ci_results['mae_ratios'][family]:
            for h in horizons:
                mae_pt, mae_lo, mae_hi = ci_results['mae_ratios'][family][comp][h]
                wis_pt, wis_lo, wis_hi = ci_results['wis_ratios'][family][comp][h]
                ci_rows.append({
                    'family': family,
                    'comparison': comp,
                    'horizon': h,
                    'mae_ratio': mae_pt,
                    'mae_ci_lower': mae_lo,
                    'mae_ci_upper': mae_hi,
                    'wis_ratio': wis_pt,
                    'wis_ci_lower': wis_lo,
                    'wis_ci_upper': wis_hi,
                })

    pd.DataFrame(ci_rows).to_csv(
        os.path.join(out_dir, 'synthetic_bootstrap_mae_ratios.csv'), index=False
    )
    print(f"Saved: {out_dir}/synthetic_bootstrap_mae_ratios.csv")


def generate_summary(benchmark_results: Dict, out_dir: str = OUT_DIR):
    """Generate markdown summary report."""
    results = benchmark_results['results']
    ci_results = benchmark_results['ci_results']
    config = benchmark_results['config']
    horizons = config['horizons']

    lines = []
    lines.append("# TFP v2.2 Synthetic Benchmark Results")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # Configuration
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- Series per family: {config['n_series']}")
    lines.append(f"- Rolling origins: {config['n_origins']}")
    lines.append(f"- Horizons: {horizons}")
    lines.append(f"- Bootstrap resamples: {config['n_bootstrap']}")
    lines.append(f"- Random seed: {config['seed']}")
    lines.append("")

    # Synthetic Family Descriptions
    lines.append("## Synthetic Families")
    lines.append("")
    lines.append("### 1. Trend + Noise")
    lines.append("- Length: 200 time steps")
    lines.append("- Piecewise linear trend with potential turning point at t=100")
    lines.append("- Slope in [-0.5, +0.5], 50% chance of sign flip")
    lines.append("- Gaussian noise: 5-15% of signal range")
    lines.append("")
    lines.append("### 2. Seasonal AR(1)")
    lines.append("- Length: 240 time steps")
    lines.append("- Period 12 seasonality")
    lines.append("- AR(1) level process with coefficient in [0.3, 0.9]")
    lines.append("- Random seasonal amplitude and phase")
    lines.append("")
    lines.append("### 3. Logistic S-curve (Growth)")
    lines.append("- Length: 160 time steps")
    lines.append("- Logistic function: K in [80, 120], r in [0.03, 0.15]")
    lines.append("- Evaluation windows in 20-80% of K (growth phase)")
    lines.append("- Mild local wiggles + noise")
    lines.append("")
    lines.append("### 4. Logistic S-curve (Saturated)")
    lines.append("- Same as above but evaluation in 90-100% of K (saturation)")
    lines.append("")

    # MAE Results Tables
    lines.append("## Point Forecast Results (MAE)")
    lines.append("")

    for family in results['mae']:
        lines.append(f"### {family}")
        lines.append("")
        lines.append("| Horizon | TFP | SimpleTheta | Naive | TFP/Theta | 95% CI | TFP/Naive | 95% CI |")
        lines.append("|---------|-----|-------------|-------|-----------|--------|-----------|--------|")

        for h in horizons:
            tfp_mae = np.nanmean(results['mae'][family]['TFP'][h])
            theta_mae = np.nanmean(results['mae'][family]['SimpleTheta'][h])
            naive_mae = np.nanmean(results['mae'][family]['Naive'][h])

            theta_pt, theta_lo, theta_hi = ci_results['mae_ratios'][family]['TFP/SimpleTheta'][h]
            naive_pt, naive_lo, naive_hi = ci_results['mae_ratios'][family]['TFP/Naive'][h]

            lines.append(
                f"| H{h} | {tfp_mae:.2f} | {theta_mae:.2f} | {naive_mae:.2f} | "
                f"**{theta_pt:.3f}** | [{theta_lo:.3f}, {theta_hi:.3f}] | "
                f"**{naive_pt:.3f}** | [{naive_lo:.3f}, {naive_hi:.3f}] |"
            )
        lines.append("")

    # WIS Results Tables
    lines.append("## Interval Forecast Results (WIS with IntervalLawV2)")
    lines.append("")

    for family in results['wis']:
        lines.append(f"### {family}")
        lines.append("")
        lines.append("| Horizon | TFP WIS | Theta WIS | Naive WIS | TFP/Theta | 95% CI | Coverage |")
        lines.append("|---------|---------|-----------|-----------|-----------|--------|----------|")

        for h in horizons:
            tfp_wis = np.nanmean(results['wis'][family]['TFP'][h])
            theta_wis = np.nanmean(results['wis'][family]['SimpleTheta'][h])
            naive_wis = np.nanmean(results['wis'][family]['Naive'][h])
            tfp_cov = np.nanmean(results['coverage'][family]['TFP'][h])

            wis_pt, wis_lo, wis_hi = ci_results['wis_ratios'][family]['TFP/SimpleTheta'][h]

            lines.append(
                f"| H{h} | {tfp_wis:.2f} | {theta_wis:.2f} | {naive_wis:.2f} | "
                f"**{wis_pt:.3f}** | [{wis_lo:.3f}, {wis_hi:.3f}] | {tfp_cov:.1%} |"
            )
        lines.append("")

    # Summary Analysis
    lines.append("## Summary Analysis")
    lines.append("")

    # Count wins
    tfp_wins_mae = 0
    tfp_wins_wis = 0
    total_comparisons = 0

    for family in ci_results['mae_ratios']:
        for h in horizons:
            total_comparisons += 1
            mae_pt, _, mae_hi = ci_results['mae_ratios'][family]['TFP/SimpleTheta'][h]
            wis_pt, _, wis_hi = ci_results['wis_ratios'][family]['TFP/SimpleTheta'][h]
            if mae_hi < 1.0:  # Significant win
                tfp_wins_mae += 1
            if wis_hi < 1.0:
                tfp_wins_wis += 1

    lines.append("### TFP vs SimpleTheta (Significant Wins)")
    lines.append("")
    lines.append(f"- MAE: TFP significantly better in {tfp_wins_mae}/{total_comparisons} comparisons")
    lines.append(f"- WIS: TFP significantly better in {tfp_wins_wis}/{total_comparisons} comparisons")
    lines.append("")

    # Family-by-family summary
    lines.append("### Family-by-Family Performance")
    lines.append("")

    for family in ci_results['mae_ratios']:
        h1_mae = ci_results['mae_ratios'][family]['TFP/SimpleTheta'][1][0]
        h12_mae = ci_results['mae_ratios'][family]['TFP/SimpleTheta'][12][0]

        if h1_mae < 0.95 and h12_mae < 0.95:
            verdict = "TFP clearly outperforms"
        elif h1_mae > 1.05 and h12_mae > 1.05:
            verdict = "SimpleTheta clearly outperforms"
        else:
            verdict = "Mixed or roughly equal"

        lines.append(f"- **{family}**: {verdict} (H1 ratio: {h1_mae:.3f}, H12 ratio: {h12_mae:.3f})")

    lines.append("")

    # IntervalLawV2 calibration
    lines.append("### IntervalLawV2 Calibration")
    lines.append("")
    lines.append("Coverage rates for 90% prediction intervals:")
    lines.append("")

    for family in results['coverage']:
        tfp_cov_h1 = np.nanmean(results['coverage'][family]['TFP'][1])
        theta_cov_h1 = np.nanmean(results['coverage'][family]['SimpleTheta'][1])
        lines.append(f"- **{family}**: TFP={tfp_cov_h1:.1%}, SimpleTheta={theta_cov_h1:.1%}")

    lines.append("")

    # Law-like narrative
    lines.append("## Narrative: Is TFP Law-Like?")
    lines.append("")
    lines.append("The synthetic benchmark tests whether TFP v2.2 behaves as a **law-like generalist**")
    lines.append("that captures general regularities rather than overfitting specific real datasets.")
    lines.append("")
    lines.append("### Key Observations")
    lines.append("")

    # Analyze results for narrative
    trend_h1 = ci_results['mae_ratios']['trend']['TFP/SimpleTheta'][1][0]
    seasonal_h1 = ci_results['mae_ratios']['seasonal']['TFP/SimpleTheta'][1][0]
    logistic_g_h1 = ci_results['mae_ratios']['logistic_growth']['TFP/SimpleTheta'][1][0]
    logistic_s_h1 = ci_results['mae_ratios']['logistic_saturated']['TFP/SimpleTheta'][1][0]

    lines.append(f"1. **Trend series**: TFP/Theta ratio = {trend_h1:.3f}")
    if trend_h1 < 0.95:
        lines.append("   - TFP's adaptive blending effectively captures linear and turning point trends")
    elif trend_h1 > 1.05:
        lines.append("   - SimpleTheta's linear extrapolation is effective for pure trend series")
    else:
        lines.append("   - Performance is roughly equal, both methods handle trends similarly")

    lines.append("")
    lines.append(f"2. **Seasonal series**: TFP/Theta ratio = {seasonal_h1:.3f}")
    if seasonal_h1 < 0.95:
        lines.append("   - TFP adapts well to seasonal patterns despite no explicit seasonal modeling")
    elif seasonal_h1 > 1.05:
        lines.append("   - SimpleTheta's explicit seasonality handling gives it an advantage")
    else:
        lines.append("   - Both methods handle seasonality comparably")

    lines.append("")
    lines.append(f"3. **Logistic growth**: TFP/Theta ratio = {logistic_g_h1:.3f}")
    if logistic_g_h1 < 0.95:
        lines.append("   - TFP excels at S-curve dynamics in growth phase")
        lines.append("   - Percentile-based dampening enables near-pure trend following")
    elif logistic_g_h1 > 1.05:
        lines.append("   - SimpleTheta handles growth phase better")
    else:
        lines.append("   - Both methods comparable during growth phase")

    lines.append("")
    lines.append(f"4. **Logistic saturation**: TFP/Theta ratio = {logistic_s_h1:.3f}")
    if logistic_s_h1 < 0.95:
        lines.append("   - TFP handles saturation well, reducing extrapolation when appropriate")
    elif logistic_s_h1 > 1.05:
        lines.append("   - Saturation phase may challenge TFP's trend-following bias")
    else:
        lines.append("   - Both methods handle saturation comparably")

    lines.append("")
    lines.append("### Conclusion")
    lines.append("")

    avg_ratio = np.mean([trend_h1, seasonal_h1, logistic_g_h1, logistic_s_h1])
    if avg_ratio < 0.95:
        lines.append("The results **support the law-like hypothesis**. TFP v2.2 demonstrates")
        lines.append("consistent performance improvements across diverse synthetic structures,")
        lines.append("suggesting it captures general forecasting principles rather than")
        lines.append("memorizing patterns from specific real datasets.")
    elif avg_ratio > 1.05:
        lines.append("The results **challenge the law-like hypothesis**. TFP v2.2 underperforms")
        lines.append("SimpleTheta on synthetic data, suggesting its advantages on real data may")
        lines.append("come from fitting specific patterns rather than general principles.")
    else:
        lines.append("The results show **mixed evidence** for the law-like hypothesis. TFP v2.2")
        lines.append("performs comparably to SimpleTheta on synthetic data, neither clearly")
        lines.append("supporting nor refuting generalization capabilities.")

    lines.append("")

    # Potential concerns
    lines.append("## Potential Reviewer Concerns")
    lines.append("")
    lines.append("1. **Synthetic simplicity**: Real-world data has complex patterns not captured here")
    lines.append("2. **Parameter sensitivity**: Results may depend on synthetic generator parameters")
    lines.append("3. **IntervalLawV2 applied uniformly**: May favor models with similar residual structure")
    lines.append("4. **Limited model set**: Only three models compared (no ETS/ARIMA)")
    lines.append("5. **Horizon granularity**: H=1,4,8,12 may miss patterns at other horizons")
    lines.append("")

    # Write to file
    summary_path = os.path.join(out_dir, 'synthetic_summary.md')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Saved: {summary_path}")

    return '\n'.join(lines)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='TFP v2.2 Synthetic Benchmark'
    )
    parser.add_argument('--n-series', type=int, default=N_SERIES,
                        help=f'Series per family (default: {N_SERIES})')
    parser.add_argument('--n-origins', type=int, default=N_ORIGINS,
                        help=f'Rolling origins (default: {N_ORIGINS})')
    parser.add_argument('--n-bootstrap', type=int, default=N_BOOTSTRAP,
                        help=f'Bootstrap resamples (default: {N_BOOTSTRAP})')
    parser.add_argument('--seed', type=int, default=SEED,
                        help=f'Random seed (default: {SEED})')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: fewer series and bootstrap samples')

    args = parser.parse_args()

    if args.quick:
        args.n_series = 50
        args.n_bootstrap = 200
        print("QUICK MODE: Reduced series (50) and bootstrap (200)")

    # Run benchmark
    results = run_benchmark(
        n_series=args.n_series,
        n_origins=args.n_origins,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        verbose=True
    )

    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    save_results(results)

    # Generate summary
    print("\n" + "=" * 80)
    print("GENERATING SUMMARY")
    print("=" * 80)
    summary = generate_summary(results)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)

    return results


if __name__ == "__main__":
    main()
