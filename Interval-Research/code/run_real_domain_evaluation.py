"""
Real Domain Evaluation for Interval Paper
==========================================

Runs empirical quantile and conformal methods on real data:
- Flu US hospitalizations
- Bass technology adoption
- Finance (from synthesis results)

Generates:
- Coverage comparisons
- Reliability diagram data
- Bootstrap confidence intervals
"""

import numpy as np
import pandas as pd
import sys
import os
from typing import Dict, List, Tuple

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from empirical_quantile_intervals import generate_intervals, compute_wis, compute_coverage, QGRID_23, _qname
from conformal_baseline import conformal_full_quantiles


def load_flu_data(filepath: str) -> Dict[str, np.ndarray]:
    """Load Flu hospitalization data by state."""
    df = pd.read_csv(filepath)

    series = {}
    for loc in df['location'].unique():
        loc_df = df[df['location'] == loc].sort_values('date')
        if len(loc_df) >= 50:
            series[f"flu_{loc}"] = loc_df['value'].values.astype(float)

    return series


def load_bass_data(filepath: str) -> Dict[str, np.ndarray]:
    """Load Bass technology adoption data."""
    df = pd.read_csv(filepath)

    series = {}
    for entity in df['Entity'].unique():
        entity_df = df[df['Entity'] == entity].sort_values('Year')
        values = entity_df['Technology Diffusion (Comin and Hobijn (2004) and others)'].values
        if len(values) >= 20:
            series[f"bass_{entity}"] = values.astype(float)

    return series


def evaluate_series(
    y: np.ndarray,
    horizons: List[int] = [1, 2, 4],
    n_windows: int = 10
) -> List[Dict]:
    """Evaluate both methods on a single series."""
    results = []
    min_train = 50

    if len(y) < min_train + max(horizons):
        return results

    # Evaluation windows
    start = max(min_train, len(y) - n_windows - max(horizons))

    for t in range(start, len(y) - max(horizons)):
        y_train = y[:t]
        point = float(y_train[-1])

        for h in horizons:
            if t + h >= len(y):
                continue

            actual = float(y[t + h])

            # Empirical method
            emp_q = generate_intervals(y_train, point, h)

            # Conformal method
            try:
                conf_q = conformal_full_quantiles(y_train, point, h)
            except:
                conf_q = emp_q  # Fallback

            for method, quantiles in [('empirical', emp_q), ('conformal', conf_q)]:
                record = {
                    'method': method,
                    'origin': t,
                    'horizon': h,
                    'actual': actual,
                    'point': point,
                }

                # All quantiles
                for q in QGRID_23:
                    q_name = _qname(q)
                    if q_name in quantiles:
                        record[q_name] = quantiles[q_name]

                # Coverage
                for level in [0.50, 0.80, 0.90, 0.95]:
                    try:
                        record[f'cov_{int(level*100)}'] = 1 if compute_coverage(actual, quantiles, level) else 0
                    except:
                        record[f'cov_{int(level*100)}'] = np.nan

                # WIS
                try:
                    record['wis'] = compute_wis(actual, quantiles)
                except:
                    record['wis'] = np.nan

                results.append(record)

    return results


def compute_reliability_data(df: pd.DataFrame) -> pd.DataFrame:
    """Compute reliability diagram data."""
    reliability = []

    for method in df['method'].unique():
        method_df = df[df['method'] == method]

        for q in QGRID_23:
            q_name = _qname(q)
            if q_name not in method_df.columns:
                continue

            # Proportion of actuals below this quantile
            valid = method_df[~method_df[q_name].isna()]
            if len(valid) == 0:
                continue

            below = (valid['actual'] <= valid[q_name]).mean()

            reliability.append({
                'method': method,
                'nominal': q,
                'observed': below,
                'n': len(valid)
            })

    return pd.DataFrame(reliability)


def bootstrap_coverage_ci(
    df: pd.DataFrame,
    method: str,
    level: float = 0.90,
    n_boot: int = 1000,
    ci_level: float = 0.95
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for coverage."""
    method_df = df[df['method'] == method]
    cov_col = f'cov_{int(level*100)}'

    if cov_col not in method_df.columns:
        return np.nan, np.nan, np.nan

    values = method_df[cov_col].dropna().values
    if len(values) < 10:
        return np.nan, np.nan, np.nan

    point_est = values.mean()

    # Bootstrap
    boot_means = []
    for _ in range(n_boot):
        sample = np.random.choice(values, size=len(values), replace=True)
        boot_means.append(sample.mean())

    alpha = (1 - ci_level) / 2
    ci_lo = np.percentile(boot_means, alpha * 100)
    ci_hi = np.percentile(boot_means, (1 - alpha) * 100)

    return point_est, ci_lo, ci_hi


def main():
    print("=" * 70)
    print("REAL DOMAIN EVALUATION FOR INTERVAL PAPER")
    print("=" * 70)

    # File paths
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_root = os.path.dirname(base_path)

    flu_path = os.path.join(repo_root, "Flu-Update", "target-hospital-admissions-NEW.csv")
    bass_path = os.path.join(repo_root, "Bass-Research", "data",
                             "technology-adoption-by-households-in-the-united-states.csv")

    all_results = []

    # Load and evaluate Flu data
    print("\n1. Loading Flu data...")
    if os.path.exists(flu_path):
        flu_series = load_flu_data(flu_path)
        print(f"   Loaded {len(flu_series)} Flu series")

        for name, y in list(flu_series.items())[:20]:  # Limit for speed
            results = evaluate_series(y, horizons=[1, 2, 4], n_windows=5)
            for r in results:
                r['domain'] = 'flu'
                r['series'] = name
            all_results.extend(results)
    else:
        print(f"   Flu data not found at {flu_path}")

    # Load and evaluate Bass data
    print("\n2. Loading Bass data...")
    if os.path.exists(bass_path):
        bass_series = load_bass_data(bass_path)
        print(f"   Loaded {len(bass_series)} Bass series")

        for name, y in bass_series.items():
            results = evaluate_series(y, horizons=[1, 2, 4], n_windows=5)
            for r in results:
                r['domain'] = 'bass'
                r['series'] = name
            all_results.extend(results)
    else:
        print(f"   Bass data not found at {bass_path}")

    if not all_results:
        print("\nNo results generated. Check data paths.")
        return

    df = pd.DataFrame(all_results)
    print(f"\nTotal forecasts: {len(df)}")

    # Coverage summary
    print("\n" + "=" * 70)
    print("COVERAGE SUMMARY (90% Nominal)")
    print("=" * 70)

    summary = df.groupby(['method', 'domain']).agg({
        'cov_90': 'mean',
        'wis': 'mean',
        'actual': 'count'
    }).reset_index()

    print(f"\n{'Method':<12} {'Domain':<10} {'90% Cov':>10} {'Mean WIS':>12} {'N':>8}")
    print("-" * 55)
    for _, row in summary.iterrows():
        cov = row['cov_90'] * 100 if not np.isnan(row['cov_90']) else 0
        print(f"{row['method']:<12} {row['domain']:<10} {cov:>9.1f}% {row['wis']:>12.2f} {row['actual']:>8.0f}")

    # Bootstrap CIs
    print("\n" + "=" * 70)
    print("BOOTSTRAP 95% CONFIDENCE INTERVALS (90% Coverage)")
    print("=" * 70)

    for method in ['empirical', 'conformal']:
        est, lo, hi = bootstrap_coverage_ci(df, method, level=0.90, n_boot=1000)
        print(f"\n{method.capitalize()}:")
        print(f"  Coverage: {est*100:.1f}% [{lo*100:.1f}%, {hi*100:.1f}%]")

    # Reliability data
    print("\n" + "=" * 70)
    print("RELIABILITY DATA")
    print("=" * 70)

    reliability = compute_reliability_data(df)

    print(f"\n{'Quantile':>10} {'Empirical':>12} {'Conformal':>12} {'Perfect':>10}")
    print("-" * 50)

    for q in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]:
        emp = reliability[(reliability['method'] == 'empirical') & (reliability['nominal'] == q)]
        conf = reliability[(reliability['method'] == 'conformal') & (reliability['nominal'] == q)]

        emp_val = emp['observed'].values[0] if len(emp) > 0 else np.nan
        conf_val = conf['observed'].values[0] if len(conf) > 0 else np.nan

        print(f"{q:>10.2f} {emp_val:>12.3f} {conf_val:>12.3f} {q:>10.2f}")

    # Save results
    output_dir = os.path.dirname(os.path.abspath(__file__))
    df.to_csv(os.path.join(output_dir, 'real_domain_raw_forecasts.csv'), index=False)
    reliability.to_csv(os.path.join(output_dir, 'real_domain_reliability.csv'), index=False)

    print("\n" + "=" * 70)
    print("Saved: real_domain_raw_forecasts.csv, real_domain_reliability.csv")
    print("=" * 70)

    return df, reliability


if __name__ == "__main__":
    df, reliability = main()
