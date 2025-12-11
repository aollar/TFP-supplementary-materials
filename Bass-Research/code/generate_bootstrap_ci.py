#!/usr/bin/env python3
"""
Generate Bootstrap Confidence Intervals for Bass-Research paper.

Entity-level block bootstrap with 10,000 resamples.
Computes MAE ratios (TFP / baseline) and 95% CIs.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def entity_block_bootstrap(
    df: pd.DataFrame,
    n_resamples: int = 10000,
    seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Perform entity-level block bootstrap to compute MAE ratio CIs.

    Resamples at the technology (entity) level to respect within-entity correlation.
    """
    np.random.seed(seed)

    entities = df['entity'].unique()
    n_entities = len(entities)

    # Baselines to compare
    baselines = {
        'bass': 'bass_mae',
        'gompertz': 'gompertz_mae',
        'logistic': 'logistic_mae',
        'theta': 'theta_mae',
        'naive': 'naive_mae'
    }

    results = {}

    for baseline_name, baseline_col in baselines.items():
        # Filter to windows where baseline has valid data
        if baseline_name in ['bass', 'gompertz', 'logistic']:
            valid_df = df[df[baseline_col].notna()].copy()
        else:
            valid_df = df.copy()

        valid_entities = valid_df['entity'].unique()
        n_valid = len(valid_entities)

        # Point estimates
        tfp_mae = valid_df['tfp_mae'].mean()
        baseline_mae = valid_df[baseline_col].mean()
        ratio = tfp_mae / baseline_mae
        improvement = (1 - ratio) * 100

        # Bootstrap
        bootstrap_ratios = []

        for _ in range(n_resamples):
            # Sample entities with replacement
            sampled_entities = np.random.choice(valid_entities, size=n_valid, replace=True)

            # Get all windows for sampled entities
            boot_dfs = [valid_df[valid_df['entity'] == e] for e in sampled_entities]
            boot_df = pd.concat(boot_dfs, ignore_index=True)

            # Compute ratio
            boot_tfp = boot_df['tfp_mae'].mean()
            boot_baseline = boot_df[baseline_col].mean()

            if boot_baseline > 0:
                bootstrap_ratios.append(boot_tfp / boot_baseline)

        bootstrap_ratios = np.array(bootstrap_ratios)
        ci_low = np.percentile(bootstrap_ratios, 2.5)
        ci_high = np.percentile(bootstrap_ratios, 97.5)

        results[baseline_name] = {
            'comparison': f'TFP vs {baseline_name.capitalize()}',
            'mae_ratio': ratio,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'improvement_pct': improvement,
            'n_windows': len(valid_df),
            'n_entities': n_valid,
            'significant': ci_high < 1.0
        }

        print(f"TFP vs {baseline_name.capitalize()}: ratio={ratio:.3f} [{ci_low:.3f}, {ci_high:.3f}] "
              f"improvement={improvement:.1f}% (n={len(valid_df)} windows)")

    return results


def regime_bootstrap(
    df: pd.DataFrame,
    regimes: List[str],
    n_resamples: int = 10000,
    seed: int = 42
) -> List[Dict]:
    """
    Compute regime-level bootstrap CIs for TFP vs Bass and TFP vs Theta.
    """
    np.random.seed(seed)

    results = []

    for regime in regimes:
        regime_df = df[df['regime'] == regime].copy()

        if len(regime_df) == 0:
            continue

        entities = regime_df['entity'].unique()
        n_entities = len(entities)

        for baseline_name, baseline_col in [('Bass', 'bass_mae'), ('SimpleTheta', 'theta_mae')]:
            if baseline_name == 'Bass':
                valid_df = regime_df[regime_df[baseline_col].notna()].copy()
            else:
                valid_df = regime_df.copy()

            if len(valid_df) == 0:
                continue

            valid_entities = valid_df['entity'].unique()
            n_valid = len(valid_entities)

            # Point estimates
            tfp_mae = valid_df['tfp_mae'].mean()
            baseline_mae = valid_df[baseline_col].mean()
            ratio = tfp_mae / baseline_mae
            improvement = (1 - ratio) * 100

            # Bootstrap
            bootstrap_ratios = []

            for _ in range(n_resamples):
                sampled_entities = np.random.choice(valid_entities, size=n_valid, replace=True)
                boot_dfs = [valid_df[valid_df['entity'] == e] for e in sampled_entities]
                boot_df = pd.concat(boot_dfs, ignore_index=True)

                boot_tfp = boot_df['tfp_mae'].mean()
                boot_baseline = boot_df[baseline_col].mean()

                if boot_baseline > 0:
                    bootstrap_ratios.append(boot_tfp / boot_baseline)

            bootstrap_ratios = np.array(bootstrap_ratios)
            ci_low = np.percentile(bootstrap_ratios, 2.5)
            ci_high = np.percentile(bootstrap_ratios, 97.5)

            results.append({
                'regime': regime,
                'comparison': f'TFP vs {baseline_name}',
                'mae_ratio': ratio,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'improvement_pct': improvement,
                'n_windows': len(valid_df),
                'significant': ci_high < 1.0
            })

            print(f"{regime} - TFP vs {baseline_name}: ratio={ratio:.3f} [{ci_low:.3f}, {ci_high:.3f}] "
                  f"improvement={improvement:.1f}%")

    return results


def main():
    # Load raw data
    df = pd.read_csv('results/bass_revalidation_raw.csv')

    print(f"Loaded {len(df)} evaluation windows across {df['entity'].nunique()} technologies")
    print()

    # Overall bootstrap CIs
    print("=" * 60)
    print("OVERALL BOOTSTRAP CIs (10,000 resamples, entity-level block)")
    print("=" * 60)
    overall_results = entity_block_bootstrap(df, n_resamples=10000)

    # Save overall results
    overall_df = pd.DataFrame([
        {
            'comparison': v['comparison'],
            'mae_ratio': v['mae_ratio'],
            'ci_95_low': v['ci_low'],
            'ci_95_high': v['ci_high'],
            'improvement_pct': v['improvement_pct'],
            'n_windows': v['n_windows'],
            'significant_at_95': v['significant']
        }
        for v in overall_results.values()
    ])
    overall_df.to_csv('results/bass_bootstrap_ci_10k.csv', index=False)
    print(f"\nSaved: results/bass_bootstrap_ci_10k.csv")

    # Regime-level bootstrap CIs
    print()
    print("=" * 60)
    print("REGIME-LEVEL BOOTSTRAP CIs")
    print("=" * 60)

    regimes = ['Early', 'Growth', 'Mature', 'Saturated']
    regime_results = regime_bootstrap(df, regimes, n_resamples=10000)

    regime_df = pd.DataFrame(regime_results)
    regime_df.to_csv('results/bass_regime_bootstrap_ci_10k.csv', index=False)
    print(f"\nSaved: results/bass_regime_bootstrap_ci_10k.csv")

    # Matched window summary
    print()
    print("=" * 60)
    print("MATCHED WINDOW SUMMARY (where Bass converges)")
    print("=" * 60)

    matched_df = df[df['bass_mae'].notna()].copy()
    print(f"Matched windows: {len(matched_df)}")

    matched_summary = []
    for baseline_name, baseline_col in [('Bass', 'bass_mae'), ('Gompertz', 'gompertz_mae'),
                                         ('Logistic', 'logistic_mae')]:
        tfp_mae = matched_df['tfp_mae'].mean()
        baseline_mae = matched_df[baseline_col].mean()
        ratio = tfp_mae / baseline_mae
        improvement = (1 - ratio) * 100

        matched_summary.append({
            'baseline': baseline_name,
            'tfp_mae': tfp_mae,
            'baseline_mae': baseline_mae,
            'mae_ratio': ratio,
            'improvement_pct': improvement
        })
        print(f"TFP vs {baseline_name}: TFP={tfp_mae:.2f}%, {baseline_name}={baseline_mae:.2f}%, "
              f"improvement={improvement:.1f}%")

    matched_summary_df = pd.DataFrame(matched_summary)
    matched_summary_df.to_csv('results/bass_matched_window_summary.csv', index=False)
    print(f"\nSaved: results/bass_matched_window_summary.csv")


if __name__ == '__main__':
    main()
