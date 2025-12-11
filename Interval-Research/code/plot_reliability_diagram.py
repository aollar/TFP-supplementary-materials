"""
Reliability Diagram Plotting
=============================

Creates publication-quality reliability diagrams showing
observed vs nominal coverage across quantile levels.
"""

import numpy as np
import pandas as pd
from typing import Optional


def print_reliability_diagram(
    reliability_df: pd.DataFrame,
    method: str = 'empirical',
    title: str = "Reliability Diagram"
) -> str:
    """
    Create ASCII reliability diagram for console output.

    Args:
        reliability_df: DataFrame with 'nominal', 'observed', 'method' columns
        method: Method to plot
        title: Plot title

    Returns:
        ASCII string representation of diagram
    """
    df = reliability_df[reliability_df['method'] == method].copy()
    df = df.sort_values('nominal')

    lines = []
    lines.append("=" * 60)
    lines.append(f"  {title}")
    lines.append(f"  Method: {method}")
    lines.append("=" * 60)
    lines.append("")
    lines.append("  Observed")
    lines.append("  Coverage")
    lines.append("    1.0 |" + " " * 50 + "*")

    # Create grid
    for obs_level in np.arange(0.9, -0.05, -0.1):
        row = f"   {obs_level:.1f} |"
        for nom in np.arange(0.05, 1.0, 0.05):
            # Find closest data point
            closest = df.iloc[(df['nominal'] - nom).abs().argsort()[:1]]
            if len(closest) > 0:
                obs = closest['observed'].values[0]
                if abs(obs - obs_level) < 0.05:
                    row += "●"
                elif abs(nom - obs_level) < 0.05:
                    row += "·"  # Perfect calibration line
                else:
                    row += " "
            else:
                row += " "
        lines.append(row)

    lines.append("    0.0 |" + "-" * 50)
    lines.append("        " + "".join([f"{x:.1f}" if x in [0.2, 0.4, 0.6, 0.8, 1.0] else "   " for x in np.arange(0.0, 1.05, 0.1)]))
    lines.append("                      Nominal Coverage")
    lines.append("")
    lines.append("  Legend: ● = observed, · = perfect calibration")
    lines.append("")

    # Calibration error
    df['error'] = (df['observed'] - df['nominal']).abs()
    mean_error = df['error'].mean()
    max_error = df['error'].max()

    lines.append(f"  Mean Calibration Error: {mean_error:.3f}")
    lines.append(f"  Max Calibration Error:  {max_error:.3f}")
    lines.append("")

    return "\n".join(lines)


def create_reliability_table(reliability_df: pd.DataFrame) -> str:
    """
    Create markdown table of reliability data.
    """
    lines = []
    lines.append("## Reliability Data")
    lines.append("")
    lines.append("| Nominal | Empirical | Conformal | Perfect |")
    lines.append("|---------|-----------|-----------|---------|")

    # Get both methods
    emp = reliability_df[reliability_df['method'] == 'empirical'].set_index('nominal')
    conf = reliability_df[reliability_df['method'] == 'conformal'].set_index('nominal')

    for q in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]:
        emp_obs = emp.loc[q, 'observed'] if q in emp.index else np.nan
        conf_obs = conf.loc[q, 'observed'] if q in conf.index else np.nan

        lines.append(f"| {q:.2f} | {emp_obs:.3f} | {conf_obs:.3f} | {q:.2f} |")

    return "\n".join(lines)


def compute_calibration_metrics(reliability_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute calibration metrics for each method.
    """
    metrics = []

    for method in reliability_df['method'].unique():
        df = reliability_df[reliability_df['method'] == method]

        # Mean absolute calibration error
        mace = (df['observed'] - df['nominal']).abs().mean()

        # Root mean squared calibration error
        rmsce = np.sqrt(((df['observed'] - df['nominal'])**2).mean())

        # Coverage at key levels
        cov_90 = df[df['nominal'] == 0.90]['observed'].values
        cov_90 = cov_90[0] if len(cov_90) > 0 else np.nan

        # Over/under coverage
        over_coverage = (df['observed'] > df['nominal']).mean()

        metrics.append({
            'method': method,
            'mace': mace,
            'rmsce': rmsce,
            'cov_at_90': cov_90,
            'over_coverage_rate': over_coverage
        })

    return pd.DataFrame(metrics)


if __name__ == "__main__":
    # Demo with sample data
    print("Reliability Diagram Demo")
    print("=" * 60)

    # Create sample reliability data
    np.random.seed(42)

    quantiles = [0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99]

    # Empirical method: slight over-coverage
    emp_data = []
    for q in quantiles:
        # Observed is slightly higher than nominal (over-coverage)
        obs = min(1.0, q + np.random.uniform(0.02, 0.06))
        emp_data.append({'method': 'empirical', 'nominal': q, 'observed': obs, 'n': 1000})

    # Conformal method: well-calibrated
    conf_data = []
    for q in quantiles:
        # Observed close to nominal
        obs = q + np.random.uniform(-0.02, 0.02)
        obs = max(0, min(1, obs))
        conf_data.append({'method': 'conformal', 'nominal': q, 'observed': obs, 'n': 1000})

    reliability_df = pd.DataFrame(emp_data + conf_data)

    # Print diagrams
    print(print_reliability_diagram(reliability_df, 'empirical', 'Empirical Quantile Method'))
    print(print_reliability_diagram(reliability_df, 'conformal', 'Conformal Prediction'))

    # Calibration metrics
    print("\n" + "=" * 60)
    print("CALIBRATION METRICS")
    print("=" * 60)

    metrics = compute_calibration_metrics(reliability_df)
    print(metrics.to_string(index=False))

    # Markdown table
    print("\n" + "=" * 60)
    print("MARKDOWN TABLE")
    print("=" * 60)
    print(create_reliability_table(reliability_df))
