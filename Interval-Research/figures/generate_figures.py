#!/usr/bin/env python3
"""Generate figures for the ERS paper from existing evaluation data."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13

FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(os.path.dirname(FIGURES_DIR), 'code')


def generate_reliability_diagram():
    """Generate reliability diagram showing observed vs nominal coverage."""
    # Load reliability data
    rel_path = os.path.join(CODE_DIR, 'full_11domain_reliability.csv')
    df = pd.read_csv(rel_path)

    # Columns are: method, domain, nominal, observed, n
    # Aggregate across domains by method and nominal level
    quantiles = sorted(df['nominal'].unique())

    fig, ax = plt.subplots(figsize=(8, 6))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=1.5)

    # Aggregate results by method
    for method, color, marker in [('empirical', '#2ecc71', 'o'), ('conformal', '#3498db', 's')]:
        method_df = df[df['method'] == method]

        # Weight by n to get weighted average observed coverage
        observed = []
        for q in quantiles:
            q_df = method_df[method_df['nominal'] == q]
            if len(q_df) > 0:
                weighted_obs = (q_df['observed'] * q_df['n']).sum() / q_df['n'].sum()
                observed.append(weighted_obs)
            else:
                observed.append(np.nan)

        label = 'ERS (Empirical)' if method == 'empirical' else 'Split Conformal'
        ax.plot(quantiles, observed, marker=marker, markersize=8, linewidth=2,
                color=color, label=label)

    ax.set_xlabel('Nominal Quantile Level')
    ax.set_ylabel('Observed Coverage')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right')
    ax.set_title('Reliability Diagram: Observed vs Nominal Coverage')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'reliability_diagram.png'), dpi=150)
    plt.savefig(os.path.join(FIGURES_DIR, 'reliability_diagram.pdf'))
    plt.close()
    print("Generated: reliability_diagram.png/pdf")


def generate_coverage_by_domain():
    """Generate bar chart of 90% coverage by domain."""
    # Load summary data
    summary_path = os.path.join(CODE_DIR, 'full_11domain_summary.csv')
    df = pd.read_csv(summary_path)

    # Get unique domains
    domains = df['Domain'].unique()

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(domains))
    width = 0.35

    emp_cov = []
    conf_cov = []

    for domain in domains:
        emp = df[(df['Domain'] == domain) & (df['Method'] == 'empirical')]['Cov90'].values
        conf = df[(df['Domain'] == domain) & (df['Method'] == 'conformal')]['Cov90'].values
        emp_cov.append(emp[0] * 100 if len(emp) > 0 else 0)
        conf_cov.append(conf[0] * 100 if len(conf) > 0 else 0)

    bars1 = ax.bar(x - width/2, emp_cov, width, label='ERS (Empirical)', color='#2ecc71')
    bars2 = ax.bar(x + width/2, conf_cov, width, label='Split Conformal', color='#3498db')

    # Target line
    ax.axhline(y=90, color='red', linestyle='--', linewidth=1.5, label='90% Target')

    ax.set_xlabel('Domain')
    ax.set_ylabel('Coverage (%)')
    ax.set_title('90% Prediction Interval Coverage by Domain')
    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=45, ha='right')
    ax.legend(loc='lower left')
    ax.set_ylim(70, 105)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'coverage_by_domain.png'), dpi=150)
    plt.savefig(os.path.join(FIGURES_DIR, 'coverage_by_domain.pdf'))
    plt.close()
    print("Generated: coverage_by_domain.png/pdf")


def generate_wis_comparison():
    """Generate WIS comparison chart."""
    # Load summary data
    summary_path = os.path.join(CODE_DIR, 'full_11domain_summary.csv')
    df = pd.read_csv(summary_path)

    domains = df['Domain'].unique()

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(domains))

    emp_wis = []
    conf_wis = []

    for domain in domains:
        emp = df[(df['Domain'] == domain) & (df['Method'] == 'empirical')]['MeanWIS'].values
        conf = df[(df['Domain'] == domain) & (df['Method'] == 'conformal')]['MeanWIS'].values
        emp_wis.append(emp[0] if len(emp) > 0 else 0)
        conf_wis.append(conf[0] if len(conf) > 0 else 0)

    # Calculate ratio (conformal/empirical) for each domain
    ratios = [c/e if e > 0 else 0 for e, c in zip(emp_wis, conf_wis)]

    colors = ['#2ecc71' if r < 1 else '#e74c3c' for r in ratios]
    bars = ax.bar(x, ratios, color=colors)

    # Reference line at 1.0
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5)

    ax.set_xlabel('Domain')
    ax.set_ylabel('WIS Ratio (Conformal / ERS)')
    ax.set_title('Weighted Interval Score Comparison\n(Green = Conformal better, Red = ERS better)')
    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'wis_comparison.png'), dpi=150)
    plt.savefig(os.path.join(FIGURES_DIR, 'wis_comparison.pdf'))
    plt.close()
    print("Generated: wis_comparison.png/pdf")


if __name__ == '__main__':
    print("Generating figures for ERS paper...")
    generate_reliability_diagram()
    generate_coverage_by_domain()
    generate_wis_comparison()
    print("Done! All figures saved to:", FIGURES_DIR)
