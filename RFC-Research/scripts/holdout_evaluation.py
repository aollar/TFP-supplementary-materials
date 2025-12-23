#!/usr/bin/env python3
"""
Holdout Evaluation for RFC Updates vs Obsoletes analysis.

This script performs internal validation by:
1. Loading the frozen pipeline configuration
2. Splitting edges into 75% dev / 25% holdout (stratified by relation type)
3. Running the analysis ONCE on holdout with NO tuning
4. Reporting results and generating holdout-specific figures

This is NOT a true prospective holdout (data was seen during development),
but demonstrates the effect replicates on a held-out subset under a frozen pipeline.

Usage: python holdout_evaluation.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy import stats

# Paths
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Fixed seed for reproducibility
RANDOM_SEED = 42


def load_config():
    """Load frozen pipeline configuration."""
    config_path = SCRIPT_DIR / "holdout_config.json"
    with open(config_path) as f:
        return json.load(f)


def load_edges():
    """Load per-edge data."""
    results_path = RESULTS_DIR / "rfc_snapshot_results.json"
    with open(results_path) as f:
        data = json.load(f)
    return data["results"]


def cliffs_delta(x, y):
    """Compute Cliff's delta effect size."""
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.0
    more = sum(1 for xi in x for yi in y if xi > yi)
    less = sum(1 for xi in x for yi in y if xi < yi)
    return (more - less) / (n_x * n_y)


def bootstrap_ci(data, stat_func=np.median, n_bootstrap=10000, ci=0.95):
    """Compute bootstrap confidence interval."""
    np.random.seed(RANDOM_SEED)
    n = len(data)
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(stat_func(sample))

    alpha = 1 - ci
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    return lower, upper


def stratified_split(edges, dev_fraction=0.75, seed=42):
    """
    Split edges into dev and holdout sets, stratified by relation type.
    Returns edge indices for reproducibility.
    """
    np.random.seed(seed)

    # Separate by type
    updates_idx = [i for i, e in enumerate(edges) if e["relation_type"] == "updates"]
    obsoletes_idx = [i for i, e in enumerate(edges) if e["relation_type"] == "obsoletes"]

    # Shuffle
    np.random.shuffle(updates_idx)
    np.random.shuffle(obsoletes_idx)

    # Split
    n_updates_dev = int(len(updates_idx) * dev_fraction)
    n_obsoletes_dev = int(len(obsoletes_idx) * dev_fraction)

    dev_idx = updates_idx[:n_updates_dev] + obsoletes_idx[:n_obsoletes_dev]
    holdout_idx = updates_idx[n_updates_dev:] + obsoletes_idx[n_obsoletes_dev:]

    return dev_idx, holdout_idx


def run_analysis(edges, indices, config):
    """Run the frozen pipeline on a subset of edges."""
    subset = [edges[i] for i in indices]

    updates = [e for e in subset if e["relation_type"] == "updates"]
    obsoletes = [e for e in subset if e["relation_type"] == "obsoletes"]

    updates_takeover = [e["takeover_ratio"] for e in updates]
    obsoletes_takeover = [e["takeover_ratio"] for e in obsoletes]

    # Compute statistics
    median_updates = np.median(updates_takeover) if updates_takeover else 0
    median_obsoletes = np.median(obsoletes_takeover) if obsoletes_takeover else 0

    # Effect size
    delta = cliffs_delta(obsoletes_takeover, updates_takeover)

    # Statistical test
    if len(updates_takeover) > 0 and len(obsoletes_takeover) > 0:
        stat, p_value = stats.mannwhitneyu(
            obsoletes_takeover, updates_takeover,
            alternative='greater'
        )
    else:
        stat, p_value = 0, 1.0

    # Bootstrap CIs
    ci_updates = bootstrap_ci(updates_takeover) if updates_takeover else (0, 0)
    ci_obsoletes = bootstrap_ci(obsoletes_takeover) if obsoletes_takeover else (0, 0)

    return {
        "n_total": len(subset),
        "n_updates": len(updates),
        "n_obsoletes": len(obsoletes),
        "median_updates": median_updates,
        "median_obsoletes": median_obsoletes,
        "ci_updates": ci_updates,
        "ci_obsoletes": ci_obsoletes,
        "cliffs_delta": delta,
        "mann_whitney_stat": stat,
        "p_value": p_value,
        "updates_takeover": updates_takeover,
        "obsoletes_takeover": obsoletes_takeover
    }


def generate_holdout_figure(results, output_dir):
    """Generate violin plot for holdout evaluation."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6))

    updates = results["updates_takeover"]
    obsoletes = results["obsoletes_takeover"]

    # Violin plot
    parts = ax.violinplot([updates, obsoletes], positions=[1, 2],
                          showmeans=False, showmedians=False, showextrema=False)

    colors = ['#3498db', '#e74c3c']
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)

    # Box plot overlay
    bp = ax.boxplot([updates, obsoletes], positions=[1, 2], widths=0.15,
                    patch_artist=True, showfliers=False)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    for element in ['whiskers', 'caps', 'medians']:
        plt.setp(bp[element], color='black', linewidth=1.5)

    # Annotations
    med_u = results["median_updates"]
    med_o = results["median_obsoletes"]

    ax.annotate(f'Median: {med_u:.3f}', xy=(1, med_u), xytext=(0.5, 0.25),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='#3498db'))
    ax.annotate(f'Median: {med_o:.3f}', xy=(2, med_o), xytext=(2.5, 0.55),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='#e74c3c'))

    # Labels
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f'Updates\n(n={results["n_updates"]})',
                        f'Obsoletes\n(n={results["n_obsoletes"]})'])
    ax.set_ylabel('Snapshot Takeover Ratio')
    ax.set_ylim(-0.05, 1.05)

    delta = results["cliffs_delta"]
    p = results["p_value"]
    p_str = f"p = {p:.2e}" if p >= 0.0001 else "p < 0.0001"
    ax.set_title(f"HOLDOUT Evaluation: Citation Takeover by Relation Type\n"
                 f"(Cliff's δ = {delta:.3f}, {p_str})")

    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Neutral (0.5)')
    ax.legend(loc='upper left')

    # Add "HOLDOUT" watermark
    ax.text(0.98, 0.02, 'HOLDOUT SET (25%)', transform=ax.transAxes,
            fontsize=12, color='green', alpha=0.7,
            ha='right', va='bottom', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    png_path = output_dir / 'figure_holdout_violin.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()

    return png_path


def main():
    print("=" * 60)
    print("RFC Updates vs Obsoletes - HOLDOUT EVALUATION")
    print(f"Run time: {datetime.now().isoformat()}")
    print("=" * 60)

    # Load config
    print("\n[1/5] Loading frozen pipeline configuration...")
    config = load_config()
    print(f"  Min citers: {config['pipeline_spec']['min_citers']}")
    print(f"  Dev/Holdout split: {config['split_spec']['dev_fraction']:.0%} / {config['split_spec']['holdout_fraction']:.0%}")
    print(f"  Random seed: {config['split_spec']['random_seed']}")

    # Load edges
    print("\n[2/5] Loading edges...")
    edges = load_edges()
    print(f"  Total edges: {len(edges)}")

    # Stratified split
    print("\n[3/5] Performing stratified split...")
    dev_idx, holdout_idx = stratified_split(
        edges,
        dev_fraction=config['split_spec']['dev_fraction'],
        seed=config['split_spec']['random_seed']
    )

    # Save holdout indices for reproducibility
    holdout_ids = {
        "holdout_indices": holdout_idx,
        "dev_indices": dev_idx,
        "random_seed": RANDOM_SEED,
        "created": datetime.now().isoformat()
    }
    holdout_ids_path = RESULTS_DIR / "holdout_edge_ids.json"
    with open(holdout_ids_path, 'w') as f:
        json.dump(holdout_ids, f, indent=2)
    print(f"  Saved holdout indices to: {holdout_ids_path.name}")

    # Run on dev set (for comparison)
    print("\n[4/5] Running analysis...")
    dev_results = run_analysis(edges, dev_idx, config)
    holdout_results = run_analysis(edges, holdout_idx, config)

    print("\n  --- DEV SET (75%) ---")
    print(f"  n_updates: {dev_results['n_updates']}")
    print(f"  n_obsoletes: {dev_results['n_obsoletes']}")
    print(f"  Median Updates: {dev_results['median_updates']:.4f}")
    print(f"  Median Obsoletes: {dev_results['median_obsoletes']:.4f}")
    print(f"  Cliff's δ: {dev_results['cliffs_delta']:.4f}")
    print(f"  p-value: {dev_results['p_value']:.2e}")

    print("\n  --- HOLDOUT SET (25%) ---")
    print(f"  n_updates: {holdout_results['n_updates']}")
    print(f"  n_obsoletes: {holdout_results['n_obsoletes']}")
    print(f"  Median Updates: {holdout_results['median_updates']:.4f}")
    print(f"  Median Obsoletes: {holdout_results['median_obsoletes']:.4f}")
    print(f"  Cliff's δ: {holdout_results['cliffs_delta']:.4f}")
    print(f"  p-value: {holdout_results['p_value']:.2e}")

    # Check acceptance criteria
    criteria = config['acceptance_criteria']
    passed = (
        holdout_results['cliffs_delta'] >= criteria['cliffs_delta_min'] and
        holdout_results['p_value'] <= criteria['p_value_max'] and
        holdout_results['median_obsoletes'] > holdout_results['median_updates']
    )

    print(f"\n  Acceptance criteria:")
    print(f"    δ ≥ {criteria['cliffs_delta_min']}: {'✓' if holdout_results['cliffs_delta'] >= criteria['cliffs_delta_min'] else '✗'}")
    print(f"    p ≤ {criteria['p_value_max']}: {'✓' if holdout_results['p_value'] <= criteria['p_value_max'] else '✗'}")
    print(f"    Obsoletes > Updates: {'✓' if holdout_results['median_obsoletes'] > holdout_results['median_updates'] else '✗'}")

    # Generate figure
    print("\n[5/5] Generating holdout figure...")
    fig_path = generate_holdout_figure(holdout_results, FIGURES_DIR)
    print(f"  Generated: {fig_path.name}")

    # Save results
    output = {
        "evaluation_type": "internal_validation_split",
        "note": "This is a retrospective split, not a true prospective holdout. "
                "Data was seen during development but analysis used frozen pipeline.",
        "config": config,
        "dev_results": {
            "n_total": dev_results["n_total"],
            "n_updates": dev_results["n_updates"],
            "n_obsoletes": dev_results["n_obsoletes"],
            "median_updates": round(dev_results["median_updates"], 4),
            "median_obsoletes": round(dev_results["median_obsoletes"], 4),
            "cliffs_delta": round(dev_results["cliffs_delta"], 4),
            "p_value": f"{dev_results['p_value']:.2e}"
        },
        "holdout_results": {
            "n_total": holdout_results["n_total"],
            "n_updates": holdout_results["n_updates"],
            "n_obsoletes": holdout_results["n_obsoletes"],
            "median_updates": round(holdout_results["median_updates"], 4),
            "median_obsoletes": round(holdout_results["median_obsoletes"], 4),
            "ci_updates": [round(x, 4) for x in holdout_results["ci_updates"]],
            "ci_obsoletes": [round(x, 4) for x in holdout_results["ci_obsoletes"]],
            "cliffs_delta": round(holdout_results["cliffs_delta"], 4),
            "p_value": f"{holdout_results['p_value']:.2e}"
        },
        "acceptance_criteria_met": bool(passed),
        "generated_at": datetime.now().isoformat()
    }

    output_path = RESULTS_DIR / "holdout_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {output_path.name}")

    # Final status
    print("\n" + "=" * 60)
    if passed:
        print("HOLDOUT EVALUATION: PASSED")
        print("The effect replicates on held-out data under frozen pipeline.")
    else:
        print("HOLDOUT EVALUATION: FAILED ACCEPTANCE CRITERIA")
    print("=" * 60)

    return passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
