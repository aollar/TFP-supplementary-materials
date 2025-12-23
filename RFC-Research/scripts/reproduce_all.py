#!/usr/bin/env python3
"""
Reproducibility script for RFC Updates vs Obsoletes analysis.

This script:
1. Loads raw data and validates structure
2. Runs sanity checks against reported statistics
3. Regenerates figures from actual per-edge data
4. Computes SHA-256 hashes for all outputs
5. Produces a manifest for verification

Usage: python reproduce_all.py
"""

import json
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy import stats

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data" / "processed"
RESULTS_DIR = SCRIPT_DIR / "results"
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Expected values (from paper)
EXPECTED = {
    "n_updates": 859,
    "n_obsoletes": 512,
    "n_total": 1371,
    "median_updates": 0.083,
    "median_obsoletes": 0.773,
    "cliffs_delta": 0.796,
    "tolerance": 0.01  # Allow 1% tolerance for floating point
}


def sha256_file(filepath):
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def load_data():
    """Load the snapshot results with per-edge takeover values."""
    results_path = RESULTS_DIR / "rfc_snapshot_results.json"
    with open(results_path) as f:
        data = json.load(f)
    return data


def cliffs_delta(x, y):
    """Compute Cliff's delta effect size."""
    n_x, n_y = len(x), len(y)
    more = sum(1 for xi in x for yi in y if xi > yi)
    less = sum(1 for xi in x for yi in y if xi < yi)
    return (more - less) / (n_x * n_y)


def run_sanity_checks(data):
    """Run sanity checks and return pass/fail status with details."""
    results = data["results"]
    analysis = data["analysis"]

    checks = []
    all_passed = True

    # Extract per-edge data
    updates = [r for r in results if r["relation_type"] == "updates"]
    obsoletes = [r for r in results if r["relation_type"] == "obsoletes"]

    updates_takeover = [r["takeover_ratio"] for r in updates]
    obsoletes_takeover = [r["takeover_ratio"] for r in obsoletes]
    all_takeover = updates_takeover + obsoletes_takeover

    # Check 1: All takeover values in [0, 1]
    in_range = all(0 <= t <= 1 for t in all_takeover)
    checks.append({
        "name": "Takeover values in [0, 1]",
        "passed": in_range,
        "expected": "All values in [0, 1]",
        "actual": f"Min={min(all_takeover):.4f}, Max={max(all_takeover):.4f}"
    })
    if not in_range:
        all_passed = False

    # Check 2: Sample sizes match
    n_updates_actual = len(updates)
    n_obsoletes_actual = len(obsoletes)
    size_match = (n_updates_actual == EXPECTED["n_updates"] and
                  n_obsoletes_actual == EXPECTED["n_obsoletes"])
    checks.append({
        "name": "Sample sizes match",
        "passed": size_match,
        "expected": f"Updates={EXPECTED['n_updates']}, Obsoletes={EXPECTED['n_obsoletes']}",
        "actual": f"Updates={n_updates_actual}, Obsoletes={n_obsoletes_actual}"
    })
    if not size_match:
        all_passed = False

    # Check 3: Median updates
    median_updates_actual = np.median(updates_takeover)
    median_updates_ok = abs(median_updates_actual - EXPECTED["median_updates"]) < EXPECTED["tolerance"]
    checks.append({
        "name": "Median takeover (Updates)",
        "passed": median_updates_ok,
        "expected": f"{EXPECTED['median_updates']:.3f} ± {EXPECTED['tolerance']}",
        "actual": f"{median_updates_actual:.4f}"
    })
    if not median_updates_ok:
        all_passed = False

    # Check 4: Median obsoletes
    median_obsoletes_actual = np.median(obsoletes_takeover)
    median_obsoletes_ok = abs(median_obsoletes_actual - EXPECTED["median_obsoletes"]) < EXPECTED["tolerance"]
    checks.append({
        "name": "Median takeover (Obsoletes)",
        "passed": median_obsoletes_ok,
        "expected": f"{EXPECTED['median_obsoletes']:.3f} ± {EXPECTED['tolerance']}",
        "actual": f"{median_obsoletes_actual:.4f}"
    })
    if not median_obsoletes_ok:
        all_passed = False

    # Check 5: Cliff's delta
    delta_actual = cliffs_delta(obsoletes_takeover, updates_takeover)
    delta_ok = abs(delta_actual - EXPECTED["cliffs_delta"]) < EXPECTED["tolerance"]
    checks.append({
        "name": "Cliff's delta",
        "passed": delta_ok,
        "expected": f"{EXPECTED['cliffs_delta']:.3f} ± {EXPECTED['tolerance']}",
        "actual": f"{delta_actual:.4f}"
    })
    if not delta_ok:
        all_passed = False

    # Check 6: Mann-Whitney U test significance
    stat, p_value = stats.mannwhitneyu(obsoletes_takeover, updates_takeover, alternative='greater')
    p_ok = p_value < 0.0001
    checks.append({
        "name": "Mann-Whitney p < 0.0001",
        "passed": p_ok,
        "expected": "p < 0.0001",
        "actual": f"p = {p_value:.2e}"
    })
    if not p_ok:
        all_passed = False

    # Check 7: Direction check (swapping labels flips direction)
    # If we swap, delta should flip sign
    delta_swapped = cliffs_delta(updates_takeover, obsoletes_takeover)
    direction_ok = delta_swapped < 0 and delta_actual > 0
    checks.append({
        "name": "Direction check (swap flips sign)",
        "passed": direction_ok,
        "expected": "Original δ > 0, Swapped δ < 0",
        "actual": f"Original δ = {delta_actual:.3f}, Swapped δ = {delta_swapped:.3f}"
    })
    if not direction_ok:
        all_passed = False

    return all_passed, checks, {
        "updates_takeover": updates_takeover,
        "obsoletes_takeover": obsoletes_takeover,
        "n_updates": n_updates_actual,
        "n_obsoletes": n_obsoletes_actual,
        "median_updates": median_updates_actual,
        "median_obsoletes": median_obsoletes_actual,
        "cliffs_delta": delta_actual,
        "p_value": p_value
    }


def generate_figure1(updates_takeover, obsoletes_takeover, output_dir):
    """Generate Figure 1: Takeover distribution violin plot from REAL data."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6))

    # Violin plot with ACTUAL data
    parts = ax.violinplot([updates_takeover, obsoletes_takeover], positions=[1, 2],
                          showmeans=False, showmedians=False, showextrema=False)

    colors = ['#3498db', '#e74c3c']
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)

    # Box plot overlay
    bp = ax.boxplot([updates_takeover, obsoletes_takeover], positions=[1, 2], widths=0.15,
                    patch_artist=True, showfliers=False)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    for element in ['whiskers', 'caps', 'medians']:
        plt.setp(bp[element], color='black', linewidth=1.5)

    # Compute actual medians
    med_updates = np.median(updates_takeover)
    med_obsoletes = np.median(obsoletes_takeover)

    # Add median annotations
    ax.annotate(f'Median: {med_updates:.3f}', xy=(1, med_updates), xytext=(0.5, 0.25),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='#3498db'))
    ax.annotate(f'Median: {med_obsoletes:.3f}', xy=(2, med_obsoletes), xytext=(2.5, 0.55),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='#e74c3c'))

    # Labels
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f'Updates\n(n={len(updates_takeover)})',
                        f'Obsoletes\n(n={len(obsoletes_takeover)})'])
    ax.set_ylabel('Snapshot Takeover Ratio')
    ax.set_ylim(-0.05, 1.05)

    # Compute effect size for title
    delta = cliffs_delta(obsoletes_takeover, updates_takeover)
    ax.set_title(f"Figure 1: Citation Takeover by Relation Type\n(Cliff's δ = {delta:.3f}, p < 0.0001)")

    # Reference line
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Neutral (0.5)')
    ax.legend(loc='upper left')

    plt.tight_layout()

    png_path = output_dir / 'figure1_takeover_distribution.png'
    pdf_path = output_dir / 'figure1_takeover_distribution.pdf'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()

    return png_path, pdf_path


def write_edges_csv(data, output_path):
    """Write per-edge data to CSV for full reproducibility."""
    results = data["results"]

    with open(output_path, 'w') as f:
        f.write("edge_id,old_rfc,new_rfc,relation_type,total_citers,cites_old,cites_new,cites_both,takeover_ratio\n")
        for i, r in enumerate(results):
            f.write(f"{i},{r['old_rfc']},{r['new_rfc']},{r['relation_type']},"
                    f"{r['total_citers']},{r['cites_old']},{r['cites_new']},"
                    f"{r['cites_both']},{r['takeover_ratio']:.6f}\n")

    return output_path


def main():
    print("=" * 60)
    print("RFC Updates vs Obsoletes - Reproducibility Check")
    print(f"Run time: {datetime.now().isoformat()}")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading data...")
    data = load_data()
    print(f"  Loaded {len(data['results'])} edges")

    # Run sanity checks
    print("\n[2/5] Running sanity checks...")
    all_passed, checks, computed = run_sanity_checks(data)

    for check in checks:
        status = "✓ PASS" if check["passed"] else "✗ FAIL"
        print(f"  {status}: {check['name']}")
        print(f"         Expected: {check['expected']}")
        print(f"         Actual:   {check['actual']}")

    if not all_passed:
        print("\n" + "=" * 60)
        print("SANITY CHECK FAILED - Stopping execution")
        print("=" * 60)
        return False

    print("\n  All sanity checks passed!")

    # Generate figures from real data
    print("\n[3/5] Generating figures from actual data...")
    png_path, pdf_path = generate_figure1(
        computed["updates_takeover"],
        computed["obsoletes_takeover"],
        FIGURES_DIR
    )
    print(f"  Generated: {png_path.name}")

    # Write per-edge CSV
    print("\n[4/5] Writing per-edge CSV...")
    edges_csv_path = RESULTS_DIR / "edges_with_takeover.csv"
    write_edges_csv(data, edges_csv_path)
    print(f"  Written: {edges_csv_path.name}")

    # Compute hashes
    print("\n[5/5] Computing SHA-256 hashes...")
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "sanity_checks": "PASSED",
        "computed_statistics": {
            "n_updates": computed["n_updates"],
            "n_obsoletes": computed["n_obsoletes"],
            "median_updates": round(computed["median_updates"], 4),
            "median_obsoletes": round(computed["median_obsoletes"], 4),
            "cliffs_delta": round(computed["cliffs_delta"], 4),
            "p_value": f"{computed['p_value']:.2e}"
        },
        "file_hashes": {}
    }

    files_to_hash = [
        RESULTS_DIR / "rfc_snapshot_results.json",
        edges_csv_path,
        png_path,
    ]

    for filepath in files_to_hash:
        if filepath.exists():
            h = sha256_file(filepath)
            manifest["file_hashes"][filepath.name] = h
            print(f"  {filepath.name}: {h[:16]}...")

    # Write manifest
    manifest_path = RESULTS_DIR / "reproducibility_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Manifest written to: {manifest_path.name}")

    print("\n" + "=" * 60)
    print("REPRODUCIBILITY CHECK COMPLETE - ALL PASSED")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
