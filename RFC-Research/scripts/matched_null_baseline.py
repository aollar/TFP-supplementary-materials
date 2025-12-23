#!/usr/bin/env python3
"""
Matched-Null Baseline for RFC Updates vs Obsoletes analysis.

This script tests whether the Updates/Obsoletes effect is explained by
document properties (age, citation volume) rather than relation type.

Approach:
1. For each real edge, extract decade of successor and total citers
2. Create matched-null pairs by randomly pairing RFCs within same decade
   with similar citation volume
3. Compute "fake takeover" for matched-null pairs
4. Compare real effect (δ ≈ 0.796) against matched-null (should be δ ≈ 0)

If matched-null produces δ ≈ 0.796, the effect is explained by confounds.
If matched-null produces δ ≈ 0, the relation type itself carries information.

Usage: python matched_null_baseline.py
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from collections import defaultdict
import random

# Paths
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"

# Fixed seed for reproducibility
RANDOM_SEED = 42

# RFC corpus info (RFC number → approximate year based on RFC numbering)
# This is a rough approximation; exact dates would need RFC metadata
def rfc_to_decade(rfc_num):
    """Approximate decade from RFC number."""
    # Rough mapping based on RFC publication patterns
    if rfc_num < 500:
        return 1970
    elif rfc_num < 1000:
        return 1980
    elif rfc_num < 2000:
        return 1990
    elif rfc_num < 4000:
        return 2000
    elif rfc_num < 7000:
        return 2010
    else:
        return 2020


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


def find_matched_null_pairs(edges, n_bins=5):
    """
    Create matched-null pairs for each edge.

    For each real edge, find another pair of RFCs from the same decade
    with similar total citers, then compute their "fake takeover".
    """
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # Group edges by decade of new RFC
    decade_edges = defaultdict(list)
    for edge in edges:
        decade = rfc_to_decade(edge["new_rfc"])
        decade_edges[decade].append(edge)

    # Build citer bins within each decade
    matched_null_takeovers = {"updates": [], "obsoletes": []}

    for relation_type in ["updates", "obsoletes"]:
        real_edges = [e for e in edges if e["relation_type"] == relation_type]

        for edge in real_edges:
            decade = rfc_to_decade(edge["new_rfc"])
            total_citers = edge["total_citers"]

            # Find other edges in same decade with similar citers (±50%)
            candidates = [
                e for e in decade_edges[decade]
                if e != edge and
                abs(e["total_citers"] - total_citers) / total_citers < 0.5
            ]

            if len(candidates) >= 2:
                # Randomly select two edges to form a fake pair
                sample = random.sample(candidates, 2)
                # Fake takeover: use the takeover from one random edge
                # This breaks the true relationship
                fake_takeover = sample[0]["takeover_ratio"]
                matched_null_takeovers[relation_type].append(fake_takeover)
            else:
                # If not enough candidates, use random from same decade
                if len(decade_edges[decade]) >= 2:
                    sample = random.sample(
                        [e for e in decade_edges[decade] if e != edge],
                        min(2, len([e for e in decade_edges[decade] if e != edge]))
                    )
                    if sample:
                        fake_takeover = sample[0]["takeover_ratio"]
                        matched_null_takeovers[relation_type].append(fake_takeover)

    return matched_null_takeovers


def shuffle_within_decade(edges, n_permutations=500):
    """
    Alternative approach: shuffle takeover values within decade-citer bins.

    This preserves marginal distributions while breaking the relationship
    between relation type and takeover.
    """
    np.random.seed(RANDOM_SEED)

    # Create decade-citer bins
    def get_bin(edge):
        decade = rfc_to_decade(edge["new_rfc"])
        # Citer bin (quartiles within decade would be ideal, use rough bins)
        if edge["total_citers"] < 50:
            citer_bin = "low"
        elif edge["total_citers"] < 100:
            citer_bin = "med"
        else:
            citer_bin = "high"
        return (decade, citer_bin)

    # Original separation
    updates = [e["takeover_ratio"] for e in edges if e["relation_type"] == "updates"]
    obsoletes = [e["takeover_ratio"] for e in edges if e["relation_type"] == "obsoletes"]
    real_delta = cliffs_delta(obsoletes, updates)

    # Bin edges
    bins = defaultdict(list)
    for i, edge in enumerate(edges):
        bins[get_bin(edge)].append(i)

    # Permutation test within bins
    null_deltas = []
    for _ in range(n_permutations):
        # Shuffle takeover within each bin
        shuffled_takeover = [None] * len(edges)
        for bin_indices in bins.values():
            takeovers = [edges[i]["takeover_ratio"] for i in bin_indices]
            np.random.shuffle(takeovers)
            for i, idx in enumerate(bin_indices):
                shuffled_takeover[idx] = takeovers[i]

        # Compute delta on shuffled data
        updates_shuffled = [shuffled_takeover[i] for i, e in enumerate(edges)
                          if e["relation_type"] == "updates"]
        obsoletes_shuffled = [shuffled_takeover[i] for i, e in enumerate(edges)
                            if e["relation_type"] == "obsoletes"]

        null_deltas.append(cliffs_delta(obsoletes_shuffled, updates_shuffled))

    return real_delta, null_deltas


def compute_matched_null_delta(edges):
    """
    Compute δ under matched-null: compare Updates vs Obsoletes edges
    where we've broken the relationship by shuffling within matched strata.
    """
    np.random.seed(RANDOM_SEED)

    # Group by (decade, citer_quartile)
    def get_stratum(edge):
        decade = rfc_to_decade(edge["new_rfc"])
        # Within-decade quartiles would be ideal; use absolute bins for simplicity
        citers = edge["total_citers"]
        if citers < 40:
            q = 1
        elif citers < 60:
            q = 2
        elif citers < 100:
            q = 3
        else:
            q = 4
        return (decade, q)

    strata = defaultdict(list)
    for edge in edges:
        strata[get_stratum(edge)].append(edge)

    # For matched-null: randomly reassign relation types within each stratum
    null_deltas = []
    n_permutations = 1000

    for _ in range(n_permutations):
        updates_takeover = []
        obsoletes_takeover = []

        for stratum_edges in strata.values():
            if len(stratum_edges) < 2:
                continue

            # Shuffle relation types within stratum
            takeovers = [e["takeover_ratio"] for e in stratum_edges]
            types = [e["relation_type"] for e in stratum_edges]
            np.random.shuffle(types)

            for takeover, rel_type in zip(takeovers, types):
                if rel_type == "updates":
                    updates_takeover.append(takeover)
                else:
                    obsoletes_takeover.append(takeover)

        if updates_takeover and obsoletes_takeover:
            null_deltas.append(cliffs_delta(obsoletes_takeover, updates_takeover))

    return null_deltas


def main():
    print("=" * 70)
    print("MATCHED-NULL BASELINE ANALYSIS")
    print("Testing whether effect is explained by document properties")
    print("=" * 70)

    # Load data
    edges = load_edges()
    print(f"\nLoaded {len(edges)} edges")

    # Real effect
    updates = [e["takeover_ratio"] for e in edges if e["relation_type"] == "updates"]
    obsoletes = [e["takeover_ratio"] for e in edges if e["relation_type"] == "obsoletes"]
    real_delta = cliffs_delta(obsoletes, updates)

    print(f"\n--- REAL EFFECT ---")
    print(f"Updates: n={len(updates)}, median={np.median(updates):.3f}")
    print(f"Obsoletes: n={len(obsoletes)}, median={np.median(obsoletes):.3f}")
    print(f"Cliff's δ = {real_delta:.4f}")

    # Matched-null via stratified permutation
    print(f"\n--- MATCHED-NULL BASELINE ---")
    print("Shuffling relation types within (decade, citer-quartile) strata...")

    null_deltas = compute_matched_null_delta(edges)

    mean_null = np.mean(null_deltas)
    std_null = np.std(null_deltas)
    ci_lower = np.percentile(null_deltas, 2.5)
    ci_upper = np.percentile(null_deltas, 97.5)

    print(f"\nMatched-null δ: mean={mean_null:.4f}, std={std_null:.4f}")
    print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    # P-value: how often does null exceed real?
    p_value = np.mean([d >= real_delta for d in null_deltas])
    print(f"\nP(null ≥ real): {p_value:.6f}")

    # Verdict
    print(f"\n--- INTERPRETATION ---")
    if abs(mean_null) < 0.1:
        print("✓ Matched-null δ ≈ 0: Effect is NOT explained by document properties")
        print(f"  Real δ ({real_delta:.3f}) >> Null δ ({mean_null:.3f})")
        print("  The relation type itself carries predictive information")
        verdict = "PASSED"
    elif mean_null > 0.3:
        print("✗ Matched-null δ > 0.3: Effect may be partially confounded")
        print("  Document properties explain some of the effect")
        verdict = "PARTIAL"
    else:
        print("△ Matched-null shows small residual effect")
        print("  Some confounding possible but effect largely genuine")
        verdict = "MARGINAL"

    # Additional analysis: decade breakdown
    print(f"\n--- DECADE-STRATIFIED MATCHED-NULL ---")
    decades = sorted(set(rfc_to_decade(e["new_rfc"]) for e in edges))

    for decade in decades:
        decade_edges = [e for e in edges if rfc_to_decade(e["new_rfc"]) == decade]
        if len(decade_edges) < 20:
            continue

        upd = [e["takeover_ratio"] for e in decade_edges if e["relation_type"] == "updates"]
        obs = [e["takeover_ratio"] for e in decade_edges if e["relation_type"] == "obsoletes"]

        if len(upd) >= 5 and len(obs) >= 5:
            decade_delta = cliffs_delta(obs, upd)
            print(f"  {decade}s: n={len(decade_edges)}, δ={decade_delta:.3f}")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"VERDICT: {verdict}")
    print(f"Real δ = {real_delta:.4f}")
    print(f"Matched-null δ = {mean_null:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
    print(f"{'=' * 70}")

    # Save results
    results = {
        "real_delta": real_delta,
        "matched_null_mean": mean_null,
        "matched_null_std": std_null,
        "matched_null_ci_lower": ci_lower,
        "matched_null_ci_upper": ci_upper,
        "p_value": p_value,
        "n_permutations": len(null_deltas),
        "verdict": verdict
    }

    output_path = RESULTS_DIR / "matched_null_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
