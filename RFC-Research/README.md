# RFC Updates vs Obsoletes: Reproduction Package

## Citation

Ollar, A. (2025). Updates Predict Low, Obsoletes Predict High Citation Takeover in Internet Standards.
Zenodo. https://doi.org/10.5281/zenodo.18028423

## Overview

This package contains all materials necessary to reproduce the results from:

**"Updates Predict Low, Obsoletes Predict High Citation Takeover in Internet Standards: Evidence for Propagation vs Reorientation Deployment Modes (Q4A/Q4B) in RFC Revisions"**

## Key Finding

RFC "Obsoletes" relationships show systematically higher citation takeover than "Updates" relationships:
- **Cliff's delta = 0.796** (95% CI: [0.77, 0.82])
- **p < 0.0001** (Mann-Whitney U test)
- Updates median takeover: 0.083
- Obsoletes median takeover: 0.773

## Directory Structure

```
rfc_updates_obsoletes/
├── README.md                  # This file
├── rfc_updates_obsoletes_paper.md   # Full paper (Markdown)
├── rfc_updates_obsoletes_paper.pdf  # Full paper (PDF)
├── data/
│   ├── edges_with_takeover.csv      # Per-edge data (n=1,371)
│   └── rfc_snapshot_results.json    # Snapshot analysis results
├── figures/
│   └── *.png                        # Paper figures
├── scripts/
│   ├── reproduce_all.py             # Main reproduction script
│   ├── matched_null_baseline.py     # Matched-null baseline analysis
│   ├── holdout_evaluation.py        # Holdout validation
│   └── holdout_config.json          # Holdout parameters
└── results/
    ├── holdout_results.json         # Holdout validation results
    ├── matched_null_results.json    # Matched-null results
    └── reproducibility_manifest.json # SHA-256 hashes
```

## Quick Start

### Requirements

```bash
pip install numpy scipy pandas matplotlib
```

### Run Main Reproduction Script

```bash
cd scripts
python reproduce_all.py
```

This will:
1. Load the per-edge data
2. Verify statistics match reported values (n=859/512, median=0.083/0.773, delta=0.796)
3. Regenerate Figure 1 from raw data
4. Compute SHA-256 hashes for verification

### Run Matched-Null Baseline

```bash
python matched_null_baseline.py
```

Verifies that document properties (age, citation volume) cannot explain the effect:
- Real delta: 0.796
- Matched-null delta: 0.156

### Run Holdout Validation

```bash
python holdout_evaluation.py
```

Verifies effect replicates on held-out data (25% holdout, frozen pipeline):
- Dev set delta: 0.789
- Holdout delta: 0.817

## Data Description

### edges_with_takeover.csv

Per-edge data for 1,371 RFC revision relationships:

| Column | Description |
|--------|-------------|
| old_rfc | Predecessor RFC number |
| new_rfc | Successor RFC number |
| relation_type | "updates" or "obsoletes" |
| total_citers | Documents citing old OR new |
| takeover_ratio | Fraction citing new among total |

### Expected Sanity Checks

The reproduce_all.py script verifies:

| Statistic | Expected | Tolerance |
|-----------|----------|-----------|
| n_updates | 859 | exact |
| n_obsoletes | 512 | exact |
| median_updates | 0.083 | +/- 0.01 |
| median_obsoletes | 0.773 | +/- 0.01 |
| cliffs_delta | 0.796 | +/- 0.01 |

## License

- Code: MIT License
- Data and manuscript: CC-BY 4.0

## Contact

Austin Ollar
austin@austinollar.com

## Acknowledgments

The conceptual inspiration for TFP's "Information Emergence Cycle" and the Q4A/Q4B distinction
came from work on the Objective Personality System by Dave Powers and Shan Renee.
