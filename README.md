# The First Pattern (TFP) — Supplementary Materials

## Overview

The First Pattern (TFP) is a proposed lifecycle grammar for adaptive systems. This repository contains supplementary materials, benchmark tools, and the handoff package for outside researchers.

## Research Papers

| Paper | Title | Status | DOI |
|-------|-------|--------|-----|
| **Paper 4** (Flagship) | Structured Reseeding After Failure | Preprint | [10.5281/zenodo.19157916](https://zenodo.org/records/19157916) |
| **Paper 5** (Flagship) | Non-Interchangeable Stage Functions Across Staged Adaptive Systems | Preprint | TBD |
| Paper A | Dual-Memory Dissociation Under Partial Observability | In review (ESWA) | [10.5281/zenodo.18929188](https://zenodo.org/records/18929188) |
| Paper B | Dissociation Test for Causal Attribution Under Partial Observability | In review (J. Power Sources) | [10.5281/zenodo.18809446](https://zenodo.org/records/18809446) |
| Paper C | Depletion-Gated Reorientation: A Minimal Mechanism for Efficient Spatial Foraging | Preprint | [10.5281/zenodo.18930571](https://zenodo.org/records/18930571) |
| **Handoff Paper** | The First Pattern Research Program: A Portable Framework | Preprint | TBD |

## Repository Structure

```
TFP-supplementary-materials/
├── README.md                    # This file
├── OSF_Preregistration.md       # Pre-registration for Paper 5
├── papers/
│   ├── TFP_Paper5_Flagship.docx # Paper 5 manuscript
│   └── TFP_Handoff_Paper.docx   # Handoff paper for outside researchers
├── figures/
│   ├── iec_cycle.png            # IEC diagram
│   ├── fig_killshot.png         # Permutation mapping search result
│   ├── fig2_heatmap.png         # Cross-domain operator-stage heatmap
│   ├── fig3_wrong_mapping.png   # Wrong-mapping collapse
│   ├── fig4_lodo.png            # Leave-one-domain-out analysis
│   └── fig5_ctgov_enrichment.png # CT.gov size-controlled enrichment
└── benchmarks/                  # Benchmark tools (coming soon)
```

## Key Results

**Paper 4** demonstrated that post-failure reorientation is structured, not random: systems target specifically novel options and return to abandoned alternatives within a characteristic timeframe. This was shown across macroeconomic, avian, mammalian, and human decision-making domains.

**Paper 5** demonstrated that lifecycle stages carry non-interchangeable functional signatures across four institutionally independent domains (IETF, USPTO, ClinicalTrials.gov, FR-EPA). Of six possible operator-stage assignments, only TFP's prediction recovers the observed structure, with a 92% gap over the next best alternative.

## How to Get Involved

See the Handoff Paper (`papers/TFP_Handoff_Paper.docx`) for:
- What TFP claims and what has been validated
- How to test TFP on a new domain (step-by-step)
- Benchmark tasks and rival models
- Open problems and contribution paths

## Author

Austin Ollar — ORCID: [0009-0002-9998-9287](https://orcid.org/0009-0002-9998-9287)

## License

CC BY 4.0
