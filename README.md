# The First Pattern (TFP) Research Program

**A proposed lifecycle grammar for adaptive systems -- papers, benchmarks, and handoff materials**

The First Pattern (TFP) proposes that adaptive systems under resource constraints converge on a common lifecycle grammar: four stages with non-interchangeable functional roles, connected by unidirectional transitions. This is not a metaphor -- it is a testable structural claim. The research program tests it empirically across domains including patent prosecution, clinical trials, federal rulemaking, internet standards, avian foraging, and human decision-making.

This repository contains the full paper canon, benchmark specifications, and everything an outside researcher needs to test, replicate, or challenge TFP.

---

**New here? Read [START_HERE.md](START_HERE.md).**

---

## Paper Canon

| Paper | Role | Title | Status | DOI |
|-------|------|-------|--------|-----|
| **Paper 4** | Behavioral flagship | Structured Reseeding After Failure | Preprint | [10.5281/zenodo.19157916](https://zenodo.org/records/19157916) |
| **Paper 5** | Structural flagship | Non-Interchangeable Stage Functions Across Staged Adaptive Systems | Preprint | TBD |
| Paper A | Simulation proof | Dual-Memory Dissociation Under Partial Observability | In review (ESWA) | [10.5281/zenodo.18929188](https://zenodo.org/records/18929188) |
| Paper B | Empirical bridge | Dissociation Test for Causal Attribution Under Partial Observability | In review (J. Power Sources) | [10.5281/zenodo.18809446](https://zenodo.org/records/18809446) |
| Paper C | Mechanism | Depletion-Gated Reorientation | Preprint | [10.5281/zenodo.18930571](https://zenodo.org/records/18930571) |
| Handoff Paper | Portable framework | The First Pattern Research Program: A Portable Framework | Preprint | TBD |

## Repository Structure

    TFP-supplementary-materials/
    |-- START_HERE.md                 New researcher entry point
    |-- README.md                     This file
    |-- VALIDATED_VS_OPEN.md          Status of every testable claim
    |-- BENCHMARKS.md                 Benchmark families with falsification criteria
    |-- CONTRIBUTING.md               How to contribute, report results, propose rivals
    |-- ROADMAP.md                    Research priorities and open problems
    |-- CITATION.cff                  Citation metadata
    |-- OSF_Preregistration.md        Pre-registration for Paper 5
    |-- papers/
    |   |-- TFP_Paper4_Flagship.pdf   Paper 4: behavioral flagship
    |   |-- TFP_Paper5_Flagship.docx  Paper 5: structural flagship
    |   |-- TFP_PaperA_DualMemory.pdf
    |   |-- TFP_PaperB_Battery.pdf
    |   |-- TFP_PaperC_Foraging.pdf
    |   +-- TFP_Handoff_Paper.docx    Step-by-step guide for outside researchers
    |-- figures/
    |   |-- iec_cycle.png             IEC cycle diagram
    |   |-- fig_killshot.png          Permutation mapping search result
    |   |-- fig2_heatmap.png          Cross-domain operator-stage heatmap
    |   |-- fig3_wrong_mapping.png    Wrong-mapping collapse
    |   |-- fig4_lodo.png             Leave-one-domain-out analysis
    |   +-- fig5_ctgov_enrichment.png
    +-- benchmarks/                   Benchmark tools (coming soon)

## Quick Links

| Resource | Description |
|----------|-------------|
| [Start Here](START_HERE.md) | Entry point for new researchers |
| [Benchmarks](BENCHMARKS.md) | How to test or attack TFP |
| [Validated vs. Open](VALIDATED_VS_OPEN.md) | What is confirmed and what is not |
| [Contributing](CONTRIBUTING.md) | How to contribute (including negative results) |
| [Roadmap](ROADMAP.md) | Research priorities and open problems |
| [Handoff Paper](papers/TFP_Handoff_Paper.docx) | Step-by-step guide for testing TFP on a new domain |

## Key Results

**Paper 4** demonstrated that post-failure reorientation is structured, not random. Across macroeconomic, avian, mammalian, and human decision-making domains, systems target specifically novel options after failure and return to abandoned alternatives within a characteristic timeframe. This rules out random re-exploration and generic explore-exploit models.

**Paper 5** demonstrated that lifecycle stages carry non-interchangeable functional signatures across four institutionally independent domains (IETF, USPTO, ClinicalTrials.gov, FR-EPA). Of six possible operator-stage assignments, only TFP prediction recovers the observed structure, with a 92% gap over the next best alternative. HMMs trained without TFP constraints independently converge on the same binding.

## Author

Austin Ollar -- ORCID: [0009-0002-9998-9287](https://orcid.org/0009-0002-9998-9287)

Website: [austinollar.com/the-first-pattern/](https://austinollar.com/the-first-pattern/)

## License

CC BY 4.0
