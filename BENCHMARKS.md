# TFP Benchmark Families

Each benchmark family below specifies what it tests, what rival models it competes against, what counts as support, and what counts as falsification. If you want to challenge TFP, pick a benchmark and run it.

---

## 1. Stage-Function Binding

**What it tests:** Whether lifecycle stages carry non-interchangeable functional signatures — that is, whether TFP's specific assignment of operators to stages is the only one that recovers the observed structure.

**Data needed:** Domain with identifiable lifecycle stages and measurable operator-like features (exploration intensity, stabilization signals, reseeding behavior).

**Rival models:** All 3! = 6 alternative operator-stage permutations (for 3 operators); all 4! = 24 permutations when B_in is included.

**Support criteria:** TFP's predicted permutation ranks #1 with a significant gap over the next best alternative.

**Falsification criteria:** An alternative permutation ties or beats TFP on the binding search.

**Current status:** Validated in Paper 5 across 4 domains (IETF, USPTO, ClinicalTrials.gov, FR-EPA). 92% gap over next best.

---

## 2. Transition Topology

**What it tests:** Whether lifecycle phases follow an ordered, unidirectional flow (Q1 to Q2 to Q3 to Q4) rather than random or symmetric switching.

**Data needed:** Temporal sequence data with enough resolution to identify phase transitions.

**Rival models:** Symmetric explore-exploit models; random-switching models; bidirectional phase models.

**Support criteria:** Asymmetry index is statistically significant — transitions follow the predicted direction more often than chance.

**Falsification criteria:** Symmetric flow is observed, or reverse transitions (Q3 to Q2, Q4 to Q1) occur at comparable rates to forward transitions.

**Current status:** Validated in Paper 4 across 4 domains.

---

## 3. Reseeding vs. Re-exploration

**What it tests:** Whether post-failure behavior (Q4B) is a distinct reseeding process — targeting specifically novel options and returning to abandoned alternatives — rather than generic re-exploration.

**Data needed:** Choice data following failure events, with enough history to distinguish novel-seeking from random exploration.

**Rival models:** UCB (Upper Confidence Bound); Thompson Sampling; epsilon-greedy; Optimal Foraging Theory (OFT) patch-leaving.

**Support criteria:** Q4B behavior shows targeted novelty-seeking followed by return to previously abandoned options, distinct from Q2 exploration patterns.

**Falsification criteria:** Q4B is statistically indistinguishable from Q2 exploration — reseeding is just re-exploration with a different name.

**Current status:** Validated in Paper 4 (human bandit task, macroeconomic data, avian/mammalian foraging).

---

## 4. Dissociation Under Observability

**What it tests:** Whether systems maintain dual-memory separation (short-term exploration memory vs. long-term stabilization memory) under partial observability conditions.

**Data needed:** System with identifiable memory channels operating under varying observability.

**Rival models:** Single-channel models; unified-memory architectures.

**Support criteria:** Dissociation signature is present — the two memory channels respond differently to observability manipulations.

**Falsification criteria:** A single memory channel is sufficient to explain all observed behavior; dual-channel architecture adds no explanatory power.

**Current status:** Validated-simulation in Paper A. Real-data replication needed.

---

## 5. Learned-Rival Tests

**What it tests:** Whether data-driven models (HMMs, latent-state models) independently recover TFP's operator-stage binding when trained without TFP constraints.

**Data needed:** Same lifecycle data used for supervised TFP tests, but analyzed with unsupervised methods.

**Rival models:** Unconstrained k-state HMMs; latent Dirichlet allocation; any latent-variable model that can discover structure without TFP priors.

**Support criteria:** The learned model recovers states that align with TFP's predicted operator-stage binding.

**Note:** Convergence supports TFP (the structure is discoverable, not imposed). Divergence challenges TFP (the imposed structure may be an artifact of feature engineering).

**Current status:** Validated in Paper 5. HMM independently converges on TFP-predicted structure.

---

## 6. Cross-Domain Replication

**What it tests:** Whether the same TFP signatures appear across institutionally and mechanistically independent domains.

**Data needed:** Lifecycle data from multiple unrelated domains.

**Rival models:** Domain-specific models that explain each domain independently without shared structure.

**Support criteria:** Leave-one-domain-out analysis survives — removing any single domain does not eliminate the TFP signal.

**Falsification criteria:** A single domain drives the entire result; removing it collapses the cross-domain finding.

**Current status:** Validated in Paper 5 (4 domains, leave-one-out analysis shows robustness).

---

## How to Run a Benchmark

1. Pick a benchmark family above.
2. Read the Handoff Paper () for detailed methodology.
3. Pre-register your predictions before running the test.
4. Report all results, including negatives. See [CONTRIBUTING.md](CONTRIBUTING.md).
5. Open an issue in this repository with your findings.
