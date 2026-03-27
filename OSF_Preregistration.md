# OSF Preregistration: Non-Interchangeable Stage Functions in Staged Adaptive Systems

**Registration date:** 2026-03-26
**Author:** Austin Ollar (ORCID: 0009-0002-9998-9287)

---

## 1. Study Information

### Title
Non-Interchangeable Stage Functions in Staged Adaptive Systems: A Forced-Choice Test of Operator-Stage Binding Across Four Domains

### Research Questions
1. Do lifecycle stages in staged adaptive systems carry non-interchangeable functional signatures, as predicted by the Information Emergence Cycle (IEC)?
2. Is the IEC-predicted operator-stage assignment uniquely best among all possible assignments?
3. Does an unconstrained learned model independently recover the same operator-stage binding?

### Hypotheses
**H1 (Primary):** The IEC operator-stage mapping (E_out at Q2, E_in at Q3, B_out at Q4) produces stronger signal than all five alternative permutations across four domains.

**H2 (Secondary):** Wrong-stage permutation and wrong-operator assignment destroy the observed functional signatures in each domain individually.

**H3 (Tertiary):** A Gaussian Hidden Markov Model with k=4 latent states, fit without IEC labels, independently recovers operator-dominant states consistent with the IEC Q2-Q4 binding.

---

## 2. Design Plan

### Study Type
Observational, cross-domain analysis of existing institutional lifecycle data.

### Study Design
For each of four institutionally independent domains, we measure three operator-type metrics at three lifecycle stages (Q2, Q3, Q4). We test all six possible operator-stage permutations and compare total signal strength. The study is scoped to Q2-Q4; Q1 (generation/potential) is not testable in these domains and is reserved for future work.

### Domains (locked before analysis)
1. **IETF RFC Standards** (N = 7,748 documents)
2. **USPTO Patent Prosecution** (N = 15,600 full-cycle applications)
3. **ClinicalTrials.gov Registered Trials** (N = 46,669 studies)
4. **Federal Register EPA Rulemaking** (N = 148 matched NPRM-Final pairs)

---

## 3. Sampling Plan

### Existing Data
All datasets are pre-existing institutional records. No new data collection.

### Data Exclusion
- IETF: Documents without valid Q2 (WG adoption) dates excluded
- Patents: Applications without full-cycle stage data (Q1-Q4) excluded
- CT.gov: Studies with censored outcomes treated as censored, not as failures
- FR-EPA: Only matched NPRM-Final pairs with docket access included

### Sample Sizes
Fixed by available institutional records. No sample size calculation performed; institutional datasets are population-level.

---

## 4. Variables

### Stage Boundary Definitions (locked)

| Domain | Q2 Boundary | Q3 Boundary | Q4 Boundary |
|--------|-------------|-------------|-------------|
| IETF | WG adoption date | Last Call date | RFC publication date |
| Patents | First Office Action | NOA (Q4A) or Final Rejection (Q4B) | Grant or abandonment |
| CT.gov | Enrollment start date | Primary completion date | Results posted date |
| FR-EPA | NPRM publication date | (Q2-Q4 compressed) | Final Rule publication + effective date |

### Operator Definitions (locked)

**E_out (External coupling):** Increase in externally sourced connections, references, collaborators, or stakeholder engagement at a given stage.

**E_in (Internal transformation):** Magnitude of internal content change, refinement, narrowing, or analytical work at a given stage.

**B_out (Irreversible commitment):** Degree of permanent, publicly visible deployment or commitment that cannot be privately reversed.

### Operator Metrics Per Domain (locked)

| Domain | E_out Metric | E_in Metric | B_out Metric |
|--------|-------------|-------------|-------------|
| IETF | d2>d3 timing pattern (81.1%) | Text diff ratio d2 vs d3 (80.2%) | Irreversibility rate (100%) |
| Patents | Coupling rate tx_between (δ=0.34) | Boundary-excluded coupling (δ=0.97) | Grant/abandon definitional |
| CT.gov | T3 monotonicity (98.2%) | Outcome count enrichment (d=0.285) | Results posting irreversibility |
| FR-EPA | Comment volume (66.2%) | Text diff matched vs placebo (d=3.01) | Effective date markers (effect=1.452) |

---

## 5. Analysis Plan

### Primary Analysis: Permutation Mapping Search

For three operators (E_out, E_in, B_out) assigned to three stages (Q2, Q3, Q4), there are 3! = 6 possible assignments. For each domain, we construct a 3x3 signal matrix (operator × stage) where each cell represents the measured signal strength of that operator at that stage. For each of the 6 permutations, we compute the total "diagonal" signal (the sum of signals when each operator is at its assigned stage). The IEC prediction is that the assignment E_out→Q2, E_in→Q3, B_out→Q4 produces the highest total signal.

**Success criterion:** IEC assignment ranks #1 of 6 in total signal AND #1 in each individual domain.

**Kill criterion:** If any alternative assignment scores within 20% of the IEC assignment, the uniqueness claim fails.

### Secondary Analysis: Wrong-Mapping Collapse

For each domain, compare the signal under the correct IEC mapping against:
- Permuted stage labels (random assignment)
- Wrong-operator at correct stage (e.g., E_in metrics at Q2)
- Jittered stage boundaries (±90d, ±180d)

**Success criterion:** Permuted/wrong signals collapse to chance or null in all domains.

### Tertiary Analysis: HMM Learned Rival

Fit a Gaussian HMM with k=3 and k=4 latent states on pooled lifecycle feature sequences (E_out, E_in, B_out features at each stage). The HMM discovers states without IEC labels. Check whether the discovered states have operator-dominant profiles consistent with IEC binding.

**If HMM recovers IEC structure:** The binding is independently discoverable, supporting its reality.
**If HMM does not recover:** IEC adds unique theoretical value beyond data-driven discovery.

### Cross-Domain Analysis

- **Directional consistency:** Binomial test across 12 tests (3 signatures × 4 domains), H0: p = 0.5.
- **Leave-one-domain-out:** Remove each domain and retest. Result must survive at p < 0.01 in all four configurations.
- **Fisher's method:** Combine primary p-values across domains.

### Confound Controls

- CT.gov S3: Stratify by enrollment quintiles and trial phase. Effect must survive in ≥4/5 quintiles and ≥7/9 phases.
- Patents: Year stratification, tech center stratification, entity size, engaged-Q4B cohort.
- FR-EPA: Year stratification, office type, CFR part.

---

## 6. Multiple Comparisons

Primary analysis (permutation search) is a single forced-choice test, not requiring correction. Secondary analyses (12 directional tests) are reported with both uncorrected and Holm-Bonferroni-corrected p-values.

---

## 7. Scope and Limitations (pre-declared)

1. **Q1 not tested.** The generation/potential stage is not observable in these four institutional domains. The paper tests Q2-Q4 non-interchangeability.

2. **Institutional domains only.** All four domains are formal staged institutional systems. Generalization to biological, ecological, or informal systems is future work.

3. **Stage boundaries are researcher-assigned** from institutional structure, not algorithmically discovered. The permutation search and HMM rival partially address this concern.

4. **Metrics vary across domains.** Each domain uses domain-native observables. Cross-domain comparability rests on the operator-level theoretical mapping, not on identical metrics.

---

## 8. Kill Criteria

The flagship claim dies if:

1. Any alternative permutation scores within 20% of the IEC assignment in total signal.
2. The IEC assignment is not #1 in at least 3 of 4 domains.
3. Wrong-mapping does not produce signal collapse in at least 3 of 4 domains.
4. Removing any single domain causes the cross-domain binomial to become non-significant (p > 0.05).
5. CT.gov S3 does not survive size control in at least 3/5 enrollment quintiles.
6. The HMM rival recovers a clearly superior alternative structure that contradicts IEC predictions.

---

## 9. Data Availability

All datasets are publicly available:
- IETF: https://datatracker.ietf.org
- Patents: Google BigQuery patents-public-data.uspto_oce_pair
- CT.gov: https://aact.ctti-clinicaltrials.org
- FR-EPA: https://www.federalregister.gov + https://api.regulations.gov
