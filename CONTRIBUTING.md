# Contributing to TFP Research

## Who This Repo Is For

This repository is for researchers who want to test, replicate, challenge, or extend The First Pattern. It is not a fan club. The most valuable contributions are the ones that could prove TFP wrong.

## Types of Contributions

### Replication
Test TFP predictions on a domain we have already studied. Confirm or deny our results with independent data or methods. See the Handoff Paper for methodology.

### Adversarial Testing
Try to break TFP. Find domains where the permutation search does NOT rank TFP first. Find datasets where wrong-mapping does NOT collapse the signal. Build a case that TFP is an artifact of feature engineering, domain selection, or statistical methodology.

### Rival Models
Propose and test alternative lifecycle models that make specific, quantitative operator-stage predictions. If a rival model matches or beats TFP on the permutation search, that is a major finding. See [BENCHMARKS.md](BENCHMARKS.md) for the benchmark families and falsification criteria.

### New Domains
Apply TFP to a domain not yet tested — especially non-institutional domains (biological, ecological, neural). The Handoff Paper has step-by-step instructions.

### Formalization
Mathematical formalization of the invariant hierarchy, operator algebra, lifecycle grammar, or transition topology. Moving TFP from empirical pattern to formal theory is a major open problem.

### Benchmark Building
Design new benchmarks or improve existing ones. Propose new rival models, sharper falsification criteria, or better statistical tests.

## Replication Standards

All contributions should meet these standards:

1. **Pre-register predictions.** Before running a TFP test on new data, state what you expect to find. This prevents post-hoc rationalization.
2. **Report all results, including negatives.** A null result is a contribution, not a failure. TFP needs to know where it breaks.
3. **Use holdout validation.** Split your data. Do not optimize on the same sample you test on.
4. **Report effect sizes, not just p-values.** Statistical significance without practical significance is not interesting.
5. **Make your analysis reproducible.** Share code, data (where possible), and methodology.

## Reporting Null or Negative Findings

Null results are genuinely important to this research program. If you test TFP and it fails:

1. Open an issue in this repository.
2. Tag it with .
3. Include: domain tested, stage mapping used, permutation search results (all 6 or 24 assignments), effect sizes, p-values, and your interpretation.
4. Do not bury it. Do not apologize for it. A clean negative is worth more than a noisy positive.

## Proposing a Benchmark or Rival Model

To propose a new benchmark:

1. Open an issue with the tag .
2. Specify: what it tests, what data it needs, what rival models it competes against, what counts as support, and what counts as falsification.
3. If possible, include a proof-of-concept analysis.

To propose a rival model:

1. Open an issue with the tag .
2. Specify: what the model predicts, how it differs from TFP, and what data would distinguish them.
3. Ideally, run both models on the same data and report comparative results.

## Code of Conduct

- Be rigorous.
- Be honest.
- Report negative results.
- Do not oversell.
- Disagree with evidence, not rhetoric.
- Credit prior work, including work that contradicts your findings.
