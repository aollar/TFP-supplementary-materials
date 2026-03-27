# Contributing to TFP Research

Thank you for your interest in The First Pattern (TFP) research program.

## Ways to Contribute

### Test TFP on a New Domain
The most valuable contribution is testing TFP predictions on a domain we have not studied. See the Handoff Paper (`papers/TFP_Handoff_Paper.docx`) for step-by-step instructions.

### Adversarial Testing
Try to break TFP. Find domains where the permutation search does NOT rank TFP first. Find datasets where wrong-mapping does NOT collapse the signal. Negative results are genuinely valuable.

### Build Stronger Rivals
Propose and test alternative lifecycle models that make specific operator-stage predictions. If a rival model matches or beats TFP on the permutation search, that is an important finding.

### Improve Operator Metrics
Develop better ways to measure B_in (internal stabilization), which is currently the least-tested operator.

### Formalize the Theory
Mathematical formalization of the invariant hierarchy, operator algebra, or lifecycle grammar would be a major contribution.

## How to Report Results

Open an issue on this repository with:
1. Domain tested
2. Stage mapping used
3. Permutation search results (all 6 or 24 assignments)
4. Whether TFP was #1 or not
5. Effect sizes and p-values
6. Any negative results

## Code of Conduct

Be rigorous. Be honest. Report negative results. Do not oversell.
