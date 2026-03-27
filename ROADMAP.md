# TFP Research Roadmap

## Current Canon (Completed Papers)

| Paper | Title | Status |
|-------|-------|--------|
| Paper 4 | Structured Reseeding After Failure | Preprint ([Zenodo](https://zenodo.org/records/19157916)) |
| Paper 5 | Non-Interchangeable Stage Functions Across Staged Adaptive Systems | Preprint (Zenodo pending) |
| Paper A | Dual-Memory Dissociation Under Partial Observability | In review (ESWA) |
| Paper B | Dissociation Test for Causal Attribution Under Partial Observability | In review (J. Power Sources) |
| Paper C | Depletion-Gated Reorientation | Preprint ([Zenodo](https://zenodo.org/records/18930571)) |
| Handoff Paper | The First Pattern Research Program: A Portable Framework | Preprint |

## Immediate Priorities

- [ ] Paper 5 upload to Zenodo with DOI
- [ ] OSF pre-registration for cross-domain replication
- [ ] Theory bible: comprehensive reference document covering all operators, quadrants, mappings, and invariants

## Open Benchmark Families

These are the active testing fronts. See [BENCHMARKS.md](BENCHMARKS.md) for full specifications.

- [ ] **Stage-Function Binding** — Can an alternative permutation beat TFP? (Currently: no, across 4 domains)
- [ ] **Transition Topology** — Is flow truly unidirectional? (Currently: yes, across 4 domains)
- [ ] **Reseeding vs. Re-exploration** — Is Q4B distinct from Q2? (Currently: yes, in bandit + foraging data)
- [ ] **Dissociation Under Observability** — Does dual-memory hold in real data? (Currently: simulation only)
- [ ] **Learned-Rival Tests** — Do unsupervised models recover TFP structure? (Currently: HMM convergence in Paper 5)
- [ ] **Cross-Domain Replication** — Does the signal survive leave-one-out? (Currently: yes, 4 domains)

## Mechanism-Search Frontier

The empirical pattern is established. The next question is: what generates it? Candidate generator classes to investigate:

- [ ] Minimal dynamical systems that produce TFP-like phase structure
- [ ] Information-theoretic constraints that force operator-stage binding
- [ ] Evolutionary game-theoretic models with lifecycle stages
- [ ] Thermodynamic or free-energy formulations of adaptive cycling
- [ ] Neural circuit architectures that naturally dissociate exploration and stabilization memory

## Long-Range Goals

- [ ] **Formalization**: Translate TFP from empirical grammar to formal mathematical theory (automata theory, category theory, or dynamical systems)
- [ ] **Cross-scale isomorphism**: Test whether the same grammar appears at cellular, organismal, and institutional scales simultaneously
- [ ] **Non-institutional domains**: Biological lifecycle data (cell division, immune response), ecological succession, neural development
- [ ] **Full 4-operator grammar**: Validate B_in and complete the 4! = 24-way permutation search
- [ ] **TFP-inspired architectures**: RL agents or optimization algorithms that use TFP structure as an inductive bias
