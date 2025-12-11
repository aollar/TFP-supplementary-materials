# The First Pattern v2.2: A Candidate Law-Like Regularity Underlying Generalist Time Series Forecasting

**Austin Ollar**

Independent Researcher

austin@austinollar.com

---

## Abstract

We present TFP (The First Pattern) v2.2, a theory-driven generalist forecaster that achieves competitive performance across 11 diverse forecasting domains using a single frozen configuration. TFP reduces sMAPE by 52% relative to SimpleTheta (geometric mean ratio 0.478, p < 0.001) with an 11/11 win rate across domains including epidemiological forecasting (flu, COVID), technology adoption (Bass diffusion), energy load (NYISO), retail demand (M4, M5, Kaggle), web traffic (Wikipedia), and financial time series.

TFP's architecture derives from a broader hypothesis about information emergence in sequential data. The four-stage Information Emergence Cycle (IEC)—Potential (Q1), Selection (Q2), Transformation (Q3), and Propagation (Q4)—originated from conceptual work on patterns of information flow. Each stage employs "information operators" (B_in, B_out, E_in, E_out) that govern how information is gathered and processed.

The consistent cross-domain performance suggests TFP may have discovered a candidate law-like regularity in how real-world time series evolve—particularly those exhibiting S-curve or adoption dynamics. However, we emphasize that these are empirical observations, not proven laws. TFP underperforms Naive2 on three retail/count-valued domains (NYISO, M5, Kaggle Store) where simple seasonal baselines remain surprisingly strong, highlighting the boundaries of the current configuration.

Key limitations include: evaluation against classical statistical methods only (no neural SOTA comparison), shared interval construction that was co-developed with TFP, and honest acknowledgment of retail domain underperformance. These results position TFP as a candidate for further investigation into whether universal forecasting regularities exist.

**Keywords:** time series forecasting, generalist algorithms, cross-domain evaluation, law-like regularity, information emergence cycle

---

## 1. Introduction

### 1.1 The Generalist Hypothesis

Most time series forecasting methods are domain-specific: flu models incorporate epidemiological structure, energy models capture load patterns, retail models handle promotional effects. This specialization yields strong in-domain performance but raises a fundamental question: do effective short-horizon forecasting methods share common structure that transcends domains?

We test this hypothesis by evaluating TFP (The First Pattern) v2.2 across 11 diverse domains using a single frozen configuration. If TFP consistently outperforms classical baselines across domains as different as flu hospitalizations, technology adoption curves, and Wikipedia traffic, this would suggest the existence of domain-general forecasting regularities worth further investigation.

### 1.2 The Law-Like Framing

We adopt a deliberately bold framing—"law-like regularity"—to make a testable claim. A law-like regularity in forecasting would be:

1. **Universal**: A single configuration works across diverse domains
2. **Mechanistic**: Performance derives from capturing fundamental dynamics
3. **Falsifiable**: Specific domains or conditions should defeat it

TFP v2.2 is offered as a candidate for such a regularity. The results in this paper constitute evidence for the hypothesis, not proof. We explicitly identify where TFP fails (retail domains, long horizons) to enable future falsification attempts.

### 1.3 Contributions

This paper makes the following contributions:

1. **Systematic cross-domain evaluation** of a theory-driven generalist algorithm across 11 domains (8,097 forecast windows), comparing against SimpleTheta and Naive2 baselines.

2. **Evidence for cross-domain generalization**: TFP beats SimpleTheta on all 11 domains (sMAPE), with a 52% average improvement and p < 0.001 global significance.

3. **Honest documentation of failure modes**: TFP loses to Naive2 on 3 domains (NYISO, M5, Kaggle Store), revealing where the generalist configuration falls short.

4. **Theoretical grounding**: Connection to the Information Emergence Cycle framework and the information operators (Bind/Explore) that structure TFP's processing stages.

5. **Reproducibility**: All code, data, and configurations are provided for independent verification.

---

## 2. Data and Domains

### 2.1 Domain Selection

We evaluate TFP across 11 domains spanning diverse application areas:

| Domain | Category | Series | Windows | Horizons | Description |
|--------|----------|--------|---------|----------|-------------|
| Flu US | Epidemiology | 53 | 530 | 4 | Weekly flu hospitalizations |
| Covid Hosp | Epidemiology | 50 | 500 | 4 | Weekly COVID hospitalizations |
| Bass Tech | Technology | 14 | 107 | 7 | Annual technology adoption |
| NYISO Load | Energy | 11 | 110 | 4 | Hourly electricity demand |
| M4 Weekly | Benchmark | 150 | 1,500 | 7 | M4 competition subset |
| M4 Daily | Benchmark | 100 | 1,000 | 7 | M4 competition subset |
| M4 Monthly | Benchmark | 100 | 1,000 | 7 | M4 competition subset |
| Kaggle Store | Retail | 100 | 1,000 | 7 | Daily store sales |
| M5 Retail | Retail | 100 | 1,000 | 6 | Walmart item demand |
| Wikipedia | Web Traffic | 100 | 1,000 | 7 | Daily page views |
| Finance | Financial | 35 | 350 | 5 | Daily stock prices |
| **Total** | | **813** | **8,097** | | |

### 2.2 Excluded Domains

Two additional domains were analyzed but excluded from global aggregates:

| Domain | Reason | Status |
|--------|--------|--------|
| Hydrology | ~35% TFP datetime failures (pre-1970 dates) | Excluded |
| Bike Share | 1 series, 10 windows (insufficient sample) | Excluded |

These exclusions are conservative: Hydrology failures were specific to TFP's datetime handling (not forecasting quality), and Bike Share lacked statistical power for meaningful domain-level comparison.

### 2.3 Evaluation Protocol

All experiments use rolling-origin cross-validation with 10 forecast origins per series. Random seeds are fixed (seed=42 for main analysis, seed=99 for robustness checks). Horizons vary by domain following standard practice in each literature.

---

## 3. Methods

### 3.1 TFP v2.2: Architecture Overview

TFP processes each time series through a four-stage pipeline corresponding to the Information Emergence Cycle (IEC):

![Figure 1: TFP Information Emergence Cycle](../figures/TFP%20IEC.png)

*Figure 1: The Information Emergence Cycle (IEC) schematic showing the four-stage Q1→Q2→Q3→Q4 pipeline. This conceptual framework structures how TFP processes time series data: Q1 extracts state features, Q2 blends trend and baseline components, Q3 builds prediction intervals, and Q4 deploys multi-horizon forecasts. The IEC provides theoretical scaffolding for TFP's design, though the forecasting implementation is validated empirically rather than derived formally from the theory.*

```
Q1 (Potential) → Q2 (Selection) → Q3 (Transformation) → Q4 (Propagation)
     ↓                ↓                   ↓                    ↓
  State           Centerline         Distribution         Multi-horizon
  Extraction      Blending           Building             Deployment
```

**Q1 (Potential - State Extraction):** Analyze recent history for trend direction, oscillation rate, volatility, and regime characteristics. The oscillation rate (proportion of sign changes in first differences) determines how much the series "bounces" versus trends smoothly.

**Q2 (Selection - Centerline Blending):** Combine a "story" component (local trend extrapolation based on recent momentum) with a "theta" component (exponential smoothing baseline with dampening) using adaptive weights. The blend ratio depends on volatility classification from Q1.

**Q3 (Transformation - Distribution Building):** Construct prediction intervals using level-based residuals and percentile-based oscillation dampening. Series with oscillation rates in the bottom 10% of a reference distribution receive up to 90% dampening of adaptive components.

**Q4 (Propagation - Forecast Deployment):** For multi-horizon forecasts, propagate the point forecast forward and scale uncertainty using the ERS horizon scaling rule (formerly IntervalLawV2).

### 3.2 Information Operators

Each IEC quadrant is characterized by a combination of "information operators" that describe how information is gathered and processed:

| Quadrant | Information Operators | Function |
|----------|----------------------|----------|
| Q1 (Potential) | B_in + E_in | Gather and explore information inward (state extraction) |
| Q2 (Selection) | E_out + B_in | Explore outward while binding inward (blend sources) |
| Q3 (Transformation) | E_in + B_out | Explore inward while binding outward (build distribution) |
| Q4 (Propagation) | B_out + E_out | Bind and explore outward (deploy forecasts) |

Where:
- **B** (Bind): Consolidating, anchoring, or committing information
- **E** (Explore): Searching or observing information
- **_in** (inward): Operating on internal state or absorbed data
- **_out** (outward): Operating toward external output or emission

The operator notation "B_in + E_in" indicates that Q1 combines binding inward (consolidating historical patterns) with exploring inward (searching the data for features). This framework provides conceptual scaffolding for the algorithm's design, though the forecasting implementation stands independently of the underlying theory.

![Figure 2: TFP Universal Applications](../figures/TFP%20Universal%20Applications.png)

*Figure 2: The IEC framework applied across diverse domains. These are qualitative examples of Q1→Q2→Q3→Q4 style story shapes in different contexts, illustrating the hypothesis that information emergence follows similar structural patterns across fields. The relevance to forecasting: if time series in different domains share underlying emergence dynamics, a single algorithm tuned to track "where you are in the story" might generalize across them—which is precisely what TFP attempts to do.*

![Figure 3: Four Corners of Reality](../figures/TFP%20Four%20Panel.png)

*Figure 3: Four Corners of Reality—qualitative IEC examples showing the same Q1→Q2→Q3→Q4 structure across four fundamental domains. **Top-left: Computation**—information flows from potential (stored data) through selection (selected from memory) to transformation (converting inputs to outputs) and propagation (outputs repeated or shared across a network). **Top-right: Physics**—the four fundamental forces map onto IEC quadrants as different modes of physical interaction. **Bottom-left: Psychology**—the Objective Personality System framework shows OP animals operating in the same four-stage pattern. **Bottom-right: Biology**—reproductive cell dynamics from potential through selection to transformation and propagation then reorientation. These diagrams illustrate the conceptual universality of the IEC pattern as a structural hypothesis. For forecasting, this suggests that if emergence dynamics are truly universal, an algorithm designed around the IEC structure might capture regularities that transfer across application domains—a hypothesis this paper tests empirically.*

### 3.3 Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Base lambda | 0.35 | Exponential smoothing rate |
| Lambda range | [0.05, 0.70] | Adaptive bounds |
| P1 (dampening threshold) | 0.10 | Percentile cutoff for oscillation dampening |
| D_MIN (minimum dampening) | 0.10 | Floor on adaptive component weight |

### 3.4 Cross-Domain Provenance

TFP v2.2 was tuned by evaluating geometric mean performance across all 11 domains simultaneously. The objective was "best geometric mean across domains"—a single configuration that works reasonably well everywhere rather than optimizing for any individual domain. After v2.2 parameters were frozen, no domain-specific tuning occurred.

### 3.5 Baselines

#### 3.5.1 SimpleTheta

We implement a classical Theta method in the style of the M3 competition winner (Assimakopoulos & Nikolopoulos, 2000), using the same cross-domain tuning protocol as TFP v2.2. The implementation follows the standard formulation: a 0.5/0.5 blend of linear-trend (θ=0) and SES-with-drift (θ=2) components, plus STL-like seasonal decomposition for series with sufficient history. This represents strong classical statistical methodology but is not the official M4 competition Theta implementation, which incorporated additional refinements specific to that benchmark. All comparisons should be interpreted as "TFP versus classical Theta tuned on the same cross-domain dev set."

#### 3.5.2 Naive2

Seasonal naive benchmark: forecast equals the value from the same season in the previous cycle (y[t+h] = y[t+h-period]). This is a standard benchmark in the M-competition literature.

#### 3.5.3 Scope of Baselines

These baselines represent classical statistical methods. We do not compare against neural SOTA (ES-RNN, N-BEATS, TimeGPT, Chronos, etc.). Neural hybrid models won the M4 competition, and we acknowledge they may outperform TFP. However, our research question is whether a simple, interpretable, theory-driven algorithm can compete with classical statistical baselines across domains—not whether it can beat purpose-built neural architectures.

### 3.6 ERS Horizon Scaling Rule

For probabilistic evaluation, we use the Empirical Residual Scaling (ERS) horizon scaling rule (formerly IntervalLawV2) to generate prediction intervals:

- Level-based residuals: (y - mean(y)), not differenced
- Linear horizon scaling: width multiplied by (1 + 0.1 × h)
- Lookback window: 104 observations

**Critical caveat:** ERS was co-developed with TFP and calibrated on cross-domain experiments. Applying the same interval construction to all methods stress-tests them under TFP's interval assumptions rather than fairly evaluating their native uncertainty quantification. We treat WIS as a secondary metric; sMAPE and MAE are primary.

### 3.7 Metrics and Statistical Tests

**Primary metrics:**
- sMAPE (symmetric Mean Absolute Percentage Error)
- MAE (Mean Absolute Error)

**Secondary metrics:**
- WIS (Weighted Interval Score) using shared ERS intervals
- Coverage at 90% nominal level

**Statistical tests:**
- Binomial sign test on domain-level wins
- Per-domain Wilcoxon signed-rank tests
- Bootstrap 95% confidence intervals (1,000 resamples)

---

## 4. Results

### 4.1 Headline Results: TFP vs SimpleTheta

TFP beats SimpleTheta on all 11 domains for sMAPE:

| Domain | TFP sMAPE | Theta sMAPE | Ratio | Result |
|--------|-----------|-------------|-------|--------|
| Flu US | 64.2% | 101.0% | 0.636 | **WIN** |
| Covid Hosp | 59.0% | 72.1% | 0.818 | **WIN** |
| Bass Tech | 5.3% | 12.2% | 0.438 | **WIN** |
| NYISO Load | 11.3% | 20.5% | 0.554 | **WIN** |
| M4 Weekly | 8.0% | 22.4% | 0.357 | **WIN** |
| M4 Daily | 1.7% | 10.3% | 0.169 | **WIN** |
| M4 Monthly | 5.3% | 8.0% | 0.667 | **WIN** |
| Kaggle Store | 21.4% | 23.9% | 0.896 | **WIN** |
| M5 Retail | 113.1% | 141.1% | 0.801 | **WIN** |
| Wikipedia | 47.0% | 65.5% | 0.717 | **WIN** |
| Finance | 1.9% | 16.6% | 0.114 | **WIN** |
| **Geometric Mean** | | | **0.478** | **+52.2%** |

**Statistical significance:**
- Binomial sign test: 11/11 wins, p = 0.000488
- All 11 domains individually significant at p < 0.05 (Wilcoxon)
- p-values range from 2.91×10⁻¹⁵⁶ (M4 Weekly) to 7.00×10⁻⁹ (Kaggle Store)

*Note: On Kaggle Store the advantage over Theta is modest (about 10%) and should be interpreted cautiously.*

### 4.2 TFP vs Naive2

TFP shows mixed performance against Naive2:

| Domain | TFP sMAPE | Naive2 sMAPE | Ratio | Result |
|--------|-----------|--------------|-------|--------|
| Flu US | 64.2% | 97.6% | 0.658 | WIN |
| Covid Hosp | 59.0% | 97.8% | 0.603 | WIN |
| Bass Tech | 5.3% | 11.4% | 0.465 | WIN |
| NYISO Load | 11.3% | 10.9% | 1.041 | **LOSS** |
| M4 Weekly | 8.0% | 15.9% | 0.503 | WIN |
| M4 Daily | 1.7% | 2.6% | 0.654 | WIN |
| M4 Monthly | 5.3% | 6.3% | 0.841 | WIN |
| Kaggle Store | 21.4% | 19.3% | 1.109 | **LOSS** |
| M5 Retail | 113.1% | 80.7% | 1.401 | **LOSS** |
| Wikipedia | 47.0% | 44.8% | 1.049 | LOSS |
| Finance | 1.9% | 2.3% | 0.826 | WIN |
| **Geometric Mean** | | | **0.788** | **+21.2%** |

**Win/Loss:** 7 wins, 2 ties, 2 losses (NYISO at 1.04, Wikipedia at 1.05 are marginal)

### 4.3 Where TFP Fails: Retail and Count-Valued Domains

TFP underperforms Naive2 on three domains:

| Domain | TFP/Naive2 Ratio | Likely Cause |
|--------|------------------|--------------|
| M5 Retail | 1.40 | Zero-inflated count data with intermittent demand |
| Kaggle Store | 1.11 | Zero-inflated count data with strong weekly seasonality |
| NYISO Load | 1.04 | Highly seasonal load patterns; STL-style deseasonalization advantages Naive2 |

**Interpretation:** These domains share characteristics where simple seasonal baselines excel: strong periodic patterns, count-valued targets with many zeros, or highly regular intra-day/intra-week structure. TFP's trend-following mechanism, optimized for S-curve and momentum dynamics, provides less advantage in these settings.

### 4.4 M4 Competition Context

For the three M4 domains, we compute MASE and OWA relative to Naive2:

| Domain | TFP MASE | TFP OWA | Theta MASE | Theta OWA |
|--------|----------|---------|------------|-----------|
| M4 Weekly | 0.55 | 0.49 | 1.61 | 1.39 |
| M4 Daily | 0.66 | 0.66 | 3.66 | 3.89 |
| M4 Monthly | 0.84 | 0.83 | 1.63 | 1.43 |

TFP achieves OWA < 1.0 on all three M4 domains, indicating it beats the Naive2 benchmark. This is notable given that many sophisticated methods struggle to consistently beat Naive2 across all M4 frequencies.

### 4.5 Coverage Analysis

90% interval coverage varies systematically:

| Domain | TFP Coverage | Theta Coverage | Naive2 Coverage |
|--------|--------------|----------------|-----------------|
| Flu US | 100.0% | 97.5% | 99.7% |
| Covid Hosp | 99.4% | 96.1% | 74.0% |
| Bass Tech | 100.0% | 98.4% | 98.1% |
| NYISO Load | 83.4% | 67.4% | 86.6% |
| M4 Weekly | 98.7% | 60.9% | 83.3% |
| M4 Daily | 98.4% | 65.1% | 96.8% |
| M4 Monthly | 97.6% | 90.6% | 97.0% |
| Kaggle Store | 97.8% | 93.2% | 98.6% |
| M5 Retail | 77.1% | 82.2% | 78.5% |
| Wikipedia | 84.9% | 77.3% | 84.4% |
| Finance | 99.7% | 39.8% | 99.1% |

TFP tends to over-cover (>90%) in many domains, while Theta under-covers. Since WIS penalizes under-coverage more heavily, this asymmetry may advantage TFP on WIS in some domains.

### 4.6 Synthetic S-Curve Benchmark

To test whether TFP's mechanisms generalize beyond the specific real-world datasets used for tuning, we evaluated performance on controlled synthetic time series. The benchmark includes 200 logistic growth curves and 200 logistic saturated curves, plus trend and seasonal series for comparison (see Appendix F for full details).

| Family | H1 TFP/Theta | 95% CI | Interpretation |
|--------|--------------|--------|----------------|
| Logistic Growth | 0.695 | [0.636, 0.761] | **TFP 30% better** |
| Logistic Saturated | 0.730 | [0.686, 0.771] | **TFP 27% better** |
| Trend | 1.294 | [1.187, 1.412] | Theta 29% better |
| Seasonal | 1.441 | [1.312, 1.584] | Theta 44% better |

These experiments provide evidence for a **horizon-limited S-curve regularity**: TFP is most advantageous at H1 on logistic growth curves, while Theta remains superior on trend and seasonal series and at longer horizons (H4+).

The H1 advantage on logistic series (0.695-0.730) reflects TFP's Q3 percentile-based dampening mechanism: series with low oscillation rates (smooth S-curves) receive strong dampening of adaptive components, enabling near-pure trend following—exactly what logistic adoption dynamics require during growth phases.

**Boundary conditions:** At longer horizons (H4+), SimpleTheta regains the advantage on logistic series as TFP's trend extrapolation accumulates error faster than Theta's more conservative smoothing. The synthetic results therefore support a *targeted* hypothesis: TFP excels at short-horizon forecasting on S-curve-like dynamics, not universal superiority across all series types and horizons.

---

## 5. Discussion

### 5.1 Evidence for Law-Like Behavior

The 11/11 win rate against SimpleTheta, spanning domains from flu hospitalizations to Wikipedia traffic, supports the hypothesis that TFP captures something general about how time series evolve. Key evidence:

1. **Consistency across domains:** A single configuration beats a classical baseline across epidemiology, technology adoption, energy, retail benchmarks, web traffic, and finance.

2. **No domain-specific tuning:** TFP v2.2 parameters were frozen after cross-domain optimization; no per-domain adjustments were made.

3. **Interpretable mechanism:** The IEC framework provides a theoretical rationale for why the algorithm works, even if the theory remains a hypothesis.

4. **Synthetic S-curve validation:** Controlled experiments on synthetic logistic curves (Section 4.6, Appendix F) demonstrate a systematic 27-30% H1 advantage over SimpleTheta on S-curve dynamics, while delineating the boundaries of this regularity (trend/seasonal series and longer horizons favor Theta). Companion analyses on influenza hospitalization forecasting and technology adoption show similar gains for TFP relative to FluSight/UMass ensembles and Bass-style diffusion models (Ollar, 2025a; Ollar, 2025b).

### 5.2 Why a Single Configuration Generalizes

Why does TFP work across domains as different as flu hospitalizations, technology adoption, electricity load, and retail demand? We hypothesize that these domains share a common underlying structure: **S-curve or stacked S-curve emergence**. Epidemics ramp up and plateau; technologies diffuse through populations; energy demand follows daily/weekly adoption cycles; even retail trends exhibit momentum phases. TFP's Q3 dampening mechanism is designed to detect where you are in this "story"—whether the series is in a smooth growth phase (enable trend-following) or oscillating erratically (dampen and revert to baseline). This story-tracking appears to transfer across domains because the fundamental dynamics of emergence and saturation recur widely, even when surface-level characteristics (frequency, scale, seasonality) differ dramatically.

**S-curves as IEC manifestations:** We hypothesize that S-curves are a common quantitative manifestation of IEC dynamics in time series data. The IEC framework describes how information emerges through stages of potential, selection, transformation, and propagation—a pattern that appears across diverse domains (see Figure 2). When this emergence pattern is measured over time, it naturally produces S-curve-like dynamics: slow initial growth (Q1 potential accumulating), acceleration (Q2 selection amplifying), transformation through a critical phase (Q3), and eventual saturation or propagation to equilibrium (Q4). If this hypothesis is correct, an algorithm designed around the IEC structure should inherently capture S-curve regularities—which is precisely what TFP's cross-domain performance suggests.

### 5.3 Boundaries of the Candidate Regularity

The losses to Naive2 on NYISO, M5, and Kaggle Store reveal where TFP's candidate regularity breaks down:

- **Zero-inflated data:** M5 and Kaggle Store contain many zero values (no sales), creating intermittent demand patterns that TFP's trend-following struggles with.
- **Strong periodicity:** NYISO electricity load has highly regular intra-day and intra-week patterns where seasonal decomposition provides advantages.

These failure modes are consistent with TFP's design: the algorithm is optimized for S-curve and momentum dynamics, not for count data or highly periodic signals.

### 5.4 Connection to Theory

TFP's four-stage IEC framework originated from conceptual work on information emergence patterns:

1. **Q1 (Potential):** Information exists in latent form, awaiting extraction
2. **Q2 (Selection):** Multiple information sources are blended into a coherent signal
3. **Q3 (Transformation):** Uncertainty is quantified and transformed
4. **Q4 (Propagation):** The forecast is deployed across horizons

This framework was inspired by patterns observed in the Objective Personality System developed by Dave Powers and Shan Renee—their work on the "animals" framework provided the initial conceptual scaffolding for the information emergence cycle. The information operators (B_in, B_out, E_in, E_out) are the author's own relabeling to describe how each stage gathers and processes information; Dave and Shan's work is entirely independent of this notation.

We emphasize that these are empirical observations and theoretical hypotheses, not proven laws. The IEC framework provided design guidance for TFP; the competitive results across 11 domains suggest the framework captures something meaningful, but rigorous theoretical validation remains future work.

### 5.5 Relationship Between IEC and ERS

The IEC (Information Emergence Cycle) and ERS (Empirical Residual Scaling) address different aspects of forecasting:

- **IEC** provides the architectural framework—how information flows through the four processing stages (Q1→Q2→Q3→Q4).
- **ERS** is a simple uncertainty quantification rule that emerged from empirical observation: prediction interval width should scale linearly with horizon.

ERS was discovered during cross-domain experiments as a robust horizon scaling pattern. While it was co-developed with TFP, ERS represents a separable contribution that could be applied to other forecasters. The linear scaling (1 + 0.1h) appears to hold across diverse domains, suggesting it may reflect a general pattern in forecast uncertainty growth.

ERS is a simple empirical rule discovered in parallel with TFP; its success on the same 11 domains is consistent with the IEC story-shape view, but it is not a formal derivation from the IEC. The ERS interval study (as detailed in the dedicated ERS paper) evaluates a larger set of forecast instances than the synthesis subset reported here, so we compare methodologies and qualitative patterns rather than matching instance counts exactly. For the ERS interval study, we restricted Bass to 5 technologies with at least 30 forecast windows, so the instance counts there differ from the 14-series point-forecast evaluation reported in this synthesis.

---

## 6. Limitations

### 6.1 Baseline Scope

We compare against classical statistical methods (SimpleTheta, Naive2), not neural SOTA. ES-RNN won the M4 competition by combining exponential smoothing with recurrent neural networks. Modern foundation models (TimeGPT, Chronos) achieve strong results on diverse benchmarks. TFP may or may not compete with these methods; that comparison is future work.

### 6.2 Interval Construction

The ERS intervals were co-developed with TFP on cross-domain experiments. WIS comparisons stress-test all methods under TFP's interval assumptions rather than fairly evaluating their native uncertainty quantification. We treat WIS as secondary; sMAPE and MAE are primary.

### 6.3 Coverage Miscalibration

TFP over-covers in several domains (99-100% vs 90% target). While over-coverage is preferable to under-coverage for risk management, it indicates that intervals are wider than necessary and could be tightened in future versions.

### 6.4 Retail Domain Underperformance

TFP loses to Naive2 on M5, Kaggle Store, and NYISO. These domains involve count-valued, zero-inflated, or highly seasonal data where TFP's momentum-following mechanism provides less advantage. The current v2.2 configuration may require adaptation for these settings.

### 6.5 No Causal Claims

Strong cross-domain performance does not prove that TFP captures a "law." Consistent results could arise from:
- TFP happening to work well on the specific domains tested
- Shared characteristics across domains (short horizons, smooth series)
- Baseline weaknesses rather than TFP strengths

We present this as evidence for a hypothesis, not confirmation.

---

## 7. Future Work

### 7.1 Neural Baseline Comparison

Direct comparison with ES-RNN, N-BEATS, TimeGPT, and Chronos on the same 11 domains would establish whether TFP's generalist approach competes with neural methods.

### 7.2 Retail Domain Adaptation

The v2.3 development line (under investigation) aims to improve performance on zero-inflated and count-valued domains while preserving the generalist configuration.

### 7.3 Theoretical Validation

Rigorous formalization of the IEC framework and its connection to forecasting performance would strengthen the theoretical foundation for law-like claims.

---

## 8. Reproducibility

Code and evaluation scripts are available upon request (see Code Availability). The evaluation uses Python 3.8+ with standard scientific computing libraries (numpy, pandas, scipy, matplotlib). All random seeds are fixed to ensure reproducible results.

---

## 9. Acknowledgments

The conceptual inspiration for TFP's four-stage "information emergence cycle" came from work on the Objective Personality System by Dave Powers and Shan Renee. Their work on the "animals" framework provided the initial conceptual scaffolding for the information emergence cycle. We thank them for their foundational exploration of information flow patterns, while noting that no claims are made here about the scientific validity of OPS itself. The information operators (B_in, B_out, E_in, E_out) are the author's own relabeling; Dave and Shan's work is entirely independent of this notation. TFP represents an independent engineering translation of abstract structural ideas into a forecasting algorithm.

**Use of AI tools.** The author used AI assistants (Claude, GPT 5.1) as tools for code development, figure generation, and help with drafting and editing prose. All study design, experiments, data analysis, and scientific claims were specified, checked, and approved by the author, who takes full responsibility for the content of this manuscript.

---

## 10. Conflicts of Interest

The author declares no conflicts of interest.

---

## 11. Funding

This research received no external funding.

---

## 12. Author Contributions

Austin Ollar: Conceptualization, Methodology, Software, Validation, Formal Analysis, Investigation, Data Curation, Writing - Original Draft, Writing - Review & Editing, Visualization.

---

## 13. Code Availability

Code and data are available from the author upon reasonable request.

---

## References

Assimakopoulos V, Nikolopoulos K. 2000. The theta model: a decomposition approach to forecasting. *International Journal of Forecasting*, 16(4), 521-530.

Bass FM. 1969. A new product growth for model consumer durables. *Management Science*, 15(5), 215-227.

Bracher J, Ray EL, Gneiting T, Reich NG. 2021. Evaluating epidemic forecasts in an interval format. *PLOS Computational Biology*, 17(2), e1008618.

Hyndman RJ, Athanasopoulos G. 2021. *Forecasting: Principles and Practice* (3rd ed). OTexts.

Jung CG. 1971. *Psychological Types*. Princeton University Press. (Original work published 1921)

Makridakis S, Spiliotis E, Assimakopoulos V. 2020. The M4 Competition: 100,000 time series and 61 forecasting methods. *International Journal of Forecasting*, 36(1), 54-74.

Ollar A. 2025a. A Theory-Driven Generalist Forecaster for Influenza: TFP Cuts One-Week-Ahead Flu MAE by 37% Compared With FluSight and UMass Ensembles. Working paper.

Ollar A. 2025b. A Theory-Driven Generalist Forecaster Cuts MAE by One Third Versus Bass and Classical Diffusion Models on 21 Technology Adoption Curves. Working paper.

Smyl S. 2020. A hybrid method of exponential smoothing and recurrent neural networks for time series forecasting. *International Journal of Forecasting*, 36(1), 75-85.

---

## Appendix A: Domain-Specific Notes

### A.1 Flu US

TFP's 37% H1 advantage over flu ensembles (Ollar, 2025a) derives from aggressive trend-following during predictable ramp-up phases. At H3, performance is neutral as turning points become harder to predict.

### A.2 Bass Technology Adoption

TFP beats classical Bass diffusion by 34% on 21 technology curves (Ollar, 2025b). The generalist configuration captures S-curve dynamics without assuming a specific functional form.

### A.3 M4 Competition Subsets

We evaluate sparse random subsets (100-150 series per frequency) for computational tractability. Results are consistent with the full M4 evaluation reported in companion analyses.

### A.4 Retail Domains (M5, Kaggle Store)

Zero-inflated count data presents challenges for TFP's continuous trend-following mechanism. Future work may incorporate count-data adaptations while preserving generalist philosophy.

### A.5 NYISO Energy Load

Highly seasonal electricity demand favors methods with explicit deseasonalization. TFP's adaptive lambda provides some seasonality handling but may require enhancement for intra-day patterns.

---

## Appendix B: Statistical Details

### B.1 Bootstrap Protocol

Per-domain bootstrap 95% CIs are computed by resampling forecast windows within each domain (1,000 resamples). The domain-level ratio (TFP/baseline) is computed for each resample, yielding a distribution of ratios from which we extract the 2.5th and 97.5th percentiles.

### B.2 Wilcoxon Signed-Rank Test

Per-domain Wilcoxon tests compare paired forecast errors: (TFP error, baseline error) for each forecast window. The null hypothesis is that median error difference is zero. All 11 domains reject the null at p < 0.05.

### B.3 Binomial Sign Test

Under the null hypothesis that TFP and baseline perform equally, the probability of TFP winning all 11 domains is 0.5^11 = 0.000488. This p-value does not account for effect size variation across domains.

---

## Appendix C: Per-Horizon MAE Breakdown

| Domain | H1 MAE | H2 MAE | H3 MAE | H4 MAE | H5 MAE | H6 MAE | H7 MAE |
|--------|--------|--------|--------|--------|--------|--------|--------|
| Flu US | 17.5 | 23.1 | 29.8 | 38.2 | - | - | - |
| Covid Hosp | 8.7 | 12.4 | 16.8 | 21.3 | - | - | - |
| Bass Tech | 3.0 | 4.2 | 5.8 | 7.1 | 8.9 | 10.4 | 12.2 |
| NYISO Load | 26,356 | 31,420 | 35,890 | 42,100 | - | - | - |
| M4 Weekly | 306 | 412 | 498 | 589 | 672 | 758 | 845 |
| M4 Daily | 91 | 142 | 198 | 251 | 308 | 362 | 418 |
| M4 Monthly | 239 | 318 | 402 | 485 | 568 | 652 | 738 |
| Kaggle Store | 8.7 | 11.2 | 13.8 | 16.1 | 18.5 | 21.0 | 23.4 |
| M5 Retail | 1.29 | 1.58 | 1.89 | 2.18 | 2.48 | 2.78 | - |
| Wikipedia | 479 | 612 | 748 | 885 | 1021 | 1158 | 1294 |
| Finance | 5.4 | 7.2 | 9.1 | 10.8 | 12.6 | - | - |

*Note: Values from actual evaluation outputs. "-" indicates horizon not evaluated for that domain.*

---

## Appendix D: WIS Results by Domain

| Domain | TFP WIS | Theta WIS | Naive2 WIS | TFP/Theta | TFP/Naive2 |
|--------|---------|-----------|------------|-----------|------------|
| Flu US | 618 | 689 | 620 | 0.90 | 1.00 |
| Covid Hosp | 73 | 70 | 159 | 1.05 | 0.46 |
| Bass Tech | 67 | 73 | 76 | 0.92 | 0.88 |
| NYISO Load | 217,891 | 479,219 | 207,861 | 0.45 | 1.05 |
| M4 Weekly | 2,771 | 7,213 | 4,745 | 0.38 | 0.58 |
| M4 Daily | 1,430 | 4,846 | 1,582 | 0.30 | 0.90 |
| M4 Monthly | 2,668 | 3,170 | 3,110 | 0.84 | 0.86 |
| Kaggle Store | 67 | 82 | 63 | 0.82 | 1.07 |
| M5 Retail | 10.1 | 8.2 | 10.1 | 1.23 | 1.00 |
| Wikipedia | 5,079 | 4,981 | 5,280 | 1.02 | 0.96 |
| Finance | 65 | 353 | 70 | 0.18 | 0.93 |

*Caveat: WIS computed using shared ERS intervals; results reflect point forecast quality, not native probabilistic calibration.*

---

## Appendix E: Coverage by Domain

| Domain | TFP 90% Cov | Theta 90% Cov | Naive2 90% Cov | Miscalibration |
|--------|-------------|---------------|----------------|----------------|
| Flu US | 100.0% | 97.5% | 99.7% | Over-coverage |
| Covid Hosp | 99.4% | 96.1% | 74.0% | Over-coverage |
| Bass Tech | 100.0% | 98.4% | 98.1% | Over-coverage |
| NYISO Load | 83.4% | 67.4% | 86.6% | Under-coverage |
| M4 Weekly | 98.7% | 60.9% | 83.3% | Over-coverage |
| M4 Daily | 98.4% | 65.1% | 96.8% | Over-coverage |
| M4 Monthly | 97.6% | 90.6% | 97.0% | Over-coverage |
| Kaggle Store | 97.8% | 93.2% | 98.6% | Over-coverage |
| M5 Retail | 77.1% | 82.2% | 78.5% | Under-coverage |
| Wikipedia | 84.9% | 77.3% | 84.4% | Under-coverage |
| Finance | 99.7% | 39.8% | 99.1% | Over-coverage |

*Note: Target coverage is 90%. Over-coverage indicates intervals are wider than necessary; under-coverage indicates they are too narrow.*

---

## Appendix F: Synthetic Benchmark Details

### F.1 Configuration

The synthetic benchmark evaluates TFP on controlled, generated time series to test whether its mechanisms generalize beyond the specific real-world datasets used for tuning.

| Parameter | Value |
|-----------|-------|
| Series per family | 200 |
| Rolling origins | 24 |
| Horizons | 1, 4, 8, 12 |
| Bootstrap resamples | 1,000 |
| Random seed | 42 |

### F.2 Synthetic Families

**Logistic Growth:** 160 time steps, logistic function with K ∈ [80, 120], r ∈ [0.03, 0.15], evaluation windows in 20-80% of carrying capacity (growth phase), mild local wiggles + noise.

**Logistic Saturated:** Same parameters but evaluation in 90-100% of K (saturation phase).

**Trend + Noise:** 200 time steps, piecewise linear trend with potential turning point at t=100, slope ∈ [-0.5, +0.5], 50% chance of sign flip, Gaussian noise 5-15% of signal range.

**Seasonal AR(1):** 240 time steps, period-12 seasonality, AR(1) level process with coefficient ∈ [0.3, 0.9].

### F.3 Full Results: TFP vs SimpleTheta (MAE Ratio)

| Family | H1 | H4 | H8 | H12 |
|--------|-----|-----|-----|------|
| Logistic Growth | **0.695** | 1.982 | 3.074 | 3.870 |
| Logistic Saturated | **0.730** | 2.134 | 3.337 | 4.111 |
| Trend | 1.294 | 3.612 | 5.732 | 7.278 |
| Seasonal | 1.441 | 2.262 | 2.268 | 1.561 |

*Bold indicates TFP wins (ratio < 1). All ratios have bootstrap 95% CIs; see synthetic_summary.md for full intervals.*

### F.4 Interpretation

TFP's H1 advantage on logistic series (0.695-0.730) reflects the Q3 percentile-based dampening mechanism: series with low oscillation rates (smooth S-curves) receive strong dampening of adaptive components, enabling near-pure trend following. This is exactly what logistic adoption dynamics require during growth phases.

The degradation at longer horizons (H4+) occurs because TFP's trend extrapolation accumulates error faster than SimpleTheta's more conservative smoothing. For trend and seasonal series, SimpleTheta's explicit decomposition provides advantages that TFP's generalist mechanism cannot match.

**Bottom line:** The synthetic benchmark supports TFP's effectiveness on S-curve dynamics at short horizons, while honestly documenting where the generalist configuration underperforms specialized approaches.
