# quantile_regression_pdlp — Ecosystem Gap Audit & Gap-Filling Plan

## Purpose

This document is a deeper audit of the current `quantile_regression_pdlp` package and the surrounding quantile / uncertainty ecosystem.

It focuses on two questions:

1. **What meaningful gaps still exist in the package, relative to the broader ecosystem?**
2. **What should be built next to fill those gaps without diluting the package’s core advantage?**

This is a **practical ecosystem audit**, not a full literature review. It is based on:
- the current repository structure, docs, examples, changelog, and benchmark scripts
- current public package ecosystems around quantile modeling, conformal prediction, forest-based quantiles, and probabilistic prediction

---

# Executive Summary

The package is already stronger than a typical “new quantile regressor” project.

It already has:
- joint multi-quantile fitting
- non-crossing support
- multi-output regression
- analytical / kernel / cluster-robust / bootstrap SEs
- conformalized quantile regression
- evaluation metrics
- crossing diagnostics and rearrangement
- censored quantile regression
- formula support
- sparse solving
- validation against sklearn and R `quantreg`

That means the next gaps are **not** mostly “missing core methods.”

The biggest remaining gaps are:

## Tier 0 — must fix now
1. **Benchmark visibility gap**
2. **Benchmark artifact / reporting gap**
3. **Cross-library interoperability gap**

## Tier 1 — strongest product differentiation
4. **Calibration diagnostics depth gap**
5. **Advanced conformal methods gap**
6. **Decision-support / model-selection gap**

## Tier 2 — model family expansion
7. **Expectile regression gap**
8. **Additive / spline quantile modeling gap**
9. **Distributional interface gap**

## Tier 3 — advanced depth
10. **Expanded censored / survival workflow gap**
11. **Time-series / non-exchangeable coverage gap**
12. **Published reproducible benchmark dashboard gap**

If addressed well, the package can position itself as:

> The most credible sklearn-native toolkit for constrained linear quantile modeling, inference, calibration, and evaluation — with benchmarked comparisons against the rest of the quantile ecosystem.

---

# What the Repository Already Does Well

The current repository already includes a broad set of capabilities.

## Core strengths already present
- optimization-based quantile regression on OR-Tools (PDLP, GLOP, scipy sparse)
- sklearn-style API (`fit`/`predict`/`score`, pipeline-compatible)
- statsmodels-style `summary()` with coefficients, SEs, p-values, CIs
- joint multi-quantile fitting with non-crossing LP constraints
- multi-output regression
- L1, elastic net, SCAD, MCP regularizers
- analytical, kernel (sandwich), cluster-robust, and bootstrap SEs
- conformalized quantile regression (`ConformalQuantileRegression`)
- standalone metrics module: pinball loss, coverage, interval score, crossing rate/magnitude, full evaluation report
- crossing diagnostics (`check_crossing`, `crossing_summary`) and rearrangement (`rearrange_quantiles`)
- censored quantile regression (`CensoredQuantileRegression`)
- prediction intervals, pseudo R², quantile process plots
- formula interface via patsy
- 3 examples: conformal intervals, evaluation, crossing fix
- docs pages: Home, Usage, Evaluation, Crossing, Conformal, API, Theory
- CI: Python 3.9/3.10/3.11 matrix on GitHub Actions
- v0.3.0 on PyPI

## Important repo observation
The package already includes benchmark-style validation scripts:
- `tests/bench_vs_sklearn.py` — 7 tests comparing coefficients, predictions, pinball loss, weighted and L1-regularized fits against sklearn's `QuantileRegressor`
- `tests/bench_vs_r.py` — 7 tests comparing against R's `quantreg` (coefficients, multiple quantiles, analytical/kernel SEs, extreme quantiles, weighted, censored QR vs `crq`)

But these are:
- stored under `tests/`
- excluded from default pytest runs via `pyproject.toml` `--ignore` flags
- not exposed in the docs navigation (nav has: Home, Usage, Evaluation, Crossing, Conformal, API, Theory — no Benchmarks)
- not surfaced as benchmark charts or tables in the README/docs
- produce no CSV, table, or figure artifacts

The regular test suite has **136 tests** across 5 test files (core, advanced features, conformal, metrics, postprocess) — all passing.

So the package has **benchmark evidence in development form**, but not **benchmark presentation in user-facing form**.

That is the top-priority product gap.

---

# Ecosystem Audit: Where Other Tools Currently Win

This package does **not** need to become all things to all users. But it should understand the major categories it sits beside.

## 1. Linear quantile regression tools
Examples:
- sklearn `QuantileRegressor`
- statsmodels `QuantReg`
- R `quantreg`

### What they are good at
- simple baseline quantile fitting
- familiar APIs
- classical statistical workflows
- trusted references for validation

### Where this package already wins
- joint multi-quantile fitting
- non-crossing support
- richer inference
- more regularization options
- conformal wrapper
- evaluation and crossing tools
- multi-output support
- censored QR support

### Remaining gap
The package needs to **prove this win more visibly** through published benchmarks, comparison tables, and reproducible result pages.

---

## 2. Tree / boosting quantile models
Examples:
- XGBoost quantile regression
- CatBoost `Quantile`, `MultiQuantile`, `Expectile`
- LightGBM quantile objective

### What they are good at
- nonlinear tabular relationships
- strong predictive performance
- scalable production workflows
- familiar model training patterns

### Where they still win
- practical nonlinear accuracy on many real datasets
- widespread industry adoption
- tree-based feature interactions out of the box

### Important implication
This package should **not** try to rebuild a full gradient boosting stack.

### The real gap
The package needs:
- adapters
- benchmark comparisons
- documentation on when linear constrained QR beats tree methods
- interoperability with external quantile predictors for evaluation / post-processing / conformalization

---

## 3. Quantile forests
Example:
- Zillow `quantile-forest`

### What they are good at
- nonparametric conditional quantiles
- arbitrary quantile prediction at inference time
- drop-in tree-ensemble workflows
- useful uncertainty estimates without retraining per quantile

### Gap relative to this package
The package lacks:
- an external adapter layer
- benchmark inclusion for quantile forests
- a “when to use forest vs LP-based QR” guide

---

## 4. Conformal prediction libraries
Example:
- MAPIE

### What they are good at
- broader conformal coverage options
- broader uncertainty workflows
- stronger packaging around calibration and risk control
- time-series and beyond-regression framing

### Where this package currently lags
The repo has split conformal calibration, but not:
- jackknife+
- CV+
- subgroup / conditional coverage diagnostics
- exchangeability checks
- richer calibration reporting
- time-aware conformal workflows

### Implication
Conformal support exists, but the package does not yet feel like a **calibration toolkit**.

---

## 5. Probabilistic prediction libraries
Example:
- NGBoost

### What they are good at
- full predictive distributions
- sampling
- likelihood-based evaluation
- uncertainty beyond a few fixed quantiles

### Gap relative to this package
This package has quantiles and intervals, but lacks:
- an approximate distribution interface over quantile grids
- CDF / PPF / sampling helpers
- probabilistic diagnostic tools beyond interval metrics

---

## 6. Smooth / additive quantile modeling
Examples in the wider ecosystem:
- spline-based or additive quantile workflows
- GAM-like approaches

### Current gap
The package is strongest in linear constrained QR, but not yet in:
- nonlinear interpretable quantile effects
- spline / basis expansion utilities
- additive quantile pipelines
- partial-effect style quantile docs

---

# Deep Gap List and How to Fill It

---

## Gap 1: Benchmark visibility gap
### Problem
Benchmark code exists, but users cannot easily see results.

### Why it matters
A package claiming to go beyond sklearn / statsmodels needs visible proof:
- accuracy parity or superiority where expected
- speed tradeoffs
- calibration tradeoffs
- crossing behavior
- sparse / large-scale benefits

### What to build
Create first-class benchmark reporting.

### Deliverables
- `docs/benchmarks.md`
- README benchmark section
- benchmark charts checked into the repo
- CSV benchmark artifacts under versioned paths
- benchmark reproduction instructions

### File plan
```text
docs/benchmarks.md
docs/assets/benchmarks/
benchmarks/README.md
benchmarks/results/
```

### Success criteria
A user should be able to answer:
- how does this compare to sklearn?
- how does it compare to R `quantreg`?
- what is the speed/accuracy tradeoff?
- when does PDLP help?

---

## Gap 2: Benchmark artifact / reporting gap
### Problem
Current benchmark scripts are validation-oriented tests, not publishing-oriented benchmark pipelines.

### What to change
Split “benchmark assertions” from “benchmark reports”.

### Recommendation
Keep:
- lightweight parity assertions in tests

Move / add:
- a separate `benchmarks/` directory
- scripts that produce CSV + plots + markdown summary

### Proposed structure
```text
benchmarks/
  README.md
  datasets.py
  run_linear_baselines.py
  run_sparse_benchmarks.py
  run_conformal_benchmarks.py
  run_cross_library.py
  report.py
  results/
    raw/
    tables/
    figures/
```

### Outputs to publish
- runtime by dataset size
- memory by dataset size if feasible
- pinball loss by tau
- interval coverage
- interval width
- crossing rate
- sparse problem scaling

---

## Gap 3: Cross-library interoperability gap
### Problem
The package can evaluate its own models, but it is not yet positioned as the central evaluation workbench for external quantile models.

### What to build
Add adapter utilities for external model predictions.

### Adapter targets
- sklearn `QuantileRegressor`
- statsmodels `QuantReg`
- XGBoost quantile
- CatBoost quantile / multi-quantile
- quantile-forest
- MAPIE outputs where sensible

### Why this matters
This turns the package from:
- “a quantile solver”

into:
- “a quantile modeling toolkit”

### Deliverables
```text
src/quantile_regression_pdlp/interoperability.py
benchmarks/adapters/
examples/compare_external_models.py
```

### API direction
Possible helpers:
- normalize external predictions to canonical `{tau -> output -> array}` form
- evaluate external models with internal metric suite
- rearrange / conformalize external quantile outputs

---

## Gap 4: Calibration diagnostics depth gap
### Problem
The package has conformal intervals and interval metrics, but calibration diagnostics are still shallow.

### Missing pieces
- coverage by subgroup
- coverage by prediction width bucket
- coverage by feature buckets
- observed vs nominal coverage plots
- sharpness vs coverage tables
- conditional coverage warnings

### What to build
A calibration diagnostics module.

### Proposed functions
```python
coverage_by_group(...)
coverage_by_bin(...)
nominal_vs_empirical_coverage(...)
sharpness_summary(...)
interval_miscoverage_profile(...)
```

### Docs to add
- “How to diagnose interval calibration”
- “Marginal vs conditional coverage”
- “Why conformal can still fail in subgroups”

---

## Gap 5: Advanced conformal methods gap
### Problem
Split conformal is a strong baseline, but users will increasingly expect more.

### Prioritized additions
1. jackknife+
2. CV+
3. grouped conformal diagnostics
4. time-aware conformal sketch
5. exchangeability diagnostics / warnings

### Important note
Do **not** build an over-ambitious conformal framework immediately.

### Best path
Start with:
- split conformal done very well
- richer diagnostics
- one additional method: `jackknife+` or `CV+`

### Proposed structure
```text
src/quantile_regression_pdlp/conformal.py
src/quantile_regression_pdlp/conformal_diagnostics.py
docs/conformal_advanced.md
examples/conformal_diagnostics.py
```

---

## Gap 6: Decision-support / model-selection gap
### Problem
Users need help deciding:
- which taus to fit
- whether to use raw QR or conformal
- when to use linear QR vs trees vs forests
- how to choose regularization
- how to tune for interval quality, not just pinball loss

### What to build
A model-selection guide plus utilities.

### Missing utilities
- quantile CV scorers
- interval-score-based tuning helper
- multi-objective summaries: pinball + coverage + width
- “recommendation report” style benchmark outputs

### Docs to add
- “Choosing the right quantile model”
- “When to use this package vs sklearn, XGBoost, CatBoost, MAPIE, quantile forests”

---

## Gap 7: Expectile regression gap
### Problem
The ecosystem increasingly includes expectile support, but the package currently appears quantile-only.

### Why this matters
Expectiles are:
- natural asymmetric regression analogues
- smoother to optimize than pinball loss
- useful for tail-sensitive modeling
- already surfaced in tools like CatBoost

### What to build
- `ExpectileRegression`
- optional `JointExpectileRegression`
- comparison docs: quantile vs expectile

### Recommended priority
Medium-high, after benchmark visibility and interoperability.

### What not to do
Do not mix the expectile implementation into quantile internals too aggressively if the optimization path differs substantially.

---

## Gap 8: Additive / spline quantile modeling gap
### Problem
The package’s linear core is strong, but users need a path to interpretable nonlinearity.

### Best solution
Do not implement a full GAM system first.

### Start with
- basis expansion helpers
- spline-compatible preprocessing
- sklearn pipeline examples
- quantile effect visualization

### Deliverables
```text
src/quantile_regression_pdlp/preprocessing.py
examples/spline_quantile_regression.py
docs/nonlinear.md
```

### Why this matters
This gives the package a middle lane between:
- strict linear QR
- full tree boosting

---

## Gap 9: Distributional interface gap
### Problem
Quantiles are available, but there is no object representing the implied conditional distribution.

### Why this matters
A distribution layer unlocks:
- interpolation between fitted quantiles
- approximate sampling
- CDF / PPF access
- risk analytics like VaR / CVaR approximations
- richer evaluation workflows

### What to build
A quantile-distribution wrapper.

### Proposed API
```python
qd = QuantileDistribution(quantiles=[0.1, 0.5, 0.9], values=pred)
qd.cdf(x)
qd.ppf(q)
qd.sample(1000)
qd.mean_approx()
qd.var_approx()
```

### Recommended priority
Medium. Strong differentiator, but not urgent before benchmarks and calibration depth.

---

## Gap 10: Expanded censored / survival workflow gap
### Problem
Censored QR exists, but it is still positioned as an advanced feature rather than a polished workflow.

### Missing pieces
- more examples
- stronger docs
- benchmark comparisons against R `crq`
- censoring diagnostics
- warnings under severe censoring
- guidance on when estimates are unstable

### Recommended additions
- `docs/censored.md`
- examples with right-censoring and left-censoring
- benchmark report vs R where feasible
- diagnostic summaries on censoring fraction and effective information

---

## Gap 11: Time-series / non-exchangeable coverage gap
### Problem
The current conformal framing is standard IID / exchangeable split calibration.

### Why it matters
Many interval users care about:
- forecasting
- drift
- time splits
- rolling recalibration

### Recommendation
Do not build a full forecasting library.

### Instead
Add a focused time-aware calibration note and examples:
- rolling calibration split
- blocked split
- warnings about exchangeability failure
- “not guaranteed under drift” guidance

This is more about correct usage and credibility than about adding a massive new subsystem.

---

## Gap 12: Published reproducible benchmark dashboard gap
### Problem
Even with `docs/benchmarks.md`, results can become stale or non-reproducible.

### What to build
A lightweight benchmark publication workflow.

### Recommended workflow
- manual or nightly GitHub Action
- generate benchmark CSVs and plots
- commit or publish docs artifacts
- include environment metadata
- label results by package version and commit SHA

### Suggested files
```text
.github/workflows/benchmarks.yml
benchmarks/report.py
benchmarks/results/metadata.json
```

---

# Benchmark Visibility: Dedicated Fix Plan

This deserves its own section because it is the most obvious current product gap.

## Current state
Benchmarks exist, but only as hidden development validation:
- `tests/bench_vs_sklearn.py` — 7 tests: median/multi-quantile coefficients, predictions, L1 regularization, weighted, extreme quantiles, pinball loss parity
- `tests/bench_vs_r.py` — 7 tests: median/multi-quantile coefficients, analytical/kernel SEs, extreme quantiles, weighted, censored QR vs `crq`

They are excluded from the default pytest run via `pyproject.toml`:
```
addopts = "-q --ignore=tests/bench_vs_sklearn.py --ignore=tests/bench_vs_r.py"
```

No `benchmarks/` directory exists. No benchmark results, CSVs, plots, or markdown summaries are published.

## Why this is not enough
Users cannot see:
- benchmark outputs
- benchmark plots
- hardware / environment assumptions
- calibration comparisons
- sparse scaling results
- crossing behavior comparisons

## Minimum viable benchmark publishing plan

### 1. Add benchmark docs page
Add:
```text
docs/benchmarks.md
```

Include:
- benchmark goals
- datasets used
- metrics used
- environment notes
- current results tables
- interpretation notes

### 2. Add benchmark nav entry
Update `mkdocs.yml`:
```yaml
nav:
  - Home: index.md
  - Usage: usage.md
  - Evaluation: evaluation.md
  - Crossing: crossing.md
  - Conformal: conformal.md
  - Benchmarks: benchmarks.md
  - API: api.md
  - Theory: theory.md
```

### 3. Move benchmark runners out of tests
Keep parity assertions in `tests/`, but create:
```text
benchmarks/
```

### 4. Publish benchmark figures
Recommended charts:
- runtime vs sample size
- runtime vs number of features
- pinball loss vs tau
- interval coverage vs nominal coverage
- interval width vs model
- crossing rate by model
- sparse runtime / memory comparison

### 5. Add README summary
Add a short benchmark section:
- one table
- one chart
- link to full benchmark page

### 6. Add reproducibility command block
Example:
```bash
python benchmarks/run_cross_library.py --out benchmarks/results/raw/latest.csv
python benchmarks/report.py --input benchmarks/results/raw/latest.csv --docs docs/assets/benchmarks/
```

---

# Recommended Benchmark Matrix

## Baseline matrix
These are the first comparisons to surface publicly.

| Category | Model | Why include |
|---------|------|-------------|
| Linear baseline | sklearn `QuantileRegressor` | Closest Python baseline |
| Classical stats | statsmodels `QuantReg` | Reference econometrics baseline |
| Gold reference | R `quantreg` | Trusted quantile regression reference |
| Internal | `QuantileRegression` | Main package model |

## Phase 2 matrix
| Category | Model | Why include |
|---------|------|-------------|
| Tree quantile | XGBoost quantile | Common nonlinear baseline |
| Tree quantile | CatBoost quantile / multi-quantile | Strong production baseline |
| Forest quantile | `quantile-forest` | Popular nonparametric quantile baseline |
| Calibration | MAPIE + sklearn base | Strong uncertainty baseline |

## Metrics to show
- train time
- predict time
- memory where practical
- pinball loss by quantile
- average pinball loss
- empirical coverage
- mean interval width
- crossing rate
- crossing magnitude
- sparse scaling behavior

---

# What the Package Should Not Try to Do

## Do not do these early
- build a full native boosting system
- build a full native quantile forest
- become a full forecasting library
- become a general probabilistic programming toolkit
- chase every conformal niche before benchmark credibility is established

## Why
The strongest lane is:
- constrained linear quantile methods
- inference
- calibration
- evaluation
- high-credibility comparisons

That lane is already distinctive.

---

# Prioritized Roadmap After This Audit

## P0 — immediate
1. Add benchmark page and benchmark artifacts
2. Move benchmark runners into a real `benchmarks/` directory
3. Add README benchmark summary
4. Add cross-library benchmark harness foundation

## P1 — next
5. Add calibration diagnostics
6. Add one advanced conformal method
7. Add interoperability adapters for external quantile models
8. Add “when to use what” decision guide

## P2 — expansion
9. Add expectile regression
10. Add spline / additive quantile workflow
11. Add quantile distribution wrapper

## P3 — advanced depth
12. Expand censored workflow and docs
13. Add time-aware calibration guidance
14. Add benchmark publishing workflow

---

# Concrete File Plan

## Benchmarks (Phase 1 — linear baselines)
```text
benchmarks/
  README.md
  run_linear_baselines.py       # PDLP vs sklearn vs statsmodels
  report.py                     # CSV → markdown tables + figures
  results/
    raw/                        # CSV outputs
    tables/                     # markdown tables
    figures/                    # PNG charts
```

## Benchmarks (Phase 2 — cross-library, deferred)
```text
benchmarks/
  run_cross_library.py          # adds XGBoost, CatBoost, quantile-forest
  run_conformal_benchmarks.py   # CQR vs MAPIE
  run_sparse_benchmarks.py      # large-scale scaling
```

Note: the large adapter directory (`adapters/` with 7 files) is over-engineered for Phase 1. Start with inline comparison in `run_linear_baselines.py` and factor adapters out only if reuse demands it.

## Docs
```text
docs/benchmarks.md
docs/model_selection.md
docs/nonlinear.md
docs/censored.md
docs/conformal_advanced.md
docs/assets/benchmarks/
```

## Package additions
```text
src/quantile_regression_pdlp/interoperability.py
src/quantile_regression_pdlp/calibration_diagnostics.py
src/quantile_regression_pdlp/distributions.py
src/quantile_regression_pdlp/expectile_regression.py
src/quantile_regression_pdlp/preprocessing.py
```

## Examples
```text
examples/compare_external_models.py
examples/benchmark_reproduction.py
examples/calibration_diagnostics.py
examples/spline_quantile_regression.py
examples/censored_quantile_workflow.py
```

---

# Success Criteria

This audit is successfully acted on if the package reaches the point where a user can immediately see:

## Benchmark credibility
- benchmark tables and figures are public
- comparisons are reproducible
- README points to benchmark results

## Ecosystem clarity
- clear explanation of where the package wins
- clear explanation of when to use tree / forest / conformal alternatives
- clear explanation of linear QR vs calibration vs nonlinear alternatives

## Product depth
- interval calibration is diagnosable, not just available
- external quantile models can be evaluated and post-processed
- model families expand in deliberate ways, not random ones

---

# Final Recommendation

The next best move is **not** another large modeling subsystem first.

The next best move is to turn the package’s existing strengths into visible, benchmarked, ecosystem-aware credibility.

## Recommended next build order
1. benchmark page + benchmark artifacts
2. benchmark harness + adapters
3. calibration diagnostics
4. advanced conformal method
5. expectile regression
6. spline / additive workflow
7. distribution wrapper
8. censored workflow polish

That order improves:
- trust
- discoverability
- differentiation
- adoption

before expanding the modeling surface area further.

---

# Short Version

If only three things get done next, make them:

1. **Show the benchmarks**
2. **Add cross-library benchmark / adapter infrastructure**
3. **Deepen calibration diagnostics**

That will do more for the package’s real standing in the ecosystem than adding a random new estimator first.
