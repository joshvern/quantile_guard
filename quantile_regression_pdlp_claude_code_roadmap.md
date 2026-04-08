# quantile_regression_pdlp — Roadmap & Implementation Spec

## Purpose

Expand `quantile_regression_pdlp` from a strong optimization-based quantile regression library into a broader **quantile modeling toolkit** for tabular data.

### Current strengths (v0.2.0)
- LP-based quantile regression via OR-Tools (PDLP, GLOP) and scipy (HiGHS)
- Joint multi-quantile fitting with non-crossing constraints
- Analytical, bootstrap, kernel, and cluster-robust standard errors
- L1 / elastic net / SCAD / MCP regularization via LLA
- Censored quantile regression (right/left, Powell iterative)
- Prediction intervals, quantile process plots, pseudo R²
- Formula interface (patsy), multi-output, weighted regression
- sklearn-compatible API (`fit`, `predict`, `score`, `get_params`, `set_params`)

### What to build next
1. Standalone evaluation metrics and diagnostics
2. Crossing detection and rearrangement utilities
3. Conformalized quantile regression for calibrated intervals
4. Benchmark harness for cross-library comparison
5. Expectile regression, spline workflows, distribution abstraction

---

## Guiding Principles

1. **Do not compete with tree libraries** — wrap and benchmark against them instead
2. **Preserve sklearn compatibility** — `fit`/`predict`/`score`/`get_params`/`set_params`
3. **Keep optimization explicit** — users should know when they use LP vs post-hoc calibration
4. **Make uncertainty evaluation first-class** — coverage, width, crossing diagnostics
5. **Prefer modular composition** — reusable metrics, calibration, and postprocessing primitives

---

## Repository Structure

### Current layout
```text
src/quantile_regression_pdlp/
  __init__.py                    # exports QuantileRegression, CensoredQuantileRegression
  quantile_regression.py         # main estimator module (~1200 lines)
tests/
  test_quantile_regression.py    # 28 core tests
  test_advanced_features.py      # 46 feature tests
  bench_vs_sklearn.py            # benchmark (excluded from CI)
  bench_vs_r.py                  # benchmark (excluded from CI)
docs/
  index.md, usage.md, api.md, theory.md
```

### Planned additions (flat modules, not deep subpackages)
```text
src/quantile_regression_pdlp/
  metrics.py                     # Milestone 1: evaluation functions
  postprocess.py                 # Milestone 2: crossing detection + rearrangement
  conformal.py                   # Milestone 3: conformalized quantile regression
tests/
  test_metrics.py
  test_postprocess.py
  test_conformal.py
docs/
  evaluation.md
  crossing.md
  conformal.md
examples/
  evaluate_quantile_model.py
  fix_crossing_quantiles.py
  conformal_intervals.py
```

### API naming convention
The existing estimator names are **`QuantileRegression`** and **`CensoredQuantileRegression`** (not `QuantileRegressor`). All new code must use these names consistently.

---

## Phase A — Evaluation & Crossing (Milestones 1–2)

### Milestone 1: Metrics & Diagnostics

**Module**: `src/quantile_regression_pdlp/metrics.py`

Functions to implement:
```python
pinball_loss(y_true, y_pred, tau) -> float
multi_quantile_pinball_loss(y_true, y_pred_dict, taus) -> dict[float, float]
empirical_coverage(y_true, lower, upper) -> float
mean_interval_width(lower, upper) -> float
crossing_rate(predictions, taus) -> float
crossing_magnitude(predictions, taus) -> float
interval_score(y_true, lower, upper, alpha) -> float
quantile_evaluation_report(y_true, predictions, taus, lower=None, upper=None) -> dict
```

Notes:
- `predictions` is a 2D array `(n_samples, n_quantiles)` with columns corresponding to `taus`
- `QuantileRegression.score()` already computes negative mean pinball loss internally; these standalone functions allow evaluation of any model's predictions
- `quantile_evaluation_report` returns a flat dict summarizing all metrics (not a class)

**Tests**: `tests/test_metrics.py`
- Hand-checked pinball loss values
- Coverage = 1.0 when all points inside interval
- Coverage = 0.0 when all points outside
- Zero crossing rate for monotone predictions
- Positive crossing rate for intentionally crossed predictions
- Interval score matches known formula

**Docs**: `docs/evaluation.md`
**Example**: `examples/evaluate_quantile_model.py`

---

### Milestone 2: Crossing Detection & Rearrangement

**Module**: `src/quantile_regression_pdlp/postprocess.py`

Functions and class:
```python
check_crossing(predictions, taus) -> np.ndarray       # boolean mask (n_samples,)
crossing_summary(predictions, taus) -> dict            # rate, magnitude, worst rows
rearrange_quantiles(predictions, taus) -> np.ndarray   # per-row monotone sort
```

Notes:
- `QuantileRegression` already enforces non-crossing via LP constraints + isotonic projection at predict time. These utilities are for evaluating/fixing predictions from *any* quantile model.
- No `RearrangedQuantileEstimator` wrapper in this milestone — keep it simple.

**Tests**: `tests/test_postprocess.py`
- Rearranged output is monotone
- Shape preserved
- Crossing rate drops to zero after rearrangement
- Already-monotone input unchanged

**Docs**: `docs/crossing.md`
**Example**: `examples/fix_crossing_quantiles.py`

---

## Phase B — Conformal Calibration (Milestone 3)

### Milestone 3: Conformalized Quantile Regression

**Module**: `src/quantile_regression_pdlp/conformal.py`

```python
from quantile_regression_pdlp.conformal import ConformalQuantileRegression

cqr = ConformalQuantileRegression(
    base_estimator=QuantileRegression(tau=[0.05, 0.5, 0.95]),
    coverage=0.90,
    random_state=42,
)
cqr.fit(X_train, y_train)
intervals = cqr.predict_interval(X_test)  # dict with 'lower', 'upper', 'width'
cqr.empirical_coverage(X_val, y_val)
```

Scope:
- Split conformal calibration (train/calibration split)
- Nonconformity scores: max(lower - y, y - upper)
- Interval prediction with calibrated offset
- Coverage and width reporting

Not in scope (defer):
- CV+ / jackknife+ variants
- Locally adaptive conformal
- Rolling / time-aware calibration

**Tests**: `tests/test_conformal.py`
- Coverage ≥ nominal - epsilon on synthetic data
- Intervals have lower ≤ upper
- Width is positive
- Reproducible with fixed seed
- Clear error when base estimator has < 2 quantiles

**Docs**: `docs/conformal.md`
**Example**: `examples/conformal_intervals.py`

---

## Phase C — Benchmarking & Interop (Milestone 4)

### Milestone 4: Benchmark Harness

**Directory**: `benchmarks/` (outside `src/`, excluded from package)

```text
benchmarks/
  run_benchmark.py              # main runner script
  datasets.py                   # synthetic data generators
  adapters.py                   # optional wrappers for sklearn, statsmodels, etc.
  results/                      # output CSV/JSON
```

Target comparisons:
- Internal `QuantileRegression` (PDLP, GLOP, sparse)
- sklearn `QuantileRegressor`
- statsmodels `QuantReg` (if installed)
- LightGBM / XGBoost quantile (optional extras)

Metrics: pinball loss, coverage, width, crossing rate, runtime.

Keep full benchmarks out of CI. Add smoke test only.

**Docs**: `docs/benchmarks.md`

---

## Phase D — Model Expansion (Milestones 5–7)

### Milestone 5: Expectile Regression
- `ExpectileRegression` estimator using IRLS (asymmetric squared loss)
- Joint multi-expectile variant
- Comparison docs: quantiles vs expectiles

### Milestone 6: Spline / Additive Quantile Workflow
- Basis expansion utilities compatible with sklearn pipelines
- Example using `sklearn.preprocessing.SplineTransformer` + `QuantileRegression`
- No custom GAM framework

### Milestone 7: Distribution Wrapper & Censored Improvements
- `QuantileDistribution`: interpolation, CDF, PPF, sampling from a quantile grid
- Improved censored QR diagnostics: effective sample size, censoring summary

---

## Dependencies Strategy

All new modules (metrics, postprocess, conformal) use only existing core dependencies (numpy, scipy, pandas, sklearn). No new required dependencies.

Optional extras already in `pyproject.toml`:
```toml
formula = ["patsy>=0.5.0"]
plot = ["matplotlib>=3.1.0"]
all = ["patsy>=0.5.0", "matplotlib>=3.1.0"]
```

Benchmark adapters use lazy imports with clear error messages.

---

## Non-Goals

- Native gradient boosting or random forest implementations
- Deep learning quantile architectures
- Probabilistic programming backends
- AutoML machinery
- Renaming existing estimator classes

---

## Success Criteria

After implementation, users can:
1. Evaluate any quantile model with standardized metrics
2. Detect and fix crossing quantiles from external models
3. Get calibrated prediction intervals via conformal calibration
4. Compare quantile methods across the ecosystem
5. Do all of this with a consistent sklearn-native API
