[![PyPI][pypi-badge]][pypi-link]
[![Python Versions][py-badge]][pypi-link]
[![CI][ci-badge]][ci-link]
[![Docs][docs-badge]][docs-link]

[pypi-badge]: https://img.shields.io/pypi/v/quantile-guard.svg
[py-badge]: https://img.shields.io/pypi/pyversions/quantile-guard.svg
[ci-badge]: https://github.com/joshvern/quantile_guard/actions/workflows/ci.yml/badge.svg
[docs-badge]: https://github.com/joshvern/quantile_guard/actions/workflows/docs.yml/badge.svg

[pypi-link]: https://pypi.org/project/quantile-guard/
[ci-link]: https://github.com/joshvern/quantile_guard/actions/workflows/ci.yml
[docs-link]: https://joshvern.github.io/quantile_guard/

# quantile-guard

**The quantile modeling toolkit with non-crossing guarantees.**

Fit multiple quantiles jointly with monotonicity constraints that guarantee predictions never cross. Built-in inference, conformal calibration, evaluation metrics, and crossing diagnostics. Scikit-learn compatible.

---

## Why Not Just Fit Quantiles Independently?

When you fit quantiles one at a time (as sklearn and statsmodels do), nothing prevents the 90th percentile prediction from falling *below* the 10th. On real data with heavy tails or many quantile levels, **this happens frequently**:

| n | features | quantiles | Crossing rate (independent) | Crossing rate (quantile-guard) |
|---:|---:|---:|---:|---:|
| 500 | 10 | 13 | **30.0%** | **0%** |
| 1,000 | 10 | 13 | **16.5%** | **0%** |
| 2,000 | 20 | 13 | **11.0%** | **0%** |

quantile-guard eliminates crossings by construction — and the joint formulation acts as beneficial regularization, achieving **equal or better pinball loss** than independent fitting.

[Full benchmark results](https://joshvern.github.io/quantile_guard/benchmarks/) | [Reproduce locally](https://joshvern.github.io/quantile_guard/benchmarks/#reproducing-these-results)

## Who Is This For

- **Data scientists** building prediction intervals for production systems where crossed quantiles break downstream logic
- **Researchers & econometricians** who need valid statistical inference (SEs, p-values, CIs) on quantile regression coefficients
- **ML engineers** who want a drop-in sklearn-compatible estimator with monotone quantile guarantees
- **Risk analysts & actuaries** modeling conditional tail distributions with censored or survival data
- **Anyone evaluating quantile models** — the metrics and diagnostics modules work with predictions from XGBoost, LightGBM, or any other source

## Workflows

This is a toolkit, not a single estimator. It covers the pipeline from raw quantile regression through calibrated prediction intervals:

| Workflow | What it does |
|----------|-------------|
| **Joint Quantile Regression** | Fit multiple quantiles in one call with non-crossing guarantees |
| **Conformalized Quantile Regression** | Calibrate intervals for finite-sample coverage guarantees |
| **Censored Quantile Regression** | Handle right- or left-censored (survival) data |
| **Evaluation & Metrics** | Pinball loss, coverage, interval score, crossing diagnostics |
| **Calibration Diagnostics** | Coverage by group/bin, nominal vs empirical, sharpness analysis |
| **Crossing Detection & Repair** | Diagnose and fix crossings from any quantile model |

<details>
<summary><strong>Feature comparison vs sklearn & statsmodels</strong></summary>

| Feature | quantile-guard | sklearn | statsmodels |
|---------|:---:|:---:|:---:|
| Multiple quantiles (joint fit) | Yes | No | No |
| Non-crossing guarantee | Yes | No | No |
| Multi-output regression | Yes | No | No |
| Analytical / kernel / cluster / bootstrap SEs | Yes | No | Partial |
| L1 / Elastic Net / SCAD / MCP | Yes | L1 only | No |
| Conformal calibration (CQR) | Yes | No | No |
| Calibration diagnostics | Yes | No | No |
| Evaluation metrics suite | Yes | Partial | No |
| Crossing detection + fix | Yes | No | No |
| Censored QR | Yes | No | No |
| Prediction intervals | Yes | No | No |
| Pseudo R-squared | Yes | No | Yes |
| Formula interface | Yes | No | Yes |
| Sklearn pipeline compatible | Yes | Yes | No |

</details>

## Installation

```bash
pip install quantile-guard
```

Optional extras:

```bash
pip install quantile-guard[all]      # formula interface + plots
pip install quantile-guard[plot]     # matplotlib only
pip install quantile-guard[formula]  # patsy only
```

> **Migrating from `quantile-regression-pdlp`?** Just change your install and imports — the API is the same:
> ```python
> # before
> from quantile_regression_pdlp import QuantileRegression
> # after
> from quantile_guard import QuantileRegression
> ```

## Quick Start

```python
import numpy as np
from quantile_guard import QuantileRegression

X = np.random.default_rng(0).normal(size=(200, 3))
y = X @ [2.0, -1.5, 0.8] + np.random.default_rng(1).normal(scale=0.5, size=200)

# Fit 3 quantiles jointly — guaranteed non-crossing
model = QuantileRegression(tau=[0.1, 0.5, 0.9], se_method='analytical')
model.fit(X, y)

# Summaries with coefficients, SEs, p-values, and 95% CIs
print(model.summary()[0.5]['y'])

# Prediction intervals (guaranteed monotone: lower < median < upper)
interval = model.predict_interval(X[:5], coverage=0.80)
print(interval['y']['lower'], interval['y']['upper'])
```

### Conformal Calibration

Turn raw quantile predictions into intervals with coverage guarantees:

```python
from quantile_guard.conformal import ConformalQuantileRegression

base = QuantileRegression(tau=[0.05, 0.5, 0.95], se_method='analytical')
cqr = ConformalQuantileRegression(base_estimator=base, coverage=0.90)
cqr.fit(X_train, y_train)

intervals = cqr.predict_interval(X_test)
print(cqr.empirical_coverage(X_test, y_test))  # should be >= 0.90
```

### Censored Quantile Regression

For survival data with right- or left-censoring:

```python
from quantile_guard import CensoredQuantileRegression

model = CensoredQuantileRegression(tau=0.5, censoring='right', se_method='analytical')
model.fit(X, observed_time, event_indicator=delta)
```

### Evaluate Any Quantile Model

The metrics and diagnostics modules work with predictions from any source — not just this package:

```python
from quantile_guard.metrics import quantile_evaluation_report
from quantile_guard.postprocess import crossing_summary

# Evaluate predictions from XGBoost, LightGBM, or any other model
report = quantile_evaluation_report(y_true, predictions, taus)
crossings = crossing_summary(predictions, taus)
```

### Regularization

```python
QuantileRegression(tau=0.5, regularization='l1', alpha=0.1)       # Lasso
QuantileRegression(tau=0.5, regularization='elasticnet', alpha=0.1, l1_ratio=0.5)
QuantileRegression(tau=0.5, regularization='scad', alpha=0.3)     # Less bias on large coefficients
QuantileRegression(tau=0.5, regularization='mcp', alpha=0.3)
```

### Inference Options

```python
QuantileRegression(tau=0.5, se_method='analytical')   # Fast asymptotic SEs
QuantileRegression(tau=0.5, se_method='kernel')        # Heteroscedasticity-robust
QuantileRegression(tau=0.5, se_method='bootstrap', n_bootstrap=500)
# Cluster-robust SEs
model.fit(X, y, clusters=group_labels)
```

## Benchmarks

Tested on heavy-tailed heteroscedastic data (Student-t noise, 10-20 features, up to 13 quantiles). Independent fitters cross; quantile-guard does not — while matching or improving prediction quality:

| n | features | quantiles | Crossing (quantile-guard) | Crossing (sklearn) | Pinball (quantile-guard) | Pinball (sklearn) |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 10 | 7 | **0%** | 11.0% | **0.5148** | 0.5166 |
| 500 | 10 | 13 | **0%** | 30.0% | **0.5095** | 0.5240 |
| 1,000 | 10 | 13 | **0%** | 16.5% | **0.5048** | 0.5071 |
| 2,000 | 20 | 13 | **0%** | 11.0% | **0.5599** | 0.5611 |

**Speed tradeoff:** quantile-guard solves a single joint LP with non-crossing constraints, which is slower than fitting each quantile independently. The value is the guarantee and the richer downstream workflows. For single-quantile fits where speed matters most, sklearn or statsmodels may be more appropriate.

[Full benchmark results](https://joshvern.github.io/quantile_guard/benchmarks/) | [Reproduce locally](https://joshvern.github.io/quantile_guard/benchmarks/#reproducing-these-results)

## When to Use This vs Alternatives

**Use quantile-guard when you need:**
- Multiple quantile predictions that must not cross (production pipelines, interval forecasts)
- Statistical inference on quantile coefficients (SEs, p-values, confidence intervals)
- Calibrated prediction intervals (conformal quantile regression)
- Censored/survival quantile models
- A complete evaluation workflow for any quantile model's predictions

**Use sklearn or statsmodels when:**
- You only need a single quantile (e.g., median regression)
- Raw speed matters more than crossing guarantees
- You don't need inference, calibration, or evaluation tooling

## Documentation

Full docs: [joshvern.github.io/quantile_guard](https://joshvern.github.io/quantile_guard/)

## Under the Hood

Quantile regression is naturally a linear program. quantile-guard solves joint multi-quantile LPs with non-crossing constraints using:

- **PDLP** — first-order primal-dual solver (default, from Google OR-Tools)
- **GLOP** — revised simplex (faster on small/medium problems)
- **HiGHS** — via scipy's sparse LP interface (memory-efficient)

```python
QuantileRegression(tau=0.5, solver_backend='GLOP')   # simplex
QuantileRegression(tau=0.5, use_sparse=True)          # scipy sparse
```

## Dependencies

**Required:** numpy, pandas, scipy, scikit-learn, ortools, tqdm, joblib

**Optional:** matplotlib (plots), patsy (formulas), statsmodels (benchmarks)

## Contributing

Contributions welcome. Open an issue or pull request on [GitHub](https://github.com/joshvern/quantile_guard).

## License

MIT
