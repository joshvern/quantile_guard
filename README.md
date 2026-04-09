[![PyPI][pypi-badge]][pypi-link]
[![Python Versions][py-badge]][pypi-link]
[![CI][ci-badge]][ci-link]
[![Docs][docs-badge]][docs-link]

[pypi-badge]: https://img.shields.io/pypi/v/quantile-regression-pdlp.svg
[py-badge]: https://img.shields.io/pypi/pyversions/quantile-regression-pdlp.svg
[ci-badge]: https://github.com/joshvern/quantile_regression_pdlp/actions/workflows/ci.yml/badge.svg
[docs-badge]: https://github.com/joshvern/quantile_regression_pdlp/actions/workflows/docs.yml/badge.svg

[pypi-link]: https://pypi.org/project/quantile-regression-pdlp/
[ci-link]: https://github.com/joshvern/quantile_regression_pdlp/actions/workflows/ci.yml
[docs-link]: https://joshvern.github.io/quantile_regression_pdlp/

# quantile-regression-pdlp

Optimization-based quantile regression built on Google OR-Tools. Scikit-learn API, statsmodels-style summaries, and features that go beyond what either package offers.

**What makes this different from sklearn or statsmodels?**

- Fits **multiple quantiles jointly** with non-crossing constraints
- **Multi-output** regression in a single model
- **SCAD, MCP, and elastic net** penalties (not just L1)
- **Analytical, bootstrap, kernel, and cluster-robust** standard errors
- **Conformalized quantile regression** for calibrated prediction intervals
- **Evaluation metrics**: pinball loss, coverage, interval score, crossing diagnostics
- **Calibration diagnostics**: coverage by group/bin, nominal vs empirical, sharpness analysis
- **Crossing detection and rearrangement** for any quantile model's predictions
- **Prediction intervals**, quantile process plots, and pseudo R²
- **Censored quantile regression** for survival data
- Scipy sparse solver for **large-scale** problems
- Validated against sklearn, statsmodels, and R's `quantreg`

| Feature | This package | sklearn | statsmodels |
|---------|:---:|:---:|:---:|
| Multiple quantiles (joint) | Yes | No | No |
| Non-crossing constraints | Yes | No | No |
| Multi-output | Yes | No | No |
| Analytical SEs | Yes | No | Yes |
| Kernel (robust) SEs | Yes | No | Yes |
| Cluster-robust SEs | Yes | No | No |
| Bootstrap SEs | Yes | No | No |
| L1 / Elastic Net / SCAD / MCP | Yes | L1 only | No |
| Conformal calibration (CQR) | Yes | No | No |
| Evaluation metrics suite | Yes | Partial | No |
| Crossing detection + fix | Yes | No | No |
| Calibration diagnostics | Yes | No | No |
| Prediction intervals | Yes | No | No |
| Pseudo R² | Yes | No | Yes |
| Formula interface | Yes | No | Yes |
| Censored QR | Yes | No | No |
| Sklearn pipeline compatible | Yes | Yes | No |

## Installation

```bash
pip install quantile-regression-pdlp
```

Optional extras:

```bash
pip install quantile-regression-pdlp[all]   # formula interface + plots
pip install quantile-regression-pdlp[plot]   # matplotlib only
pip install quantile-regression-pdlp[formula] # patsy only
```

## Quick Start

```python
import numpy as np
from quantile_regression_pdlp import QuantileRegression

X = np.random.default_rng(0).normal(size=(200, 3))
y = X @ [2.0, -1.5, 0.8] + np.random.default_rng(1).normal(scale=0.5, size=200)

model = QuantileRegression(tau=[0.1, 0.5, 0.9], n_bootstrap=200, random_state=0)
model.fit(X, y)

# Summaries with coefficients, SEs, p-values, and 95% CIs
print(model.summary()[0.5]['y'])

# Prediction intervals
interval = model.predict_interval(X[:5], coverage=0.80)
print(interval['y']['lower'], interval['y']['upper'])

# Pseudo R²
print(model.pseudo_r_squared_)
```

## Features at a Glance

### Regularization

```python
# L1 (Lasso)
QuantileRegression(tau=0.5, regularization='l1', alpha=0.1)

# Elastic net
QuantileRegression(tau=0.5, regularization='elasticnet', alpha=0.1, l1_ratio=0.5)

# SCAD (less bias on large coefficients)
QuantileRegression(tau=0.5, regularization='scad', alpha=0.3)

# MCP
QuantileRegression(tau=0.5, regularization='mcp', alpha=0.3)
```

### Inference Options

```python
# Fast analytical SEs (no bootstrapping needed)
model = QuantileRegression(tau=0.5, se_method='analytical')
model.fit(X, y)

# Heteroscedasticity-robust kernel sandwich SEs
model = QuantileRegression(tau=0.5, se_method='kernel')
model.fit(X, y)

# Cluster-robust SEs
model = QuantileRegression(tau=0.5, se_method='analytical')
model.fit(X, y, clusters=group_labels)
```

### Quantile Process Plot

```python
model = QuantileRegression(
    tau=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
    se_method='analytical'
)
model.fit(X, y)
model.plot_quantile_process(feature='X1')
```

### Formula Interface

```python
model = QuantileRegression(tau=0.5, se_method='analytical')
model.fit_formula('y ~ x1 + x2 + C(region)', data=df)
```

### Censored Quantile Regression

```python
from quantile_regression_pdlp import CensoredQuantileRegression

model = CensoredQuantileRegression(tau=0.5, censoring='right', se_method='analytical')
model.fit(X, observed_time, event_indicator=delta)
```

### Solver Options

```python
# GLOP simplex (faster on small/medium problems)
QuantileRegression(tau=0.5, solver_backend='GLOP')

# Scipy sparse solver (memory-efficient for large datasets)
QuantileRegression(tau=0.5, use_sparse=True)

# Solver tuning
QuantileRegression(tau=0.5, solver_tol=1e-8, solver_time_limit=60.0)
```

## Benchmarks

Tested on heavy-tailed heteroscedastic data (Student-t noise, 10-20 features, up to 13 quantiles). The key advantage: **zero quantile crossings** where independent fitters produce 4-30% crossing rates.

| n | features | quantiles | Crossing rate (this) | Crossing rate (sklearn) | Pinball loss (this) | Pinball loss (sklearn) |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 10 | 7 | **0%** | 11.0% | **0.5148** | 0.5166 |
| 500 | 10 | 13 | **0%** | 30.0% | **0.5095** | 0.5240 |
| 1,000 | 10 | 13 | **0%** | 16.5% | **0.5048** | 0.5071 |
| 2,000 | 20 | 13 | **0%** | 11.0% | **0.5599** | 0.5611 |

The joint non-crossing formulation also achieves slightly better pinball loss as the constraints act as beneficial regularization.

Full results and methodology: [Benchmarks](https://joshvern.github.io/quantile_regression_pdlp/benchmarks/)

```bash
pip install -e ".[benchmark]"
python benchmarks/run_linear_baselines.py
python benchmarks/report.py
```

## Documentation

Full docs: [joshvern.github.io/quantile_regression_pdlp](https://joshvern.github.io/quantile_regression_pdlp/)

## Why PDLP?

Quantile regression is naturally a linear program. OR-Tools' PDLP is a first-order solver designed for large-scale LPs, making it efficient for high-dimensional problems. For smaller problems, the package also supports GLOP (simplex) and scipy's HiGHS solver.

## Dependencies

**Required:** ortools, numpy, pandas, scipy, tqdm, joblib, scikit-learn

**Optional:** matplotlib (plots), patsy (formulas)

## Contributing

Contributions welcome! Open an issue or submit a pull request on [GitHub](https://github.com/joshvern/quantile_regression_pdlp).

## License

MIT
