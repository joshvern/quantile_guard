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
- **Analytical, bootstrap, and cluster-robust** standard errors
- **Prediction intervals**, quantile process plots, and pseudo R²
- **Censored quantile regression** for survival data
- Scipy sparse solver for **large-scale** problems
- Validated against sklearn, statsmodels, and R's `quantreg`

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
