# Changelog

This project follows Semantic Versioning (SemVer): https://semver.org/

## 0.2.0 - Feature release

### New features
- **Analytical standard errors** — Koenker-Bassett (1978) IID sandwich estimator via `se_method='analytical'`. No bootstrapping needed.
- **Kernel (heteroscedasticity-robust) SEs** — Powell (1991) sandwich estimator via `se_method='kernel'`.
- **Cluster-robust SEs** — pass `clusters=` to `fit()` for grouped/panel data with small-sample correction.
- **Elastic net regularization** — `regularization='elasticnet'` with `l1_ratio` parameter.
- **SCAD penalty** — `regularization='scad'` (Fan & Li, 2001) via Local Linear Approximation.
- **MCP penalty** — `regularization='mcp'` (Zhang, 2010) via Local Linear Approximation.
- **Prediction intervals** — `predict_interval(X, coverage=0.90)` using fitted quantile bounds.
- **Quantile process plots** — `plot_quantile_process(feature=)` with confidence bands (requires matplotlib).
- **Formula interface** — `fit_formula('y ~ x1 + C(group)', data=df)` with automatic categorical encoding (requires patsy).
- **Censored quantile regression** — `CensoredQuantileRegression` for right/left-censored (survival) data using Powell's iterative algorithm.
- **Scipy sparse solver** — `use_sparse=True` for memory-efficient solving via HiGHS on large datasets.
- **Pseudo R-squared** — Koenker-Machado (1999) goodness-of-fit measure in `pseudo_r_squared_`.
- **Multi-output regression** — fit multiple response variables in a single model.
- **Solver diagnostics** — `solver_info_` dict with status, wall time, variable/constraint counts.

### Changes
- Objective normalization now uses `(1/n) * loss + alpha * penalty`, consistent with scikit-learn's `QuantileRegressor`. The same `alpha` value produces comparable regularization across different sample sizes.
- Added optional dependency groups: `[formula]` (patsy), `[plot]` (matplotlib), `[all]`.
- Benchmark tests against sklearn and R's `quantreg` package (excluded from default test suite).

## 0.1.1 - Documentation site
- Added MkDocs Material documentation under `docs/`.
- Added GitHub Pages deployment workflow via GitHub Actions.
- Linked docs from the README and added a Docs badge.

## 0.1.0 - Initial release
- Initial public release on PyPI.
