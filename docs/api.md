# API Reference

## `QuantileRegression`

```python
from quantile_regression_pdlp import QuantileRegression
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tau` | `float` or `list[float]` | `0.5` | Quantile(s) in (0, 1). Lists are sorted automatically. |
| `n_bootstrap` | `int` | `1000` | Bootstrap samples for SE estimation. Only used with `se_method='bootstrap'`. |
| `random_state` | `int` or `None` | `None` | Random seed for reproducibility. |
| `regularization` | `str` | `'none'` | `'none'`, `'l1'`, `'elasticnet'`, `'scad'`, or `'mcp'`. |
| `alpha` | `float` | `0.0` | Regularization strength. Normalized as `(1/n)*loss + alpha*penalty`. |
| `l1_ratio` | `float` | `1.0` | Elastic net mixing (0=L2, 1=L1). Only for `regularization='elasticnet'`. |
| `n_jobs` | `int` | `1` | Parallel jobs for bootstrapping. `-1` = all cores. |
| `solver_backend` | `str` | `'PDLP'` | LP solver: `'PDLP'` or `'GLOP'`. |
| `solver_tol` | `float` or `None` | `None` | Solver optimality tolerance. |
| `solver_time_limit` | `float` or `None` | `None` | Max solver time in seconds. |
| `enforce_non_crossing_predict` | `bool` | `True` | Isotonic projection on predictions to prevent crossing. |
| `se_method` | `str` | `'bootstrap'` | `'bootstrap'`, `'analytical'` (IID), or `'kernel'` (robust). |
| `use_sparse` | `bool` | `False` | Use scipy sparse LP solver instead of OR-Tools. |

### Methods

#### `fit(X, y, weights=None, clusters=None)`

Fit the quantile regression model.

- **X**: `(n_samples, n_features)` -- NumPy array or pandas DataFrame.
- **y**: `(n_samples,)` or `(n_samples, n_outputs)` -- NumPy array, Series, or DataFrame.
- **weights**: `(n_samples,)` optional -- observation weights.
- **clusters**: `(n_samples,)` optional -- cluster labels for cluster-robust SEs.
- **Returns**: `self`

#### `fit_formula(formula, data, weights=None, clusters=None)`

Fit using an R-style formula. Requires `patsy`.

- **formula**: `str` -- e.g. `"y ~ x1 + x2 + C(group)"`.
- **data**: `DataFrame` -- contains all referenced variables.
- **weights**: `str` or array -- column name or weight array.
- **clusters**: `str` or array -- column name or cluster array.
- **Returns**: `self`

#### `predict(X)`

Returns `dict[float, dict[str, np.ndarray]]` -- `{tau: {output: predictions}}`.

#### `predict_interval(X, coverage=0.90)`

Returns `dict[str, dict]` -- `{output: {'lower': array, 'median': array, 'upper': array, 'tau_lower': float, ...}}`.

Requires at least 2 fitted quantiles. Uses the fitted quantiles closest to the
requested coverage bounds.

#### `score(X, y, sample_weight=None)`

Negative mean pinball loss (higher is better, sklearn convention).

#### `summary()`

Returns `dict[float, dict[str, DataFrame]]` -- tables with Coefficient, Std. Error,
t-value, P>|t|, CI 2.5%, CI 97.5%.

#### `plot_quantile_process(feature=None, figsize=None, ax=None)`

Plot coefficients across tau values with confidence intervals. Requires `matplotlib`.
Returns a matplotlib `Figure`.

### Attributes (after fitting)

| Attribute | Type | Description |
|-----------|------|-------------|
| `coef_` | `dict` | `{tau: {output: array}}` -- slope coefficients. |
| `intercept_` | `dict` | `{tau: {output: float}}` -- intercept terms. |
| `stderr_` | `dict` | `{tau: {output: array}}` -- standard errors (intercept + slopes). |
| `tvalues_` | `dict` | `{tau: {output: array}}` -- t-statistics. |
| `pvalues_` | `dict` | `{tau: {output: array}}` -- p-values. |
| `confidence_intervals_` | `dict` | `{tau: {output: DataFrame}}` -- 95% CIs. |
| `solver_info_` | `dict` | Status, wall time, variable/constraint counts, objective value. |
| `pseudo_r_squared_` | `dict` | `{tau: {output: float}}` -- Koenker-Machado (1999) pseudo R². |
| `feature_names_` | `list` | Feature names (from pandas or auto-generated). |
| `output_names_` | `list` | Output names (from pandas or auto-generated). |

---

## `CensoredQuantileRegression`

```python
from quantile_regression_pdlp import CensoredQuantileRegression
```

Subclass of `QuantileRegression` for right- or left-censored data.

### Additional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `censoring` | `str` | `'right'` | `'right'` or `'left'`. |
| `max_censor_iter` | `int` | `50` | Maximum Powell iterations. |
| `censor_tol` | `float` | `1e-4` | Convergence tolerance. |

All `QuantileRegression` parameters are also accepted.

### Methods

#### `fit(X, y, event_indicator, weights=None, clusters=None)`

- **event_indicator**: `(n_samples,)` -- 1 = event observed, 0 = censored.
- All other parameters same as `QuantileRegression.fit()`.
