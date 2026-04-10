# API Reference

## Joint Quantile Regression

### `QuantileRegression`

```python
from quantile_guard import QuantileRegression
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tau` | `float` or `list[float]` | `0.5` | Unique quantile(s) in (0, 1). Lists are sorted automatically. |
| `n_bootstrap` | `int` | `1000` | Bootstrap samples for SE estimation. Only used with `se_method='bootstrap'`. |
| `random_state` | `int` or `None` | `None` | Random seed for reproducibility. |
| `regularization` | `str` | `'none'` | `'none'`, `'l1'`, `'elasticnet'`, `'scad'`, or `'mcp'`. |
| `alpha` | `float` | `0.0` | Regularization strength. Normalized as `(1/n)*loss + alpha*penalty`. |
| `l1_ratio` | `float` | `1.0` | Elastic net mixing (0=L2, 1=L1). Only for `regularization='elasticnet'`. |
| `n_jobs` | `int` | `1` | Parallel jobs for bootstrapping. `-1` = all cores. |
| `solver_backend` | `str` | `'PDLP'` | LP solver: `'PDLP'` or `'GLOP'`. |
| `solver_tol` | `float` or `None` | `None` | Solver optimality tolerance. |
| `solver_time_limit` | `float` or `None` | `None` | Max solver time in seconds. |
| `enforce_non_crossing_predict` | `bool` | `True` | Apply row-wise rearrangement when prediction-time crossings are detected. |
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

## Censored Quantile Regression

### `CensoredQuantileRegression`

```python
from quantile_guard import CensoredQuantileRegression
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

---

## Conformalized Quantile Regression

### `ConformalQuantileRegression`

```python
from quantile_guard.conformal import ConformalQuantileRegression
```

Split conformal calibration for prediction intervals with finite-sample coverage
guarantees. Wraps any base quantile estimator.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_estimator` | estimator | required | Quantile model with ≥ 2 quantiles in `tau`. |
| `coverage` | `float` | `0.90` | Target coverage level in (0, 1). |
| `calibration_size` | `float` | `0.25` | Fraction of data held out for calibration. |
| `random_state` | `int` or `None` | `None` | Seed for train/calibration split. |

### Methods

#### `fit(X, y)`

Fit base estimator on training split and calibrate on held-out data.

#### `predict_interval(X)`

Returns `dict[str, dict]` -- `{output: {'lower': array, 'upper': array, 'width': array}}`.

#### `empirical_coverage(X, y)`

Returns `dict[str, float]` -- empirical coverage per output.

#### `interval_width(X)`

Returns `dict[str, float]` -- mean interval width per output.

### Attributes (after fitting)

| Attribute | Type | Description |
|-----------|------|-------------|
| `offset_` | `dict` | `{output: float}` -- calibrated conformity score quantile. |
| `lower_tau_` | `float` | Lower quantile from base estimator. |
| `upper_tau_` | `float` | Upper quantile from base estimator. |
| `base_estimator_` | estimator | Fitted clone of the base estimator. |

---

## Evaluation & Metrics

Standalone evaluation functions that work with predictions from **any** quantile model — not just this package.

```python
from quantile_guard.metrics import (
    pinball_loss,
    multi_quantile_pinball_loss,
    empirical_coverage,
    mean_interval_width,
    crossing_rate,
    crossing_magnitude,
    interval_score,
    quantile_evaluation_report,
)
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `pinball_loss` | `(y_true, y_pred, tau) -> float` | Mean pinball loss for a single quantile. |
| `multi_quantile_pinball_loss` | `(y_true, predictions, taus) -> dict` | Per-quantile pinball loss. |
| `empirical_coverage` | `(y_true, lower, upper) -> float` | Fraction of points inside interval. |
| `mean_interval_width` | `(lower, upper) -> float` | Mean interval width. |
| `crossing_rate` | `(predictions, taus) -> float` | Fraction of samples with crossing quantiles. |
| `crossing_magnitude` | `(predictions, taus) -> float` | Mean severity of crossings. |
| `interval_score` | `(y_true, lower, upper, alpha) -> float` | Gneiting-Raftery interval score. |
| `quantile_evaluation_report` | `(...) -> dict` | Full evaluation summary. |

---

## Crossing Detection & Repair

Diagnose and fix quantile crossings from any model's predictions.

```python
from quantile_guard.postprocess import (
    check_crossing,
    crossing_summary,
    rearrange_quantiles,
)
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `check_crossing` | `(predictions, taus) -> ndarray[bool]` | Boolean mask of samples with crossings. |
| `crossing_summary` | `(predictions, taus) -> dict` | Rate, magnitude, worst rows. |
| `rearrange_quantiles` | `(predictions, taus) -> ndarray` | Per-row sort to enforce monotonicity. |
