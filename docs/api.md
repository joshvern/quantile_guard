# API

## `QuantileRegression`

Import:

```python
from quantile_regression_pdlp import QuantileRegression
```

### Parameters

- `tau` (`float` or `list[float]`, default `0.5`)
  - Quantile(s) in `(0, 1)`. If a list is provided it is sorted.
- `n_bootstrap` (`int`, default `1000`)
  - Number of bootstrap samples used to estimate coefficient standard errors.
- `random_state` (`int | None`, default `None`)
  - Random seed used for bootstrapping.
- `regularization` (`str`, default `'none'`)
  - `'none'` or `'l1'` (Lasso-style) regularization.
- `alpha` (`float`, default `0.0`)
  - Regularization strength (only used if `regularization='l1'`).
- `n_jobs` (`int`, default `1`)
  - Parallelism for bootstrapping. Use `-1` for all cores.

### Methods

#### `fit(X, y, weights=None)`

- `X`: array-like of shape `(n_samples, n_features)` (NumPy or pandas DataFrame)
- `y`: array-like of shape `(n_samples,)` or `(n_samples, n_outputs)` (NumPy/pandas)
- `weights`: optional array-like of shape `(n_samples,)`

Returns `self`.

#### `predict(X)`

Returns a nested dictionary:

- `y_pred: dict[float, dict[str, np.ndarray]]`
- Structure: `{tau: {output_name: predictions}}`
- Each prediction array has shape `(n_samples,)`.

#### `summary()`

Returns a nested dictionary:

- `summary_dict: dict[float, dict[str, pandas.DataFrame]]`
- Each DataFrame includes coefficient estimates and bootstrap-based standard errors.

Note: reported t-values and p-values use a t reference distribution (approximate).
