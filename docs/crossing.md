# Handling Quantile Crossing

## The problem

When quantile models are fitted independently for each tau, predictions can
**cross**: the 90th percentile prediction may be lower than the 10th. This
violates the definition of quantiles and produces nonsensical intervals.

## Joint constraints vs post-hoc rearrangement

`QuantileRegression` prevents crossing in two ways:

1. **LP constraints** during fitting enforce `Q(tau_j) <= Q(tau_{j+1})` on
   training data
2. **Isotonic projection** at prediction time fixes any remaining violations
   on new data

If you're using external models (LightGBM, XGBoost, etc.) that don't enforce
non-crossing, the `postprocess` module provides tools to detect and fix
crossings.

## Detecting crossings

```python
from quantile_regression_pdlp.postprocess import check_crossing, crossing_summary
import numpy as np

# predictions: (n_samples, n_quantiles), one column per tau
taus = [0.1, 0.5, 0.9]
predictions = np.array([
    [1.0, 2.0, 3.0],  # ok
    [3.0, 2.0, 1.0],  # crossed
    [1.5, 1.5, 2.0],  # ok (equal is fine)
])

mask = check_crossing(predictions, taus)
# array([False, True, False])

summary = crossing_summary(predictions, taus)
# {
#     'crossing_rate': 0.333,
#     'crossing_magnitude': 0.667,
#     'max_magnitude': 2.0,
#     'n_crossing_samples': 1,
#     'n_total_samples': 3,
#     'worst_row_index': 1,
# }
```

## Fixing crossings with rearrangement

The simplest fix: sort each row so predictions are non-decreasing.

```python
from quantile_regression_pdlp.postprocess import rearrange_quantiles

fixed = rearrange_quantiles(predictions, taus)
# array([[1., 2., 3.],
#        [1., 2., 3.],
#        [1.5, 1.5, 2.]])
```

This implements the rearrangement approach of Chernozhukov, Fernández-Val, and
Galichon (2010). It guarantees monotonicity but may change the marginal
distribution of each quantile's predictions.

## When to use what

| Situation | Recommendation |
|-----------|----------------|
| Fitting with this package | Non-crossing is automatic |
| Evaluating external predictions | Use `crossing_summary` to diagnose |
| Fixing external predictions | Use `rearrange_quantiles` |
| Need guaranteed non-crossing | Fit with `QuantileRegression(tau=[...])` |
