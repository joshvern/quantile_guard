# Evaluating Quantile Models

## Why dedicated metrics?

Standard regression metrics (MSE, MAE) don't capture what matters for quantile
predictions: calibration, interval quality, and monotonicity. The `metrics`
module provides standalone functions that work with predictions from any model.

## Pinball Loss

The pinball (check) loss is the natural scoring rule for quantile predictions.
Lower is better.

```python
from quantile_regression_pdlp.metrics import pinball_loss

loss = pinball_loss(y_true, y_pred, tau=0.5)
```

For multiple quantiles at once:

```python
from quantile_regression_pdlp.metrics import multi_quantile_pinball_loss
import numpy as np

taus = [0.1, 0.5, 0.9]
predictions = np.column_stack([pred_q10, pred_q50, pred_q90])
losses = multi_quantile_pinball_loss(y_true, predictions, taus)
# {0.1: 0.12, 0.5: 0.08, 0.9: 0.15}
```

## Coverage and Interval Width

Check whether prediction intervals actually contain the right fraction of
observations:

```python
from quantile_regression_pdlp.metrics import empirical_coverage, mean_interval_width

cov = empirical_coverage(y_true, lower=pred_q10, upper=pred_q90)
width = mean_interval_width(lower=pred_q10, upper=pred_q90)
```

A well-calibrated 80% interval should have `cov ≈ 0.80`. If coverage is much
lower, the intervals are too narrow. If much higher, they're too wide.

## Crossing Diagnostics

When fitting multiple quantiles independently, predictions can cross: the 90th
percentile prediction may fall below the 10th. This is a sign of model
misspecification or insufficient constraints.

```python
from quantile_regression_pdlp.metrics import crossing_rate, crossing_magnitude

rate = crossing_rate(predictions, taus)       # fraction of rows with crossings
mag = crossing_magnitude(predictions, taus)   # average severity
```

Note: `QuantileRegression` with multiple `tau` values already enforces
non-crossing via LP constraints, so crossing rate should be zero for its
predictions. These functions are most useful when evaluating external models.

## Interval Score

The interval score (Gneiting & Raftery, 2007) combines width and coverage
penalties into a single number. Lower is better.

```python
from quantile_regression_pdlp.metrics import interval_score

score = interval_score(y_true, lower, upper, alpha=0.1)  # for 90% intervals
```

## Full Evaluation Report

Get all metrics at once:

```python
from quantile_regression_pdlp.metrics import quantile_evaluation_report

report = quantile_evaluation_report(y_true, predictions, taus)
print(report)
# {
#     'pinball_losses': {0.1: ..., 0.5: ..., 0.9: ...},
#     'mean_pinball_loss': ...,
#     'crossing_rate': 0.0,
#     'crossing_magnitude': 0.0,
#     'coverage': 0.82,
#     'mean_width': 2.45,
#     'median_width': 2.31,
#     'interval_score': 3.12,
# }
```

By default, coverage and interval metrics use the first and last quantile
columns as the lower/upper bounds. You can override this with explicit
`lower` and `upper` arrays.
