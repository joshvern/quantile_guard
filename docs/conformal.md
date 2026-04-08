# Conformalized Quantile Regression

## Why conformal calibration?

Raw quantile predictions don't come with coverage guarantees. A model's 90th
percentile prediction might actually contain only 70% of observations. Conformal
calibration adds a data-driven correction that provides **finite-sample marginal
coverage guarantees**.

Reference: Romano, Patterson, Candès (2019). "Conformalized Quantile
Regression." NeurIPS 2019.

## How it works

1. **Split** the training data into a proper training set and a calibration set
2. **Fit** the base quantile model on the training set
3. **Compute nonconformity scores** on the calibration set:
   `score_i = max(lower_i - y_i, y_i - upper_i)`
4. **Find the calibration quantile** of the scores at level `ceil((1-α)(n+1))/n`
5. **Expand intervals** by adding/subtracting this offset at prediction time

The result: prediction intervals that contain at least `1-α` fraction of
future observations in expectation.

## Basic usage

```python
from quantile_regression_pdlp import QuantileRegression
from quantile_regression_pdlp.conformal import ConformalQuantileRegression

# Base model with quantiles that bracket the target coverage
base = QuantileRegression(tau=[0.05, 0.5, 0.95], se_method='analytical')

# Wrap with conformal calibration
cqr = ConformalQuantileRegression(
    base_estimator=base,
    coverage=0.90,
    calibration_size=0.25,   # fraction held out for calibration
    random_state=42,
)
cqr.fit(X_train, y_train)

# Calibrated intervals
intervals = cqr.predict_interval(X_test)
print(intervals['y']['lower'][:5])
print(intervals['y']['upper'][:5])
print(intervals['y']['width'][:5])
```

## Evaluating coverage

```python
# Check actual coverage on held-out data
coverage = cqr.empirical_coverage(X_test, y_test)
print(f"Empirical coverage: {coverage['y']:.3f}")  # should be ≥ 0.90

# Mean interval width
width = cqr.interval_width(X_test)
print(f"Mean width: {width['y']:.3f}")
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_estimator` | estimator | required | Quantile model with `tau` attribute (≥ 2 quantiles) |
| `coverage` | `float` | `0.90` | Target coverage in (0, 1) |
| `calibration_size` | `float` | `0.25` | Fraction of data for calibration |
| `random_state` | `int` or `None` | `None` | Seed for train/calibration split |

## Choosing quantiles for the base model

The base model's extreme quantiles should bracket the desired coverage. For
90% coverage, `tau=[0.05, 0.95]` is natural. The conformal offset then adjusts
these bounds to hit the target coverage.

Using more extreme base quantiles (e.g., `tau=[0.01, 0.99]` for 90% coverage)
gives conformal more room to shrink intervals, potentially producing tighter
bounds. Using less extreme quantiles forces conformal to expand more.

## Caveats

- **Marginal coverage only**: conformal guarantees average coverage, not
  conditional coverage for each subgroup. Intervals may be too wide in
  low-variance regions and too narrow in high-variance regions.
- **Exchangeability assumed**: the calibration and test data must be from the
  same distribution.
- **Calibration set size**: larger calibration sets give tighter offset
  estimates but leave less data for training.
