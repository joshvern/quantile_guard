# Calibration Diagnostics

## Why calibration matters

A model's 90th percentile prediction should contain 90% of observations. In
practice, this often fails — especially in subgroups or regions where the
model is less accurate. The `calibration` module provides tools to diagnose
these patterns.

## Marginal vs conditional coverage

**Marginal coverage** is the overall fraction of observations inside the
interval. Even a well-calibrated model (marginal coverage = 90%) may have
poor **conditional coverage** — e.g., 70% in one subgroup and 99% in another.

Conformal calibration (see [Conformal](conformal.md)) guarantees marginal
coverage, but not conditional coverage. Calibration diagnostics help you
find where coverage breaks down.

## Coverage by group

Check whether coverage holds across categorical subgroups:

```python
from quantile_guard.calibration import coverage_by_group

result = coverage_by_group(y_test, lower, upper, groups=region_labels)
for group, stats in result.items():
    print(f"{group}: coverage={stats['coverage']:.3f}, "
          f"width={stats['mean_width']:.3f}, n={stats['n']}")
```

## Coverage by feature bin

Check whether coverage degrades in certain regions of a continuous feature:

```python
from quantile_guard.calibration import coverage_by_bin

bins = coverage_by_bin(y_test, lower, upper, feature=X_test[:, 0], n_bins=5)
for b in bins:
    print(f"bin [{b['bin_lower']:.2f}, {b['bin_upper']:.2f}]: "
          f"coverage={b['coverage']:.3f}, width={b['mean_width']:.3f}")
```

This is particularly useful for detecting heteroscedasticity: if coverage
drops in bins where the true variance is high, the intervals are too narrow
in those regions.

## Nominal vs empirical coverage

For models with multiple quantile predictions, compare expected vs actual
coverage for each symmetric quantile pair:

```python
from quantile_guard.calibration import nominal_vs_empirical_coverage

taus = [0.05, 0.25, 0.5, 0.75, 0.95]
result = nominal_vs_empirical_coverage(y_test, predictions, taus)
for entry in result:
    print(f"[{entry['tau_lower']}, {entry['tau_upper']}]: "
          f"nominal={entry['nominal_coverage']:.0%}, "
          f"empirical={entry['empirical_coverage']:.1%}, "
          f"gap={entry['coverage_gap']:+.1%}")
```

## Sharpness summary

Narrower intervals are better, as long as coverage is maintained. The
sharpness summary reports interval width statistics:

```python
from quantile_guard.calibration import sharpness_summary

stats = sharpness_summary(lower, upper)
print(f"Mean width: {stats['mean_width']:.3f}")
print(f"Median width: {stats['median_width']:.3f}")
print(f"IQR of widths: {stats['iqr_width']:.3f}")
```

## Full calibration report

Combine all diagnostics into a single summary:

```python
from quantile_guard.calibration import calibration_summary

report = calibration_summary(
    y_test, lower, upper,
    nominal_coverage=0.90,
    groups=region_labels,       # optional
    feature=X_test[:, 0],       # optional
    n_bins=5,
)
print(f"Coverage: {report['empirical_coverage']:.3f} "
      f"(target: {report['nominal_coverage']:.3f})")
print(f"Gap: {report['coverage_gap']:+.3f}")
print(f"Mean width: {report['sharpness']['mean_width']:.3f}")
```

## When conformal can still fail in subgroups

Conformal prediction guarantees marginal coverage under exchangeability, but:

- **Subgroup coverage** is not guaranteed. A group that is systematically
  harder to predict may have lower coverage.
- **Heteroscedastic data** can lead to intervals that are too wide in
  low-variance regions and too narrow in high-variance regions.
- **Distribution shift** between calibration and test data breaks the
  exchangeability assumption entirely.

Use `coverage_by_group` and `coverage_by_bin` to detect these patterns, and
consider fitting separate models or using adaptive conformal methods when
conditional coverage matters.
