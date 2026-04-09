# quantile-guard

**The quantile modeling toolkit with non-crossing guarantees.**

Fit multiple quantiles jointly with monotonicity constraints that guarantee predictions never cross. Built-in inference, conformal calibration, evaluation metrics, and crossing diagnostics. Scikit-learn compatible.

---

## Why Not Fit Quantiles Independently?

When you fit quantiles independently (as sklearn and statsmodels do), nothing prevents the 90th percentile prediction from falling below the 10th.

!!! warning "Crossing is common on real data"
    At n=500 with 13 quantile levels, **30% of test samples** show crossed predictions when using sklearn or statsmodels. quantile-guard produces zero crossings by construction — and the joint formulation achieves equal or better pinball loss. See [Benchmarks](benchmarks.md) for full results.

## Workflows

This is a toolkit, not a single estimator. It covers the pipeline from raw quantile regression through calibrated prediction intervals:

| Workflow | Module | What it does |
|----------|--------|-------------|
| **Joint Quantile Regression** | `QuantileRegression` | Fit multiple quantiles with non-crossing guarantees, inference, and regularization |
| **Conformalized QR** | `conformal` | Calibrate intervals for finite-sample coverage guarantees |
| **Censored QR** | `CensoredQuantileRegression` | Right- or left-censored (survival) quantile models |
| **Evaluation & Metrics** | `metrics` | Pinball loss, coverage, interval score, crossing rate |
| **Calibration Diagnostics** | `calibration` | Coverage by group/bin, nominal vs empirical, sharpness |
| **Crossing Detection & Repair** | `postprocess` | Diagnose and fix crossings from any quantile model |

??? note "Feature comparison vs sklearn & statsmodels"

    | Feature | quantile-guard | sklearn | statsmodels |
    |---------|:---:|:---:|:---:|
    | Multiple quantiles (joint fit) | Yes | No | No |
    | Non-crossing guarantee | Yes | No | No |
    | Multi-output regression | Yes | No | No |
    | Analytical / kernel / cluster / bootstrap SEs | Yes | No | Partial |
    | L1 / Elastic Net / SCAD / MCP | Yes | L1 only | No |
    | Conformal calibration (CQR) | Yes | No | No |
    | Calibration diagnostics | Yes | No | No |
    | Evaluation metrics suite | Yes | Partial | No |
    | Crossing detection + fix | Yes | No | No |
    | Censored QR | Yes | No | No |
    | Prediction intervals | Yes | No | No |
    | Pseudo R-squared | Yes | No | Yes |
    | Formula interface | Yes | No | Yes |
    | Sklearn pipeline compatible | Yes | Yes | No |

## Who Is This For

**Data scientists** building prediction intervals for production systems where crossed quantiles break downstream logic.

**Researchers & econometricians** who need valid statistical inference — SEs, p-values, confidence intervals — on quantile regression coefficients.

**ML engineers** who want a drop-in sklearn-compatible estimator with monotone quantile guarantees.

**Risk analysts & actuaries** modeling conditional tail distributions with censored or survival data.

**Anyone evaluating quantile models** — the metrics and diagnostics modules work with predictions from XGBoost, LightGBM, or any other source.

## Installation

```bash
pip install quantile-guard
```

With optional extras:

=== "All extras"
    ```bash
    pip install quantile-guard[all]
    ```

=== "Plots only"
    ```bash
    pip install quantile-guard[plot]
    ```

=== "Formulas only"
    ```bash
    pip install quantile-guard[formula]
    ```

!!! tip "Migrating from `quantile-regression-pdlp`?"
    For the core estimator, the import path changes but the call pattern stays the same:
    ```python
    # before
    from quantile_regression_pdlp import QuantileRegression
    # after
    from quantile_guard import QuantileRegression
    ```

## Quick Example

```python
import numpy as np
from quantile_guard import QuantileRegression

X = np.random.default_rng(0).normal(size=(200, 3))
y = X @ [2.0, -1.5, 0.8] + np.random.default_rng(1).normal(scale=0.5, size=200)

# Fit 3 quantiles jointly — guaranteed non-crossing
model = QuantileRegression(tau=[0.1, 0.5, 0.9], se_method='analytical')
model.fit(X, y)

print(model.summary()[0.5]['y'])
print(model.pseudo_r_squared_)
```

## Benchmark Evidence

Independent fitters produce 4-30% crossing rates on challenging data; quantile-guard produces zero — while matching or improving pinball loss:

| n | features | quantiles | Crossing (quantile-guard) | Crossing (sklearn) | Pinball (quantile-guard) | Pinball (sklearn) |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 10 | 13 | **0%** | 30.0% | **0.5095** | 0.5240 |
| 1,000 | 10 | 13 | **0%** | 16.5% | **0.5048** | 0.5071 |
| 2,000 | 20 | 13 | **0%** | 11.0% | **0.5599** | 0.5611 |

**Speed tradeoff:** the joint LP is slower than fitting quantiles independently. The value is the guarantee and the richer downstream workflows.

[Full benchmark results](benchmarks.md) | [Reproduce locally](benchmarks.md#reproducing-these-results)

## Next Steps

- [Usage Guide](usage.md) — fit, predict, inference, regularization, formulas
- [Benchmarks](benchmarks.md) — crossing rates, pinball loss, timing, coverage
- [Conformal](conformal.md) — calibrated prediction intervals
- [Calibration](calibration.md) — coverage diagnostics by group, bin, and feature
- [Crossing](crossing.md) — detection, diagnostics, rearrangement
- [Evaluation](evaluation.md) — pinball loss, coverage, interval score
- [API Reference](api.md) — full parameter and method documentation
- [Theory](theory.md) — LP formulation, non-crossing constraints, SE estimation
