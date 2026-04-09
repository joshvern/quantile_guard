# quantile-guard

**Non-crossing quantile regression with inference, calibration, and evaluation — in one toolkit.**

Fit multiple quantiles jointly with monotonicity constraints that guarantee predictions never cross. Get standard errors, p-values, conformal calibration, and evaluation metrics out of the box. Scikit-learn compatible.

---

## Who Is This For

<div class="grid" markdown>

**Data scientists** building prediction intervals for production systems where crossed quantiles break downstream logic.

**Researchers & econometricians** who need valid statistical inference — SEs, p-values, confidence intervals — on quantile regression coefficients.

**ML engineers** who want a drop-in sklearn-compatible estimator that guarantees monotone quantile predictions.

**Risk analysts & actuaries** modeling conditional tail distributions with censored or survival data.

**Anyone evaluating quantile models** — the metrics and diagnostics modules work with predictions from XGBoost, LightGBM, or any other source.

</div>

---

## Why This Exists

When you fit quantiles independently (as sklearn and statsmodels do), nothing prevents the 90th percentile prediction from falling below the 10th. This package solves that by fitting all quantiles in a single joint optimization with non-crossing constraints — and wraps the result in a complete modeling workflow.

!!! warning "Crossing is common on real data"
    At n=500 with 13 quantile levels, **30% of test samples** show crossed predictions when using sklearn or statsmodels. This package produces zero crossings by construction. See [Benchmarks](benchmarks.md) for full results.

## Workflows

| Workflow | Module | What it does |
|----------|--------|-------------|
| **Joint Quantile Regression** | `QuantileRegression` | Fit multiple quantiles with non-crossing guarantees, inference, and regularization |
| **Conformalized QR** | `conformal` | Calibrate intervals for finite-sample coverage guarantees |
| **Censored QR** | `CensoredQuantileRegression` | Right- or left-censored (survival) quantile models |
| **Evaluation & Metrics** | `metrics` | Pinball loss, coverage, interval score, crossing rate |
| **Calibration Diagnostics** | `calibration` | Coverage by group/bin, nominal vs empirical, sharpness |
| **Crossing Detection & Repair** | `postprocess` | Diagnose and fix crossings from any quantile model |

??? note "Feature comparison vs sklearn & statsmodels"

    | Feature | This package | sklearn | statsmodels |
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

## Benchmarked

Tested on heavy-tailed heteroscedastic data. Independent fitters produce 4-30% crossing rates; this package produces zero — while matching or improving pinball loss.

!!! tip "See full results"
    [Benchmarks](benchmarks.md) includes crossing rates, pinball loss, timing, and coverage across multiple dataset sizes, with reproduction instructions.

## Next Steps

- [Usage Guide](usage.md) — fit, predict, inference, regularization, formulas
- [Evaluation](evaluation.md) — pinball loss, coverage, interval score
- [Crossing](crossing.md) — detection, diagnostics, rearrangement
- [Conformal](conformal.md) — calibrated prediction intervals
- [Calibration](calibration.md) — coverage diagnostics by group, bin, and feature
- [Benchmarks](benchmarks.md) — comparisons against sklearn and statsmodels
- [API Reference](api.md) — full parameter and method documentation
- [Theory](theory.md) — LP formulation, non-crossing constraints, SE estimation
