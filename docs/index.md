# quantile-guard

**Non-crossing quantile models with built-in inference, calibration, and evaluation.**

A quantile modeling toolkit that fits multiple quantiles jointly with monotonicity constraints — guaranteeing predictions never cross. From raw quantile regression through calibrated prediction intervals, with inference, diagnostics, and evaluation built in.

## Why This Exists

When you fit quantiles independently (as sklearn and statsmodels do), nothing prevents the 90th percentile prediction from falling below the 10th. This package solves that by fitting all quantiles in a single joint optimization with non-crossing constraints — and wraps the result in a complete modeling workflow.

## Workflows

| Workflow | Module | What it does |
|----------|--------|-------------|
| **Joint Quantile Regression** | `QuantileRegression` | Fit multiple quantiles with non-crossing guarantees, inference, and regularization |
| **Conformalized QR** | `conformal` | Calibrate intervals for finite-sample coverage guarantees |
| **Censored QR** | `CensoredQuantileRegression` | Right- or left-censored (survival) quantile models |
| **Evaluation & Metrics** | `metrics` | Pinball loss, coverage, interval score, crossing rate |
| **Calibration Diagnostics** | `calibration` | Coverage by group/bin, nominal vs empirical, sharpness |
| **Crossing Detection & Repair** | `postprocess` | Diagnose and fix crossings from any quantile model |

## Key Differentiators

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
| Pseudo R² | Yes | No | Yes |
| Formula interface | Yes | No | Yes |
| Sklearn pipeline compatible | Yes | Yes | No |

## Benchmarked

Tested on heavy-tailed heteroscedastic data. Independent fitters produce 4-30% crossing rates; this package produces zero — while matching or improving pinball loss.

See [Benchmarks](benchmarks.md) for full results and reproduction instructions.

## Installation

```bash
pip install quantile-guard
```

With optional extras:

```bash
pip install quantile-guard[all]  # formulas + plots
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

## Next Steps

- [Usage Guide](usage.md) — fit, predict, inference, regularization, formulas
- [Evaluation](evaluation.md) — pinball loss, coverage, interval score
- [Crossing](crossing.md) — detection, diagnostics, rearrangement
- [Conformal](conformal.md) — calibrated prediction intervals
- [Calibration](calibration.md) — coverage diagnostics by group, bin, and feature
- [Benchmarks](benchmarks.md) — comparisons against sklearn and statsmodels
- [API Reference](api.md) — full parameter and method documentation
- [Theory](theory.md) — LP formulation, non-crossing constraints, SE estimation
