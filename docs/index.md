# quantile-regression-pdlp

Optimization-based quantile regression built on Google OR-Tools.
Scikit-learn API, statsmodels-style summaries, and features beyond what either offers.

## Installation

```bash
pip install quantile-regression-pdlp
```

With optional extras:

```bash
pip install quantile-regression-pdlp[all]  # formulas + plots
```

## Quick Example

```python
import numpy as np
from quantile_regression_pdlp import QuantileRegression

X = np.random.default_rng(0).normal(size=(200, 3))
y = X @ [2.0, -1.5, 0.8] + np.random.default_rng(1).normal(scale=0.5, size=200)

model = QuantileRegression(tau=[0.1, 0.5, 0.9], se_method='analytical')
model.fit(X, y)

print(model.summary()[0.5]['y'])
print(model.pseudo_r_squared_)
model.plot_quantile_process(feature='X1')
```

## Key Features

| Feature | This package | sklearn | statsmodels |
|---------|:---:|:---:|:---:|
| Multiple quantiles (joint) | Yes | No | No |
| Non-crossing constraints | Yes | No | No |
| Multi-output | Yes | No | No |
| Analytical SEs | Yes | No | Yes |
| Kernel (robust) SEs | Yes | No | Yes |
| Cluster-robust SEs | Yes | No | No |
| Bootstrap SEs | Yes | No | No |
| Empirical p-values + CIs | Yes | No | No |
| L1 / Elastic Net / SCAD / MCP | Yes | L1 only | No |
| Prediction intervals | Yes | No | No |
| Conformal calibration (CQR) | Yes | No | No |
| Evaluation metrics suite | Yes | Partial | No |
| Crossing detection + fix | Yes | No | No |
| Quantile process plots | Yes | No | No |
| Pseudo R² | Yes | No | Yes |
| Formula interface | Yes | No | Yes |
| Censored QR | Yes | No | No |
| Sparse solver mode | Yes | No | No |
| Sklearn pipeline compatible | Yes | Yes | No |

Next: [Usage](usage.md) | [Evaluation](evaluation.md) | [Crossing](crossing.md) | [Conformal](conformal.md) | [API Reference](api.md) | [Theory](theory.md)
