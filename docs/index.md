# quantile-regression-pdlp

`quantile-regression-pdlp` is an optimization-based quantile regression implementation built on Google OR-Tools' PDLP solver.
It provides a familiar `scikit-learn`-style API and optional statistical summaries via bootstrapping.

## Installation

```bash
pip install quantile-regression-pdlp
```

## 10-line example

```python
import numpy as np
from quantile_regression_pdlp import QuantileRegression

X = np.random.default_rng(0).normal(size=(100, 2))
y = 1.5 * X[:, 0] - 2.0 * X[:, 1] + np.random.default_rng(1).normal(scale=0.5, size=100)

model = QuantileRegression(tau=0.5, n_bootstrap=200, random_state=0)
model.fit(X, y)
print(model.summary()[0.5]["y"])
print(model.predict(X[:3])[0.5]["y"])
```

Next: see [Usage](usage.md) and [API](api.md).
