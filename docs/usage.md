# Usage

This package follows a `scikit-learn`-style workflow: instantiate, `fit`, then `predict`.

## Fit / predict

```python
import numpy as np
from quantile_regression_pdlp import QuantileRegression

X = np.random.default_rng(0).normal(size=(50, 1))
y = 2.0 * X[:, 0] + np.random.default_rng(1).normal(scale=0.3, size=50)

model = QuantileRegression(tau=0.5, n_bootstrap=200, random_state=0)
model.fit(X, y)

pred = model.predict(X[:5])
# pred is a nested dict: {tau: {output_name: np.ndarray}}
print(pred[0.5]["y"].shape)
```

## Multiple quantiles (single output)

When `tau` is a list, the model solves all quantiles jointly and adds non-crossing constraints on the training data.

```python
taus = [0.25, 0.5, 0.75]
model = QuantileRegression(tau=taus, n_bootstrap=200, random_state=0)
model.fit(X, y)

pred = model.predict(X[:3])
for q in taus:
    print(q, pred[q]["y"])
```

## Multi-output targets

If `y` is 2D (or a pandas DataFrame), you get one set of coefficients per output.

```python
Y = np.column_stack([y, y + 0.5])
model = QuantileRegression(tau=[0.4, 0.6], n_bootstrap=200, random_state=0)
model.fit(X, Y)

pred = model.predict(X[:2])
print(pred[0.4].keys())
```

## Weighted regression

Pass `weights` to up/down-weight observations:

```python
weights = np.ones(X.shape[0])
weights[:10] = 2.0
model = QuantileRegression(tau=0.5, n_bootstrap=200, random_state=0)
model.fit(X, y, weights=weights)
```
