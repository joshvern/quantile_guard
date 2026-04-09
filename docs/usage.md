# Usage

## Basic Fit / Predict

```python
import numpy as np
from quantile_guard import QuantileRegression

X = np.random.default_rng(0).normal(size=(200, 2))
y = 2.0 * X[:, 0] - 1.0 * X[:, 1] + np.random.default_rng(1).normal(scale=0.5, size=200)

model = QuantileRegression(tau=0.5, n_bootstrap=200, random_state=0)
model.fit(X, y)

pred = model.predict(X[:5])
print(pred[0.5]["y"])
```

## Multiple Quantiles

Pass a list of quantiles to fit them jointly with non-crossing constraints.

```python
model = QuantileRegression(tau=[0.1, 0.25, 0.5, 0.75, 0.9],
                           n_bootstrap=200, random_state=0)
model.fit(X, y)

for q in model.tau:
    print(f"tau={q}: intercept={model.intercept_[q]['y']:.3f}")
```

## Prediction Intervals

```python
model = QuantileRegression(tau=[0.05, 0.5, 0.95], se_method='analytical')
model.fit(X, y)

interval = model.predict_interval(X[:10], coverage=0.90)
print(interval['y']['lower'])   # 5th percentile predictions
print(interval['y']['median'])  # 50th percentile
print(interval['y']['upper'])   # 95th percentile
```

## Inference Methods

### Bootstrap (default)

Empirical p-values and percentile confidence intervals from bootstrap resampling.

```python
model = QuantileRegression(tau=0.5, se_method='bootstrap',
                           n_bootstrap=500, random_state=0, n_jobs=-1)
model.fit(X, y)
print(model.summary()[0.5]['y'])
```

### Analytical (fast)

Asymptotic IID standard errors using Koenker-Bassett (1978) sandwich estimator.
No bootstrapping needed.

```python
model = QuantileRegression(tau=0.5, se_method='analytical')
model.fit(X, y)
```

### Kernel (heteroscedasticity-robust)

Powell (1991) kernel sandwich estimator. Robust to non-constant error variance.

```python
model = QuantileRegression(tau=0.5, se_method='kernel')
model.fit(X, y)
```

### Cluster-Robust

For grouped/panel data with intra-cluster correlation.

```python
model = QuantileRegression(tau=0.5, se_method='analytical')
model.fit(X, y, clusters=group_ids)
```

## Regularization

### L1 (Lasso)

```python
model = QuantileRegression(tau=0.5, regularization='l1', alpha=0.1)
model.fit(X, y)
```

Alpha is normalized as `(1/n) * pinball_loss + alpha * penalty`, consistent with
sklearn's `QuantileRegressor`. The same alpha value produces comparable results
across different dataset sizes.

### Elastic Net

Combines L1 and L2 penalties. `l1_ratio=1` is pure L1, `l1_ratio=0` is pure L2.

```python
model = QuantileRegression(tau=0.5, regularization='elasticnet',
                           alpha=0.1, l1_ratio=0.5)
```

### SCAD

Fan & Li (2001). Less bias on large coefficients than L1, better variable selection.

```python
model = QuantileRegression(tau=0.5, regularization='scad', alpha=0.3)
```

### MCP

Zhang (2010). Similar properties to SCAD with a different shape.

```python
model = QuantileRegression(tau=0.5, regularization='mcp', alpha=0.3)
```

## Quantile Process Plots

Visualize how coefficients change across quantiles.

```python
model = QuantileRegression(
    tau=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
    se_method='analytical'
)
model.fit(X, y)
fig = model.plot_quantile_process(feature='X1')
```

## Formula Interface

R-style formulas with automatic categorical encoding (requires `patsy`).

```python
import pandas as pd

df = pd.DataFrame({'y': y, 'x1': X[:, 0], 'x2': X[:, 1],
                   'group': np.random.choice(['A', 'B'], size=len(y))})

model = QuantileRegression(tau=0.5, se_method='analytical')
model.fit_formula('y ~ x1 + x2 + C(group)', df)
```

## Model Evaluation

```python
# Pinball loss (negative, higher = better, sklearn convention)
print(model.score(X, y))

# Koenker-Machado pseudo R²
print(model.pseudo_r_squared_)
```

## Censored Quantile Regression

For right- or left-censored (survival) data using Powell's iterative algorithm.

```python
from quantile_guard import CensoredQuantileRegression

model = CensoredQuantileRegression(tau=0.5, censoring='right',
                                    se_method='analytical')
model.fit(X, observed_time, event_indicator=delta)
# delta: 1 = event observed, 0 = censored
```

## Solver Options

```python
# GLOP simplex solver (often faster on small/medium problems)
QuantileRegression(tau=0.5, solver_backend='GLOP')

# Scipy sparse solver (memory-efficient for large n)
QuantileRegression(tau=0.5, use_sparse=True)

# Tuning
QuantileRegression(tau=0.5, solver_tol=1e-8, solver_time_limit=60.0)
```

Solver diagnostics are stored after fitting:

```python
print(model.solver_info_)
# {'status': 'OPTIMAL', 'wall_time_seconds': 0.12, 'num_variables': 800, ...}
```

## Multi-Output Regression

```python
Y = np.column_stack([y, y + 1.0])
model = QuantileRegression(tau=[0.25, 0.75], se_method='analytical')
model.fit(X, Y)

pred = model.predict(X[:3])
print(pred[0.25].keys())  # {'y1', 'y2'}
```

## Weighted Regression

```python
weights = np.where(X[:, 0] > 0, 2.0, 1.0)
model = QuantileRegression(tau=0.5, se_method='analytical')
model.fit(X, y, weights=weights)
```
