[![PyPI][pypi-badge]][pypi-link]
[![Python Versions][py-badge]][pypi-link]
[![CI][ci-badge]][ci-link]
[![Docs][docs-badge]][docs-link]

[pypi-badge]: https://img.shields.io/pypi/v/quantile-regression-pdlp.svg
[py-badge]: https://img.shields.io/pypi/pyversions/quantile-regression-pdlp.svg
[ci-badge]: https://github.com/joshvern/quantile_regression_pdlp/actions/workflows/ci.yml/badge.svg
[docs-badge]: https://github.com/joshvern/quantile_regression_pdlp/actions/workflows/docs.yml/badge.svg

[pypi-link]: https://pypi.org/project/quantile-regression-pdlp/
[ci-link]: https://github.com/joshvern/quantile_regression_pdlp/actions/workflows/ci.yml
[docs-link]: https://joshvern.github.io/quantile_regression_pdlp/

**TL;DR**: `quantile-regression-pdlp` is an optimization-based quantile regression implementation built on Google OR-Tools' PDLP solver.
It provides a familiar `scikit-learn`-style API and optional statistical summaries via bootstrapping.

## Quantile Regression PDLP

A Python package for performing quantile regression using the PDLP solver from Google's OR-Tools, with an interface and summaries similar to `statsmodels`, and fully compatible with `scikit-learn` and `pandas`.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Quantile Regression as a Linear Programming Problem](#quantile-regression-as-a-linear-programming-problem)
    - [Mathematical Formulation](#mathematical-formulation)
    - [LP Formulation](#lp-formulation)
- [Features](#features)
- [Documentation](#documentation)
    - [Class: `QuantileRegression`](#class-quantileregression)
- [Examples](#examples)
    - [Example 1: Basic Usage](#example-1-basic-usage)
    - [Example 2: Estimating Multiple Quantiles Simultaneously](#example-2-estimating-multiple-quantiles-simultaneously)
    - [Example 3: Multi-Output Quantile Regression with Weighted Regression and L1 Regularization](#example-3-multi-output-quantile-regression-with-weighted-regression-and-l1-regularization)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Installation
### Prerequisites
- Python 3.9 or higher.
- `ortools` package.

### Install from PyPI
```bash
pip install quantile-regression-pdlp
```

### Install from Source
Clone the repository:
```bash
git clone https://github.com/joshvern/quantile_regression_pdlp.git
cd quantile_regression_pdlp
pip install -U pip
pip install .
```

## Documentation
Docs site: https://joshvern.github.io/quantile_regression_pdlp/

### Quick sanity check
```bash
python -c "from quantile_regression_pdlp import QuantileRegression; print(QuantileRegression)"
```

## Why PDLP?
- Quantile regression can be written as a linear program (LP) by introducing slack variables for positive/negative residuals.
- OR-Tools' PDLP is a first-order method aimed at solving large LPs efficiently compared to many traditional simplex/interior-point approaches.

## Usage
```python
from quantile_regression_pdlp import QuantileRegression
import numpy as np


# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 2)
y = X @ np.array([1.5, -2.0]) + np.random.randn(100) * 0.5


# Initialize and fit the model
model = QuantileRegression(tau=0.5, n_bootstrap=500, random_state=42)
model.fit(X, y)


# Print the summary
print(model.summary())


# Make predictions
X_new = np.array([[0.1, 0.2], [0.5, 0.8]])
y_pred = model.predict(X_new)
print('Predictions:', y_pred)
```

## Quantile Regression as a Linear Programming Problem

Quantile regression aims to estimate the conditional quantiles of a response variable given certain predictor variables. Unlike ordinary least squares regression, which minimizes the sum of squared residuals, quantile regression minimizes a weighted sum of absolute residuals and can incorporate regularization. Estimating multiple quantiles simultaneously allows for a more comprehensive understanding of the conditional distribution and can enforce non-crossing constraints on the training data.

### Mathematical Formulation

For multiple quantiles $\tau_1 < \tau_2 < \dots < \tau_K$ and multiple outputs $M$, the weighted quantile regression problem can be formulated as:

$$
\min_{\beta^{(k, m)}} \sum_{k=1}^{K} \sum_{m=1}^{M} \sum_{i=1}^{n} w_i \rho_{\tau_k}(y_i^{(m)} - x_i^{\top} \beta^{(k, m)})
$$

where:
- $y_i^{(m)}$ is the response variable for the $m^{th}$ output.
- $x_i$ is the vector of predictor variables.
- $w_i$ is the weight assigned to the $i$-th observation.
- $\beta^{(k, m)}$ is the vector of coefficients for quantile $\tau_k$ and output $m$.
- $\rho_{\tau_k}(u)$ is the quantile loss function defined as:

$$
\rho_{\tau_k}(u) = u(\tau_k - \mathbb{I}(u < 0))
$$

To prevent quantile crossing, we impose additional constraints ensuring that the predicted quantile levels are ordered appropriately for each output:

$$
x_i^{\top} \beta^{(1, m)} \leq x_i^{\top} \beta^{(2, m)} \leq \dots \leq x_i^{\top} \beta^{(K, m)}, \quad \forall i = 1, \dots, n; \quad \forall m = 1, \dots, M
$$

### LP Formulation

By introducing auxiliary variables and incorporating regularization, the problem can be rewritten:

#### Variables

- For each quantile $\tau_k$ and output $m$, introduce:
  - Coefficients $\beta^{(k, m)}$.
  - Non-negative slack variables $r_{i}^{+(k, m)}$ and $r_{i}^{-(k, m)}$ for each observation:

$$
y_i^{(m)} - x_i^{\top} \beta^{(k, m)} = r_{i}^{+(k, m)} - r_{i}^{-(k, m)}, \quad \forall i, \forall k, \forall m
$$

- For L1 regularization, introduce auxiliary variables $z_j^{(k, m)}$ for each coefficient $\beta_j^{(k, m)}$ (excluding the intercept):

$$
z_j^{(k, m)} \geq \beta_j^{(k, m)} \\
z_j^{(k, m)} \geq -\beta_j^{(k, m)}, \quad \forall j, \forall k, \forall m
$$

#### Objective Function

The objective function incorporates both the weighted quantile loss for all quantiles and outputs, and the L1 regularization term:

$$
\min_{\beta^{(k, m)}, r^{+(k, m)}, r^{-(k, m)}, z^{(k, m)}} \sum_{k=1}^{K} \sum_{m=1}^{M} \left( \sum_{i=1}^{n} \left( \tau_k w_i r_{i}^{+(k, m)} + (1 - \tau_k) w_i r_{i}^{-(k, m)} \right) + \lambda \sum_{j=1}^{p} z_j^{(k, m)} \right)
$$

where:
- $\lambda$ is the regularization strength parameter.
- $p$ is the number of predictor variables (excluding the intercept).

#### Constraints

Subject to the constraints:

1. **Residual Constraints**:

$$
y_i^{(m)} - x_i^{\top} \beta^{(k, m)} = r_{i}^{+(k, m)} - r_{i}^{-(k, m)}, \quad \forall i = 1, \dots, n; \quad \forall k = 1, \dots, K; \quad \forall m = 1, \dots, M
$$

2. **Non-negativity Constraints**:

$$
r_{i}^{+(k, m)} \geq 0, \quad r_{i}^{-(k, m)} \geq 0, \quad \forall i = 1, \dots, n; \quad \forall k = 1, \dots, K; \quad \forall m = 1, \dots, M
$$

3. **L1 Regularization Constraints**:

$$
z_j^{(k, m)} \geq \beta_j^{(k, m)} \\
z_j^{(k, m)} \geq -\beta_j^{(k, m)}, \quad \forall j = 1, \dots, p; \quad \forall k = 1, \dots, K; \quad \forall m = 1, \dots, M
$$

4. **Non-Crossing Constraints**:

For each output $m$:

$$
x_i^{\top} \beta^{(k, m)} \leq x_i^{\top} \beta^{(k+1, m)}, \quad \forall i = 1, \dots, n; \quad \forall k = 1, \dots, K-1; \quad \forall m = 1, \dots, M
$$

This LP formulation can be efficiently solved using the PDLP solver provided by Google's OR-Tools.

## Features
- **Sklearn and Pandas Compliant**: Seamlessly integrates with `scikit-learn` pipelines and accepts `pandas DataFrame` inputs.
- **Custom Quantiles**: Supports estimation for any quantile $\tau \in (0, 1)$.
- **Multiple Quantiles**: Estimates multiple quantiles simultaneously with non-crossing constraints enforced on the training data.
- **Multi-Output Regression**: Supports regression tasks with multiple target variables, training them jointly.
- **Weighted Quantile Regression**: Assigns different weights to observations, allowing differential influence on regression estimates.
- **L1 Regularization (Lasso)**: Promotes sparsity in the model by penalizing the absolute values of the coefficients.
- **Statistical Summaries**: Provides bootstrap standard errors; reported t-values and p-values use a t reference distribution (approximate).
- **Bootstrap Estimation**: Standard errors are estimated using bootstrap resampling.
- **Parallel Bootstrapping**: Utilizes multiple CPU cores for bootstrapping to speed up computations.
- **Progress Indicators**: Displays progress bars during bootstrapping to inform users of training progress.
- **Simple API**: Designed to be similar to `statsmodels` and `scikit-learn` for ease of use.
- **Efficient Solver**: Utilizes the PDLP solver for efficient computation, suitable for large datasets.

## Documentation
### Class: `QuantileRegression`
*Initialization*
```python
QuantileRegression(
    tau=0.5,
    n_bootstrap=1000,
    random_state=None,
    regularization='none',
    alpha=0.0,
    n_jobs=1
)
```
- **Parameters**:
    - `tau` (float or list of floats, default=0.5): The quantile(s) to estimate, each must be between 0 and 1. Can be a single float or a list of floats.
    - `n_bootstrap` (int, default=1000): Number of bootstrap samples for estimating standard errors.
    - `random_state` (int, default=None): Seed for the random number generator.
    - `regularization` (str, default='none'): Type of regularization to apply. Options are `'l1'` for Lasso regularization or `'none'` for no regularization.
    - `alpha` (float, default=0.0): Regularization strength. Must be a non-negative float. Higher values imply stronger regularization.
    - `n_jobs` (int, default=1): The number of jobs to run in parallel for bootstrapping. `-1` means using all processors.

*Attributes*
- `coef_` (dict): Estimated coefficients for each quantile and output. Structure: {tau: {output: array of coefficients}}.
- `intercept_` (dict): Estimated intercept term for each quantile and output. Structure: {tau: {output: float}}.
- `stderr_` (dict): Standard errors of the coefficients for each quantile and output. Structure: {tau: {output: array of standard errors}}.
- `tvalues_` (dict): T-statistics of the coefficients for each quantile and output. Structure: {tau: {output: array of t-values}}.
- `pvalues_` (dict): P-values of the coefficients for each quantile and output. Structure: {tau: {output: array of p-values}}.
- `feature_names_` (list): List of feature names. If input X is a pandas DataFrame, the column names are used; otherwise, generic names are assigned.
- `output_names_` (list): List of output names. If input y is a pandas DataFrame, the column names are used; otherwise, generic names are assigned.

*Methods*
- `fit(X, y, weights=None)`: Fit the quantile regression model to the data.
    - **Parameters**:
        - `X`: array-like of shape (n_samples, n_features). Training data. Can be a NumPy array or a pandas DataFrame.
        - `y`: array-like of shape (n_samples,) or (n_samples, n_outputs). Target values. Can be a NumPy array, pandas Series, or pandas DataFrame.
        - `weights` (ndarray, optional): Weights for each observation, shape (n_samples,). Default is `None`, which assigns equal weight to all observations.
    - **Returns**: `self`

- `predict(X)`: Predict using the quantile regression model.
    - **Parameters**:
        - `X`: array-like of shape (n_samples, n_features). Samples. Can be a NumPy array or a pandas DataFrame.
    - **Returns**: `y_pred` (dict of dicts of ndarrays). Predicted values for each quantile and output. Structure: {tau: {output: array of predictions}}

- `summary()`: Return a summary of the regression results.
    - **Returns**: `summary_dict` (dict of dicts of pandas DataFrames). Summary tables for each quantile and output with coefficients, standard errors, t-values, and p-values. Structure: {tau: {output: DataFrame}}

## Examples
### Example 1: Basic Usage
```python
from quantile_regression_pdlp import QuantileRegression
import numpy as np

# Data
np.random.seed(0)
X = np.random.rand(50, 1)
y = 2 * X.squeeze() + np.random.randn(50) * 0.5

# Model
model = QuantileRegression(tau=0.5, n_bootstrap=500, random_state=0)
model.fit(X, y)
print(model.summary()[0.5])
```
**Output**
```
Bootstrapping: 100%|████████████████████████████| 500/500 [00:02<00:00, 237.43it/s]
{'y':            Coefficient  Std. Error   t-value     P>|t|
Intercept     0.053282    0.274354  0.194209  0.846815
X1            1.718165    0.372070  4.617858  0.000028}
```

### Example 2: Estimating Multiple Quantiles Simultaneously
```python
from quantile_regression_pdlp import QuantileRegression
import numpy as np

# Data
np.random.seed(0)
X = np.random.rand(50, 1)
y = np.random.rand(50) * 2 + np.random.randn(50) * 0.5

# Model
quantiles = [0.25, 0.5, 0.75]
model = QuantileRegression(tau=quantiles, n_bootstrap=500, random_state=0)
model.fit(X, y)

# Accessing summaries
summaries = model.summary()
for tau, summary_df in summaries.items():
    print(f"\nQuantile: {tau}")
    print(summary_df)

# Making predictions
X_new = np.array([[0.1], [0.5], [0.9]])
predictions = model.predict(X_new)
for tau, y_pred in predictions.items():
    print(f"\nPredictions for quantile {tau}: {y_pred}")
```
**Output**
```
Bootstrapping: 100%|█████████████████████████████| 500/500 [00:09<00:00, 53.90it/s]

Quantile: 0.25
{'y':            Coefficient  Std. Error   t-value     P>|t|
Intercept     0.989803    0.232086  4.264809  0.000091
X1           -0.827212    0.359381 -2.301768  0.025639}

Quantile: 0.5
{'y':            Coefficient  Std. Error   t-value     P>|t|
Intercept     1.000221    0.231796  4.315086  0.000077
X1           -0.054280    0.531496 -0.102128  0.919072}

Quantile: 0.75
{'y':            Coefficient  Std. Error   t-value     P>|t|
Intercept     1.612909    0.383085  4.210315  0.000109
X1           -0.163154    0.670333 -0.243393  0.808717}

Predictions for quantile 0.25: {'y': array([0.90708228, 0.57619763, 0.24531299])}

Predictions for quantile 0.5: {'y': array([0.99479299, 0.97308079, 0.95136859])}

Predictions for quantile 0.75: {'y': array([1.59659361, 1.53133192, 1.46607023])}
```

### Example 3: Multi-Output Quantile Regression with Weighted Regression and L1 Regularization
```python
from quantile_regression_pdlp import QuantileRegression
import pandas as pd
import numpy as np

# Generate sample data
np.random.seed(42)
n_samples = 200
X = pd.DataFrame({
    'Feature1': np.random.rand(n_samples),
    'Feature2': np.random.rand(n_samples),
    'Feature3': np.random.rand(n_samples)
})

# Generate two target variables
y1 = 1.0 + 2.5 * X['Feature1'] - 1.5 * X['Feature2'] + 0.5 * X['Feature3'] + np.random.randn(n_samples) * 0.3
y2 = 2.0 - 1.5 * X['Feature1'] + 2.0 * X['Feature3'] + np.random.randn(n_samples) * 0.4

y = pd.DataFrame({
    'Target1': y1,
    'Target2': y2
})

# Assign weights: higher weights to observations with y1 > median of y1
weights = np.where(y['Target1'] > np.median(y['Target1']), 1.5, 1.0)

# Initialize the model with multiple quantiles and outputs
model = QuantileRegression(
    tau=[0.25, 0.5, 0.75],
    n_bootstrap=500,
    random_state=42,
    regularization='l1',
    alpha=0.1,
    n_jobs=1  
)

# Fit the model with weighted regression
model.fit(X, y, weights=weights)

# Print the summaries for each quantile and output
summaries = model.summary()
for tau, outputs in summaries.items():
    print(f"\nQuantile: {tau}")
    for output, summary_df in outputs.items():
        print(f"\nOutput: {output}")
        print(summary_df)

# Make predictions
X_new = pd.DataFrame({
    'Feature1': [0.2, 0.6],
    'Feature2': [0.3, 0.7],
    'Feature3': [0.4, 0.9]
})
predictions = model.predict(X_new)
for tau, outputs in predictions.items():
    print(f"\nPredictions for Quantile {tau}:")
    for output, y_pred in outputs.items():
        print(f"  {output}: {y_pred}")
```
**Output**
```
Bootstrapping: 100%|█████████████████████████████| 500/500 [03:25<00:00,  2.43it/s]

Quantile: 0.25

Output: Target1
           Coefficient  Std. Error    t-value     P>|t|
Intercept     0.876363    0.091465   9.581398  0.000000
Feature1      2.568977    0.096322  26.670669  0.000000
Feature2     -1.575274    0.111982 -14.067201  0.000000
Feature3      0.338673    0.114226   2.964943  0.003401

Output: Target2
            Coefficient  Std. Error       t-value  P>|t|
Intercept  1.775060e+00    0.137418  1.291719e+01    0.0
Feature1  -1.666380e+00    0.140054 -1.189817e+01    0.0
Feature2  -1.611076e-09    0.112536 -1.431609e-08    1.0
Feature3   2.085970e+00    0.142150  1.467440e+01    0.0

Quantile: 0.5

Output: Target1
           Coefficient  Std. Error    t-value         P>|t|
Intercept     0.933077    0.086188  10.826043  0.000000e+00
Feature1      2.540118    0.100953  25.161316  0.000000e+00
Feature2     -1.526466    0.104012 -14.675813  0.000000e+00
Feature3      0.612945    0.082269   7.450508  2.858602e-12

Output: Target2
           Coefficient  Std. Error    t-value     P>|t|
Intercept     2.027951    0.143698  14.112589  0.000000
Feature1     -1.595173    0.163475  -9.757886  0.000000
Feature2     -0.021603    0.144032  -0.149988  0.880928
Feature3      2.045145    0.134148  15.245390  0.000000

Quantile: 0.75

Output: Target1
           Coefficient  Std. Error    t-value         P>|t|
Intercept     1.243473    0.117297  10.601072  0.000000e+00
Feature1      2.355551    0.099152  23.756937  0.000000e+00
Feature2     -1.647682    0.128211 -12.851339  0.000000e+00
Feature3      0.665442    0.103556   6.425893  9.650365e-10

Output: Target2
           Coefficient  Std. Error    t-value     P>|t|
Intercept     2.386047    0.131412  18.156995  0.000000
Feature1     -1.676041    0.135823 -12.339850  0.000000
Feature2     -0.139892    0.147441  -0.948801  0.343884
Feature3      2.049981    0.138002  14.854732  0.000000

Predictions for Quantile 0.25:
  Target1: [1.05304581 1.61986352]
  Target2: [2.27617155 2.65260442]

Predictions for Quantile 0.5:
  Target1: [1.22833866 1.9402722 ]
  Target2: [2.52049365 2.89635584]

Predictions for Quantile 0.75:
  Target1: [1.48645538 2.10232417]
  Target2: [2.82886323 3.12748038]
```

## Dependencies
- `ortools` (Google's OR-Tools)
- `numpy`
- `pandas`
- `scipy`
- `tqdm`
- `joblib`
- `scikit-learn`

Install dependencies using:
```bash
pip install numpy pandas scipy tqdm ortools joblib scikit-learn
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request on GitHub.
1. Fork the repository on GitHub.
2. Create a new branch for your feature or bugfix.
3. Commit your changes with clear messages.
4. Submit a pull request.

## License
This project is licensed under the MIT License.
