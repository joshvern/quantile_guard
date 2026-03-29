# Theory

## Quantile Regression as a Linear Program

Quantile regression estimates the conditional $\tau$-th quantile of $y$ given $X$
by minimizing the **check (pinball) loss**:

$$
\min_{\beta} \sum_{i=1}^{n} \rho_\tau(y_i - x_i^\top \beta)
$$

where $\rho_\tau(u) = u(\tau - \mathbf{1}(u < 0))$.

By introducing slack variables $u_i^+, u_i^- \geq 0$ for each residual
($y_i - x_i^\top \beta = u_i^+ - u_i^-$), this becomes a standard LP:

$$
\min \;\; \frac{1}{n} \sum_{i=1}^{n} \left[ \tau \, u_i^+ + (1 - \tau) \, u_i^- \right]
$$

subject to $X\beta + u^- - u^+ = y$, and $u^+, u^- \geq 0$.

The $1/n$ normalization makes the regularization parameter $\alpha$ comparable
across different sample sizes, consistent with scikit-learn's convention.

## Non-Crossing Constraints

When fitting multiple quantiles $\tau_1 < \tau_2 < \cdots < \tau_K$ jointly,
the LP adds constraints to prevent quantile crossing on the training data:

$$
x_i^\top \beta_{\tau_j} \leq x_i^\top \beta_{\tau_{j+1}}, \quad \forall\, i,\; j = 1, \ldots, K{-}1
$$

At prediction time, an isotonic regression projection is applied as a fallback
to enforce monotonicity on new data points.

## Solvers

| Solver | Type | Best for |
|--------|------|----------|
| **PDLP** | First-order primal-dual | Large-scale problems ($n > 10{,}000$) |
| **GLOP** | Revised simplex | Small/medium problems, exact solutions |
| **HiGHS** (scipy) | Simplex + interior point | Memory-efficient sparse problems |

## Regularization

### L1 (Lasso)

Adds an $\ell_1$ penalty on slopes (intercept excluded):

$$
\frac{1}{n} \sum_i \rho_\tau(y_i - x_i^\top \beta) + \alpha \sum_{j=1}^{p} |\beta_j|
$$

The L1 penalty is linearized and absorbed into the LP directly.

### Elastic Net

Combines L1 and L2 penalties:

$$
\alpha \left[ \lambda \|\beta\|_1 + \frac{1 - \lambda}{2} \|\beta\|_2^2 \right]
$$

where $\lambda$ is `l1_ratio`. Solved via **Local Linear Approximation (LLA)**:
the L2 term is linearized around the current estimate, converting the problem
to a weighted L1 penalty at each iteration.

### SCAD (Fan & Li, 2001)

The Smoothly Clipped Absolute Deviation penalty reduces bias on large coefficients:

$$
p_\alpha(|\beta_j|) = \begin{cases}
\alpha |\beta_j| & \text{if } |\beta_j| \leq \alpha \\
\frac{2a\alpha|\beta_j| - \beta_j^2 - \alpha^2}{2(a-1)} & \text{if } \alpha < |\beta_j| \leq a\alpha \\
\frac{(a+1)\alpha^2}{2} & \text{if } |\beta_j| > a\alpha
\end{cases}
$$

with $a = 3.7$ (default). Also solved via LLA.

### MCP (Zhang, 2010)

The Minimax Concave Penalty has similar properties to SCAD with a different shape:

$$
p_\alpha(|\beta_j|) = \begin{cases}
\alpha |\beta_j| - \frac{\beta_j^2}{2\gamma} & \text{if } |\beta_j| \leq \gamma\alpha \\
\frac{\gamma \alpha^2}{2} & \text{if } |\beta_j| > \gamma\alpha
\end{cases}
$$

with $\gamma = 3.0$ (default).

### LLA Algorithm

SCAD, MCP, and elastic net are non-convex and cannot be expressed as LPs directly.
The **Local Linear Approximation** iteratively:

1. Compute penalty weights $w_j = p'_\alpha(|\hat\beta_j|)$ from the current estimate
2. Solve a weighted L1 problem: $\frac{1}{n}\sum_i \rho_\tau(\cdot) + \sum_j w_j |\beta_j|$
3. Repeat until convergence (typically 3--10 iterations)

Each sub-problem is a standard LP, so the existing solver infrastructure is reused.

## Standard Error Estimation

### Bootstrap (default)

Nonparametric paired bootstrap:

1. Resample $(X_i, y_i)$ with replacement $B$ times
2. Refit the model on each bootstrap sample
3. Standard errors = standard deviation of bootstrap coefficient estimates
4. P-values = empirical (fraction of bootstrap estimates crossing zero)
5. Confidence intervals = percentile method

### Analytical IID (Koenker & Bassett, 1978)

The asymptotic sandwich estimator under IID errors:

$$
\text{Var}(\hat\beta) = \frac{\tau(1-\tau)}{n \, f_\varepsilon(0)^2} (X^\top X)^{-1}
$$

where $f_\varepsilon(0)$ is the error density at zero, estimated via the
**Hall-Sheather bandwidth**:

$$
h_n = n^{-1/3} \, z_\alpha^{2/3} \left[\frac{1.5\, \phi(z_\alpha)^2}{2 z_\alpha^2 + 1}\right]^{1/3}
$$

with $z_\alpha = \Phi^{-1}(1 - \alpha/2)$ for 95% confidence.

### Kernel (Powell, 1991)

Heteroscedasticity-robust sandwich estimator:

$$
\text{Var}(\hat\beta) = H_n^{-1} \, J_n \, H_n^{-1}
$$

where:

- $H_n = \frac{1}{nh} \sum_i f_i(0) \, x_i x_i^\top$ (estimated via kernel density at each observation)
- $J_n = \frac{\tau(1-\tau)}{n} X^\top X$

This relaxes the IID assumption and is valid under heteroscedastic errors.

### Cluster-Robust

For data with intra-cluster correlation (panel data, grouped observations):

$$
\text{Var}(\hat\beta) = H_n^{-1} \left(\sum_{g=1}^{G} S_g S_g^\top \right) H_n^{-1} \cdot \frac{G}{G-1} \cdot \frac{n-1}{n-k}
$$

where $S_g = \sum_{i \in g} \psi_\tau(y_i - x_i^\top\hat\beta) \, x_i$ is the
cluster-level score, and the multiplicative factor is a small-sample correction.

## Censored Quantile Regression

For right-censored data (e.g., survival analysis), the observed outcome is
$Y_i = \min(T_i, C_i)$ with event indicator $\delta_i = \mathbf{1}(T_i \leq C_i)$.

This package implements **Powell's (1986) iterative algorithm**:

1. Fit standard quantile regression to get initial $\hat\beta$
2. For censored observations ($\delta_i = 0$): if $x_i^\top\hat\beta > Y_i$,
   treat as uncensored; otherwise, set weight to zero
3. Refit with updated weights
4. Repeat until convergence

Left-censored data is handled symmetrically by reflecting the problem.

## Pseudo R-Squared

The Koenker & Machado (1999) pseudo R-squared:

$$
R^1(\tau) = 1 - \frac{\sum_i \rho_\tau(y_i - x_i^\top\hat\beta)}{\sum_i \rho_\tau(y_i - \hat{q}_\tau)}
$$

where $\hat{q}_\tau$ is the unconditional sample quantile. Values closer to 1
indicate better fit relative to the intercept-only model.

## References

- Koenker, R. & Bassett, G. (1978). Regression quantiles. *Econometrica*, 46(1), 33--50.
- Powell, J. L. (1986). Censored regression quantiles. *Journal of Econometrics*, 32(1), 143--155.
- Powell, J. L. (1991). Estimation of monotonic regression models under quantile restrictions. In *Nonparametric and Semiparametric Methods in Econometrics*.
- Koenker, R. & Machado, J. A. F. (1999). Goodness of fit and related inference processes for quantile regression. *JASA*, 94(448), 1296--1310.
- Fan, J. & Li, R. (2001). Variable selection via nonconcave penalized likelihood. *JASA*, 96(456), 1348--1360.
- Zhang, C.-H. (2010). Nearly unbiased variable selection under minimax concave penalty. *Annals of Statistics*, 38(2), 894--942.
- Koenker, R. (2005). *Quantile Regression*. Cambridge University Press.
