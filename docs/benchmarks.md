# Benchmarks

## Zero Crossings, Even on Hard Data

The central promise of this package: quantile predictions that never cross, by construction. These benchmarks demonstrate that promise on deliberately challenging data — and show that the non-crossing constraint also acts as beneficial regularization, improving prediction quality.

**Test conditions:** heavy-tailed heteroscedastic noise (Student-t, df=3), 10-20 features, up to 13 quantile levels. This is data designed to stress quantile estimators — in practice, your data may be gentler, but the guarantee still matters for production pipelines.

**What's compared:** this package (joint multi-quantile LP with non-crossing constraints) vs sklearn `QuantileRegressor` and statsmodels `QuantReg` (both fit each quantile independently).

### Crossing Rate (fraction of test samples with at least one violation)

| n | p | quantiles | This package | sklearn | statsmodels |
|---:|---:|---:|---:|---:|---:|
| 500 | 10 | 7 | **0%** | 11.0% | 11.0% |
| 500 | 10 | 13 | **0%** | 30.0% | 30.0% |
| 1,000 | 10 | 7 | **0%** | 6.0% | 4.0% |
| 1,000 | 10 | 13 | **0%** | 16.5% | 15.0% |
| 2,000 | 20 | 7 | **0%** | 4.5% | 4.5% |
| 2,000 | 20 | 13 | **0%** | 11.0% | 11.0% |
| 5,000 | 20 | 13 | **0%** | 0.4% | 0.4% |

Crossings are worst when:

- the data has heavy tails and heteroscedasticity
- many closely-spaced quantiles are fitted (13 vs 7)
- the sample is small relative to the number of features

At n=500 with 13 quantiles, **30% of test samples have crossings** in sklearn/statsmodels. This package has zero by construction.

## Pinball Loss

The joint non-crossing formulation achieves equal or slightly better pinball loss than independent fitting. The improvement is most visible at small n with many quantiles, where the non-crossing constraints act as beneficial regularization.

| n | p | quantiles | This package | sklearn | statsmodels |
|---:|---:|---:|---:|---:|---:|
| 500 | 10 | 7 | **0.5148** | 0.5166 | 0.5166 |
| 500 | 10 | 13 | **0.5095** | 0.5240 | 0.5240 |
| 1,000 | 10 | 7 | **0.5082** | 0.5091 | 0.5084 |
| 1,000 | 10 | 13 | **0.5048** | 0.5071 | 0.5067 |
| 2,000 | 20 | 7 | **0.5604** | 0.5606 | 0.5606 |
| 2,000 | 20 | 13 | **0.5599** | 0.5611 | 0.5611 |
| 5,000 | 20 | 13 | **0.5893** | 0.5896 | 0.5896 |

At n=500, 13 quantiles: this package achieves 0.5095 vs 0.5240 — a **2.8% improvement** from the joint formulation.

## The Speed Tradeoff (Honest)

This package is **slower** on raw wall-clock time. That's the tradeoff for solving a single joint LP with non-crossing constraints, rather than 7 or 13 separate small LPs.

| n | p | quantiles | This package (sparse) | sklearn (sum of fits) | statsmodels (sum of fits) |
|---:|---:|---:|---:|---:|---:|
| 500 | 10 | 7 | 0.8s | 0.1s | 0.3s |
| 500 | 10 | 13 | 2.8s | 0.2s | 0.4s |
| 1,000 | 10 | 7 | 2.7s | 0.3s | 0.1s |
| 1,000 | 10 | 13 | 10.3s | 0.6s | 0.3s |
| 2,000 | 20 | 7 | 19.7s | 1.3s | 0.5s |
| 2,000 | 20 | 13 | 163.0s | 2.4s | 0.6s |
| 5,000 | 20 | 7 | 125.1s | 6.7s | 0.5s |

### Why the tradeoff is worth it

The extra time buys you:

- **Zero crossings** — no post-hoc fixes, no downstream pipeline failures
- **Joint estimation** — all quantiles fitted together, sharing information
- **Better pinball loss** — the non-crossing constraints regularize beneficially
- **One fit call** — inference, intervals, and diagnostics all come from the same model

If you need only a single quantile with no crossing constraints, sklearn or statsmodels will be faster. This package's value is in the joint multi-quantile fit with guarantees — and the inference, calibration, and evaluation tools built around it.

!!! tip "Speeding things up"
    For smaller problems, use `solver_backend='GLOP'` for the simplex solver.
    For memory-constrained settings, use `use_sparse=True`.

## Empirical Coverage

Coverage of the interval formed by the outermost quantile predictions (e.g., [0.05, 0.95] for 7 quantiles). All methods achieve similar marginal coverage since they solve the same underlying optimization.

| n | p | quantiles | Nominal | This package | sklearn | statsmodels |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 10 | 7 | 90% | 88% | 88% | 88% |
| 500 | 10 | 13 | 98% | 96% | 96% | 96% |
| 1,000 | 10 | 7 | 90% | 89% | 89% | 89% |
| 1,000 | 10 | 13 | 98% | 97% | 97% | 97% |
| 2,000 | 20 | 7 | 90% | 84% | 84% | 84% |
| 5,000 | 20 | 7 | 90% | 82% | 82% | 82% |

For better-calibrated intervals, use [Conformalized Quantile Regression](conformal.md).

## Test Data

All benchmarks use synthetic data with:

- **Heavy-tailed noise**: Student-t with 3 degrees of freedom
- **Heteroscedasticity**: noise scale grows with feature values
- **Fixed seed**: fully deterministic and reproducible

This is deliberately challenging data. On well-behaved Gaussian data, crossing rates would be lower — but the guarantee still matters for production pipelines.

## Beyond Accuracy: The Full Toolkit

| Feature | This package | sklearn | statsmodels |
|---------|:---:|:---:|:---:|
| Joint multi-quantile fit | Yes | No | No |
| Non-crossing guarantee | Yes | No | No |
| Standard errors | Analytical, kernel, cluster, bootstrap | No | Analytical only |
| Conformal calibration | Built-in CQR | No | No |
| Calibration diagnostics | Yes | No | No |
| Evaluation metrics suite | Yes | Partial | No |
| Crossing detection + fix | Yes | No | No |
| Censored QR | Yes | No | No |
| SCAD / MCP / elastic net | Yes | L1 only | No |

## Reproducing These Results

```bash
pip install -e ".[benchmark]"
python benchmarks/run_linear_baselines.py
python benchmarks/report.py
```

Results are deterministic (fixed random seeds). Raw CSV output includes Python version, platform, and package version metadata.

See `benchmarks/README.md` for details.
