# Benchmarks

quantile-guard fits all quantiles jointly with non-crossing constraints. Independent fitters (sklearn, statsmodels) fit each quantile separately with no monotonicity enforcement. These benchmarks measure the practical difference on deliberately challenging data.

**Test conditions:** heavy-tailed heteroscedastic noise (Student-t, df=3), 10-20 features, up to 13 quantile levels. This is data designed to stress quantile estimators — in practice, your data may be gentler, but the guarantee still matters for production pipelines.

## Crossing Rate

Fraction of test samples where at least one quantile prediction violates monotonicity:

| n | p | quantiles | quantile-guard | sklearn | statsmodels |
|---:|---:|---:|---:|---:|---:|
| 500 | 10 | 7 | **0%** | 11.0% | 11.0% |
| 500 | 10 | 13 | **0%** | 30.0% | 30.0% |
| 1,000 | 10 | 7 | **0%** | 6.0% | 4.0% |
| 1,000 | 10 | 13 | **0%** | 16.5% | 15.0% |
| 2,000 | 20 | 7 | **0%** | 4.5% | 4.5% |
| 2,000 | 20 | 13 | **0%** | 11.0% | 11.0% |
| 5,000 | 20 | 7 | **0%** | 0.0% | 0.0% |
| 5,000 | 20 | 13 | **0%** | 0.4% | 0.4% |

Crossings are worst when:

- the data has heavy tails and heteroscedasticity
- many closely-spaced quantiles are fitted (13 vs 7)
- the sample is small relative to the number of features

At n=500 with 13 quantiles, **30% of test samples have crossings** in sklearn/statsmodels. quantile-guard has zero by construction.

## Pinball Loss

The joint non-crossing formulation matches pinball loss closely and improves it in the hardest small-sample settings. The improvement is most visible at small n with many quantiles, where the non-crossing constraints can act as beneficial regularization.

| n | p | quantiles | quantile-guard | sklearn | statsmodels |
|---:|---:|---:|---:|---:|---:|
| 500 | 10 | 7 | **0.5148** | 0.5166 | 0.5166 |
| 500 | 10 | 13 | **0.5095** | 0.5240 | 0.5240 |
| 1,000 | 10 | 7 | **0.5082** | 0.5091 | 0.5084 |
| 1,000 | 10 | 13 | **0.5048** | 0.5071 | 0.5067 |
| 2,000 | 20 | 7 | **0.5604** | 0.5606 | 0.5606 |
| 2,000 | 20 | 13 | **0.5599** | 0.5611 | 0.5611 |
| 5,000 | 20 | 7 | 0.5925 | **0.5925** | **0.5925** |
| 5,000 | 20 | 13 | **0.5893** | 0.5896 | 0.5896 |

At n=500, 13 quantiles: quantile-guard achieves 0.5095 vs 0.5240 — a **2.8% improvement** from the joint formulation.

## The Speed Tradeoff

quantile-guard is **slower** on raw wall-clock time. That's the cost of solving a single joint LP with non-crossing constraints, rather than 7 or 13 separate small LPs.

| n | p | quantiles | quantile-guard (sparse) | sklearn (sum of fits) | statsmodels (sum of fits) |
|---:|---:|---:|---:|---:|---:|
| 500 | 10 | 7 | 0.8s | 0.1s | 0.3s |
| 500 | 10 | 13 | 3.0s | 0.2s | 0.4s |
| 1,000 | 10 | 7 | 2.7s | 0.3s | 0.1s |
| 1,000 | 10 | 13 | 10.0s | 0.5s | 0.3s |
| 2,000 | 20 | 7 | 18.5s | 1.3s | 3.4s |
| 2,000 | 20 | 13 | 172.3s | 2.4s | 5.4s |
| 5,000 | 20 | 7 | 124.9s | 7.0s | 4.2s |
| 5,000 | 20 | 13 | 676.3s | 12.9s | 19.3s |

### What the extra time buys you

- **Zero crossings** — no post-hoc fixes, no downstream pipeline failures
- **Joint estimation** — all quantiles fitted together, sharing information
- **Better pinball loss** — the non-crossing constraints regularize beneficially
- **One fit call** — inference, intervals, and diagnostics all come from the same model

If you need only a single quantile with no crossing concerns, sklearn or statsmodels will be faster. quantile-guard's value is in the joint multi-quantile fit with guarantees — and the inference, calibration, and evaluation tools built around it.

!!! tip "Speeding things up"
    For smaller problems, use `solver_backend='GLOP'` for the simplex solver.
    For memory-constrained settings, use `use_sparse=True`.

## Empirical Coverage

Coverage of the interval formed by the outermost quantile predictions (e.g., [0.05, 0.95] for 7 quantiles). Marginal coverage is broadly similar across methods on this synthetic benchmark, though the independent fits can deviate more when crossings are severe.

| n | p | quantiles | Nominal | quantile-guard | sklearn | statsmodels |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 10 | 7 | 90% | 93.0% | 92.0% | 92.0% |
| 500 | 10 | 13 | 98% | 97.0% | 94.0% | 94.0% |
| 1,000 | 10 | 7 | 90% | 89.5% | 89.0% | 88.5% |
| 1,000 | 10 | 13 | 98% | 98.0% | 95.5% | 95.5% |
| 2,000 | 20 | 7 | 90% | 89.0% | 88.2% | 88.2% |
| 2,000 | 20 | 13 | 98% | 95.8% | 93.8% | 93.8% |
| 5,000 | 20 | 7 | 90% | 89.1% | 89.1% | 89.1% |
| 5,000 | 20 | 13 | 98% | 97.6% | 97.7% | 97.7% |

For better-calibrated intervals, use [Conformalized Quantile Regression](conformal.md).

## Test Data

All benchmarks use synthetic data with:

- **Heavy-tailed noise**: Student-t with 3 degrees of freedom
- **Heteroscedasticity**: noise scale grows with feature values
- **Fixed seed**: fully deterministic and reproducible

This is deliberately challenging data. On well-behaved Gaussian data, crossing rates would be lower — but the guarantee still matters for production pipelines.

## Beyond Accuracy: The Full Toolkit

quantile-guard is more than a quantile regressor. The benchmark comparison above covers only the core fitting — the package also provides:

| Capability | What it adds |
|---------|-------------|
| Standard errors | Analytical, kernel, cluster-robust, bootstrap |
| Conformal calibration | Built-in CQR with finite-sample coverage guarantees |
| Calibration diagnostics | Coverage by group, bin, feature; sharpness analysis |
| Evaluation metrics | Pinball loss, coverage, interval score, crossing rate |
| Crossing detection + repair | Diagnose and fix crossings from any model |
| Censored QR | Right- and left-censored survival models |
| Regularization | L1, elastic net, SCAD, MCP |

sklearn's `QuantileRegressor` does not provide this end-to-end workflow, and statsmodels' `QuantReg` covers only part of the inference story.

## Reproducing These Results

```bash
pip install -e ".[benchmark]"
python benchmarks/run_linear_baselines.py
python benchmarks/report.py
```

Results are deterministic (fixed random seeds). Raw CSV output includes Python version, platform, and package version metadata.

See `benchmarks/README.md` for details.
