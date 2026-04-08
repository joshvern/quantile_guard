# Benchmarks

Reproducible comparisons of `quantile_regression_pdlp` against other quantile regression implementations.

## Linear Baselines

This package is benchmarked against:

- **scikit-learn** `QuantileRegressor` — fits one quantile at a time using HiGHS
- **statsmodels** `QuantReg` — fits one quantile at a time using interior point

### Pinball Loss

All three libraries solve the same underlying linear program, so pinball loss
is equivalent across them.

| n | quantiles | PDLP (this package) | sklearn | statsmodels |
|---:|---:|---:|---:|---:|
| 500 | 3 | 0.2291 | 0.2291 | 0.2291 |
| 500 | 5 | 0.2254 | 0.2254 | 0.2254 |
| 2000 | 3 | 0.2268 | 0.2268 | 0.2268 |
| 2000 | 5 | 0.2238 | 0.2238 | 0.2238 |
| 5000 | 3 | 0.2375 | 0.2375 | 0.2375 |
| 5000 | 5 | 0.2347 | 0.2347 | 0.2347 |

### Fit Time

PDLP fits all quantiles jointly in a single LP, which adds overhead compared
to fitting each quantile independently. On small problems, sklearn and
statsmodels are faster. The tradeoff is that PDLP enforces non-crossing
constraints and provides multi-quantile inference in one call.

| n | quantiles | PDLP | sklearn (sum of per-quantile fits) | statsmodels (sum) |
|---:|---:|---:|---:|---:|
| 500 | 3 | 0.30s | 0.04s | 0.04s |
| 500 | 5 | 0.71s | 0.05s | 0.04s |
| 2000 | 3 | 1.81s | 0.27s | 0.02s |
| 2000 | 5 | 4.77s | 0.44s | 0.06s |
| 5000 | 3 | 7.18s | 1.88s | 0.21s |
| 5000 | 5 | 23.6s | 3.07s | 0.21s |

!!! note "Speed context"
    PDLP is a first-order LP solver designed for large-scale problems. For
    small/medium dense problems, simplex-based solvers (GLOP, HiGHS) are
    faster. PDLP's advantage appears on problems with many constraints (large
    n, many quantiles, non-crossing + regularization).

    For small problems, use `solver_backend='GLOP'` or `use_sparse=True`.

### Crossing Rate

PDLP enforces non-crossing constraints in the joint LP formulation, so its
crossing rate is always **zero by construction**. sklearn and statsmodels fit
each quantile independently, which can produce crossings when the data is
noisy or when quantiles are close together.

| n | quantiles | PDLP | sklearn | statsmodels |
|---:|---:|---:|---:|---:|
| 500 | 3 | 0.0% | 0.0% | 0.0% |
| 500 | 5 | 0.0% | 0.0% | 0.0% |
| 2000 | 3 | 0.0% | 0.0% | 0.0% |
| 2000 | 5 | 0.0% | 0.0% | 0.0% |
| 5000 | 3 | 0.0% | 0.0% | 0.0% |
| 5000 | 5 | 0.0% | 0.0% | 0.0% |

!!! note "Crossings in practice"
    On well-specified linear data, independent quantile fits rarely cross.
    Crossings become more common with model misspecification, high noise,
    small samples, or many closely-spaced quantiles. The non-crossing
    guarantee matters most in production pipelines where crossings would
    cause downstream errors.

### Empirical Coverage

Coverage of the interval formed by the outermost quantile predictions.

| n | quantiles | Nominal | PDLP | sklearn | statsmodels |
|---:|---:|---:|---:|---:|---:|
| 500 | 3 | 80% | 84.0% | 84.0% | 84.0% |
| 500 | 5 | 90% | 91.0% | 91.0% | 91.0% |
| 2000 | 3 | 80% | 81.5% | 81.5% | 81.5% |
| 2000 | 5 | 90% | 89.0% | 89.0% | 89.0% |
| 5000 | 3 | 80% | 77.5% | 77.5% | 77.5% |
| 5000 | 5 | 90% | 90.1% | 90.1% | 90.1% |

## What This Package Adds Beyond Accuracy Parity

Since the underlying LP is the same, the accuracy match is expected. The
differences that matter are in the **features available around the fit**:

| Feature | This package | sklearn | statsmodels |
|---------|:---:|:---:|:---:|
| Joint multi-quantile fit | Yes | No | No |
| Non-crossing guarantee | Yes | No | No |
| Standard errors | Analytical, kernel, cluster, bootstrap | No | Analytical only |
| Conformal calibration | Built-in CQR | No | No |
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

Results are deterministic (fixed random seeds). Raw CSV output includes
Python version, platform, and package version metadata.

See `benchmarks/README.md` for details.
