# Benchmarks

Reproducible benchmarks comparing `quantile_guard` against other quantile regression implementations.

## Quick Start

```bash
# Install benchmark dependencies
pip install -e ".[benchmark]"

# Run benchmarks
python benchmarks/run_linear_baselines.py

# Generate report tables and figures
python benchmarks/report.py
```

## What's Compared

### Linear Baselines (`run_linear_baselines.py`)

Compares this package against:

| Library | Model | Notes |
|---------|-------|-------|
| **This package** | `QuantileRegression` | Joint multi-quantile, non-crossing constraints |
| **scikit-learn** | `QuantileRegressor` | Per-quantile fitting, HiGHS solver |
| **statsmodels** | `QuantReg` | Per-quantile fitting via `QuantReg.fit()` |

Across:
- Dataset configurations: `(n=500, p=10)`, `(n=1000, p=10)`, `(n=2000, p=20)`, `(n=5000, p=20)`
- Quantile sets: 7 quantiles `[0.05, ..., 0.95]` and 13 quantiles `[0.01, ..., 0.99]`
- Noise: heavy-tailed heteroscedastic (Student-t, df=3)

### Metrics

- **Pinball loss** (per quantile and mean)
- **Fit time** (seconds)
- **Predict time** (seconds)
- **Crossing rate** (fraction of samples with quantile crossings)
- **Empirical coverage** (outer quantile interval)
- **Mean interval width**

## Results

After running, results are written to:

```
benchmarks/results/
  raw/linear_baselines.csv       # raw benchmark output
  tables/linear_baselines.md     # markdown summary tables
  figures/                       # PNG charts
    fit_time_vs_n.png
    pinball_loss_vs_n.png
    crossing_rate.png
    benchmark_overview.png
```

`benchmark_overview.png` is a compact shareable snapshot of the hardest
benchmark grid, intended for README/docs/social snippets where the full tables
would be too dense.

## Key Findings

On heavy-tailed heteroscedastic data:
- **Zero crossing rate** vs 4-30% for sklearn/statsmodels (crossings worst at small n with many quantiles)
- **Comparable or better pinball loss in most settings** — the non-crossing constraints act as beneficial regularization (up to 2.8% improvement at n=500, 13 quantiles)
- **Single fit call** for all quantiles with joint non-crossing constraints
- **Slower wall-clock time** — the trade-off for solving a joint LP with non-crossing constraints vs separate small LPs

## Reproducing

Results are deterministic (fixed seeds). To reproduce:

```bash
python benchmarks/run_linear_baselines.py --out benchmarks/results/raw/linear_baselines.csv
python benchmarks/report.py --input benchmarks/results/raw/linear_baselines.csv
```

Environment metadata (Python version, platform, package version) is included in the CSV output.
