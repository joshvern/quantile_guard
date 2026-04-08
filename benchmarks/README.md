# Benchmarks

Reproducible benchmarks comparing `quantile_regression_pdlp` against other quantile regression implementations.

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
| **statsmodels** | `QuantReg` | Per-quantile fitting, interior point |

Across:
- Dataset sizes: 500, 2000, 5000 samples
- Feature counts: 3, 5, 10
- Quantile sets: `[0.1, 0.5, 0.9]` and `[0.05, 0.25, 0.5, 0.75, 0.95]`

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
results/
  raw/linear_baselines.csv       # raw data
  tables/linear_baselines.md     # markdown summary tables
  figures/                       # PNG charts
    fit_time_vs_n.png
    pinball_loss_vs_n.png
    crossing_rate.png
```

## Key Findings

This package achieves:
- **Equivalent pinball loss** to sklearn and statsmodels (same underlying LP)
- **Zero crossing rate** due to joint non-crossing constraints (sklearn/statsmodels fit quantiles independently and can cross)
- **Single fit call** for all quantiles (sklearn/statsmodels require one fit per quantile)

## Reproducing

Results are deterministic (fixed seeds). To reproduce:

```bash
python benchmarks/run_linear_baselines.py --out benchmarks/results/raw/linear_baselines.csv
python benchmarks/report.py --input benchmarks/results/raw/linear_baselines.csv
```

Environment metadata (Python version, platform, package version) is included in the CSV output.
