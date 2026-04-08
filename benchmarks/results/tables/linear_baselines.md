# Benchmark Results: Linear Baselines

Generated from `benchmarks/results/raw/linear_baselines.csv`

## Pinball Loss Comparison

| n | quantiles | PDLP (this package) | sklearn QuantileRegressor | statsmodels QuantReg |
|---:|---:|---:|---:|---:|
| 500 | 3 | 0.2291 | 0.2291 | 0.2291 |
| 500 | 5 | 0.2254 | 0.2254 | 0.2254 |
| 2000 | 3 | 0.2268 | 0.2268 | 0.2268 |
| 2000 | 5 | 0.2238 | 0.2238 | 0.2238 |
| 5000 | 3 | 0.2375 | 0.2375 | 0.2375 |
| 5000 | 5 | 0.2347 | 0.2347 | 0.2347 |

## Fit Time (seconds)

| n | quantiles | PDLP (this package) | sklearn QuantileRegressor | statsmodels QuantReg |
|---:|---:|---:|---:|---:|
| 500 | 3 | 0.2965 | 0.0427 | 0.0398 |
| 500 | 5 | 0.7051 | 0.0491 | 0.0408 |
| 2000 | 3 | 1.8082 | 0.2685 | 0.0245 |
| 2000 | 5 | 4.7723 | 0.4408 | 0.0625 |
| 5000 | 3 | 7.1842 | 1.8837 | 0.2138 |
| 5000 | 5 | 23.6448 | 3.0735 | 0.2080 |

## Crossing Rate

| n | quantiles | PDLP (this package) | sklearn QuantileRegressor | statsmodels QuantReg |
|---:|---:|---:|---:|---:|
| 500 | 3 | 0.0000 | 0.0000 | 0.0000 |
| 500 | 5 | 0.0000 | 0.0000 | 0.0000 |
| 2000 | 3 | 0.0000 | 0.0000 | 0.0000 |
| 2000 | 5 | 0.0000 | 0.0000 | 0.0000 |
| 5000 | 3 | 0.0000 | 0.0000 | 0.0000 |
| 5000 | 5 | 0.0000 | 0.0000 | 0.0000 |

## Empirical Coverage (outer quantile interval)

| n | quantiles | PDLP (this package) | sklearn QuantileRegressor | statsmodels QuantReg |
|---:|---:|---:|---:|---:|
| 500 | 3 | 0.8400 | 0.8400 | 0.8400 |
| 500 | 5 | 0.9100 | 0.9100 | 0.9100 |
| 2000 | 3 | 0.8150 | 0.8150 | 0.8150 |
| 2000 | 5 | 0.8900 | 0.8900 | 0.8900 |
| 5000 | 3 | 0.7750 | 0.7750 | 0.7750 |
| 5000 | 5 | 0.9010 | 0.9010 | 0.9010 |
