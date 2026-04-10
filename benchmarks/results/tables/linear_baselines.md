# Benchmark Results: Linear Baselines

Generated from `benchmarks/results/raw/linear_baselines.csv`

## Benchmark Run Metadata

- Rows: 30
- Configs: n=500/p=10/heavy, n=1,000/p=10/heavy, n=2,000/p=20/heavy, n=5,000/p=20/heavy
- Quantile counts: 7, 13
- Models: PDLP (joint, non-crossing), PDLP sparse (joint, non-crossing), sklearn (independent), statsmodels (independent)
- Package version: 0.6.1
- Python version: 3.8.10
- Platform: Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.29

## Pinball Loss Comparison

| n | quantiles | PDLP (joint, non-crossing) | PDLP sparse (joint, non-crossing) | sklearn (independent) | statsmodels (independent) |
|---:|---:|---:|---:|---:|---:|
| 500 | 7 | 0.5148 | 0.5148 | 0.5166 | 0.5166 |
| 500 | 13 | 0.5095 | 0.5098 | 0.5240 | 0.5240 |
| 1000 | 7 | 0.5082 | 0.5082 | 0.5091 | 0.5084 |
| 1000 | 13 | 0.5048 | 0.5048 | 0.5071 | 0.5067 |
| 2000 | 7 | 0.5604 | 0.5604 | 0.5606 | 0.5606 |
| 2000 | 13 | 0.5599 | 0.5600 | 0.5611 | 0.5611 |
| 5000 | 7 | — | 0.5925 | 0.5925 | 0.5925 |
| 5000 | 13 | — | 0.5893 | 0.5896 | 0.5896 |

## Fit Time (seconds)

| n | quantiles | PDLP (joint, non-crossing) | PDLP sparse (joint, non-crossing) | sklearn (independent) | statsmodels (independent) |
|---:|---:|---:|---:|---:|---:|
| 500 | 7 | 2.1446 | 0.8238 | 0.1020 | 0.2860 |
| 500 | 13 | 5.6563 | 2.9811 | 0.2041 | 0.4134 |
| 1000 | 7 | 5.1052 | 2.6746 | 0.2948 | 0.1405 |
| 1000 | 13 | 30.0303 | 9.9827 | 0.5041 | 0.3055 |
| 2000 | 7 | 19.9364 | 18.4750 | 1.2737 | 3.3971 |
| 2000 | 13 | 81.0171 | 172.2700 | 2.3512 | 5.4355 |
| 5000 | 7 | — | 124.9229 | 6.9568 | 4.2353 |
| 5000 | 13 | — | 676.2534 | 12.9124 | 19.3487 |

## Crossing Rate

| n | quantiles | PDLP (joint, non-crossing) | PDLP sparse (joint, non-crossing) | sklearn (independent) | statsmodels (independent) |
|---:|---:|---:|---:|---:|---:|
| 500 | 7 | 0.0000 | 0.0000 | 0.1100 | 0.1100 |
| 500 | 13 | 0.0000 | 0.0000 | 0.3000 | 0.3000 |
| 1000 | 7 | 0.0000 | 0.0000 | 0.0600 | 0.0400 |
| 1000 | 13 | 0.0000 | 0.0000 | 0.1650 | 0.1500 |
| 2000 | 7 | 0.0000 | 0.0000 | 0.0450 | 0.0450 |
| 2000 | 13 | 0.0000 | 0.0000 | 0.1100 | 0.1100 |
| 5000 | 7 | — | 0.0000 | 0.0000 | 0.0000 |
| 5000 | 13 | — | 0.0000 | 0.0040 | 0.0040 |

## Empirical Coverage (outer quantile interval)

| n | quantiles | PDLP (joint, non-crossing) | PDLP sparse (joint, non-crossing) | sklearn (independent) | statsmodels (independent) |
|---:|---:|---:|---:|---:|---:|
| 500 | 7 | 0.9300 | 0.9300 | 0.9200 | 0.9200 |
| 500 | 13 | 0.9700 | 0.9700 | 0.9400 | 0.9400 |
| 1000 | 7 | 0.8950 | 0.8950 | 0.8900 | 0.8850 |
| 1000 | 13 | 0.9800 | 0.9800 | 0.9550 | 0.9550 |
| 2000 | 7 | 0.8900 | 0.8900 | 0.8825 | 0.8825 |
| 2000 | 13 | 0.9575 | 0.9575 | 0.9375 | 0.9375 |
| 5000 | 7 | — | 0.8910 | 0.8910 | 0.8910 |
| 5000 | 13 | — | 0.9760 | 0.9770 | 0.9770 |
