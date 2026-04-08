#!/usr/bin/env python
"""
Benchmark: quantile_regression_pdlp vs sklearn vs statsmodels.

Compares coefficient accuracy, pinball loss, runtime, crossing behavior,
and interval coverage across multiple dataset sizes and quantile levels.

Usage:
    python benchmarks/run_linear_baselines.py
    python benchmarks/run_linear_baselines.py --out benchmarks/results/raw/latest.csv
"""

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from quantile_regression_pdlp import QuantileRegression
from quantile_regression_pdlp.metrics import (
    crossing_rate,
    empirical_coverage,
    mean_interval_width,
    pinball_loss,
)


def _check_optional(name: str):
    """Return True if a package is importable."""
    try:
        __import__(name)
        return True
    except ImportError:
        return False


HAS_SKLEARN = _check_optional("sklearn")
HAS_STATSMODELS = _check_optional("statsmodels")


def generate_data(n: int, p: int, seed: int = 42):
    """Generate linear data with heteroscedastic noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    true_beta = rng.uniform(-2, 2, size=p)
    true_intercept = 3.0
    # Heteroscedastic noise: variance depends on X[:,0]
    noise_scale = 0.5 + 0.5 * np.abs(X[:, 0])
    y = true_intercept + X @ true_beta + rng.normal(scale=noise_scale, size=n)
    return X, y, true_beta, true_intercept


def fit_pdlp(X_train, y_train, X_test, taus):
    """Fit PDLP and return predictions + timing."""
    model = QuantileRegression(tau=taus, se_method="analytical")
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    preds = model.predict(X_test)
    predict_time = time.perf_counter() - t0

    # Extract prediction matrix (n_test, n_taus)
    pred_matrix = np.column_stack([preds[tau]["y"] for tau in taus])
    return pred_matrix, fit_time, predict_time


def fit_sklearn(X_train, y_train, X_test, taus):
    """Fit sklearn QuantileRegressor per-quantile and return predictions + timing."""
    from sklearn.linear_model import QuantileRegressor

    pred_cols = []
    total_fit_time = 0.0
    total_predict_time = 0.0

    for tau in taus:
        skl = QuantileRegressor(quantile=tau, alpha=0.0, solver="highs")
        t0 = time.perf_counter()
        skl.fit(X_train, y_train)
        total_fit_time += time.perf_counter() - t0

        t0 = time.perf_counter()
        pred_cols.append(skl.predict(X_test))
        total_predict_time += time.perf_counter() - t0

    pred_matrix = np.column_stack(pred_cols)
    return pred_matrix, total_fit_time, total_predict_time


def fit_statsmodels(X_train, y_train, X_test, taus):
    """Fit statsmodels QuantReg per-quantile and return predictions + timing."""
    import statsmodels.api as sm

    X_train_c = sm.add_constant(X_train)
    X_test_c = sm.add_constant(X_test)

    pred_cols = []
    total_fit_time = 0.0
    total_predict_time = 0.0

    for tau in taus:
        mod = sm.QuantReg(y_train, X_train_c)
        t0 = time.perf_counter()
        res = mod.fit(q=tau, max_iter=1000)
        total_fit_time += time.perf_counter() - t0

        t0 = time.perf_counter()
        pred_cols.append(res.predict(X_test_c))
        total_predict_time += time.perf_counter() - t0

    pred_matrix = np.column_stack(pred_cols)
    return pred_matrix, total_fit_time, total_predict_time


def evaluate(y_test, pred_matrix, taus):
    """Compute evaluation metrics for a prediction matrix."""
    losses = {tau: pinball_loss(y_test, pred_matrix[:, j], tau)
              for j, tau in enumerate(taus)}
    mean_loss = float(np.mean(list(losses.values())))

    cr = crossing_rate(pred_matrix, taus) if len(taus) >= 2 else 0.0

    if len(taus) >= 2:
        cov = empirical_coverage(y_test, pred_matrix[:, 0], pred_matrix[:, -1])
        width = mean_interval_width(pred_matrix[:, 0], pred_matrix[:, -1])
        nominal_coverage = taus[-1] - taus[0]
    else:
        cov = np.nan
        width = np.nan
        nominal_coverage = np.nan

    return {
        "mean_pinball_loss": mean_loss,
        "crossing_rate": cr,
        "empirical_coverage": cov,
        "nominal_coverage": nominal_coverage,
        "mean_interval_width": width,
        **{f"pinball_{tau}": losses[tau] for tau in taus},
    }


def run_benchmarks():
    """Run the full benchmark suite and return a DataFrame of results."""
    configs = [
        {"n": 500, "p": 3},
        {"n": 2000, "p": 5},
        {"n": 5000, "p": 10},
    ]
    taus_list = [
        [0.1, 0.5, 0.9],
        [0.05, 0.25, 0.5, 0.75, 0.95],
    ]

    results = []

    for cfg in configs:
        n, p = cfg["n"], cfg["p"]
        X, y, true_beta, true_intercept = generate_data(n, p)
        split = int(0.8 * n)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        for taus in taus_list:
            tau_label = ",".join(f"{t}" for t in taus)
            print(f"  n={n}, p={p}, taus=[{tau_label}]")

            # PDLP
            pred, ft, pt = fit_pdlp(X_train, y_train, X_test, taus)
            metrics = evaluate(y_test, pred, taus)
            results.append({
                "model": "PDLP (this package)",
                "n": n, "p": p, "n_taus": len(taus), "taus": tau_label,
                "fit_time_s": ft, "predict_time_s": pt,
                **metrics,
            })

            # sklearn
            if HAS_SKLEARN:
                pred, ft, pt = fit_sklearn(X_train, y_train, X_test, taus)
                metrics = evaluate(y_test, pred, taus)
                results.append({
                    "model": "sklearn QuantileRegressor",
                    "n": n, "p": p, "n_taus": len(taus), "taus": tau_label,
                    "fit_time_s": ft, "predict_time_s": pt,
                    **metrics,
                })

            # statsmodels
            if HAS_STATSMODELS:
                pred, ft, pt = fit_statsmodels(X_train, y_train, X_test, taus)
                metrics = evaluate(y_test, pred, taus)
                results.append({
                    "model": "statsmodels QuantReg",
                    "n": n, "p": p, "n_taus": len(taus), "taus": tau_label,
                    "fit_time_s": ft, "predict_time_s": pt,
                    **metrics,
                })

    df = pd.DataFrame(results)
    return df


def add_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Add environment metadata columns."""
    import quantile_regression_pdlp

    df = df.copy()
    df["python_version"] = platform.python_version()
    df["platform"] = platform.platform()
    df["package_version"] = getattr(quantile_regression_pdlp, "__version__", "unknown")
    return df


def main():
    parser = argparse.ArgumentParser(description="Run linear baseline benchmarks")
    parser.add_argument(
        "--out",
        default="benchmarks/results/raw/linear_baselines.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    print("Running linear baseline benchmarks...")
    print(f"  sklearn available: {HAS_SKLEARN}")
    print(f"  statsmodels available: {HAS_STATSMODELS}")

    df = run_benchmarks()
    df = add_metadata(df)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nResults written to {out_path}")
    print(f"  {len(df)} rows")

    # Print summary table
    summary_cols = ["model", "n", "n_taus", "fit_time_s", "mean_pinball_loss",
                    "crossing_rate", "empirical_coverage"]
    print("\nSummary:")
    print(df[summary_cols].to_string(index=False))

    return df


if __name__ == "__main__":
    main()
