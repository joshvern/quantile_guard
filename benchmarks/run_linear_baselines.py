#!/usr/bin/env python
"""
Benchmark: quantile_guard vs sklearn vs statsmodels.

Designed to show where this package's joint LP formulation, non-crossing
constraints, and sparse solver matter: large datasets, many features, many
quantiles, and noisy/heteroscedastic data where independent fits produce
crossings.

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

from quantile_guard import QuantileRegression
from quantile_guard.metrics import (
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


def generate_data(n: int, p: int, seed: int = 42, noise: str = "heavy"):
    """Generate data that stresses quantile estimation.

    Parameters
    ----------
    noise : str
        'heavy' — heteroscedastic + heavy-tailed (t-distributed) noise that
        makes extreme quantiles hard and encourages crossings in independent fits.
        'moderate' — heteroscedastic Gaussian.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    true_beta = rng.uniform(-2, 2, size=p)
    true_intercept = 3.0

    # Heteroscedastic scale: grows with |X[:,0]| and |X[:,1]| if available
    noise_scale = 0.5 + np.abs(X[:, 0])
    if p > 1:
        noise_scale += 0.3 * np.abs(X[:, 1])

    if noise == "heavy":
        # Heavy tails — t(3) scaled by heteroscedastic envelope
        eps = rng.standard_t(df=3, size=n) * noise_scale
    else:
        eps = rng.normal(scale=noise_scale, size=n)

    y = true_intercept + X @ true_beta + eps
    return X, y, true_beta, true_intercept


# ── Fitters ──────────────────────────────────────────────────────────

def fit_pdlp(X_train, y_train, X_test, taus, use_sparse=False):
    """Fit PDLP (joint multi-quantile) and return predictions + timing."""
    model = QuantileRegression(
        tau=taus, se_method="analytical", use_sparse=use_sparse,
    )
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    preds = model.predict(X_test)
    predict_time = time.perf_counter() - t0

    pred_matrix = np.column_stack([preds[tau]["y"] for tau in taus])
    return pred_matrix, fit_time, predict_time


def fit_sklearn(X_train, y_train, X_test, taus):
    """Fit sklearn QuantileRegressor per-quantile."""
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
    """Fit statsmodels QuantReg per-quantile."""
    import statsmodels.api as sm

    X_train_c = sm.add_constant(X_train)
    X_test_c = sm.add_constant(X_test)

    pred_cols = []
    total_fit_time = 0.0
    total_predict_time = 0.0

    for tau in taus:
        mod = sm.QuantReg(y_train, X_train_c)
        t0 = time.perf_counter()
        res = mod.fit(q=tau, max_iter=3000)
        total_fit_time += time.perf_counter() - t0

        t0 = time.perf_counter()
        pred_cols.append(res.predict(X_test_c))
        total_predict_time += time.perf_counter() - t0

    pred_matrix = np.column_stack(pred_cols)
    return pred_matrix, total_fit_time, total_predict_time


# ── Evaluation ───────────────────────────────────────────────────────

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


# ── Main benchmark ───────────────────────────────────────────────────

def run_benchmarks():
    """Run the full benchmark suite and return a DataFrame of results."""

    # Heavy-tailed heteroscedastic noise ensures crossings appear
    # for independent fitters even at moderate sample sizes.
    #
    # Key scenarios:
    # - Small+many quantiles: crossing advantage most visible
    # - Medium datasets: practical everyday scale
    # - Larger n: shows sparse solver vs baselines at scale
    configs = [
        {"n": 500, "p": 10, "noise": "heavy"},
        {"n": 1_000, "p": 10, "noise": "heavy"},
        {"n": 2_000, "p": 20, "noise": "heavy"},
        {"n": 5_000, "p": 20, "noise": "heavy"},
    ]

    taus_list = [
        # 7 quantiles — typical use case
        [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
        # 13 quantiles — dense grid, stress test for crossings
        [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
    ]

    results = []

    for cfg in configs:
        n, p, noise = cfg["n"], cfg["p"], cfg["noise"]
        X, y, true_beta, true_intercept = generate_data(n, p, noise=noise)
        split = int(0.8 * n)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        for taus in taus_list:
            tau_label = ",".join(f"{t}" for t in taus)
            n_taus = len(taus)
            print(f"\n  n={n:,}, p={p}, taus={n_taus}, noise={noise}")

            # ── PDLP (default OR-Tools solver) — skip on large configs ──
            if n <= 2_000:
                try:
                    pred, ft, pt = fit_pdlp(X_train, y_train, X_test, taus,
                                            use_sparse=False)
                    metrics = evaluate(y_test, pred, taus)
                    results.append({
                        "model": "PDLP (joint, non-crossing)",
                        "n": n, "p": p, "n_taus": n_taus, "taus": tau_label,
                        "noise": noise,
                        "fit_time_s": ft, "predict_time_s": pt,
                        **metrics,
                    })
                    print(f"    PDLP:        {ft:7.2f}s  pinball={metrics['mean_pinball_loss']:.4f}  "
                          f"crossing={metrics['crossing_rate']:.4f}")
                except Exception as e:
                    print(f"    PDLP failed: {e}")

            # ── PDLP sparse solver ──
            try:
                pred, ft, pt = fit_pdlp(X_train, y_train, X_test, taus,
                                        use_sparse=True)
                metrics = evaluate(y_test, pred, taus)
                results.append({
                    "model": "PDLP sparse (joint, non-crossing)",
                    "n": n, "p": p, "n_taus": n_taus, "taus": tau_label,
                    "noise": noise,
                    "fit_time_s": ft, "predict_time_s": pt,
                    **metrics,
                })
                print(f"    PDLP sparse: {ft:7.2f}s  pinball={metrics['mean_pinball_loss']:.4f}  "
                      f"crossing={metrics['crossing_rate']:.4f}")
            except Exception as e:
                print(f"    PDLP sparse failed: {e}")

            # ── sklearn (independent per-quantile) ──
            if HAS_SKLEARN:
                try:
                    pred, ft, pt = fit_sklearn(X_train, y_train, X_test, taus)
                    metrics = evaluate(y_test, pred, taus)
                    results.append({
                        "model": "sklearn (independent)",
                        "n": n, "p": p, "n_taus": n_taus, "taus": tau_label,
                        "noise": noise,
                        "fit_time_s": ft, "predict_time_s": pt,
                        **metrics,
                    })
                    print(f"    sklearn:     {ft:7.2f}s  pinball={metrics['mean_pinball_loss']:.4f}  "
                          f"crossing={metrics['crossing_rate']:.4f}")
                except Exception as e:
                    print(f"    sklearn failed: {e}")

            # ── statsmodels (independent per-quantile) ──
            if HAS_STATSMODELS:
                try:
                    pred, ft, pt = fit_statsmodels(X_train, y_train, X_test, taus)
                    metrics = evaluate(y_test, pred, taus)
                    results.append({
                        "model": "statsmodels (independent)",
                        "n": n, "p": p, "n_taus": n_taus, "taus": tau_label,
                        "noise": noise,
                        "fit_time_s": ft, "predict_time_s": pt,
                        **metrics,
                    })
                    print(f"    statsmodels: {ft:7.2f}s  pinball={metrics['mean_pinball_loss']:.4f}  "
                          f"crossing={metrics['crossing_rate']:.4f}")
                except Exception as e:
                    print(f"    statsmodels failed: {e}")

    df = pd.DataFrame(results)
    return df


def add_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Add environment metadata columns."""
    import quantile_guard

    df = df.copy()
    df["python_version"] = platform.python_version()
    df["platform"] = platform.platform()
    df["package_version"] = getattr(quantile_guard, "__version__", "unknown")
    return df


def main():
    parser = argparse.ArgumentParser(description="Run linear baseline benchmarks")
    parser.add_argument(
        "--out",
        default="benchmarks/results/raw/linear_baselines.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Linear Baseline Benchmarks")
    print("=" * 60)
    print(f"  sklearn available: {HAS_SKLEARN}")
    print(f"  statsmodels available: {HAS_STATSMODELS}")
    print(f"  Configs: 2K-100K samples, 5-100 features")
    print(f"  Quantile sets: 7 and 13 quantiles")
    print(f"  Noise: heavy-tailed heteroscedastic (t(3))")

    df = run_benchmarks()
    df = add_metadata(df)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n{'=' * 60}")
    print(f"Results written to {out_path}")
    print(f"  {len(df)} rows")

    # Print summary table
    summary_cols = ["model", "n", "p", "n_taus", "fit_time_s",
                    "mean_pinball_loss", "crossing_rate", "empirical_coverage"]
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(df[summary_cols].to_string(index=False))

    return df


if __name__ == "__main__":
    main()
