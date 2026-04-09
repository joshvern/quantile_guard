"""
Conformalized quantile regression for calibrated prediction intervals.

Compares raw quantile intervals with conformally calibrated intervals,
showing how calibration changes held-out coverage and width.
"""

import numpy as np
from quantile_guard import QuantileRegression
from quantile_guard.conformal import ConformalQuantileRegression
from quantile_guard.metrics import empirical_coverage, mean_interval_width

# Generate heteroscedastic data
rng = np.random.default_rng(42)
n = 800
X = rng.normal(size=(n, 3))
noise_scale = 0.5 + np.abs(X[:, 0])
y = X @ [2.0, -1.5, 0.8] + rng.normal(scale=noise_scale)

X_train, X_test = X[:600], X[600:]
y_train, y_test = y[:600], y[600:]

target_coverage = 0.90
taus = [0.05, 0.5, 0.95]

# --- Raw quantile intervals ---
raw_model = QuantileRegression(tau=taus, se_method='analytical')
raw_model.fit(X_train, y_train)
raw_pred = raw_model.predict(X_test)

raw_lower = raw_pred[0.05]['y']
raw_upper = raw_pred[0.95]['y']
raw_cov = empirical_coverage(y_test, raw_lower, raw_upper)
raw_width = mean_interval_width(raw_lower, raw_upper)

print("=== Raw Quantile Intervals ===")
print(f"  Coverage: {raw_cov:.3f} (target: {target_coverage:.2f})")
print(f"  Mean width: {raw_width:.3f}")

# --- Conformalized intervals ---
base = QuantileRegression(tau=taus, se_method='analytical')
cqr = ConformalQuantileRegression(
    base_estimator=base,
    coverage=target_coverage,
    calibration_size=0.25,
    random_state=42,
)
cqr.fit(X_train, y_train)

cqr_intervals = cqr.predict_interval(X_test)
cqr_cov = cqr.empirical_coverage(X_test, y_test)
cqr_width = cqr.interval_width(X_test)

print("\n=== Conformalized Intervals ===")
print(f"  Coverage: {cqr_cov['y']:.3f} (target: {target_coverage:.2f})")
print(f"  Mean width: {cqr_width['y']:.3f}")
print(f"  Calibration offset: {cqr.offset_['y']:.3f}")

# --- Compare ---
print("\n=== Comparison ===")
cov_diff = cqr_cov['y'] - raw_cov
width_diff = cqr_width['y'] - raw_width
print(f"  Coverage change: {cov_diff:+.3f}")
print(f"  Width change: {width_diff:+.3f}")
