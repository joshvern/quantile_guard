"""
Evaluate quantile predictions using the metrics module.

This example fits a multi-quantile model and evaluates its predictions
using pinball loss, coverage, crossing diagnostics, and interval score.
"""

import numpy as np
from quantile_guard import QuantileRegression
from quantile_guard.metrics import (
    pinball_loss,
    empirical_coverage,
    mean_interval_width,
    crossing_rate,
    interval_score,
    quantile_evaluation_report,
)

# Generate synthetic data
rng = np.random.default_rng(42)
n = 500
X = rng.normal(size=(n, 3))
y = X @ [2.0, -1.5, 0.8] + rng.normal(scale=0.5, size=n)

# Split train/test
X_train, X_test = X[:400], X[400:]
y_train, y_test = y[:400], y[400:]

# Fit multi-quantile model
taus = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
model = QuantileRegression(tau=taus, se_method='analytical')
model.fit(X_train, y_train)

# Get predictions as a matrix
pred_dict = model.predict(X_test)
predictions = np.column_stack([pred_dict[t]['y'] for t in taus])

# Individual metrics
print("=== Individual Metrics ===")
print(f"Median pinball loss: {pinball_loss(y_test, predictions[:, 3], 0.5):.4f}")
print(f"90% coverage: {empirical_coverage(y_test, predictions[:, 1], predictions[:, 5]):.3f}")
print(f"90% mean width: {mean_interval_width(predictions[:, 1], predictions[:, 5]):.3f}")
print(f"Crossing rate: {crossing_rate(predictions, taus):.3f}")
print(f"90% interval score: {interval_score(y_test, predictions[:, 1], predictions[:, 5], alpha=0.1):.3f}")

# Full report
print("\n=== Full Evaluation Report ===")
report = quantile_evaluation_report(y_test, predictions, taus)
for key, value in report.items():
    if key == "pinball_losses":
        print(f"  {key}:")
        for tau, loss in value.items():
            print(f"    tau={tau}: {loss:.4f}")
    else:
        print(f"  {key}: {value:.4f}")
