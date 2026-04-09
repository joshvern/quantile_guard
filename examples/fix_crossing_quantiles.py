"""
Detect and fix crossing quantile predictions.

This example simulates crossed predictions (as might come from independently
fitted models) and demonstrates the postprocess utilities.
"""

import numpy as np
from quantile_guard.postprocess import (
    check_crossing,
    crossing_summary,
    rearrange_quantiles,
)
from quantile_guard.metrics import crossing_rate

# Simulate predictions with intentional crossings
rng = np.random.default_rng(42)
n = 200
taus = [0.1, 0.25, 0.5, 0.75, 0.9]

# Start with reasonable predictions, then add noise to create crossings
base = np.linspace(-1.5, 1.5, len(taus))
predictions = base[None, :] + rng.normal(scale=0.8, size=(n, len(taus)))

# Diagnose crossings
print("=== Before Rearrangement ===")
mask = check_crossing(predictions, taus)
print(f"Samples with crossings: {mask.sum()} / {n}")

summary = crossing_summary(predictions, taus)
for key, val in summary.items():
    print(f"  {key}: {val}")

# Fix crossings
fixed = rearrange_quantiles(predictions, taus)

print("\n=== After Rearrangement ===")
print(f"Crossing rate: {crossing_rate(fixed, taus):.3f}")
print(f"Shape preserved: {fixed.shape == predictions.shape}")

# Verify monotonicity
diffs = np.diff(fixed, axis=1)
print(f"All monotone: {np.all(diffs >= 0)}")

# Compare a sample row
row = 0
print(f"\nRow {row} before: {predictions[row].round(3)}")
print(f"Row {row} after:  {fixed[row].round(3)}")
