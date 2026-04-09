"""Tests for the calibration diagnostics module."""

import numpy as np
import pytest

from quantile_guard.calibration import (
    calibration_summary,
    coverage_by_bin,
    coverage_by_group,
    nominal_vs_empirical_coverage,
    sharpness_summary,
)


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def perfect_intervals():
    """Intervals that cover all observations."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    lower = y - 1.0
    upper = y + 1.0
    return y, lower, upper


@pytest.fixture
def partial_intervals():
    """Intervals that miss some observations."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    lower = y - 0.5
    upper = y + 0.5
    # Make some miss: shift a few out of bounds
    lower[0] = 2.0   # y=1 < lower=2 → not covered
    upper[9] = 9.0    # y=10 > upper=9 → not covered
    return y, lower, upper


# ── coverage_by_group ──────────────────────────────────────────────────

class TestCoverageByGroup:
    def test_all_covered(self, perfect_intervals):
        y, lower, upper = perfect_intervals
        groups = np.array(["a", "a", "b", "b", "b"])
        result = coverage_by_group(y, lower, upper, groups)
        assert result["a"]["coverage"] == 1.0
        assert result["b"]["coverage"] == 1.0
        assert result["a"]["n"] == 2
        assert result["b"]["n"] == 3

    def test_partial_coverage(self, partial_intervals):
        y, lower, upper = partial_intervals
        groups = np.array(["a"] * 5 + ["b"] * 5)
        result = coverage_by_group(y, lower, upper, groups)
        # Group a: sample 0 not covered → 4/5
        assert result["a"]["coverage"] == 0.8
        # Group b: sample 9 not covered → 4/5
        assert result["b"]["coverage"] == 0.8

    def test_width_is_positive(self, perfect_intervals):
        y, lower, upper = perfect_intervals
        groups = np.array(["x"] * 5)
        result = coverage_by_group(y, lower, upper, groups)
        assert result["x"]["mean_width"] == 2.0

    def test_length_mismatch_raises(self, perfect_intervals):
        y, lower, upper = perfect_intervals
        with pytest.raises(ValueError, match="same length"):
            coverage_by_group(y, lower, upper, np.array(["a", "b"]))


# ── coverage_by_bin ────────────────────────────────────────────────────

class TestCoverageByBin:
    def test_basic_bins(self, perfect_intervals):
        y, lower, upper = perfect_intervals
        feature = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = coverage_by_bin(y, lower, upper, feature, n_bins=2)
        assert len(result) >= 1
        for bin_info in result:
            assert "coverage" in bin_info
            assert "mean_width" in bin_info
            assert "n" in bin_info
            assert bin_info["coverage"] == 1.0

    def test_coverage_varies_by_bin(self):
        """Intervals calibrated in one region but not another."""
        rng = np.random.default_rng(42)
        n = 200
        feature = rng.uniform(0, 10, n)
        y = feature + rng.normal(scale=0.5, size=n)
        # Narrow intervals that work only for small feature values
        lower = feature - 1.0
        upper = feature + 1.0
        # Add heteroscedastic noise: high feature → wider true spread
        y += feature * rng.normal(scale=0.3, size=n)

        result = coverage_by_bin(y, lower, upper, feature, n_bins=4)
        coverages = [b["coverage"] for b in result]
        # First bin should have higher coverage than last (due to heteroscedasticity)
        assert coverages[0] > coverages[-1] or len(result) < 4

    def test_n_bins_1(self, perfect_intervals):
        y, lower, upper = perfect_intervals
        feature = np.arange(5, dtype=float)
        result = coverage_by_bin(y, lower, upper, feature, n_bins=1)
        assert len(result) == 1
        assert result[0]["coverage"] == 1.0

    def test_invalid_n_bins(self, perfect_intervals):
        y, lower, upper = perfect_intervals
        with pytest.raises(ValueError, match="n_bins"):
            coverage_by_bin(y, lower, upper, np.arange(5.0), n_bins=0)


# ── nominal_vs_empirical_coverage ──────────────────────────────────────

class TestNominalVsEmpirical:
    def test_symmetric_taus(self):
        rng = np.random.default_rng(0)
        n = 1000
        y = rng.normal(size=n)
        taus = [0.1, 0.25, 0.5, 0.75, 0.9]
        # Perfect quantile predictions from the true distribution
        from scipy.stats import norm
        predictions = np.column_stack([
            np.full(n, norm.ppf(tau)) for tau in taus
        ])
        result = nominal_vs_empirical_coverage(y, predictions, taus)
        assert len(result) == 2  # (0.1, 0.9) and (0.25, 0.75)
        for entry in result:
            assert "nominal_coverage" in entry
            assert "empirical_coverage" in entry
            assert "coverage_gap" in entry
            # Coverage should be reasonably close to nominal
            assert abs(entry["coverage_gap"]) < 0.05

    def test_three_taus(self):
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        predictions = np.array([
            [-1, 0, 1],
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
        ], dtype=float)
        taus = [0.1, 0.5, 0.9]
        result = nominal_vs_empirical_coverage(y, predictions, taus)
        assert len(result) == 1
        assert result[0]["tau_lower"] == 0.1
        assert result[0]["tau_upper"] == 0.9
        assert result[0]["nominal_coverage"] == pytest.approx(0.8)

    def test_shape_mismatch(self):
        with pytest.raises(ValueError):
            nominal_vs_empirical_coverage(
                np.zeros(5), np.zeros((5, 2)), [0.1, 0.5, 0.9]
            )


# ── sharpness_summary ──────────────────────────────────────────────────

class TestSharpnessSummary:
    def test_constant_width(self):
        lower = np.zeros(100)
        upper = np.ones(100) * 2
        result = sharpness_summary(lower, upper)
        assert result["mean_width"] == 2.0
        assert result["median_width"] == 2.0
        assert result["std_width"] == pytest.approx(0.0)
        assert result["min_width"] == 2.0
        assert result["max_width"] == 2.0
        assert result["iqr_width"] == pytest.approx(0.0)

    def test_variable_width(self):
        lower = np.zeros(100)
        upper = np.arange(100, dtype=float)
        result = sharpness_summary(lower, upper)
        assert result["mean_width"] == pytest.approx(49.5)
        assert result["min_width"] == 0.0
        assert result["max_width"] == 99.0
        assert "width_percentiles" in result
        assert "p10" in result["width_percentiles"]
        assert "p90" in result["width_percentiles"]

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            sharpness_summary(np.zeros(5), np.ones(3))


# ── calibration_summary ───────────────────────────────────────────────

class TestCalibrationSummary:
    def test_basic(self, perfect_intervals):
        y, lower, upper = perfect_intervals
        result = calibration_summary(y, lower, upper, nominal_coverage=0.9)
        assert result["nominal_coverage"] == 0.9
        assert result["empirical_coverage"] == 1.0
        assert result["coverage_gap"] == pytest.approx(0.1)
        assert "sharpness" in result
        assert result["sharpness"]["mean_width"] == 2.0

    def test_with_groups(self, perfect_intervals):
        y, lower, upper = perfect_intervals
        groups = np.array(["a", "a", "b", "b", "b"])
        result = calibration_summary(
            y, lower, upper, nominal_coverage=0.9, groups=groups
        )
        assert "coverage_by_group" in result
        assert "a" in result["coverage_by_group"]

    def test_with_feature(self, perfect_intervals):
        y, lower, upper = perfect_intervals
        feature = np.arange(5, dtype=float)
        result = calibration_summary(
            y, lower, upper, nominal_coverage=0.9,
            feature=feature, n_bins=2,
        )
        assert "coverage_by_feature_bin" in result
        assert len(result["coverage_by_feature_bin"]) >= 1

    def test_with_groups_and_feature(self, perfect_intervals):
        y, lower, upper = perfect_intervals
        groups = np.array(["a", "a", "b", "b", "b"])
        feature = np.arange(5, dtype=float)
        result = calibration_summary(
            y, lower, upper, nominal_coverage=0.9,
            groups=groups, feature=feature,
        )
        assert "coverage_by_group" in result
        assert "coverage_by_feature_bin" in result
