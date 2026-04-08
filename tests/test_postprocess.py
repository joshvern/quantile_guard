"""Tests for crossing detection and rearrangement utilities."""

import numpy as np
import pytest

from quantile_regression_pdlp.postprocess import (
    check_crossing,
    crossing_summary,
    rearrange_quantiles,
)


TAUS = [0.1, 0.5, 0.9]


class TestCheckCrossing:

    def test_no_crossing(self):
        preds = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        result = check_crossing(preds, TAUS)
        assert result.shape == (2,)
        assert not np.any(result)

    def test_all_crossing(self):
        preds = np.array([[3, 2, 1], [6, 5, 4]], dtype=float)
        result = check_crossing(preds, TAUS)
        assert np.all(result)

    def test_partial_crossing(self):
        preds = np.array([
            [1, 2, 3],   # ok
            [3, 2, 1],   # crossing
            [1, 1, 1],   # ok (equal is not crossing)
        ], dtype=float)
        result = check_crossing(preds, TAUS)
        np.testing.assert_array_equal(result, [False, True, False])

    def test_returns_boolean(self):
        preds = np.array([[1, 2, 3]], dtype=float)
        assert check_crossing(preds, TAUS).dtype == bool

    def test_invalid_taus_order(self):
        with pytest.raises(ValueError, match="ascending order"):
            check_crossing([[1, 2]], [0.9, 0.1])

    def test_single_tau_rejected(self):
        with pytest.raises(ValueError, match="at least 2"):
            check_crossing([[1]], [0.5])


class TestCrossingSummary:

    def test_no_crossings(self):
        preds = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        s = crossing_summary(preds, TAUS)
        assert s["crossing_rate"] == pytest.approx(0.0)
        assert s["crossing_magnitude"] == pytest.approx(0.0)
        assert s["max_magnitude"] == pytest.approx(0.0)
        assert s["n_crossing_samples"] == 0
        assert s["n_total_samples"] == 2
        assert s["worst_row_index"] is None

    def test_all_crossings(self):
        # row 0: [3,1,2] -> diffs [-2,1] -> violations: 2
        # row 1: [5,1,3] -> diffs [-4,2] -> violations: 4
        preds = np.array([[3, 1, 2], [5, 1, 3]], dtype=float)
        s = crossing_summary(preds, TAUS)
        assert s["crossing_rate"] == pytest.approx(1.0)
        assert s["crossing_magnitude"] == pytest.approx(3.0)  # mean(2,4)
        assert s["max_magnitude"] == pytest.approx(4.0)
        assert s["n_crossing_samples"] == 2
        assert s["worst_row_index"] == 1

    def test_summary_keys(self):
        preds = np.array([[1, 2, 3]], dtype=float)
        s = crossing_summary(preds, TAUS)
        expected_keys = {
            "crossing_rate", "crossing_magnitude", "max_magnitude",
            "n_crossing_samples", "n_total_samples", "worst_row_index",
        }
        assert set(s.keys()) == expected_keys


class TestRearrangeQuantiles:

    def test_already_sorted(self):
        preds = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        result = rearrange_quantiles(preds, TAUS)
        np.testing.assert_array_equal(result, preds)

    def test_sorts_each_row(self):
        preds = np.array([[3, 1, 2]], dtype=float)
        result = rearrange_quantiles(preds, TAUS)
        np.testing.assert_array_equal(result, [[1, 2, 3]])

    def test_preserves_shape(self):
        preds = np.random.default_rng(0).normal(size=(50, 5))
        taus = [0.1, 0.25, 0.5, 0.75, 0.9]
        result = rearrange_quantiles(preds, taus)
        assert result.shape == preds.shape

    def test_eliminates_crossings(self):
        rng = np.random.default_rng(42)
        preds = rng.normal(size=(100, 3))  # random, likely has crossings
        result = rearrange_quantiles(preds, TAUS)
        diffs = np.diff(result, axis=1)
        assert np.all(diffs >= 0), "Rearranged predictions should be monotone"

    def test_crossing_rate_drops_to_zero(self):
        from quantile_regression_pdlp.metrics import crossing_rate

        rng = np.random.default_rng(42)
        preds = rng.normal(size=(100, 3))
        assert crossing_rate(preds, TAUS) > 0, "Test data should have crossings"

        result = rearrange_quantiles(preds, TAUS)
        assert crossing_rate(result, TAUS) == pytest.approx(0.0)

    def test_equal_values_preserved(self):
        preds = np.array([[2, 2, 2]], dtype=float)
        result = rearrange_quantiles(preds, TAUS)
        np.testing.assert_array_equal(result, preds)

    def test_integration_with_model(self):
        """Rearrangement works on QuantileRegression output (already monotone)."""
        from quantile_regression_pdlp import QuantileRegression

        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 2))
        y = X @ [1, -1] + rng.normal(scale=0.3, size=100)

        model = QuantileRegression(tau=TAUS, se_method='analytical')
        model.fit(X, y)
        pred_dict = model.predict(X)
        preds = np.column_stack([pred_dict[t]['y'] for t in TAUS])

        result = rearrange_quantiles(preds, TAUS)
        np.testing.assert_array_almost_equal(result, preds)
