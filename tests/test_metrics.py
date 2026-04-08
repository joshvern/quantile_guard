"""Tests for quantile evaluation metrics."""

import numpy as np
import pytest

from quantile_regression_pdlp.metrics import (
    pinball_loss,
    multi_quantile_pinball_loss,
    empirical_coverage,
    mean_interval_width,
    crossing_rate,
    crossing_magnitude,
    interval_score,
    quantile_evaluation_report,
)


class TestPinballLoss:

    def test_median_pinball_hand_checked(self):
        """At tau=0.5, pinball loss = 0.5 * mean(|y - y_pred|)."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.0, 2.5])
        # residuals: -0.5, 0.0, 0.5
        # losses: 0.5*0.5=0.25, 0, 0.5*0.5=0.25 -> mean = 0.5/3
        expected = (0.25 + 0.0 + 0.25) / 3
        assert pinball_loss(y_true, y_pred, 0.5) == pytest.approx(expected)

    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert pinball_loss(y, y, 0.5) == pytest.approx(0.0)

    def test_asymmetric_loss(self):
        """tau=0.9 penalizes underprediction more than overprediction."""
        y_true = np.array([10.0])
        y_pred_under = np.array([8.0])  # residual = 2, loss = 0.9*2 = 1.8
        y_pred_over = np.array([12.0])  # residual = -2, loss = 0.1*2 = 0.2
        assert pinball_loss(y_true, y_pred_under, 0.9) == pytest.approx(1.8)
        assert pinball_loss(y_true, y_pred_over, 0.9) == pytest.approx(0.2)

    def test_invalid_tau(self):
        with pytest.raises(ValueError, match="tau must be in"):
            pinball_loss([1], [1], 0.0)
        with pytest.raises(ValueError, match="tau must be in"):
            pinball_loss([1], [1], 1.0)
        with pytest.raises(ValueError, match="tau must be in"):
            pinball_loss([1], [1], -0.5)

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            pinball_loss([1, 2], [1], 0.5)

    def test_2d_input_rejected(self):
        with pytest.raises(ValueError, match="must be 1D"):
            pinball_loss([[1, 2]], [1], 0.5)


class TestMultiQuantilePinballLoss:

    def test_matches_individual_calls(self):
        rng = np.random.default_rng(42)
        y = rng.normal(size=50)
        taus = [0.1, 0.5, 0.9]
        preds = np.column_stack([y - 1, y, y + 1])
        result = multi_quantile_pinball_loss(y, preds, taus)
        for j, tau in enumerate(taus):
            assert result[tau] == pytest.approx(pinball_loss(y, preds[:, j], tau))

    def test_wrong_column_count(self):
        with pytest.raises(ValueError, match="columns"):
            multi_quantile_pinball_loss([1, 2], [[1, 2], [3, 4]], [0.1, 0.5, 0.9])

    def test_single_tau_rejected(self):
        with pytest.raises(ValueError, match="at least 2"):
            multi_quantile_pinball_loss([1], [[1]], [0.5])


class TestEmpiricalCoverage:

    def test_all_covered(self):
        y = np.array([1.0, 2.0, 3.0])
        lower = np.array([0.0, 1.0, 2.0])
        upper = np.array([2.0, 3.0, 4.0])
        assert empirical_coverage(y, lower, upper) == pytest.approx(1.0)

    def test_none_covered(self):
        y = np.array([10.0, 20.0])
        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])
        assert empirical_coverage(y, lower, upper) == pytest.approx(0.0)

    def test_partial_coverage(self):
        y = np.array([1.0, 5.0, 3.0, 10.0])
        lower = np.array([0.0, 0.0, 0.0, 0.0])
        upper = np.array([2.0, 4.0, 4.0, 4.0])
        # covered: 1, 3 (indices 0, 2). not: 5, 10
        assert empirical_coverage(y, lower, upper) == pytest.approx(0.5)

    def test_boundary_included(self):
        """Points exactly on lower or upper boundary are covered."""
        y = np.array([1.0, 3.0])
        lower = np.array([1.0, 2.0])
        upper = np.array([2.0, 3.0])
        assert empirical_coverage(y, lower, upper) == pytest.approx(1.0)


class TestMeanIntervalWidth:

    def test_basic(self):
        lower = np.array([0.0, 1.0, 2.0])
        upper = np.array([1.0, 3.0, 5.0])
        # widths: 1, 2, 3 -> mean = 2
        assert mean_interval_width(lower, upper) == pytest.approx(2.0)

    def test_zero_width(self):
        v = np.array([1.0, 2.0])
        assert mean_interval_width(v, v) == pytest.approx(0.0)

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            mean_interval_width([1, 2], [1])


class TestCrossingRate:

    def test_no_crossing(self):
        """Monotonically increasing columns have zero crossing."""
        preds = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        assert crossing_rate(preds, [0.1, 0.5, 0.9]) == pytest.approx(0.0)

    def test_all_crossing(self):
        """Reversed columns cross on every row."""
        preds = np.array([[3, 2, 1], [6, 5, 4]], dtype=float)
        assert crossing_rate(preds, [0.1, 0.5, 0.9]) == pytest.approx(1.0)

    def test_partial_crossing(self):
        preds = np.array([
            [1, 2, 3],   # ok
            [3, 2, 1],   # crossing
        ], dtype=float)
        assert crossing_rate(preds, [0.1, 0.5, 0.9]) == pytest.approx(0.5)

    def test_equal_values_no_crossing(self):
        """Equal adjacent predictions are not crossings."""
        preds = np.array([[1, 1, 1]], dtype=float)
        assert crossing_rate(preds, [0.1, 0.5, 0.9]) == pytest.approx(0.0)


class TestCrossingMagnitude:

    def test_no_crossing_zero_magnitude(self):
        preds = np.array([[1, 2, 3]], dtype=float)
        assert crossing_magnitude(preds, [0.1, 0.5, 0.9]) == pytest.approx(0.0)

    def test_known_magnitude(self):
        # row: [3, 1, 2] -> diffs: [-2, 1] -> violations: 2 -> sum per row: 2
        preds = np.array([[3, 1, 2]], dtype=float)
        assert crossing_magnitude(preds, [0.1, 0.5, 0.9]) == pytest.approx(2.0)

    def test_multiple_rows(self):
        preds = np.array([
            [1, 2, 3],   # violations: 0
            [3, 1, 2],   # violations: 2
        ], dtype=float)
        # mean of [0, 2] = 1.0
        assert crossing_magnitude(preds, [0.1, 0.5, 0.9]) == pytest.approx(1.0)


class TestIntervalScore:

    def test_perfect_narrow_interval(self):
        """Point inside narrow interval has low score."""
        y = np.array([5.0])
        lower = np.array([4.0])
        upper = np.array([6.0])
        # width=2, no penalty -> score=2
        assert interval_score(y, lower, upper, 0.1) == pytest.approx(2.0)

    def test_point_below_interval(self):
        """Point below interval incurs penalty."""
        y = np.array([0.0])
        lower = np.array([2.0])
        upper = np.array([4.0])
        # width=2, below penalty = (2/0.1)*(2-0)=40 -> total=42
        assert interval_score(y, lower, upper, 0.1) == pytest.approx(42.0)

    def test_point_above_interval(self):
        """Point above interval incurs penalty."""
        y = np.array([10.0])
        lower = np.array([2.0])
        upper = np.array([4.0])
        # width=2, above penalty = (2/0.1)*(10-4)=120 -> total=122
        assert interval_score(y, lower, upper, 0.1) == pytest.approx(122.0)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha must be in"):
            interval_score([1], [0], [2], 0.0)

    def test_wider_interval_higher_base_score(self):
        """All else equal, wider intervals have higher scores."""
        y = np.array([5.0])
        s_narrow = interval_score(y, np.array([4.0]), np.array([6.0]), 0.1)
        s_wide = interval_score(y, np.array([0.0]), np.array([10.0]), 0.1)
        assert s_wide > s_narrow


class TestQuantileEvaluationReport:

    def test_report_keys(self):
        rng = np.random.default_rng(42)
        y = rng.normal(size=100)
        taus = [0.1, 0.5, 0.9]
        preds = np.column_stack([y - 1, y, y + 1])
        report = quantile_evaluation_report(y, preds, taus)
        expected_keys = {
            "pinball_losses", "mean_pinball_loss",
            "crossing_rate", "crossing_magnitude",
            "coverage", "mean_width", "median_width",
            "interval_score",
        }
        assert set(report.keys()) == expected_keys

    def test_report_values_reasonable(self):
        rng = np.random.default_rng(0)
        y = rng.normal(size=200)
        taus = [0.1, 0.5, 0.9]
        preds = np.column_stack([
            np.quantile(y, 0.1) * np.ones(200),
            np.quantile(y, 0.5) * np.ones(200),
            np.quantile(y, 0.9) * np.ones(200),
        ])
        report = quantile_evaluation_report(y, preds, taus)
        assert report["crossing_rate"] == 0.0
        assert report["mean_pinball_loss"] > 0
        assert 0.6 < report["coverage"] < 1.0
        assert report["mean_width"] > 0

    def test_report_with_explicit_bounds(self):
        y = np.array([1.0, 2.0, 3.0])
        taus = [0.1, 0.5, 0.9]
        preds = np.array([[0.5, 1.0, 1.5], [1.5, 2.0, 2.5], [2.5, 3.0, 3.5]])
        lower = np.array([0.0, 1.0, 2.0])
        upper = np.array([2.0, 3.0, 4.0])
        report = quantile_evaluation_report(y, preds, taus, lower=lower, upper=upper)
        assert report["coverage"] == pytest.approx(1.0)

    def test_integration_with_model(self):
        """Metrics work with QuantileRegression predictions."""
        from quantile_regression_pdlp import QuantileRegression

        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 2))
        y = 2 * X[:, 0] - X[:, 1] + rng.normal(scale=0.5, size=100)

        taus = [0.1, 0.5, 0.9]
        model = QuantileRegression(tau=taus, se_method='analytical')
        model.fit(X, y)

        pred_dict = model.predict(X)
        preds = np.column_stack([pred_dict[t]['y'] for t in taus])
        report = quantile_evaluation_report(y, preds, taus)

        assert report["crossing_rate"] == pytest.approx(0.0)
        assert 0.7 < report["coverage"] < 1.0
        assert report["mean_pinball_loss"] > 0
