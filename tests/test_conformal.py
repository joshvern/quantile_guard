"""Tests for conformalized quantile regression."""

import numpy as np
import pytest

from quantile_guard import QuantileRegression
from quantile_guard.conformal import ConformalQuantileRegression


@pytest.fixture
def synthetic_data():
    """Homoscedastic linear data."""
    rng = np.random.default_rng(42)
    n = 500
    X = rng.normal(size=(n, 3))
    y = X @ [2.0, -1.5, 0.8] + rng.normal(scale=1.0, size=n)
    return X, y


@pytest.fixture
def heteroscedastic_data():
    """Heteroscedastic data (harder for conformal)."""
    rng = np.random.default_rng(42)
    n = 600
    X = rng.normal(size=(n, 2))
    noise_scale = 0.5 + np.abs(X[:, 0])
    y = 2 * X[:, 0] - X[:, 1] + rng.normal(scale=noise_scale)
    return X, y


class TestConformalFit:

    def test_fit_basic(self, synthetic_data):
        X, y = synthetic_data
        base = QuantileRegression(tau=[0.05, 0.5, 0.95], se_method='analytical')
        cqr = ConformalQuantileRegression(base_estimator=base, coverage=0.90, random_state=0)
        cqr.fit(X, y)
        assert hasattr(cqr, 'offset_')
        assert hasattr(cqr, 'base_estimator_')
        assert 'y' in cqr.offset_

    def test_requires_multiple_quantiles(self, synthetic_data):
        X, y = synthetic_data
        base = QuantileRegression(tau=0.5, se_method='analytical')
        cqr = ConformalQuantileRegression(base_estimator=base)
        with pytest.raises(ValueError, match="at least 2 quantiles"):
            cqr.fit(X, y)

    def test_invalid_coverage(self, synthetic_data):
        X, y = synthetic_data
        base = QuantileRegression(tau=[0.1, 0.9], se_method='analytical')
        cqr = ConformalQuantileRegression(base_estimator=base, coverage=1.5)
        with pytest.raises(ValueError, match="coverage must be in"):
            cqr.fit(X, y)

    def test_invalid_calibration_size(self, synthetic_data):
        X, y = synthetic_data
        base = QuantileRegression(tau=[0.1, 0.9], se_method='analytical')
        cqr = ConformalQuantileRegression(base_estimator=base, calibration_size=0.0)
        with pytest.raises(ValueError, match="calibration_size must be in"):
            cqr.fit(X, y)


class TestConformalPredict:

    def test_predict_interval_shape(self, synthetic_data):
        X, y = synthetic_data
        base = QuantileRegression(tau=[0.05, 0.95], se_method='analytical')
        cqr = ConformalQuantileRegression(base_estimator=base, coverage=0.90, random_state=0)
        cqr.fit(X, y)

        intervals = cqr.predict_interval(X[:10])
        assert 'y' in intervals
        assert intervals['y']['lower'].shape == (10,)
        assert intervals['y']['upper'].shape == (10,)
        assert intervals['y']['width'].shape == (10,)

    def test_lower_leq_upper(self, synthetic_data):
        X, y = synthetic_data
        base = QuantileRegression(tau=[0.05, 0.95], se_method='analytical')
        cqr = ConformalQuantileRegression(base_estimator=base, coverage=0.90, random_state=0)
        cqr.fit(X, y)

        intervals = cqr.predict_interval(X)
        assert np.all(intervals['y']['lower'] <= intervals['y']['upper'])

    def test_width_positive(self, synthetic_data):
        X, y = synthetic_data
        base = QuantileRegression(tau=[0.05, 0.95], se_method='analytical')
        cqr = ConformalQuantileRegression(base_estimator=base, coverage=0.90, random_state=0)
        cqr.fit(X, y)

        intervals = cqr.predict_interval(X)
        assert np.all(intervals['y']['width'] > 0)

    def test_predict_before_fit_fails(self):
        base = QuantileRegression(tau=[0.1, 0.9], se_method='analytical')
        cqr = ConformalQuantileRegression(base_estimator=base)
        with pytest.raises(Exception):  # NotFittedError or similar
            cqr.predict_interval(np.zeros((5, 3)))


class TestConformalCoverage:

    def test_coverage_near_nominal_homoscedastic(self, synthetic_data):
        """On homoscedastic data, coverage should be >= nominal - epsilon."""
        X, y = synthetic_data
        # Use 80% of data for fit, hold out 20% for evaluation
        X_fit, X_test = X[:400], X[400:]
        y_fit, y_test = y[:400], y[400:]

        base = QuantileRegression(tau=[0.05, 0.5, 0.95], se_method='analytical')
        cqr = ConformalQuantileRegression(
            base_estimator=base, coverage=0.90, random_state=42
        )
        cqr.fit(X_fit, y_fit)

        cov = cqr.empirical_coverage(X_test, y_test)
        # With 100 test samples and conformal guarantee, coverage should be
        # close to nominal. Allow some slack for finite sample.
        assert cov['y'] >= 0.80, f"Coverage {cov['y']:.3f} too low"
        assert cov['y'] <= 1.0

    def test_coverage_near_nominal_heteroscedastic(self, heteroscedastic_data):
        """Conformal should maintain coverage even under heteroscedasticity."""
        X, y = heteroscedastic_data
        X_fit, X_test = X[:500], X[500:]
        y_fit, y_test = y[:500], y[500:]

        base = QuantileRegression(tau=[0.05, 0.5, 0.95], se_method='analytical')
        cqr = ConformalQuantileRegression(
            base_estimator=base, coverage=0.90, random_state=42
        )
        cqr.fit(X_fit, y_fit)

        cov = cqr.empirical_coverage(X_test, y_test)
        assert cov['y'] >= 0.75, f"Coverage {cov['y']:.3f} too low"

    def test_higher_coverage_wider_intervals(self, synthetic_data):
        """Higher target coverage should produce wider intervals."""
        X, y = synthetic_data

        widths = {}
        for target_cov in [0.80, 0.95]:
            base = QuantileRegression(tau=[0.025, 0.975], se_method='analytical')
            cqr = ConformalQuantileRegression(
                base_estimator=base, coverage=target_cov, random_state=42
            )
            cqr.fit(X, y)
            w = cqr.interval_width(X[:50])
            widths[target_cov] = w['y']

        assert widths[0.95] > widths[0.80]


class TestConformalReproducibility:

    def test_fixed_seed_reproducible(self, synthetic_data):
        X, y = synthetic_data
        base = QuantileRegression(tau=[0.05, 0.95], se_method='analytical')

        cqr1 = ConformalQuantileRegression(base_estimator=base, random_state=42)
        cqr1.fit(X, y)
        i1 = cqr1.predict_interval(X[:5])

        cqr2 = ConformalQuantileRegression(base_estimator=base, random_state=42)
        cqr2.fit(X, y)
        i2 = cqr2.predict_interval(X[:5])

        np.testing.assert_array_equal(i1['y']['lower'], i2['y']['lower'])
        np.testing.assert_array_equal(i1['y']['upper'], i2['y']['upper'])

    def test_different_seed_different_results(self, synthetic_data):
        X, y = synthetic_data
        base = QuantileRegression(tau=[0.05, 0.95], se_method='analytical')

        cqr1 = ConformalQuantileRegression(base_estimator=base, random_state=1)
        cqr1.fit(X, y)
        i1 = cqr1.predict_interval(X[:5])

        cqr2 = ConformalQuantileRegression(base_estimator=base, random_state=2)
        cqr2.fit(X, y)
        i2 = cqr2.predict_interval(X[:5])

        # Offsets should differ (different cal splits)
        assert not np.allclose(i1['y']['lower'], i2['y']['lower'])


class TestConformalIntervalWidth:

    def test_interval_width_method(self, synthetic_data):
        X, y = synthetic_data
        base = QuantileRegression(tau=[0.05, 0.95], se_method='analytical')
        cqr = ConformalQuantileRegression(base_estimator=base, random_state=0)
        cqr.fit(X, y)

        w = cqr.interval_width(X[:20])
        assert 'y' in w
        assert w['y'] > 0
