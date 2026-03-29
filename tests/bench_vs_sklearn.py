"""
Benchmark / development tests comparing quantile_regression_pdlp against
sklearn's QuantileRegressor.

These are NOT part of the core test suite. Run them explicitly:
    python -m pytest tests/bench_vs_sklearn.py -v

They validate that our solver produces results consistent with sklearn's
implementation across various scenarios (unregularized, L1, weighted,
extreme quantiles, pinball loss).
"""

import numpy as np
import pytest
from sklearn.linear_model import QuantileRegressor

from quantile_regression_pdlp import QuantileRegression


class TestAgainstSklearn:
    """Compare PDLP quantile regression against sklearn's QuantileRegressor."""

    @pytest.fixture
    def linear_data(self):
        rng = np.random.default_rng(42)
        n = 300
        X = rng.normal(size=(n, 3))
        true_beta = np.array([2.0, -1.5, 0.8])
        true_intercept = 3.0
        y = true_intercept + X @ true_beta + rng.normal(scale=0.5, size=n)
        return X, y, true_beta, true_intercept

    def _fit_sklearn(self, X, y, quantile, alpha=0.0, sample_weight=None):
        """Fit sklearn QuantileRegressor for a single quantile."""
        skl = QuantileRegressor(quantile=quantile, alpha=alpha, solver='highs')
        skl.fit(X, y, sample_weight=sample_weight)
        return skl

    def test_median_coefficients_match(self, linear_data):
        """PDLP and sklearn should produce similar coefficients at tau=0.5."""
        X, y, _, _ = linear_data

        pdlp = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0)
        pdlp.fit(X, y)

        skl = self._fit_sklearn(X, y, quantile=0.5)

        np.testing.assert_allclose(
            pdlp.coef_[0.5]['y'], skl.coef_, atol=0.1,
            err_msg="Slope coefficients should match sklearn"
        )
        np.testing.assert_allclose(
            pdlp.intercept_[0.5]['y'], skl.intercept_, atol=0.1,
            err_msg="Intercept should match sklearn"
        )

    def test_multiple_quantiles_match(self, linear_data):
        """PDLP and sklearn should agree across multiple quantiles."""
        X, y, _, _ = linear_data
        taus = [0.1, 0.25, 0.5, 0.75, 0.9]

        pdlp = QuantileRegression(tau=taus, n_bootstrap=20, random_state=0)
        pdlp.fit(X, y)

        for q in taus:
            skl = self._fit_sklearn(X, y, quantile=q)
            np.testing.assert_allclose(
                pdlp.coef_[q]['y'], skl.coef_, atol=0.15,
                err_msg=f"Coefficients should match at tau={q}"
            )
            np.testing.assert_allclose(
                pdlp.intercept_[q]['y'], skl.intercept_, atol=0.15,
                err_msg=f"Intercept should match at tau={q}"
            )

    def test_predictions_match(self, linear_data):
        """Predictions should be close between PDLP and sklearn."""
        X, y, _, _ = linear_data
        X_test = X[:20]

        pdlp = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0)
        pdlp.fit(X, y)
        pred_pdlp = pdlp.predict(X_test)[0.5]['y']

        skl = self._fit_sklearn(X, y, quantile=0.5)
        pred_skl = skl.predict(X_test)

        np.testing.assert_allclose(pred_pdlp, pred_skl, atol=0.2,
                                   err_msg="Predictions should match sklearn")

    def test_l1_regularized_coefficients_match(self, linear_data):
        """L1-regularized coefficients should match sklearn.

        Both use the same convention: obj = (1/n)*pinball + alpha*L1.
        So the same alpha value should produce the same result.
        """
        X, y, _, _ = linear_data
        alpha = 0.5

        pdlp = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0,
                                   regularization='l1', alpha=alpha)
        pdlp.fit(X, y)

        skl = self._fit_sklearn(X, y, quantile=0.5, alpha=alpha)

        np.testing.assert_allclose(
            pdlp.coef_[0.5]['y'], skl.coef_, atol=0.2,
            err_msg="L1-regularized coefficients should match sklearn"
        )

    def test_weighted_regression_matches(self, linear_data):
        """Weighted quantile regression should match sklearn with sample_weight."""
        X, y, _, _ = linear_data
        rng = np.random.default_rng(99)
        weights = rng.uniform(0.5, 3.0, size=len(y))

        pdlp = QuantileRegression(tau=0.5, n_bootstrap=20, random_state=0)
        pdlp.fit(X, y, weights=weights)

        skl = self._fit_sklearn(X, y, quantile=0.5, sample_weight=weights)

        np.testing.assert_allclose(
            pdlp.coef_[0.5]['y'], skl.coef_, atol=0.15,
            err_msg="Weighted coefficients should match sklearn"
        )

    def test_extreme_quantiles_match(self, linear_data):
        """Extreme quantiles (0.05, 0.95) should still match sklearn."""
        X, y, _, _ = linear_data

        for q in [0.05, 0.95]:
            pdlp = QuantileRegression(tau=q, n_bootstrap=20, random_state=0)
            pdlp.fit(X, y)

            skl = self._fit_sklearn(X, y, quantile=q)

            np.testing.assert_allclose(
                pdlp.coef_[q]['y'], skl.coef_, atol=0.2,
                err_msg=f"Coefficients should match at extreme tau={q}"
            )

    def test_pinball_loss_comparable(self, linear_data):
        """Both models should achieve similar pinball loss on held-out data."""
        X, y, _, _ = linear_data
        X_train, X_test = X[:250], X[250:]
        y_train, y_test = y[:250], y[250:]

        q = 0.5
        pdlp = QuantileRegression(tau=q, n_bootstrap=20, random_state=0)
        pdlp.fit(X_train, y_train)
        pred_pdlp = pdlp.predict(X_test)[q]['y']

        skl = self._fit_sklearn(X_train, y_train, quantile=q)
        pred_skl = skl.predict(X_test)

        def pinball_loss(y_true, y_pred, tau):
            residual = y_true - y_pred
            return np.mean(np.where(residual >= 0, tau * residual, (tau - 1) * residual))

        loss_pdlp = pinball_loss(y_test, pred_pdlp, q)
        loss_skl = pinball_loss(y_test, pred_skl, q)

        assert abs(loss_pdlp - loss_skl) / max(loss_skl, 1e-10) < 0.2, \
            f"Pinball losses too different: PDLP={loss_pdlp:.4f} vs sklearn={loss_skl:.4f}"
