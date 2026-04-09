"""
Conformalized Quantile Regression (CQR).

Split conformal calibration that turns quantile predictions into intervals
with finite-sample coverage guarantees.

Reference: Romano, Patterson, Candès (2019).
    "Conformalized Quantile Regression."
    NeurIPS 2019.
"""

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_is_fitted


class ConformalQuantileRegression(BaseEstimator):
    """
    Conformalized Quantile Regression using split conformal calibration.

    Wraps a base quantile regression estimator and calibrates its prediction
    intervals to achieve finite-sample marginal coverage.

    Parameters
    ----------
    base_estimator : estimator
        A fitted or unfitted quantile regression estimator that supports
        ``fit(X, y)`` and ``predict(X)`` returning
        ``{tau: {output: array}}``. Must have at least 2 quantiles in
        ``tau`` attribute.

    coverage : float, default=0.90
        Target coverage level in (0, 1).

    calibration_size : float, default=0.25
        Fraction of training data held out for calibration. Must be in (0, 1).

    random_state : int or None, default=None
        Random seed for train/calibration split.

    Attributes
    ----------
    offset_ : dict
        ``{output: float}`` — calibrated conformity score quantile per output.

    lower_tau_ : float
        Lower quantile used from the base estimator.

    upper_tau_ : float
        Upper quantile used from the base estimator.

    base_estimator_ : estimator
        The fitted base estimator (clone of input).

    Examples
    --------
    >>> from quantile_guard import QuantileRegression
    >>> from quantile_guard.conformal import ConformalQuantileRegression
    >>> base = QuantileRegression(tau=[0.05, 0.5, 0.95], se_method='analytical')
    >>> cqr = ConformalQuantileRegression(base_estimator=base, coverage=0.90)
    >>> cqr.fit(X_train, y_train)
    >>> intervals = cqr.predict_interval(X_test)
    """

    def __init__(
        self,
        base_estimator,
        coverage: float = 0.90,
        calibration_size: float = 0.25,
        random_state: Optional[int] = None,
    ):
        self.base_estimator = base_estimator
        self.coverage = coverage
        self.calibration_size = calibration_size
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit the base estimator on training data and calibrate on held-out data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)

        Returns
        -------
        self
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if not 0 < self.coverage < 1:
            raise ValueError(
                f"coverage must be in (0, 1), got {self.coverage}"
            )
        if not 0 < self.calibration_size < 1:
            raise ValueError(
                f"calibration_size must be in (0, 1), got {self.calibration_size}"
            )

        base = self.base_estimator
        taus = getattr(base, 'tau', None)
        if taus is None:
            raise ValueError("base_estimator must have a 'tau' attribute")
        if isinstance(taus, (int, float)):
            taus = [taus]
        if len(taus) < 2:
            raise ValueError(
                "base_estimator must have at least 2 quantiles in tau, "
                f"got {len(taus)}"
            )

        sorted_taus = sorted(taus)
        self.lower_tau_ = sorted_taus[0]
        self.upper_tau_ = sorted_taus[-1]

        # Split into training and calibration sets
        n = X.shape[0]
        if n < 2:
            raise ValueError("fit requires at least 2 samples.")
        rng = np.random.RandomState(self.random_state)
        n_cal = max(1, int(n * self.calibration_size))
        if n_cal >= n:
            raise ValueError(
                "calibration_size leaves no samples for fitting; provide more data "
                "or use a smaller calibration_size."
            )
        indices = rng.permutation(n)
        cal_idx = indices[:n_cal]
        train_idx = indices[n_cal:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_cal, y_cal = X[cal_idx], y[cal_idx]

        # Fit base estimator on training split
        self.base_estimator_ = clone(base)
        self.base_estimator_.fit(X_train, y_train)

        # Compute nonconformity scores on calibration set
        pred_cal = self.base_estimator_.predict(X_cal)
        output_names = list(pred_cal[self.lower_tau_].keys())
        self.output_names_ = output_names

        if y_cal.ndim == 1:
            y_cal = y_cal.reshape(-1, 1)

        self.offset_ = {}
        alpha = 1.0 - self.coverage
        # Quantile level for conformal: ceil((1-alpha)(n_cal+1)) / n_cal
        q_level = min(
            np.ceil((1 - alpha) * (n_cal + 1)) / n_cal,
            1.0,
        )

        for ki, output in enumerate(output_names):
            lower_pred = pred_cal[self.lower_tau_][output]
            upper_pred = pred_cal[self.upper_tau_][output]
            y_col = y_cal[:, ki]

            # Nonconformity scores: max(lower - y, y - upper)
            scores = np.maximum(lower_pred - y_col, y_col - upper_pred)
            self.offset_[output] = float(self._conformal_quantile(scores, q_level))

        return self

    @staticmethod
    def _conformal_quantile(scores, q_level):
        """Use the conservative order statistic required for split conformal coverage."""
        try:
            return np.quantile(scores, q_level, method='higher')
        except TypeError:
            return np.quantile(scores, q_level, interpolation='higher')

    def predict_interval(self, X):
        """
        Predict calibrated intervals.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        dict
            ``{output: {'lower': array, 'upper': array, 'width': array}}``.
        """
        check_is_fitted(self, ['base_estimator_', 'offset_'])
        X = np.asarray(X)
        pred = self.base_estimator_.predict(X)

        result = {}
        for output in self.output_names_:
            lower = pred[self.lower_tau_][output] - self.offset_[output]
            upper = pred[self.upper_tau_][output] + self.offset_[output]
            result[output] = {
                'lower': lower,
                'upper': upper,
                'width': upper - lower,
            }
        return result

    def empirical_coverage(self, X, y):
        """
        Compute empirical coverage on a dataset.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)

        Returns
        -------
        dict
            ``{output: float}`` — coverage per output.
        """
        check_is_fitted(self, ['base_estimator_', 'offset_'])
        intervals = self.predict_interval(X)
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        result = {}
        for ki, output in enumerate(self.output_names_):
            lower = intervals[output]['lower']
            upper = intervals[output]['upper']
            covered = (y[:, ki] >= lower) & (y[:, ki] <= upper)
            result[output] = float(np.mean(covered))
        return result

    def interval_width(self, X):
        """
        Compute mean interval width.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        dict
            ``{output: float}`` — mean width per output.
        """
        check_is_fitted(self, ['base_estimator_', 'offset_'])
        intervals = self.predict_interval(X)
        return {
            output: float(np.mean(intervals[output]['width']))
            for output in self.output_names_
        }
