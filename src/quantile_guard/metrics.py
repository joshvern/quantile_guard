"""
Quantile regression evaluation metrics.

Standalone functions for evaluating quantile predictions from any model.
All functions accept numpy arrays and return scalars or dicts.
"""

from typing import Dict, List, Optional, Union

import numpy as np


def _validate_1d(arr: np.ndarray, name: str) -> np.ndarray:
    """Ensure array is 1D float."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {arr.shape}")
    return arr


def _validate_predictions_matrix(
    predictions: np.ndarray, taus: List[float]
) -> np.ndarray:
    """Validate predictions matrix shape against taus."""
    predictions = np.asarray(predictions, dtype=float)
    if predictions.ndim != 2:
        raise ValueError(
            f"predictions must be 2D (n_samples, n_quantiles), got shape {predictions.shape}"
        )
    if predictions.shape[1] != len(taus):
        raise ValueError(
            f"predictions has {predictions.shape[1]} columns but {len(taus)} "
            f"quantiles were given"
        )
    return predictions


def _validate_tau(tau: float) -> None:
    """Validate a single quantile value."""
    if not 0 < tau < 1:
        raise ValueError(f"tau must be in (0, 1), got {tau}")


def _validate_taus(taus: List[float]) -> List[float]:
    """Validate a list of unique quantile values."""
    taus = [float(t) for t in taus]
    if len(taus) < 2:
        raise ValueError(f"taus must have at least 2 elements, got {len(taus)}")
    for t in taus:
        _validate_tau(t)
    if len(set(taus)) != len(taus):
        raise ValueError("taus must be unique")
    return taus


def _align_predictions_with_taus(
    predictions: np.ndarray,
    taus: List[float],
) -> tuple:
    """Return predictions reordered to match ascending quantiles."""
    taus = _validate_taus(taus)
    predictions = _validate_predictions_matrix(predictions, taus)
    order = np.argsort(taus)
    taus_sorted = [taus[i] for i in order]
    if np.array_equal(order, np.arange(len(taus))):
        return predictions, taus_sorted
    return predictions[:, order], taus_sorted


def _pinball_losses_aligned(
    y_true: np.ndarray,
    predictions: np.ndarray,
    taus: List[float],
) -> Dict[float, float]:
    """Vectorized pinball loss for already-aligned predictions."""
    taus_arr = np.asarray(taus)
    residual = y_true[:, None] - predictions
    losses = np.where(
        residual >= 0,
        taus_arr * residual,
        (taus_arr - 1) * residual,
    )
    mean_losses = np.mean(losses, axis=0)
    return {tau: float(mean_losses[j]) for j, tau in enumerate(taus)}


def pinball_loss(
    y_true: np.ndarray, y_pred: np.ndarray, tau: float
) -> float:
    """
    Mean pinball (quantile) loss.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target values.
    y_pred : array-like of shape (n_samples,)
        Predicted quantile values.
    tau : float
        Quantile level in (0, 1).

    Returns
    -------
    float
        Mean pinball loss. Lower is better.

    Examples
    --------
    >>> pinball_loss([1, 2, 3], [1.5, 2.0, 2.5], 0.5)
    0.25
    """
    _validate_tau(tau)
    y_true = _validate_1d(np.asarray(y_true), "y_true")
    y_pred = _validate_1d(np.asarray(y_pred), "y_pred")
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have same length, got {len(y_true)} and {len(y_pred)}"
        )
    residual = y_true - y_pred
    loss = np.where(residual >= 0, tau * residual, (tau - 1) * residual)
    return float(np.mean(loss))


def multi_quantile_pinball_loss(
    y_true: np.ndarray,
    predictions: np.ndarray,
    taus: List[float],
) -> Dict[float, float]:
    """
    Per-quantile mean pinball loss.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target values.
    predictions : array-like of shape (n_samples, n_quantiles)
        Predicted quantile values. Column j corresponds to taus[j].
    taus : list of float
        Quantile levels in (0, 1). Column j of ``predictions`` corresponds to
        ``taus[j]``.

    Returns
    -------
    dict
        ``{tau: mean_pinball_loss}`` for each quantile.

    Examples
    --------
    >>> losses = multi_quantile_pinball_loss(
    ...     [1, 2, 3], [[0.5, 1, 1.5], [1.5, 2, 2.5], [2.5, 3, 3.5]], [0.1, 0.5, 0.9]
    ... )
    """
    y_true = _validate_1d(np.asarray(y_true), "y_true")
    predictions, taus = _align_predictions_with_taus(
        np.asarray(predictions),
        taus,
    )
    if len(y_true) != predictions.shape[0]:
        raise ValueError("y_true and predictions must have same number of rows")
    return _pinball_losses_aligned(y_true, predictions, taus)


def empirical_coverage(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> float:
    """
    Fraction of true values falling within [lower, upper].

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
    lower : array-like of shape (n_samples,)
    upper : array-like of shape (n_samples,)

    Returns
    -------
    float
        Coverage in [0, 1].
    """
    y_true = _validate_1d(np.asarray(y_true), "y_true")
    lower = _validate_1d(np.asarray(lower), "lower")
    upper = _validate_1d(np.asarray(upper), "upper")
    n = len(y_true)
    if len(lower) != n or len(upper) != n:
        raise ValueError("y_true, lower, and upper must have the same length")
    covered = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(covered))


def mean_interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """
    Mean width of prediction intervals.

    Parameters
    ----------
    lower : array-like of shape (n_samples,)
    upper : array-like of shape (n_samples,)

    Returns
    -------
    float
        Mean interval width. Non-negative when lower <= upper.
    """
    lower = _validate_1d(np.asarray(lower), "lower")
    upper = _validate_1d(np.asarray(upper), "upper")
    if len(lower) != len(upper):
        raise ValueError("lower and upper must have the same length")
    return float(np.mean(upper - lower))


def crossing_rate(predictions: np.ndarray, taus: List[float]) -> float:
    """
    Fraction of samples where quantile predictions violate monotonicity.

    A sample has a crossing if any ``predictions[i, j] > predictions[i, j+1]``
    for consecutive quantiles.

    Parameters
    ----------
    predictions : array-like of shape (n_samples, n_quantiles)
        Column j corresponds to ``taus[j]``; columns are aligned internally
        before checking crossings.
    taus : list of float
        Quantile levels in (0, 1).

    Returns
    -------
    float
        Fraction of samples with at least one crossing, in [0, 1].
    """
    predictions, taus = _align_predictions_with_taus(
        np.asarray(predictions),
        taus,
    )
    diffs = np.diff(predictions, axis=1)
    has_crossing = np.any(diffs < 0, axis=1)
    return float(np.mean(has_crossing))


def crossing_magnitude(predictions: np.ndarray, taus: List[float]) -> float:
    """
    Mean magnitude of quantile crossings across all samples.

    For each sample, sums the negative differences between consecutive quantile
    predictions. Returns the average across samples.

    Parameters
    ----------
    predictions : array-like of shape (n_samples, n_quantiles)
    taus : list of float

    Returns
    -------
    float
        Mean total crossing magnitude. Zero if no crossings.
    """
    predictions, taus = _align_predictions_with_taus(
        np.asarray(predictions),
        taus,
    )
    diffs = np.diff(predictions, axis=1)
    violations = np.clip(diffs, None, 0)  # keep only negative diffs
    return float(np.mean(np.sum(-violations, axis=1)))


def interval_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> float:
    """
    Mean interval score (Gneiting & Raftery, 2007).

    Combines interval width with penalties for under- and over-coverage.

    ``IS = (upper - lower) + (2/alpha) * (lower - y) * 1[y < lower]
                            + (2/alpha) * (y - upper) * 1[y > upper]``

    Lower is better. A well-calibrated (1-alpha) interval minimizes this score.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
    lower : array-like of shape (n_samples,)
    upper : array-like of shape (n_samples,)
    alpha : float
        Miscoverage rate in (0, 1). E.g., 0.1 for 90% intervals.

    Returns
    -------
    float
        Mean interval score.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    y_true = _validate_1d(np.asarray(y_true), "y_true")
    lower = _validate_1d(np.asarray(lower), "lower")
    upper = _validate_1d(np.asarray(upper), "upper")
    n = len(y_true)
    if len(lower) != n or len(upper) != n:
        raise ValueError("y_true, lower, and upper must have the same length")

    width = upper - lower
    below = np.clip(lower - y_true, 0, None)
    above = np.clip(y_true - upper, 0, None)
    score = width + (2 / alpha) * below + (2 / alpha) * above
    return float(np.mean(score))


def quantile_evaluation_report(
    y_true: np.ndarray,
    predictions: np.ndarray,
    taus: List[float],
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
    alpha: Optional[float] = None,
) -> dict:
    """
    Summary report of quantile prediction quality.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target values.
    predictions : array-like of shape (n_samples, n_quantiles)
        Predicted quantile values. Column j corresponds to taus[j].
    taus : list of float
        Quantile levels in (0, 1).
    lower : array-like of shape (n_samples,), optional
        Lower bound of prediction interval. If None, uses first column of
        predictions.
    upper : array-like of shape (n_samples,), optional
        Upper bound of prediction interval. If None, uses last column of
        predictions.
    alpha : float, optional
        Miscoverage rate for interval score. If None, inferred as
        ``1 - (taus[-1] - taus[0])``.

    Returns
    -------
    dict
        Keys: ``'pinball_losses'``, ``'mean_pinball_loss'``,
        ``'crossing_rate'``, ``'crossing_magnitude'``,
        ``'coverage'``, ``'mean_width'``, ``'median_width'``,
        ``'interval_score'``.
    """
    y_true = _validate_1d(np.asarray(y_true), "y_true")
    predictions, taus = _align_predictions_with_taus(
        np.asarray(predictions),
        taus,
    )

    losses = _pinball_losses_aligned(y_true, predictions, taus)
    diffs = np.diff(predictions, axis=1)
    violations = np.clip(diffs, None, 0)
    cr = float(np.mean(np.any(diffs < 0, axis=1)))
    cm = float(np.mean(np.sum(-violations, axis=1)))

    if lower is None:
        lower = predictions[:, 0]
    else:
        lower = _validate_1d(np.asarray(lower), "lower")
    if upper is None:
        upper = predictions[:, -1]
    else:
        upper = _validate_1d(np.asarray(upper), "upper")

    cov = empirical_coverage(y_true, lower, upper)
    widths = upper - lower
    mw = float(np.mean(widths))
    medw = float(np.median(widths))

    if alpha is None:
        alpha = 1.0 - (taus[-1] - taus[0])
        alpha = max(min(alpha, 0.99), 0.01)  # clamp to valid range

    iscore = interval_score(y_true, lower, upper, alpha)

    return {
        "pinball_losses": losses,
        "mean_pinball_loss": float(np.mean(list(losses.values()))),
        "crossing_rate": cr,
        "crossing_magnitude": cm,
        "coverage": cov,
        "mean_width": mw,
        "median_width": medw,
        "interval_score": iscore,
    }
