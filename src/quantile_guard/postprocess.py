"""
Crossing detection and rearrangement utilities for quantile predictions.

These functions work with predictions from any quantile model, not just
``QuantileRegression``. They are most useful when evaluating or fixing
predictions from models that don't enforce non-crossing constraints.
"""

from typing import Dict, List

import numpy as np


def _validate_predictions_taus(
    predictions: np.ndarray, taus: List[float]
) -> tuple:
    """Validate and return (predictions, sorted_taus)."""
    predictions = np.asarray(predictions, dtype=float)
    if predictions.ndim != 2:
        raise ValueError(
            f"predictions must be 2D (n_samples, n_quantiles), got shape {predictions.shape}"
        )
    taus = list(taus)
    if len(taus) < 2:
        raise ValueError(f"taus must have at least 2 elements, got {len(taus)}")
    if len(set(taus)) != len(taus):
        raise ValueError("taus must be unique")
    for t in taus:
        if not 0 < t < 1:
            raise ValueError(f"Each tau must be in (0, 1), got {t}")
    if predictions.shape[1] != len(taus):
        raise ValueError(
            f"predictions has {predictions.shape[1]} columns but {len(taus)} "
            f"quantiles were given"
        )
    sorted_taus = sorted(taus)
    if taus != sorted_taus:
        raise ValueError(
            "taus must be in ascending order. "
            f"Got {taus}, expected {sorted_taus}"
        )
    return predictions, taus


def check_crossing(predictions: np.ndarray, taus: List[float]) -> np.ndarray:
    """
    Identify samples where quantile predictions violate monotonicity.

    A sample has a crossing if any ``predictions[i, j] > predictions[i, j+1]``
    for consecutive quantiles.

    Parameters
    ----------
    predictions : array-like of shape (n_samples, n_quantiles)
        Predicted quantile values. Column j corresponds to taus[j].
    taus : list of float
        Quantile levels in (0, 1), in ascending order.

    Returns
    -------
    np.ndarray of bool, shape (n_samples,)
        True where at least one crossing occurs.

    Examples
    --------
    >>> check_crossing([[1, 2, 3], [3, 2, 1]], [0.1, 0.5, 0.9])
    array([False,  True])
    """
    predictions, taus = _validate_predictions_taus(predictions, taus)
    diffs = np.diff(predictions, axis=1)
    return np.any(diffs < 0, axis=1)


def crossing_summary(predictions: np.ndarray, taus: List[float]) -> dict:
    """
    Summary statistics about quantile crossings.

    Parameters
    ----------
    predictions : array-like of shape (n_samples, n_quantiles)
    taus : list of float

    Returns
    -------
    dict
        Keys:
        - ``'crossing_rate'``: fraction of samples with at least one crossing
        - ``'crossing_magnitude'``: mean total violation magnitude
        - ``'max_magnitude'``: worst single violation across all samples
        - ``'n_crossing_samples'``: count of samples with crossings
        - ``'n_total_samples'``: total number of samples
        - ``'worst_row_index'``: index of the row with largest total violation
          (None if no crossings)
    """
    predictions, taus = _validate_predictions_taus(predictions, taus)
    diffs = np.diff(predictions, axis=1)
    violations = np.clip(diffs, None, 0)
    row_violations = np.sum(-violations, axis=1)

    has_crossing = row_violations > 0
    n_crossing = int(np.sum(has_crossing))
    n_total = predictions.shape[0]

    if n_crossing > 0:
        worst_idx = int(np.argmax(row_violations))
        max_mag = float(np.max(-violations))
    else:
        worst_idx = None
        max_mag = 0.0

    return {
        "crossing_rate": float(n_crossing / n_total) if n_total > 0 else 0.0,
        "crossing_magnitude": float(np.mean(row_violations)),
        "max_magnitude": max_mag,
        "n_crossing_samples": n_crossing,
        "n_total_samples": n_total,
        "worst_row_index": worst_idx,
    }


def rearrange_quantiles(
    predictions: np.ndarray, taus: List[float]
) -> np.ndarray:
    """
    Enforce monotonicity by sorting each row of quantile predictions.

    For each sample, sorts the predicted quantile values so that they are
    non-decreasing across quantile levels. This is the simplest rearrangement
    strategy (Chernozhukov et al., 2010).

    Parameters
    ----------
    predictions : array-like of shape (n_samples, n_quantiles)
        Predicted quantile values. Column j corresponds to taus[j].
    taus : list of float
        Quantile levels in (0, 1), in ascending order.

    Returns
    -------
    np.ndarray of shape (n_samples, n_quantiles)
        Rearranged predictions with no crossings.

    Examples
    --------
    >>> rearrange_quantiles([[3, 1, 2]], [0.1, 0.5, 0.9])
    array([[1., 2., 3.]])
    """
    predictions, taus = _validate_predictions_taus(predictions, taus)
    return np.sort(predictions, axis=1)
