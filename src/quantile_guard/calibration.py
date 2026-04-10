"""
Calibration diagnostics for quantile predictions and prediction intervals.

Functions for assessing whether prediction intervals achieve their nominal
coverage, diagnosing coverage patterns across subgroups and feature ranges,
and summarizing sharpness (interval width) characteristics.

All functions accept numpy arrays and work with predictions from any model.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


def _validate_arrays(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and coerce interval arrays."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    lower = np.asarray(lower, dtype=float).ravel()
    upper = np.asarray(upper, dtype=float).ravel()
    n = len(y_true)
    if len(lower) != n or len(upper) != n:
        raise ValueError(
            f"y_true, lower, and upper must have the same length, "
            f"got {n}, {len(lower)}, {len(upper)}"
        )
    return y_true, lower, upper


def _validate_taus(taus: List[float]) -> List[float]:
    """Validate a list of unique quantile values."""
    taus = [float(t) for t in taus]
    if len(taus) < 2:
        raise ValueError(f"taus must have at least 2 elements, got {len(taus)}")
    if len(set(taus)) != len(taus):
        raise ValueError("taus must be unique")
    for tau in taus:
        if not 0 < tau < 1:
            raise ValueError(f"Each tau must be in (0, 1), got {tau}")
    return taus


def _align_predictions_with_taus(
    predictions: np.ndarray,
    taus: List[float],
) -> Tuple[np.ndarray, List[float]]:
    """Return predictions reordered to match ascending quantiles."""
    taus = _validate_taus(taus)
    predictions = np.asarray(predictions, dtype=float)
    if predictions.ndim != 2 or predictions.shape[1] != len(taus):
        raise ValueError(
            f"predictions must have shape (n_samples, {len(taus)}), "
            f"got {predictions.shape}"
        )
    order = np.argsort(taus)
    taus_sorted = [taus[i] for i in order]
    if np.array_equal(order, np.arange(len(taus))):
        return predictions, taus_sorted
    return predictions[:, order], taus_sorted


def coverage_by_group(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    groups: np.ndarray,
) -> Dict[str, dict]:
    """
    Compute coverage and interval width statistics per group.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target values.
    lower : array-like of shape (n_samples,)
        Lower bounds of prediction intervals.
    upper : array-like of shape (n_samples,)
        Upper bounds of prediction intervals.
    groups : array-like of shape (n_samples,)
        Group labels for each sample.

    Returns
    -------
    dict
        ``{group_label: {'coverage': float, 'mean_width': float, 'n': int}}``.

    Examples
    --------
    >>> coverage_by_group([1, 2, 3, 4], [0, 1, 2, 3], [2, 3, 4, 5], ['a', 'a', 'b', 'b'])
    {'a': {'coverage': 1.0, 'mean_width': 2.0, 'n': 2}, 'b': {'coverage': 1.0, 'mean_width': 2.0, 'n': 2}}
    """
    y_true, lower, upper = _validate_arrays(y_true, lower, upper)
    groups = np.asarray(groups)
    if len(groups) != len(y_true):
        raise ValueError("groups must have the same length as y_true")

    covered = (y_true >= lower) & (y_true <= upper)
    widths = upper - lower
    result = {}

    for g in np.unique(groups):
        mask = groups == g
        result[str(g)] = {
            "coverage": float(np.mean(covered[mask])),
            "mean_width": float(np.mean(widths[mask])),
            "n": int(np.sum(mask)),
        }
    return result


def coverage_by_bin(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    feature: np.ndarray,
    n_bins: int = 5,
) -> List[dict]:
    """
    Compute coverage and width statistics in equal-frequency bins of a feature.

    Useful for diagnosing whether coverage degrades in certain regions of the
    feature space (conditional coverage).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target values.
    lower : array-like of shape (n_samples,)
        Lower bounds of prediction intervals.
    upper : array-like of shape (n_samples,)
        Upper bounds of prediction intervals.
    feature : array-like of shape (n_samples,)
        Feature values to bin on.
    n_bins : int, default=5
        Number of equal-frequency bins.

    Returns
    -------
    list of dict
        Each dict has: ``'bin_index'``, ``'bin_lower'``, ``'bin_upper'``,
        ``'coverage'``, ``'mean_width'``, ``'n'``.
    """
    y_true, lower, upper = _validate_arrays(y_true, lower, upper)
    feature = np.asarray(feature, dtype=float).ravel()
    if len(feature) != len(y_true):
        raise ValueError("feature must have the same length as y_true")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    covered = (y_true >= lower) & (y_true <= upper)
    widths = upper - lower

    # Equal-frequency bins
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(feature, quantiles)
    # Make edges unique
    edges = np.unique(edges)
    actual_bins = len(edges) - 1

    results = []
    for i in range(actual_bins):
        if i < actual_bins - 1:
            mask = (feature >= edges[i]) & (feature < edges[i + 1])
        else:
            mask = (feature >= edges[i]) & (feature <= edges[i + 1])

        if np.sum(mask) == 0:
            continue

        results.append({
            "bin_index": i,
            "bin_lower": float(edges[i]),
            "bin_upper": float(edges[i + 1]),
            "coverage": float(np.mean(covered[mask])),
            "mean_width": float(np.mean(widths[mask])),
            "n": int(np.sum(mask)),
        })

    return results


def nominal_vs_empirical_coverage(
    y_true: np.ndarray,
    predictions: np.ndarray,
    taus: List[float],
) -> List[dict]:
    """
    Compare nominal vs empirical coverage for symmetric quantile pairs.

    For each pair of quantiles (tau_low, tau_high) that are symmetric around the
    median, computes the nominal and actual coverage.

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
    list of dict
        Each dict has: ``'tau_lower'``, ``'tau_upper'``,
        ``'nominal_coverage'``, ``'empirical_coverage'``,
        ``'coverage_gap'``.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    predictions, taus = _align_predictions_with_taus(predictions, taus)

    results = []
    n_taus = len(taus)
    for i in range(n_taus // 2):
        j = n_taus - 1 - i
        if i == j:
            continue
        tau_lo = taus[i]
        tau_hi = taus[j]
        nominal = tau_hi - tau_lo
        lower = predictions[:, i]
        upper = predictions[:, j]
        covered = (y_true >= lower) & (y_true <= upper)
        empirical = float(np.mean(covered))

        results.append({
            "tau_lower": tau_lo,
            "tau_upper": tau_hi,
            "nominal_coverage": nominal,
            "empirical_coverage": empirical,
            "coverage_gap": empirical - nominal,
        })

    return results


def sharpness_summary(
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict:
    """
    Summary statistics of interval widths (sharpness).

    Narrower intervals are "sharper" — better, as long as coverage is maintained.

    Parameters
    ----------
    lower : array-like of shape (n_samples,)
        Lower bounds.
    upper : array-like of shape (n_samples,)
        Upper bounds.

    Returns
    -------
    dict
        Keys: ``'mean_width'``, ``'median_width'``, ``'std_width'``,
        ``'min_width'``, ``'max_width'``, ``'iqr_width'``,
        ``'width_percentiles'`` (dict of 10th, 25th, 75th, 90th).
    """
    lower = np.asarray(lower, dtype=float).ravel()
    upper = np.asarray(upper, dtype=float).ravel()
    if len(lower) != len(upper):
        raise ValueError("lower and upper must have the same length")

    widths = upper - lower
    pcts = np.percentile(widths, [10, 25, 75, 90])

    return {
        "mean_width": float(np.mean(widths)),
        "median_width": float(np.median(widths)),
        "std_width": float(np.std(widths)),
        "min_width": float(np.min(widths)),
        "max_width": float(np.max(widths)),
        "iqr_width": float(pcts[2] - pcts[1]),
        "width_percentiles": {
            "p10": float(pcts[0]),
            "p25": float(pcts[1]),
            "p75": float(pcts[2]),
            "p90": float(pcts[3]),
        },
    }


def calibration_summary(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    nominal_coverage: float,
    groups: Optional[np.ndarray] = None,
    feature: Optional[np.ndarray] = None,
    n_bins: int = 5,
) -> dict:
    """
    Comprehensive calibration diagnostics report.

    Combines marginal coverage, sharpness, and optional conditional coverage
    breakdowns into a single summary.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
    lower : array-like of shape (n_samples,)
    upper : array-like of shape (n_samples,)
    nominal_coverage : float
        Target coverage level in (0, 1).
    groups : array-like of shape (n_samples,), optional
        Group labels for subgroup coverage analysis.
    feature : array-like of shape (n_samples,), optional
        Feature values for binned coverage analysis.
    n_bins : int, default=5
        Number of bins for feature-based coverage analysis.

    Returns
    -------
    dict
        Keys: ``'nominal_coverage'``, ``'empirical_coverage'``,
        ``'coverage_gap'``, ``'sharpness'``, and optionally
        ``'coverage_by_group'``, ``'coverage_by_feature_bin'``.
    """
    y_true, lower, upper = _validate_arrays(y_true, lower, upper)
    if not 0 < nominal_coverage < 1:
        raise ValueError(
            f"nominal_coverage must be in (0, 1), got {nominal_coverage}"
        )

    covered = (y_true >= lower) & (y_true <= upper)
    emp_cov = float(np.mean(covered))

    result = {
        "nominal_coverage": nominal_coverage,
        "empirical_coverage": emp_cov,
        "coverage_gap": emp_cov - nominal_coverage,
        "sharpness": sharpness_summary(lower, upper),
    }

    if groups is not None:
        result["coverage_by_group"] = coverage_by_group(
            y_true, lower, upper, groups
        )

    if feature is not None:
        result["coverage_by_feature_bin"] = coverage_by_bin(
            y_true, lower, upper, feature, n_bins=n_bins
        )

    return result
