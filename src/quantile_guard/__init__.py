"""Public package exports."""

from ._version import __version__
from .quantile_regression import QuantileRegression, CensoredQuantileRegression

__all__ = ["__version__", "QuantileRegression", "CensoredQuantileRegression"]

# Submodules available via:
#   from quantile_guard.metrics import pinball_loss, ...
#   from quantile_guard.postprocess import rearrange_quantiles, ...
#   from quantile_guard.conformal import ConformalQuantileRegression
#   from quantile_guard.calibration import calibration_summary, ...
