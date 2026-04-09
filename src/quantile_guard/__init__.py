# quantile_guard/__init__.py

from .quantile_regression import QuantileRegression, CensoredQuantileRegression

__all__ = ['QuantileRegression', 'CensoredQuantileRegression']

# Submodules available via:
#   from quantile_guard.metrics import pinball_loss, ...
#   from quantile_guard.postprocess import rearrange_quantiles, ...
#   from quantile_guard.conformal import ConformalQuantileRegression
#   from quantile_guard.calibration import calibration_summary, ...
