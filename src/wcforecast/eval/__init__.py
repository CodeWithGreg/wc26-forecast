"""Evaluation: proper scoring rules, calibration diagnostics, backtests."""

from wcforecast.eval.metrics import (  # noqa: F401
    brier_1x2,
    interval_coverage,
    log_loss_1x2,
    pit_values,
    poisson_deviance,
    rps_1x2,
)
from wcforecast.eval.backtest import run_backtest, TOURNAMENTS  # noqa: F401
