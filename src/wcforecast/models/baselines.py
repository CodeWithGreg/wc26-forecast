"""Reference baselines every serious model must beat.

* :class:`UniformBaseline` — the metric floor (1/3 on each 1X2 outcome).
* :class:`EloPoissonBaseline` — a 3-parameter Poisson GLM driven purely by
  the Elo difference. This is the "honest simple model": λ = exp(b0 +
  b1·Δelo/400 + b2·home). Most public World Cup predictors are some dressed
  up version of this (BENCHMARK.md §4), so it is the bar that justifies any
  added complexity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from wcforecast.models.poisson_core import MatchForecast, score_matrix


@dataclass
class UniformBaseline:
    """Uniform 1X2 probabilities and an average-λ scoreline matrix."""

    lam: float = 1.30  # WC-level average goals per team (BENCHMARK.md §7.1)

    def predict_score_matrix(self, *_args, **_kw) -> np.ndarray:
        return score_matrix(self.lam, self.lam, 0.0, 10)

    @property
    def p_outcomes(self) -> tuple[float, float, float]:
        return (1 / 3, 1 / 3, 1 / 3)


@dataclass
class EloPoissonBaseline:
    b0_: float = 0.25
    b1_: float = 0.6
    b2_: float = 0.25

    def fit(self, training_table: pd.DataFrame) -> "EloPoissonBaseline":
        d = training_table.dropna(subset=["home_score", "away_score"])
        de = (d["elo_home"].to_numpy() - d["elo_away"].to_numpy()) / 400.0
        home = (~d["neutral"].to_numpy()).astype(float)
        xh = d["home_score"].to_numpy(float)
        xa = d["away_score"].to_numpy(float)

        def nll(b):
            lh = np.exp(b[0] + b[1] * de + b[2] * home)
            la = np.exp(b[0] - b[1] * de)
            return -(xh * np.log(lh) - lh + xa * np.log(la) - la).sum()

        res = minimize(nll, np.array([0.25, 0.6, 0.25]), method="Nelder-Mead")
        self.b0_, self.b1_, self.b2_ = res.x
        return self

    def lambdas(self, elo_home: float, elo_away: float, neutral: bool) -> tuple[float, float]:
        de = (elo_home - elo_away) / 400.0
        home = 0.0 if neutral else 1.0
        return float(np.exp(self.b0_ + self.b1_ * de + self.b2_ * home)), float(np.exp(self.b0_ - self.b1_ * de))

    def predict_match(self, home: str, away: str, elo_home: float, elo_away: float, neutral: bool = True) -> MatchForecast:
        lh, la = self.lambdas(elo_home, elo_away, neutral)
        return MatchForecast(home, away, lh, la, score_matrix(lh, la, 0.0, 10))
