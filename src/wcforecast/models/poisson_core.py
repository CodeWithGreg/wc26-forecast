"""Scoreline-distribution primitives shared by every model.

A forecast is represented by a (G+1)×(G+1) matrix ``M`` with
``M[i, j] = P(home scores i, away scores j)``. All downstream consumers
(1X2 probabilities, totals, exact-score picks, tournament simulation, MPP
optimisation) operate on this single representation, which keeps the model
zoo interchangeable.

The Dixon-Coles low-score adjustment ``tau`` corrects the well-documented
underestimation of 0-0/1-0/0-1/1-1 outcomes by independent Poisson models
(Dixon & Coles 1997). Empirical rho at international level is small and
negative (≈ -0.05 .. -0.15).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import poisson


def dc_tau_matrix(lam_h: float, lam_a: float, rho: float, max_goals: int) -> np.ndarray:
    """Multiplicative Dixon-Coles adjustment on the low-score cells."""
    tau = np.ones((max_goals + 1, max_goals + 1))
    tau[0, 0] = 1.0 - lam_h * lam_a * rho
    tau[0, 1] = 1.0 + lam_h * rho
    tau[1, 0] = 1.0 + lam_a * rho
    tau[1, 1] = 1.0 - rho
    return tau


def score_matrix(lam_h: float, lam_a: float, rho: float = 0.0, max_goals: int = 10) -> np.ndarray:
    """Joint scoreline pmf under (adjusted) independent Poisson margins."""
    gh = poisson.pmf(np.arange(max_goals + 1), lam_h)
    ga = poisson.pmf(np.arange(max_goals + 1), lam_a)
    m = np.outer(gh, ga)
    if rho != 0.0:
        m = m * dc_tau_matrix(lam_h, lam_a, rho, max_goals)
        m = np.clip(m, 1e-12, None)
    return m / m.sum()


def mixture_score_matrix(
    lam_h: np.ndarray, lam_a: np.ndarray, rho: float = 0.0, max_goals: int = 10
) -> np.ndarray:
    """Posterior-predictive scoreline pmf: average over (λ_h, λ_a) draws.

    This is where epistemic (parameter) uncertainty enters the predictive
    distribution — the mixture is wider than any single-λ Poisson.
    """
    mats = [score_matrix(h, a, rho, max_goals) for h, a in zip(lam_h, lam_a)]
    return np.mean(mats, axis=0)


def outcome_probs(m: np.ndarray) -> tuple[float, float, float]:
    """(P_home_win, P_draw, P_away_win) from a scoreline matrix."""
    return float(np.tril(m, -1).sum()), float(np.trace(m)), float(np.triu(m, 1).sum())


def totals_pmf(m: np.ndarray) -> np.ndarray:
    """Pmf of total goals (length 2·G+1) via anti-diagonal sums."""
    g = m.shape[0] - 1
    return np.array([np.fliplr(m).trace(offset=g - t) for t in range(2 * g + 1)])


def top_scorelines(m: np.ndarray, k: int = 5) -> list[tuple[int, int, float]]:
    """The ``k`` most likely exact scores as (home, away, prob)."""
    flat = [(i, j, float(m[i, j])) for i in range(m.shape[0]) for j in range(m.shape[1])]
    return sorted(flat, key=lambda t: -t[2])[:k]


def quantile_interval(pmf: np.ndarray, level: float = 0.9) -> tuple[int, int]:
    """Central credible interval on a discrete pmf (e.g. goals for a team)."""
    cdf = np.cumsum(pmf)
    lo = int(np.searchsorted(cdf, (1 - level) / 2))
    hi = int(np.searchsorted(cdf, 1 - (1 - level) / 2))
    return lo, hi


@dataclass
class MatchForecast:
    """Full probabilistic forecast for one match."""

    home_team: str
    away_team: str
    lam_home: float  # posterior-mean expected goals
    lam_away: float
    matrix: np.ndarray  # posterior-predictive scoreline pmf
    lam_home_draws: np.ndarray = field(default=None, repr=False)  # epistemic draws
    lam_away_draws: np.ndarray = field(default=None, repr=False)

    @property
    def p_outcomes(self) -> tuple[float, float, float]:
        return outcome_probs(self.matrix)

    @property
    def home_goals_pmf(self) -> np.ndarray:
        return self.matrix.sum(axis=1)

    @property
    def away_goals_pmf(self) -> np.ndarray:
        return self.matrix.sum(axis=0)

    def goals_interval(self, side: str = "home", level: float = 0.9) -> tuple[int, int]:
        pmf = self.home_goals_pmf if side == "home" else self.away_goals_pmf
        return quantile_interval(pmf, level)

    def lam_interval(self, side: str = "home", level: float = 0.9) -> tuple[float, float]:
        """Credible interval on expected goals λ (epistemic uncertainty)."""
        draws = self.lam_home_draws if side == "home" else self.lam_away_draws
        if draws is None or len(draws) == 0:
            lam = self.lam_home if side == "home" else self.lam_away
            return lam, lam
        q = np.quantile(draws, [(1 - level) / 2, 1 - (1 - level) / 2])
        return float(q[0]), float(q[1])

    def most_likely_score(self) -> tuple[int, int, float]:
        return top_scorelines(self.matrix, 1)[0]
