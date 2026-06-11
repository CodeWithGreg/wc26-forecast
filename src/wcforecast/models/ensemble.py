"""Log-linear pooling of scoreline distributions.

The ensemble combines the Dixon-Coles and GBM forecasts at the level of the
full scoreline matrix::

    M_ens ∝ M_dc^w · M_gbm^(1-w)

Log-linear (geometric) pooling is the natural choice for combining
probabilistic forecasts that share an outcome space: it is externally
Bayesian, sharpens rather than flattens when the components agree, and a
single weight ``w`` is cheap to fit honestly out-of-sample. The weight is
selected by minimising the exact-score log-loss on held-out tournaments in
the backtest (never in-sample).

λ draws from both components are pooled by weighted resampling so that the
ensemble also carries a coherent epistemic-uncertainty estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wcforecast.models.poisson_core import MatchForecast


def pool_matrices(m_dc: np.ndarray, m_gbm: np.ndarray, w: float) -> np.ndarray:
    logm = w * np.log(np.clip(m_dc, 1e-12, None)) + (1 - w) * np.log(np.clip(m_gbm, 1e-12, None))
    m = np.exp(logm - logm.max())
    return m / m.sum()


def fit_pool_weight(
    matrices_dc: list[np.ndarray],
    matrices_gbm: list[np.ndarray],
    scores: list[tuple[int, int]],
    grid: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Choose w minimising exact-score log-loss on validation forecasts.

    Returns (w*, loss-curve over the grid) so the flatness of the optimum can
    be inspected — a flat curve means the choice barely matters, a sharp one
    means the components are complementary.
    """
    grid = np.linspace(0.0, 1.0, 21) if grid is None else grid
    losses = []
    for w in grid:
        ll = 0.0
        for md, mg, (xh, xa) in zip(matrices_dc, matrices_gbm, scores):
            m = pool_matrices(md, mg, w)
            xh_c = min(xh, m.shape[0] - 1)
            xa_c = min(xa, m.shape[1] - 1)
            ll -= np.log(max(m[xh_c, xa_c], 1e-12))
        losses.append(ll / max(len(scores), 1))
    losses = np.asarray(losses)
    return float(grid[int(np.argmin(losses))]), losses


@dataclass
class EnsembleModel:
    """Thin combinator over fitted component forecasts."""

    weight_dc: float = 0.5
    rng_seed: int = 2026

    def combine(self, f_dc: MatchForecast, f_gbm: MatchForecast) -> MatchForecast:
        m = pool_matrices(f_dc.matrix, f_gbm.matrix, self.weight_dc)
        rng = np.random.default_rng(self.rng_seed)

        def pool_draws(d1: np.ndarray, d2: np.ndarray, n: int = 400) -> np.ndarray:
            if d1 is None or d2 is None or len(d1) == 0 or len(d2) == 0:
                return d1 if d2 is None else d2
            n1 = int(round(self.weight_dc * n))
            s1 = rng.choice(d1, size=n1, replace=True)
            s2 = rng.choice(d2, size=n - n1, replace=True)
            return np.concatenate([s1, s2])

        lh = pool_draws(f_dc.lam_home_draws, f_gbm.lam_home_draws)
        la = pool_draws(f_dc.lam_away_draws, f_gbm.lam_away_draws)
        lam_h = self.weight_dc * f_dc.lam_home + (1 - self.weight_dc) * f_gbm.lam_home
        lam_a = self.weight_dc * f_dc.lam_away + (1 - self.weight_dc) * f_gbm.lam_away
        return MatchForecast(f_dc.home_team, f_dc.away_team, lam_h, lam_a, m, lh, la)
