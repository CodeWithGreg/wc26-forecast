"""Proper scoring rules and calibration diagnostics.

Primary metric: multiclass **log-loss** (ignorance score) on 1X2, following
the simulation evidence that it discriminates forecast quality better than
RPS or Brier (Constantinou & Fenton 2012; arXiv:1908.08980 — BENCHMARK.md
§6.1). **RPS** is reported alongside for comparability with the academic
literature (Groll/Zeileis line), plus exact-score log-loss and Poisson
deviance for the goal-count head, and PIT/coverage for calibration of the
full predictive distribution.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def _onehot(outcome: int) -> np.ndarray:
    v = np.zeros(3)
    v[outcome] = 1.0
    return v


def outcome_index(home_goals: int, away_goals: int) -> int:
    """0 = home win, 1 = draw, 2 = away win."""
    return 0 if home_goals > away_goals else (1 if home_goals == away_goals else 2)


def log_loss_1x2(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean negative log-likelihood of the realised outcome.

    probs: (n, 3) array of (pH, pD, pA); outcomes: (n,) in {0,1,2}.
    """
    p = np.clip(probs[np.arange(len(outcomes)), outcomes], _EPS, None)
    return float(-np.log(p).mean())


def rps_1x2(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Ranked probability score for the ordered outcome (H, D, A)."""
    out = 0.0
    for p, o in zip(probs, outcomes):
        cp = np.cumsum(p)
        co = np.cumsum(_onehot(int(o)))
        out += np.sum((cp[:-1] - co[:-1]) ** 2) / 2.0
    return float(out / len(outcomes))


def brier_1x2(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Multiclass Brier score (mean squared error of the probability vector)."""
    oh = np.stack([_onehot(int(o)) for o in outcomes])
    return float(np.mean(np.sum((probs - oh) ** 2, axis=1)))


def exact_score_log_loss(matrices: list[np.ndarray], scores: list[tuple[int, int]]) -> float:
    """Mean NLL of the realised exact scoreline under the forecast matrix."""
    ll = []
    for m, (xh, xa) in zip(matrices, scores):
        xh_c, xa_c = min(xh, m.shape[0] - 1), min(xa, m.shape[1] - 1)
        ll.append(-np.log(max(m[xh_c, xa_c], _EPS)))
    return float(np.mean(ll))


def poisson_deviance(lam: np.ndarray, goals: np.ndarray) -> float:
    """Mean unit Poisson deviance of expected-goals forecasts."""
    lam = np.clip(lam, _EPS, None)
    g = goals.astype(float)
    g_safe = np.where(g > 0, g, 1.0)  # avoid log(0); the branch is unused when g == 0
    term = np.where(g > 0, g * np.log(g_safe / lam) - (g - lam), lam)
    return float(2.0 * term.mean())


def accuracy_1x2(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float((probs.argmax(axis=1) == outcomes).mean())


def pit_values(pmfs: list[np.ndarray], observed: list[int], rng: np.random.Generator | None = None) -> np.ndarray:
    """Randomised PIT for discrete forecasts (Czado, Gneiting & Held 2009).

    For each forecast pmf and observed count x, draws u ~ U[F(x-1), F(x)].
    If the predictive distribution is calibrated, PIT values are U[0,1].
    """
    rng = rng or np.random.default_rng(7)
    out = np.empty(len(observed))
    for i, (pmf, x) in enumerate(zip(pmfs, observed)):
        cdf = np.cumsum(pmf)
        x = min(int(x), len(pmf) - 1)
        lo = cdf[x - 1] if x > 0 else 0.0
        out[i] = rng.uniform(lo, cdf[x])
    return out


def interval_coverage(
    pmfs: list[np.ndarray], observed: list[int], level: float = 0.9
) -> float:
    """Empirical coverage of central credible intervals on a discrete pmf."""
    from wcforecast.models.poisson_core import quantile_interval

    hit = 0
    for pmf, x in zip(pmfs, observed):
        lo, hi = quantile_interval(pmf, level)
        hit += int(lo <= x <= hi)
    return hit / len(observed)
