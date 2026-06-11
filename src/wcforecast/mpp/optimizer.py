"""Expected-points optimisation for Mon Petit Prono (MPP).

MPP scoring (mpp.football/rules, WC 2026 edition):

* Predicting the correct **1X2 result** earns the points displayed next to
  the outcome, which are indexed on the odds (an upset pays more than a
  favourite win).
* Predicting the **exact score** additionally earns a rarity bonus that
  depends on how many other players picked that exact score *among those
  with the correct result*: >30% → 20, 20–30% → 30, 5–20% → 50,
  0.5–5% → 70, <0.5% → 100.
* One **X2 booster** doubles a single match's points.

The optimal pick is therefore *not* the most likely scoreline: it maximises

    E[points | pick s] = P(result(s)) · pts(result(s)) + P(s) · bonus(s)

where the bonus depends on a model of the *crowd's* pick distribution. The
crowd is modelled as a sharpened version of the forecast itself blended
with the empirical popularity of "human" scorelines (2-1, 1-0, 2-0 …):
``q(s) ∝ p(s)^β · pop(s)`` — β > 1 because crowds over-concentrate on
favourite outcomes. All parameters are explicit and overridable.

When the actual MPP odds for a match are known they should be passed in;
otherwise fair odds are derived from the model's own outcome probabilities
(odds = 1/p), i.e. the optimiser then assumes MPP's odds agree with the
model. Points are odds × 10, the MPP display convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wcforecast.models.poisson_core import MatchForecast, outcome_probs

#: empirical "human pick" popularity prior on scorelines (relative weights).
#: Crowd favourites cluster on small, round scores; anything ≥5 goals a side
#: is almost never picked.
_POPULARITY = {
    (1, 0): 1.00, (2, 0): 0.95, (2, 1): 1.00, (1, 1): 0.85, (0, 0): 0.40,
    (3, 0): 0.55, (3, 1): 0.60, (0, 1): 0.55, (0, 2): 0.45, (1, 2): 0.55,
    (2, 2): 0.40, (3, 2): 0.30, (4, 0): 0.25, (4, 1): 0.25, (1, 3): 0.25,
    (0, 3): 0.20,
}
_POP_DEFAULT = 0.08


@dataclass(frozen=True)
class MppRules:
    """Tunable encoding of the MPP scoring system."""

    odds_to_points: float = 10.0  # points = odds × 10
    bonus_tiers: tuple = ((0.30, 20.0), (0.20, 30.0), (0.05, 50.0), (0.005, 70.0), (0.0, 100.0))
    crowd_sharpening: float = 1.30  # β: crowd over-concentration on favourites
    max_pick_goals: int = 5  # never pick anything wilder than 5 goals a side
    margin: float = 0.0  # bookmaker-style margin if emulating MPP odds


def crowd_distribution(matrix: np.ndarray, rules: MppRules) -> np.ndarray:
    """Model of the crowd's exact-score pick distribution."""
    g = matrix.shape[0]
    pop = np.full((g, g), _POP_DEFAULT)
    for (i, j), v in _POPULARITY.items():
        if i < g and j < g:
            pop[i, j] = v
    q = np.power(np.clip(matrix, 1e-12, None), rules.crowd_sharpening) * pop
    return q / q.sum()


def _bonus(p_exact_given_result: float, rules: MppRules) -> float:
    for thresh, pts in rules.bonus_tiers:
        if p_exact_given_result > thresh:
            return pts
    return rules.bonus_tiers[-1][1]


def outcome_points(p_outcomes: tuple[float, float, float], rules: MppRules,
                   odds: tuple[float, float, float] | None = None) -> np.ndarray:
    """Points awarded for a correct 1X2 pick (per outcome H/D/A)."""
    if odds is None:
        p = np.clip(np.asarray(p_outcomes), 1e-3, None)
        odds_arr = (1.0 + rules.margin) / p
    else:
        odds_arr = np.asarray(odds, dtype=float)
    return np.round(odds_arr * rules.odds_to_points)


def optimal_pick(
    forecast: MatchForecast,
    rules: MppRules | None = None,
    odds: tuple[float, float, float] | None = None,
    top_k: int = 3,
) -> dict:
    """Expected-points-optimal scoreline pick for one match.

    Returns the best pick, its expected points and decomposition, the most
    likely score ("banker" reference) and the ``top_k`` alternatives.
    """
    rules = rules or MppRules()
    m = forecast.matrix
    p_out = outcome_probs(m)
    pts_out = outcome_points(p_out, rules, odds)
    crowd = crowd_distribution(m, rules)
    crowd_out = outcome_probs(crowd)

    g = min(rules.max_pick_goals, m.shape[0] - 1)
    cands = []
    for i in range(g + 1):
        for j in range(g + 1):
            o = 0 if i > j else (1 if i == j else 2)
            p_exact = float(m[i, j])
            q_exact_given_res = float(crowd[i, j]) / max(float(crowd_out[o]), 1e-9)
            bonus = _bonus(q_exact_given_res, rules)
            exp_pts = p_out[o] * pts_out[o] + p_exact * bonus
            cands.append({
                "score": f"{i}-{j}", "home_goals": i, "away_goals": j,
                "outcome": "HDA"[o],
                "p_outcome": p_out[o], "outcome_pts": float(pts_out[o]),
                "p_exact": p_exact, "bonus_if_exact": bonus,
                "expected_points": float(exp_pts),
            })
    cands.sort(key=lambda c: -c["expected_points"])
    ml = forecast.most_likely_score()
    return {
        "best": cands[0],
        "alternatives": cands[1:top_k],
        "most_likely_score": {"score": f"{ml[0]}-{ml[1]}", "p": ml[2]},
        "p_outcomes": p_out,
        "outcome_points": pts_out.tolist(),
    }


def picks_table(
    forecasts: list[MatchForecast],
    rules: MppRules | None = None,
    odds_by_match: dict[tuple[str, str], tuple[float, float, float]] | None = None,
) -> pd.DataFrame:
    """Optimal picks for a slate of matches + X2 booster recommendation.

    The X2 doubles the realised points of one match; under risk neutrality
    the booster goes on the pick with the highest expected points.
    """
    rules = rules or MppRules()
    rows = []
    for f in forecasts:
        odds = (odds_by_match or {}).get((f.home_team, f.away_team))
        r = optimal_pick(f, rules, odds)
        b = r["best"]
        rows.append({
            "home_team": f.home_team, "away_team": f.away_team,
            "pick": b["score"], "expected_points": round(b["expected_points"], 2),
            "pick_outcome": b["outcome"], "p_outcome": round(b["p_outcome"], 3),
            "outcome_pts": b["outcome_pts"],
            "p_exact": round(b["p_exact"], 4), "bonus_if_exact": b["bonus_if_exact"],
            "most_likely_score": r["most_likely_score"]["score"],
            "alt_pick": r["alternatives"][0]["score"] if r["alternatives"] else "",
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["x2_booster"] = ""
        out.loc[out["expected_points"].idxmax(), "x2_booster"] = "X2 ◀"
    return out
