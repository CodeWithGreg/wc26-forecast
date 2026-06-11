"""World-Football-Elo rating engine.

Implements the eloratings.net conventions, which the literature consistently
finds to be the strongest single predictor of international results (Ley et
al. 2019; Elo beats FIFA rank 73.9% vs 68.7% on WC knockout winners — see
BENCHMARK.md §5):

* importance-dependent K factor (60 World Cup … 20 friendlies),
* goal-difference multiplier (1 / 1.5 / 1.75 / 1.75 + (N-3)/8),
* home advantage as a rating offset for non-neutral venues.

The engine replays history chronologically once and stores every pre-match
rating, so downstream feature building is leak-free by construction: the
rating attached to a match never includes that match's result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from wcforecast.config import EloConfig, get_settings

_K_BY_BUCKET = {
    "world_cup": "k_world_cup",
    "continental": "k_continental_final",
    "qualifier": "k_qualifier",
    "nations": "k_nations_league",
    "friendly": "k_friendly",
    "other": "k_qualifier",
}


@dataclass
class EloEngine:
    """Replay-based Elo computer with as-of lookups."""

    config: EloConfig = field(default_factory=lambda: get_settings().elo)
    ratings: dict[str, float] = field(default_factory=dict)
    history: list[tuple] = field(default_factory=list)  # (date, team, pre_rating)

    def rating(self, team: str) -> float:
        return self.ratings.get(team, self.config.initial_rating)

    def _k(self, importance: str) -> float:
        return getattr(self.config, _K_BY_BUCKET.get(importance, "k_qualifier"))

    def _gd_multiplier(self, goal_diff: int) -> float:
        gd = abs(goal_diff)
        if gd <= 1:
            return self.config.gd_multipliers[0]
        if gd == 2:
            return self.config.gd_multipliers[1]
        return self.config.gd_multipliers[2] + self.config.gd_extra_step * (gd - 3 if gd > 3 else 0)

    def expected(self, r_home: float, r_away: float, neutral: bool) -> float:
        """Expected score (win=1, draw=0.5) for the home side."""
        adv = 0.0 if neutral else self.config.home_advantage
        return 1.0 / (1.0 + 10.0 ** (-((r_home + adv) - r_away) / 400.0))

    def update(self, home: str, away: str, hs: int, as_: int, neutral: bool, importance: str) -> None:
        rh, ra = self.rating(home), self.rating(away)
        exp_h = self.expected(rh, ra, neutral)
        res_h = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        delta = self._k(importance) * self._gd_multiplier(hs - as_) * (res_h - exp_h)
        self.ratings[home] = rh + delta
        self.ratings[away] = ra - delta

    def fit(self, results: pd.DataFrame) -> "EloEngine":
        """Replay ``results`` (sorted by date) and record pre-match ratings."""
        results = results.sort_values("date")
        pre_h = np.empty(len(results))
        pre_a = np.empty(len(results))
        for i, row in enumerate(results.itertuples(index=False)):
            pre_h[i] = self.rating(row.home_team)
            pre_a[i] = self.rating(row.away_team)
            self.update(
                row.home_team, row.away_team, int(row.home_score), int(row.away_score),
                bool(row.neutral), str(row.importance),
            )
        self._pre = results[["date", "home_team", "away_team"]].copy()
        self._pre["elo_home_pre"] = pre_h
        self._pre["elo_away_pre"] = pre_a
        return self

    @property
    def pre_match_ratings(self) -> pd.DataFrame:
        """Pre-match ratings aligned with the fitted results frame."""
        return self._pre

    def table(self, teams: list[str] | None = None) -> pd.DataFrame:
        """Current rating table (descending)."""
        items = self.ratings.items()
        if teams is not None:
            items = [(t, self.rating(t)) for t in teams]
        df = pd.DataFrame(sorted(items, key=lambda kv: -kv[1]), columns=["team", "elo"])
        return df.reset_index(drop=True)
