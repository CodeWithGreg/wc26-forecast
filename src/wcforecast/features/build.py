"""Leak-free match feature construction.

Two entry points:

* :func:`build_training_table` — features for *played* matches, where every
  feature is strictly pre-match (Elo replayed up to but excluding the match,
  rolling form shifted by one match, rest days from the previous match).
* :func:`build_fixture_features` — features for *future* fixtures, taken
  as-of the end of the supplied results history (i.e. a pre-tournament
  freeze, the standard leave-one-tournament-out protocol from the
  literature — see BENCHMARK.md §9.3).

The same column set comes out of both, so models cannot accidentally train
and predict on differently-defined features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wcforecast.config import IMPORTANCE_ORDER
from wcforecast.features.elo import EloEngine

#: Side-view feature columns consumed by the GBM (see ``models/gbm.py``).
FEATURE_COLUMNS = [
    "elo_team", "elo_opp", "elo_diff",
    "gf5_team", "ga5_team", "gf10_team", "ga10_team", "winrate10_team",
    "gf5_opp", "ga5_opp", "gf10_opp", "ga10_opp", "winrate10_opp",
    "rest_team", "rest_opp",
    "is_home_nonneutral", "is_away_nonneutral",
    "importance_code",
]

_FORM_COLS = ["gf5", "ga5", "gf10", "ga10", "winrate10", "rest"]


def _long_view(results: pd.DataFrame) -> pd.DataFrame:
    """One row per (match, team) with goals for/against and result."""
    home = results[["date", "home_team", "away_team", "home_score", "away_score"]].copy()
    home.columns = ["date", "team", "opp", "gf", "ga"]
    away = results[["date", "away_team", "home_team", "away_score", "home_score"]].copy()
    away.columns = ["date", "team", "opp", "gf", "ga"]
    long = pd.concat([home, away], ignore_index=True).sort_values(["team", "date"], kind="stable")
    long["win"] = (long["gf"] > long["ga"]).astype(float)
    return long


def _rolling_form(long: pd.DataFrame, shifted: bool) -> pd.DataFrame:
    """Rolling form per team. ``shifted=True`` excludes the current match."""
    g = long.groupby("team", sort=False)
    out = long[["date", "team"]].copy()
    for col, src, win in [
        ("gf5", "gf", 5), ("ga5", "ga", 5),
        ("gf10", "gf", 10), ("ga10", "ga", 10),
        ("winrate10", "win", 10),
    ]:
        if shifted:
            out[col] = g[src].transform(lambda s, w=win: s.rolling(w, min_periods=1).mean().shift(1))
        else:
            out[col] = g[src].transform(lambda s, w=win: s.rolling(w, min_periods=1).mean())
    # days since the team's previous match (the state view recomputes this
    # against the fixture date downstream)
    out["rest"] = (long["date"] - g["date"].shift(1)).dt.days.clip(upper=60)
    return out


def _attach_form(matches: pd.DataFrame, form: pd.DataFrame, side: str, team_col: str) -> pd.DataFrame:
    f = form.rename(columns={c: f"{c}_{side}" for c in _FORM_COLS})
    f = f.rename(columns={"team": team_col})
    return matches.merge(f, on=["date", team_col], how="left")


def build_training_table(results: pd.DataFrame) -> pd.DataFrame:
    """Pre-match features + observed scores for every played match."""
    results = results.sort_values("date").reset_index(drop=True)
    engine = EloEngine().fit(results)
    pre = engine.pre_match_ratings
    out = results.copy()
    out["elo_home"] = pre["elo_home_pre"].to_numpy()
    out["elo_away"] = pre["elo_away_pre"].to_numpy()

    long = _long_view(results)
    form = _rolling_form(long, shifted=True).drop_duplicates(["date", "team"], keep="first")
    out = _attach_form(out, form, "home", "home_team")
    out = _attach_form(out, form, "away", "away_team")
    out["importance_code"] = pd.Categorical(
        out["importance"], categories=IMPORTANCE_ORDER
    ).codes.astype(int)
    return out


def team_state(results: pd.DataFrame) -> tuple[EloEngine, pd.DataFrame]:
    """Elo engine and latest per-team form state after replaying ``results``."""
    results = results.sort_values("date").reset_index(drop=True)
    engine = EloEngine().fit(results)
    long = _long_view(results)
    form = _rolling_form(long, shifted=False)
    form["last_date"] = long["date"].to_numpy()
    state = form.sort_values("date").groupby("team").tail(1).set_index("team")
    return engine, state


def build_fixture_features(
    results: pd.DataFrame,
    fixtures: pd.DataFrame,
    importance: str = "world_cup",
    state: tuple | None = None,
) -> pd.DataFrame:
    """As-of-now features for unplayed fixtures.

    ``fixtures`` needs columns: date, home_team, away_team, neutral.
    ``state`` optionally injects a precomputed :func:`team_state` (the replay
    over history is the expensive part; callers forecasting many fixtures
    should compute it once).
    """
    engine, state = state if state is not None else team_state(results)
    out = fixtures.copy()
    out["importance"] = importance
    out["elo_home"] = out["home_team"].map(engine.rating)
    out["elo_away"] = out["away_team"].map(engine.rating)
    for side, col in [("home", "home_team"), ("away", "away_team")]:
        for c in ["gf5", "ga5", "gf10", "ga10", "winrate10"]:
            out[f"{c}_{side}"] = out[col].map(state[c])
        last = out[col].map(state["last_date"])
        out[f"rest_{side}"] = (out["date"] - last).dt.days.clip(upper=60)
    out["importance_code"] = pd.Categorical(
        out["importance"], categories=IMPORTANCE_ORDER
    ).codes.astype(int)
    return out


def to_side_view(table: pd.DataFrame, with_target: bool = True) -> pd.DataFrame:
    """Convert a match-level table into two team-perspective rows per match.

    The GBM predicts goals scored by ``team`` against ``opp``; training on
    both orientations doubles the sample and enforces symmetry.
    """
    def one(side: str, opp: str) -> pd.DataFrame:
        d = pd.DataFrame({
            "match_id": table.index,
            "date": table["date"],
            "team": table[f"{side}_team"],
            "opp": table[f"{opp}_team"],
            "elo_team": table[f"elo_{side}"],
            "elo_opp": table[f"elo_{opp}"],
            "is_home_nonneutral": ((side == "home") & ~table["neutral"]).astype(int),
            "is_away_nonneutral": ((side == "away") & ~table["neutral"]).astype(int),
            "importance_code": table["importance_code"],
        })
        for c in ["gf5", "ga5", "gf10", "ga10", "winrate10", "rest"]:
            d[f"{c}_team"] = table[f"{c}_{side}"].to_numpy()
            d[f"{c}_opp"] = table[f"{c}_{opp}"].to_numpy()
        d["elo_diff"] = d["elo_team"] - d["elo_opp"]
        if with_target:
            d["goals"] = table[f"{side}_score"].to_numpy()
        return d

    return pd.concat([one("home", "away"), one("away", "home")], ignore_index=True)
