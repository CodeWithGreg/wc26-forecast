"""World Cup 2026 structure: groups, group fixtures, knockout bracket.

Sources: group fixtures come from the martj42 dataset (scheduled rows with NA
scores); group composition and the knockout bracket follow the official FIFA
schedule (mirrored by openfootball/worldcup, ``2026--usa``).

Bracket slot grammar
--------------------
``1A`` / ``2A``  winner / runner-up of group A
``3?ABCDF``      a third-placed team drawn from groups {A,B,C,D,F}
``W74`` / ``L101``  winner / loser of match 74 / 101

The eight best third-placed teams are assigned to the eight ``3?…`` slots by
a constraint-satisfying perfect matching (see ``wcforecast.sim.tournament``),
which operationalises FIFA's published allocation table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from wcforecast.config import get_settings


@dataclass(frozen=True)
class BracketMatch:
    match: int
    stage: str  # R32, R16, QF, SF, THIRD, FINAL
    date: str
    home: str  # slot expression
    away: str
    venue: str


def load_groups() -> pd.DataFrame:
    """Return ``team -> group`` table for the 48 qualified teams."""
    df = pd.read_csv(get_settings().data_dir / "wc2026_groups.csv")
    assert len(df) == 48, "expected 48 teams"
    return df


def load_group_fixtures() -> pd.DataFrame:
    """Return the 72 scheduled group-stage fixtures (with group labels)."""
    df = pd.read_csv(get_settings().data_dir / "wc2026_group_fixtures.csv")
    df["date"] = pd.to_datetime(df["date"])
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")
    assert len(df) == 72, "expected 72 group matches"
    return df


def load_bracket() -> list[BracketMatch]:
    """Return the 32 knockout bracket matches in match-number order."""
    raw = json.loads((get_settings().data_dir / "wc2026_bracket.json").read_text())
    out = [BracketMatch(**m) for m in raw]
    assert len(out) == 32
    return sorted(out, key=lambda m: m.match)


def host_country_of(team: str) -> str | None:
    """Return the host country a team plays 'at home' in, if any."""
    return {"United States": "United States", "Canada": "Canada", "Mexico": "Mexico"}.get(team)
