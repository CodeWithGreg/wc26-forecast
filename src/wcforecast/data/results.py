"""International match results (martj42/international_results).

The repository ships a compressed snapshot (``data/results_snapshot.csv.gz``)
so every command works offline and reproducibly. ``wcf data refresh`` updates
the snapshot from the upstream git repository when network access allows.

Schema (one row per match)::

    date, home_team, away_team, home_score, away_score,
    tournament, city, country, neutral

Unplayed scheduled matches (e.g. upcoming WC 2026 fixtures) appear upstream
with NA scores; :func:`load_results` keeps only played matches.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from wcforecast.config import TOURNAMENT_BUCKETS, get_settings

UPSTREAM_GIT = "https://github.com/martj42/international_results.git"

_DTYPES = {
    "home_team": "string",
    "away_team": "string",
    "tournament": "string",
    "city": "string",
    "country": "string",
}


def snapshot_path() -> Path:
    return get_settings().data_dir / "results_snapshot.csv.gz"


def load_results(played_only: bool = True, since: str | None = None) -> pd.DataFrame:
    """Load the historical results table.

    Parameters
    ----------
    played_only:
        Drop rows without a final score (scheduled future matches).
    since:
        Optional ISO date; rows strictly before it are dropped. Defaults to
        ``Settings.train_start``.

    Returns
    -------
    DataFrame sorted by date with parsed dtypes, an ``importance`` bucket
    column and a boolean ``neutral`` column.
    """
    s = get_settings()
    df = pd.read_csv(snapshot_path(), dtype=_DTYPES, na_values=["NA"])
    df["date"] = pd.to_datetime(df["date"])
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")
    df["importance"] = df["tournament"].map(TOURNAMENT_BUCKETS).fillna("other")
    since = since or s.train_start
    df = df[df["date"] >= pd.Timestamp(since)]
    if played_only:
        df = df.dropna(subset=["home_score", "away_score"]).copy()
        df["home_score"] = df["home_score"].astype(int)
        df["away_score"] = df["away_score"].astype(int)
    return df.sort_values("date").reset_index(drop=True)


def refresh_snapshot(verbose: bool = True) -> Path:
    """Re-download the upstream dataset (git clone) and rebuild the snapshot."""
    dest = snapshot_path()
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(
            ["git", "clone", "--depth", "1", UPSTREAM_GIT, td + "/up"],
            check=True,
            capture_output=not verbose,
        )
        src = Path(td) / "up" / "results.csv"
        import gzip

        with open(src, "rb") as fin, gzip.open(dest, "wb", compresslevel=9) as fout:
            shutil.copyfileobj(fin, fout)
    return dest
