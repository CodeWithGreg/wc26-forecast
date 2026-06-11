"""Monte-Carlo simulation of the 48-team World Cup 2026.

Pipeline per simulated world:

1. **Group stage** — every one of the 72 fixtures is sampled from its
   plug-in scoreline matrix (vectorised across simulations). Fixtures that
   already have a real result (live mode) are fixed to it.
2. **Standings** — points → goal difference → goals for → random lots.
   (The official regulations insert head-to-head between GF and lots; that
   branch is rarely decisive and is approximated by lots here — documented
   simplification.)
3. **Best thirds** — the 12 third-placed teams are ranked by the same key;
   the top 8 advance and are assigned to the eight constrained bracket
   slots by backtracking perfect matching, operationalising FIFA's
   allocation table.
4. **Knockout** — scores sampled from the scoreline matrix; if level after
   90', extra time is simulated as Poisson with λ/3 (a third of a match),
   then penalties as a fair coin. Host teams keep their venue advantage
   when the bracket schedules them in their own country.

Outputs per-team stage probabilities, champion probabilities and the
pairing distribution of every knockout slot — the latter powers forecasts
for matches whose participants are not known yet (e.g. the final).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from wcforecast.data.wc2026 import BracketMatch, host_country_of

STAGES = ["group", "R32", "R16", "QF", "SF", "THIRD", "FINAL", "champion"]

_VENUE_COUNTRY = {
    "Mexico City": "Mexico", "Guadalajara": "Mexico", "Monterrey": "Mexico",
    "Toronto": "Canada", "Vancouver": "Canada",
}


def venue_country(venue: str) -> str:
    for k, v in _VENUE_COUNTRY.items():
        if venue.startswith(k):
            return v
    return "United States"


@dataclass
class SimResults:
    n_sims: int
    teams: list[str]
    stage_counts: pd.DataFrame  # team × stage reach counts
    pairings: dict[int, Counter]  # match number -> Counter[(home, away)]
    champions: Counter

    @property
    def champion_probs(self) -> pd.Series:
        s = pd.Series(self.champions, dtype=float) / self.n_sims
        return s.sort_values(ascending=False)

    @property
    def stage_probs(self) -> pd.DataFrame:
        return (self.stage_counts / self.n_sims).sort_values("champion", ascending=False)

    def pairing_probs(self, match_no: int, top_k: int = 10) -> pd.DataFrame:
        c = self.pairings[match_no]
        rows = [{"home": h, "away": a, "prob": n / self.n_sims} for (h, a), n in c.most_common(top_k)]
        return pd.DataFrame(rows)

    @staticmethod
    def merge(parts: list[SimResults]) -> SimResults:
        """Combine independent simulation batches (e.g. chunked runs)."""
        assert parts, "nothing to merge"
        base = parts[0]
        n = sum(p.n_sims for p in parts)
        counts = sum((p.stage_counts for p in parts[1:]), base.stage_counts.copy())
        pair: dict[int, Counter] = {k: Counter(v) for k, v in base.pairings.items()}
        champ = Counter(base.champions)
        for p in parts[1:]:
            for k, v in p.pairings.items():
                pair[k].update(v)
            champ.update(p.champions)
        return SimResults(n, base.teams, counts, pair, champ)


@dataclass
class TournamentSimulator:
    """Simulator over a plug-in scoreline predictor.

    ``predict_matrix(home, away, neutral)`` must return a scoreline pmf.
    """

    predict_matrix: Callable[[str, str, bool], np.ndarray]
    groups: pd.DataFrame  # team, group
    fixtures: pd.DataFrame  # 72 group fixtures
    bracket: list[BracketMatch]
    rng_seed: int = 2026
    extra_time_factor: float = 1 / 3
    played: pd.DataFrame | None = None  # optional realised results (live mode)

    _cumsum_cache: dict = field(default_factory=dict, repr=False)

    # ----------------------------------------------------------- utilities
    def _matrix_cumsum(self, home: str, away: str, neutral: bool) -> tuple[np.ndarray, int]:
        key = (home, away, neutral)
        if key not in self._cumsum_cache:
            m = self.predict_matrix(home, away, neutral)
            self._cumsum_cache[key] = (np.cumsum(m.ravel()), m.shape[0])
        return self._cumsum_cache[key]

    def _sample_score(self, rng: np.random.Generator, home: str, away: str, neutral: bool) -> tuple[int, int]:
        c, g = self._matrix_cumsum(home, away, neutral)
        idx = int(np.searchsorted(c, rng.random()))
        return idx // g, idx % g

    # ----------------------------------------------------------- run
    def run(self, n_sims: int = 10_000, collect_pairings: bool = True) -> SimResults:
        rng = np.random.default_rng(self.rng_seed)
        teams = self.groups["team"].tolist()
        group_of = dict(zip(self.groups["team"], self.groups["group"]))
        letters = sorted(self.groups["group"].unique())
        members: dict[str, list[str]] = {
            g: self.groups[self.groups["group"] == g]["team"].tolist() for g in letters
        }

        # ---- vectorised group-stage scores -------------------------------
        fx = self.fixtures.reset_index(drop=True)
        n_fx = len(fx)
        hg = np.empty((n_fx, n_sims), dtype=np.int16)
        ag = np.empty((n_fx, n_sims), dtype=np.int16)
        realised = self._realised_lookup()
        for i, row in enumerate(fx.itertuples(index=False)):
            real = realised.get((row.home_team, row.away_team))
            if real is not None:
                hg[i, :], ag[i, :] = real
                continue
            c, g = self._matrix_cumsum(row.home_team, row.away_team, bool(row.neutral))
            idx = np.searchsorted(c, rng.random(n_sims))
            hg[i, :] = idx // g
            ag[i, :] = idx % g

        # ---- standings (vectorised per group) ----------------------------
        # rank key: points, then GD, then GF, then random lots
        first = {}
        second = {}
        third_stats = {}  # group -> (team_idx array (n_sims,), pts, gd, gf)
        for g in letters:
            tlist = members[g]
            t_index = {t: k for k, t in enumerate(tlist)}
            pts = np.zeros((4, n_sims), dtype=np.int32)
            gd = np.zeros((4, n_sims), dtype=np.int32)
            gf = np.zeros((4, n_sims), dtype=np.int32)
            rows_g = [i for i in range(n_fx) if group_of[fx["home_team"][i]] == g]
            for i in rows_g:
                h = t_index[fx["home_team"][i]]
                a = t_index[fx["away_team"][i]]
                dh, da = hg[i], ag[i]
                pts[h] += np.where(dh > da, 3, np.where(dh == da, 1, 0))
                pts[a] += np.where(da > dh, 3, np.where(dh == da, 1, 0))
                gd[h] += dh - da
                gd[a] += da - dh
                gf[h] += dh
                gf[a] += da
            lots = rng.random((4, n_sims))
            key = pts * 1e7 + (gd + 200) * 1e4 + gf * 10 + lots
            order = np.argsort(-key, axis=0)  # rank 0 = winner
            first[g] = order[0]
            second[g] = order[1]
            third_stats[g] = (order[2],
                              np.take_along_axis(pts, order[2][None, :], 0)[0],
                              np.take_along_axis(gd, order[2][None, :], 0)[0],
                              np.take_along_axis(gf, order[2][None, :], 0)[0])

        # ---- best thirds ranking ------------------------------------------
        n_groups = len(letters)
        t_pts = np.stack([third_stats[g][1] for g in letters])
        t_gd = np.stack([third_stats[g][2] for g in letters])
        t_gf = np.stack([third_stats[g][3] for g in letters])
        lots = rng.random((n_groups, n_sims))
        t_key = t_pts * 1e7 + (t_gd + 200) * 1e4 + t_gf * 10 + lots
        t_order = np.argsort(-t_key, axis=0)  # group indices ranked; top 8 qualify

        third_slots = [m for m in self.bracket if m.away.startswith("3?")]
        slot_allowed = {m.match: set(m.away[2:]) for m in third_slots}

        # ---- accumulators --------------------------------------------------
        stage_counts = pd.DataFrame(0, index=teams, columns=STAGES, dtype=int)
        stage_counts["group"] = n_sims
        pairings: dict[int, Counter] = {m.match: Counter() for m in self.bracket}
        champions: Counter = Counter()
        stage_col = {s: stage_counts.columns.get_loc(s) for s in STAGES}
        counts = stage_counts.to_numpy()
        team_row = {t: i for i, t in enumerate(teams)}

        bracket_by_no = {m.match: m for m in self.bracket}
        ko_order = sorted(bracket_by_no)

        # ---- per-sim knockout ----------------------------------------------
        for s_i in range(n_sims):
            slots: dict[str, str] = {}
            for g in letters:
                slots[f"1{g}"] = members[g][first[g][s_i]]
                slots[f"2{g}"] = members[g][second[g][s_i]]
            # third-place allocation: backtracking perfect matching
            qual_groups = [letters[int(t_order[r, s_i])] for r in range(8)]
            assign = _match_thirds(sorted(slot_allowed), slot_allowed, qual_groups)
            for match_no, g in assign.items():
                slots[f"3@{match_no}"] = members[g][int(third_stats[g][0][s_i])]

            winners: dict[int, str] = {}
            losers: dict[int, str] = {}
            for no in ko_order:
                m = bracket_by_no[no]
                home = _resolve(m.home, no, slots, winners, losers)
                away = _resolve(m.away, no, slots, winners, losers)
                vcountry = venue_country(m.venue)
                h_home = host_country_of(home) == vcountry
                a_home = host_country_of(away) == vcountry
                # treat as neutral unless exactly one side plays at home
                if a_home and not h_home:
                    sh, sa = self._sample_ko(rng, away, home, neutral=False)
                    score_a, score_h = sh, sa
                else:
                    score_h, score_a = self._sample_ko(rng, home, away, neutral=not h_home)
                if collect_pairings:
                    pairings[no][(home, away)] += 1
                win, lose = (home, away) if score_h > score_a else (away, home)
                winners[no], losers[no] = win, lose
                stage = m.stage
                counts[team_row[home], stage_col[stage]] += 1
                counts[team_row[away], stage_col[stage]] += 1
                if stage == "FINAL":
                    champions[win] += 1
                    counts[team_row[win], stage_col["champion"]] += 1

        stage_counts.iloc[:, :] = counts
        return SimResults(n_sims, teams, stage_counts, pairings, champions)

    # ------------------------------------------------------------ helpers
    def _sample_ko(self, rng: np.random.Generator, home: str, away: str, neutral: bool) -> tuple[int, int]:
        """Knockout score after 90' + ET + penalties (returns decisive score)."""
        sh, sa = self._sample_score(rng, home, away, neutral)
        if sh != sa:
            return sh, sa
        # extra time: 30 minutes ≈ λ/3
        lam_h, lam_a = self._approx_lambdas(home, away, neutral)
        eh = rng.poisson(lam_h * self.extra_time_factor)
        ea = rng.poisson(lam_a * self.extra_time_factor)
        if eh != ea:
            return sh + eh, sa + ea
        # penalties: fair coin (empirical evidence for skill at pens is weak)
        return (sh + eh + 1, sa + ea) if rng.random() < 0.5 else (sh + eh, sa + ea + 1)

    def _approx_lambdas(self, home: str, away: str, neutral: bool) -> tuple[float, float]:
        m = self.predict_matrix(home, away, neutral)
        g = np.arange(m.shape[0])
        return float(g @ m.sum(axis=1)), float(g @ m.sum(axis=0))

    def _realised_lookup(self) -> dict:
        out = {}
        if self.played is not None:
            for r in self.played.dropna(subset=["home_score", "away_score"]).itertuples(index=False):
                out[(r.home_team, r.away_team)] = (int(r.home_score), int(r.away_score))
        return out


def _resolve(expr: str, match_no: int, slots: dict, winners: dict, losers: dict) -> str:
    if expr.startswith("3?"):
        return slots[f"3@{match_no}"]
    if expr.startswith("W"):
        return winners[int(expr[1:])]
    if expr.startswith("L"):
        return losers[int(expr[1:])]
    return slots[expr]


def _match_thirds(slot_ids: list[int], slot_allowed: dict[int, set], qual_groups: list[str]) -> dict[int, str]:
    """Assign 8 qualified third-place groups to 8 constrained slots.

    Backtracking perfect matching, most-constrained slot first. Falls back to
    a greedy partial assignment if no perfect matching exists (cannot happen
    for FIFA-consistent constraint sets, but the simulator must not crash).
    """
    avail = list(qual_groups)
    order = sorted(slot_ids, key=lambda s: sum(g in slot_allowed[s] for g in avail))
    assign: dict[int, str] = {}

    def bt(i: int, remaining: list[str]) -> bool:
        if i == len(order):
            return True
        s = order[i]
        for g in [g for g in remaining if g in slot_allowed[s]]:
            assign[s] = g
            rest = list(remaining)
            rest.remove(g)
            if bt(i + 1, rest):
                return True
            del assign[s]
        return False

    if not bt(0, avail):
        rest = list(avail)
        for s in order:
            pick = next((g for g in rest if g in slot_allowed[s]), rest[0])
            assign[s] = pick
            rest.remove(pick)
    return assign


# --------------------------------------------------------------------------
def slot_forecast(
    sim: SimResults,
    match_no: int,
    forecast_fn: Callable[[str, str, bool], object],
    top_k: int = 8,
) -> dict:
    """Forecast a future match whose participants are not yet known.

    Combines the simulated pairing distribution of the slot with conditional
    match forecasts: the unconditional scoreline pmf is the pairing-weighted
    mixture of the conditional pmfs of the ``top_k`` most likely pairings
    (re-normalised). Returns pairing probabilities, the mixture matrix and
    the per-side expected-goal summary.
    """
    pairs = sim.pairing_probs(match_no, top_k)
    mats, lam_h, lam_a = [], 0.0, 0.0
    wsum = pairs["prob"].sum()
    for r in pairs.itertuples(index=False):
        f = forecast_fn(r.home, r.away, True)
        w = r.prob / wsum
        mats.append(w * f.matrix)
        lam_h += w * f.lam_home
        lam_a += w * f.lam_away
    mixture = np.sum(mats, axis=0)
    mixture = mixture / mixture.sum()
    return {"pairings": pairs, "matrix": mixture, "lam_home": lam_h, "lam_away": lam_a}
