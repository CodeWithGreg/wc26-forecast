import numpy as np
import pandas as pd
import pytest

from wcforecast.data import load_bracket, load_group_fixtures, load_groups
from wcforecast.models.poisson_core import score_matrix
from wcforecast.sim.tournament import TournamentSimulator, _match_thirds, venue_country


@pytest.fixture(scope="module")
def sim_results():
    """Small simulation driven by a deterministic strength oracle."""
    groups = load_groups()
    strengths = {t: 1.0 + 0.5 * (hash(t) % 7) / 7 for t in groups["team"]}
    strengths["France"] = 2.6  # make one clear favourite
    strengths["Spain"] = 2.5

    def predict(home, away, neutral):
        return score_matrix(strengths[home] * 0.9, strengths[away] * 0.9 * 0.8, -0.05, 10)

    sim = TournamentSimulator(
        predict_matrix=predict,
        groups=groups,
        fixtures=load_group_fixtures(),
        bracket=load_bracket(),
        rng_seed=11,
    )
    return sim.run(400)


def test_counts_consistency(sim_results):
    r = sim_results
    assert r.stage_counts["group"].sum() == 48 * r.n_sims
    # exactly 32 teams reach the R32 per sim, 2 the final, 1 champion
    assert r.stage_counts["R32"].sum() == 32 * r.n_sims
    assert r.stage_counts["FINAL"].sum() == 2 * r.n_sims
    assert r.stage_counts["champion"].sum() == r.n_sims
    assert sum(r.champions.values()) == r.n_sims


def test_stage_probs_monotone(sim_results):
    p = sim_results.stage_probs
    assert (p["R16"] <= p["R32"] + 1e-9).all()
    assert (p["champion"] <= p["FINAL"] + 1e-9).all()


def test_favourite_wins_more(sim_results):
    champs = sim_results.champion_probs
    assert champs.index[0] in ("France", "Spain")


def test_third_place_matching_respects_constraints():
    slot_allowed = {74: set("ABCDF"), 77: set("CDFGH"), 79: set("CEFHI"), 80: set("EHIJK"),
                    81: set("BEFIJ"), 82: set("AEHIJ"), 85: set("EFGIJ"), 87: set("DEIJL")}
    rng = np.random.default_rng(5)
    letters = list("ABCDEFGHIJKL")
    for _ in range(200):
        qual = list(rng.choice(letters, 8, replace=False))
        assign = _match_thirds(sorted(slot_allowed), slot_allowed, qual)
        assert sorted(assign.values()) == sorted(qual)
        for slot, g in assign.items():
            assert g in slot_allowed[slot], f"slot {slot} got group {g}"


def test_venue_country_mapping():
    assert venue_country("Mexico City") == "Mexico"
    assert venue_country("Toronto") == "Canada"
    assert venue_country("Dallas (Arlington)") == "United States"


def test_played_results_are_respected():
    groups = load_groups()
    fixtures = load_group_fixtures()

    def predict(home, away, neutral):
        return score_matrix(1.2, 1.2, 0.0, 10)

    first = fixtures.iloc[0]
    played = pd.DataFrame([{
        "home_team": first["home_team"], "away_team": first["away_team"],
        "home_score": 5, "away_score": 0,
    }])
    sim = TournamentSimulator(predict_matrix=predict, groups=groups,
                              fixtures=fixtures, bracket=load_bracket(), played=played, rng_seed=3)
    r = sim.run(50)
    # the fixed 5-0 must give the home side at least 3 points in every sim:
    # its P(advancing past groups) should exceed the loser's
    p = r.stage_probs
    assert p.loc[first["home_team"], "R32"] > p.loc[first["away_team"], "R32"]
