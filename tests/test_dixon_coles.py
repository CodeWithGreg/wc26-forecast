import numpy as np
import pandas as pd
import pytest

from wcforecast.config import DixonColesConfig
from wcforecast.models.dixon_coles import DixonColesModel


@pytest.fixture(scope="module")
def synthetic():
    """Three-tier synthetic league with known generating abilities."""
    rng = np.random.default_rng(42)
    teams = {"Strong": 0.5, "Mid": 0.0, "Weak": -0.5}
    rows = []
    date = pd.Timestamp("2022-01-01")
    names = list(teams)
    for k in range(900):
        h, a = rng.choice(names, 2, replace=False)
        lam_h = np.exp(0.15 + teams[h] - (-teams[a] * 0.5))
        lam_a = np.exp(0.15 + teams[a] - (-teams[h] * 0.5))
        rows.append({
            "date": date + pd.Timedelta(days=k),
            "home_team": h, "away_team": a,
            "home_score": rng.poisson(lam_h), "away_score": rng.poisson(lam_a),
            "neutral": True, "importance": "qualifier",
        })
    return pd.DataFrame(rows)


def test_recovers_ability_ordering(synthetic):
    cfg = DixonColesConfig(min_team_matches=5, ridge=1.0, half_life_days=10000, window_days=100000)
    m = DixonColesModel(config=cfg).fit(synthetic)
    ab = m.abilities().set_index("team")
    assert ab.loc["Strong", "strength"] > ab.loc["Mid", "strength"] > ab.loc["Weak", "strength"]


def test_forecast_uncertainty_and_coherence(synthetic):
    m = DixonColesModel(config=DixonColesConfig(min_team_matches=5, half_life_days=10000, window_days=100000)).fit(synthetic)
    f = m.predict_match("Strong", "Weak", neutral=True)
    ph, pd_, pa = f.p_outcomes
    assert ph > pa
    assert np.isclose(ph + pd_ + pa, 1.0)
    lo, hi = f.lam_interval("home")
    assert lo < f.lam_home < hi  # epistemic interval brackets the mean
    assert len(f.lam_home_draws) == m.config.n_posterior_draws


def test_unknown_team_gets_prior(synthetic):
    m = DixonColesModel(config=DixonColesConfig(min_team_matches=5, half_life_days=10000, window_days=100000)).fit(synthetic)
    f = m.predict_match("Atlantis", "Mid", neutral=True)
    # unknown team ~ average: forecast must stay finite and normalised
    assert np.isclose(f.matrix.sum(), 1.0)
    lo, hi = m.predict_match("Atlantis", "Mid").lam_interval("home")
    lo_known, hi_known = m.predict_match("Mid", "Strong").lam_interval("home")
    assert (hi - lo) > (hi_known - lo_known)  # wider uncertainty for unknown side


def test_time_decay_prefers_recent_form():
    """A team that flipped from bad to good must rate above one that did the opposite."""
    rng = np.random.default_rng(1)
    rows = []
    for k in range(400):
        date = pd.Timestamp("2018-01-01") + pd.Timedelta(days=4 * k)
        early = k < 200
        lam_r, lam_d = (0.6, 2.2) if early else (2.2, 0.6)  # Riser vs Decliner
        rows.append({
            "date": date, "home_team": "Riser", "away_team": "Decliner",
            "home_score": rng.poisson(lam_r), "away_score": rng.poisson(lam_d),
            "neutral": True, "importance": "qualifier",
        })
    df = pd.DataFrame(rows)
    cfg = DixonColesConfig(min_team_matches=5, half_life_days=200, window_days=100000)
    m = DixonColesModel(config=cfg).fit(df)
    ab = m.abilities().set_index("team")
    assert ab.loc["Riser", "strength"] > ab.loc["Decliner", "strength"]
