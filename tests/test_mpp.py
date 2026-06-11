import numpy as np

from wcforecast.models.poisson_core import MatchForecast, score_matrix
from wcforecast.mpp.optimizer import MppRules, crowd_distribution, optimal_pick, picks_table


def _forecast(lh=1.9, la=0.8):
    m = score_matrix(lh, la, -0.06, 10)
    return MatchForecast("Fav", "Dog", lh, la, m)


def test_crowd_distribution_is_distribution():
    f = _forecast()
    q = crowd_distribution(f.matrix, MppRules())
    assert np.isclose(q.sum(), 1.0)
    assert q.min() >= 0


def test_crowd_sharper_than_model_on_favourite():
    """β>1 + popularity prior ⇒ crowd over-bets the favourite's win."""
    f = _forecast()
    q = crowd_distribution(f.matrix, MppRules(crowd_sharpening=1.5))
    p_fav_model = np.tril(f.matrix, -1).sum()
    p_fav_crowd = np.tril(q, -1).sum()
    assert p_fav_crowd > p_fav_model


def test_optimal_pick_maximises_expected_points():
    f = _forecast()
    r = optimal_pick(f)
    best = r["best"]["expected_points"]
    for alt in r["alternatives"]:
        assert best >= alt["expected_points"]
    # the pick must agree with the predicted outcome of the match in
    # high-confidence cases like this one (fav strongly expected to win)
    assert r["best"]["outcome"] == "H"


def test_underdog_pick_pays_more_points_when_correct():
    f = _forecast()
    r = optimal_pick(f)
    pts = r["outcome_points"]  # H, D, A
    assert pts[2] > pts[1] > pts[0]


def test_supplied_odds_override_model_odds():
    f = _forecast()
    r = optimal_pick(f, odds=(1.5, 4.0, 8.0))
    assert r["outcome_points"] == [15.0, 40.0, 80.0]


def test_picks_table_x2_on_max_expected_points():
    fs = [_forecast(2.4, 0.5), _forecast(1.1, 1.0)]
    t = picks_table(fs)
    starred = t[t["x2_booster"] != ""]
    assert len(starred) == 1
    assert starred["expected_points"].iloc[0] == t["expected_points"].max()
