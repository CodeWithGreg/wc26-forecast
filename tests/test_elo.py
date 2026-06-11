import pandas as pd

from wcforecast.features.elo import EloEngine


def _toy_results(n_wins: int = 20) -> pd.DataFrame:
    rows = []
    for k in range(n_wins):
        rows.append({
            "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=10 * k),
            "home_team": "Alpha", "away_team": "Beta",
            "home_score": 2, "away_score": 0,
            "neutral": True, "importance": "qualifier",
        })
    return pd.DataFrame(rows)


def test_elo_zero_sum_and_direction():
    eng = EloEngine().fit(_toy_results())
    assert eng.rating("Alpha") > 1500 > eng.rating("Beta")
    assert abs((eng.rating("Alpha") - 1500) + (eng.rating("Beta") - 1500)) < 1e-9


def test_pre_match_ratings_are_leak_free():
    eng = EloEngine().fit(_toy_results())
    pre = eng.pre_match_ratings
    # the first stored pre-match rating must be the initial rating,
    # not one that already includes the first result
    assert pre["elo_home_pre"].iloc[0] == 1500
    assert pre["elo_home_pre"].iloc[1] > 1500


def test_home_advantage_raises_expectation():
    eng = EloEngine()
    assert eng.expected(1500, 1500, neutral=False) > eng.expected(1500, 1500, neutral=True) == 0.5


def test_k_factor_importance_ordering():
    eng = EloEngine()
    assert eng._k("world_cup") > eng._k("qualifier") > eng._k("friendly")
