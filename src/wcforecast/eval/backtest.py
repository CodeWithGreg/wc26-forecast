"""Leave-one-tournament-out backtest.

Protocol (BENCHMARK.md §9.3): for every held-out tournament, all models are
fit on matches **strictly before the tournament start** — Elo replay, form
features, Dixon-Coles abilities and the GBM all freeze at that date, so no
information from the evaluated tournament (or its future) leaks in. This is
the pre-tournament forecasting setting in which the published academic
models (Groll et al. 2019; Zeileis et al. 2026) are evaluated.

The ensemble pool weight is fit on an **expanding window of previously
backtested tournaments** (the first tournament uses w = 0.5), so the
ensemble's reported performance is itself out-of-sample.

Outputs a tidy per-(tournament, model) metric table, a per-match forecast
log for calibration plots, and the production pool weight refit on all
backtest tournaments at the end.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from wcforecast.config import get_settings
from wcforecast.eval import metrics as M
from wcforecast.features.build import build_fixture_features, build_training_table
from wcforecast.models.baselines import EloPoissonBaseline
from wcforecast.models.dixon_coles import DixonColesModel
from wcforecast.models.ensemble import fit_pool_weight, pool_matrices
from wcforecast.models.gbm import GbmGoalModel
from wcforecast.models.poisson_core import outcome_probs, totals_pmf


@dataclass(frozen=True)
class Tournament:
    label: str
    name: str  # tournament name in the results data
    start: str  # ISO date of first match
    importance: str


TOURNAMENTS: list[Tournament] = [
    Tournament("WC 2006", "FIFA World Cup", "2006-06-09", "world_cup"),
    Tournament("EURO 2008", "UEFA Euro", "2008-06-07", "continental"),
    Tournament("WC 2010", "FIFA World Cup", "2010-06-11", "world_cup"),
    Tournament("EURO 2012", "UEFA Euro", "2012-06-08", "continental"),
    Tournament("WC 2014", "FIFA World Cup", "2014-06-12", "world_cup"),
    Tournament("EURO 2016", "UEFA Euro", "2016-06-10", "continental"),
    Tournament("WC 2018", "FIFA World Cup", "2018-06-14", "world_cup"),
    Tournament("EURO 2020", "UEFA Euro", "2021-06-11", "continental"),
    Tournament("WC 2022", "FIFA World Cup", "2022-11-20", "world_cup"),
    Tournament("EURO 2024", "UEFA Euro", "2024-06-14", "continental"),
]

COPA: list[Tournament] = [
    Tournament("COPA 2021", "Copa América", "2021-06-13", "continental"),
    Tournament("COPA 2024", "Copa América", "2024-06-20", "continental"),
]


def _tournament_matches(results: pd.DataFrame, t: Tournament) -> pd.DataFrame:
    start = pd.Timestamp(t.start)
    sel = (
        (results["tournament"] == t.name)
        & (results["date"] >= start)
        & (results["date"] <= start + pd.Timedelta(days=45))
    )
    return results[sel].copy()


def _gbm_match_lambdas(gbm: GbmGoalModel, feats: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(n_matches, n_bags) λ draws for home and away sides, batched."""
    side = gbm.predict_fixends(feats.reset_index(drop=True))
    n = len(feats)
    bag_cols = [c for c in side.columns if c.startswith("lam_b")]
    home_rows = side.iloc[:n]
    away_rows = side.iloc[n:]
    return home_rows[bag_cols].to_numpy(), away_rows[bag_cols].to_numpy()


def run_backtest(
    results: pd.DataFrame,
    include_copa: bool = False,
    quick: bool = False,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Run the leave-one-tournament-out backtest.

    Returns (summary metrics, per-match log, production ensemble weight).
    """
    from wcforecast.models.poisson_core import mixture_score_matrix

    s = get_settings()
    tournaments = sorted(TOURNAMENTS + (COPA if include_copa else []), key=lambda t: t.start)

    rows: list[dict] = []
    match_log: list[dict] = []
    hist_dc_mats: list[np.ndarray] = []
    hist_gbm_mats: list[np.ndarray] = []
    hist_scores: list[tuple[int, int]] = []
    w_ens = 0.5

    for t in tournaments:
        start = pd.Timestamp(t.start)
        hist = results[results["date"] < start]
        test = _tournament_matches(results, t)
        if len(test) == 0:
            continue
        if verbose:
            print(f"[backtest] {t.label}: {len(test)} matches, training on {len(hist)} matches")

        table = build_training_table(hist)
        dc = DixonColesModel().fit(hist, cutoff=start)
        gbm = GbmGoalModel()
        if quick:  # smoke-test mode: smaller bag, shorter boosting
            gbm.config = type(gbm.config)(n_bags=3, n_estimators=200, early_stopping_rounds=30)
        gbm.fit(table, cutoff=start)
        gbm.rho_ = dc.rho_
        elo_base = EloPoissonBaseline().fit(table)

        feats = build_fixture_features(hist, test[["date", "home_team", "away_team", "neutral"]], t.importance)
        lam_h_gbm, lam_a_gbm = _gbm_match_lambdas(gbm, feats)

        # fit ensemble weight on previously processed tournaments (expanding)
        if hist_scores:
            w_ens, _ = fit_pool_weight(hist_dc_mats, hist_gbm_mats, hist_scores)

        per_model: dict[str, dict[str, list]] = {
            name: {"probs": [], "mats": [], "lam": [], "goals": [], "tot_pmf": [], "tot": []}
            for name in ["uniform", "elo_poisson", "dixon_coles", "gbm", "ensemble"]
        }

        for i, m_row in enumerate(test.itertuples(index=False)):
            xh, xa = int(m_row.home_score), int(m_row.away_score)
            neutral = bool(m_row.neutral)
            f_dc = dc.predict_match(m_row.home_team, m_row.away_team, neutral)
            mat_gbm = mixture_score_matrix(lam_h_gbm[i], lam_a_gbm[i], dc.rho_, dc.config.max_goals)
            f_elo = elo_base.predict_match(
                m_row.home_team, m_row.away_team,
                feats["elo_home"].iloc[i], feats["elo_away"].iloc[i], neutral,
            )
            mats = {
                "uniform": _uniform_matrix(),
                "elo_poisson": f_elo.matrix,
                "dixon_coles": f_dc.matrix,
                "gbm": mat_gbm,
                "ensemble": pool_matrices(f_dc.matrix, mat_gbm, w_ens),
            }
            lams = {
                "uniform": (1.30, 1.30),
                "elo_poisson": (f_elo.lam_home, f_elo.lam_away),
                "dixon_coles": (f_dc.lam_home, f_dc.lam_away),
                "gbm": (float(lam_h_gbm[i].mean()), float(lam_a_gbm[i].mean())),
            }
            lams["ensemble"] = (
                w_ens * lams["dixon_coles"][0] + (1 - w_ens) * lams["gbm"][0],
                w_ens * lams["dixon_coles"][1] + (1 - w_ens) * lams["gbm"][1],
            )
            for name, mat in mats.items():
                d = per_model[name]
                d["probs"].append(outcome_probs(mat))
                d["mats"].append(mat)
                d["lam"].extend(lams[name])
                d["goals"].extend([xh, xa])
                d["tot_pmf"].append(totals_pmf(mat))
                d["tot"].append(xh + xa)
            match_log.append({
                "tournament": t.label, "date": m_row.date,
                "home_team": m_row.home_team, "away_team": m_row.away_team,
                "home_score": xh, "away_score": xa,
                "p_home": outcome_probs(mats["ensemble"])[0],
                "p_draw": outcome_probs(mats["ensemble"])[1],
                "p_away": outcome_probs(mats["ensemble"])[2],
                "lam_home": lams["ensemble"][0], "lam_away": lams["ensemble"][1],
                "w_ens": w_ens,
            })
            hist_dc_mats.append(f_dc.matrix)
            hist_gbm_mats.append(mat_gbm)
            hist_scores.append((xh, xa))

        outcomes = np.array([M.outcome_index(xh, xa) for xh, xa in zip(test["home_score"], test["away_score"])])
        scores = list(zip(test["home_score"].astype(int), test["away_score"].astype(int)))
        for name, d in per_model.items():
            probs = np.asarray(d["probs"])
            rows.append({
                "tournament": t.label,
                "start": t.start,
                "model": name,
                "n_matches": len(test),
                "log_loss_1x2": M.log_loss_1x2(probs, outcomes),
                "rps": M.rps_1x2(probs, outcomes),
                "brier": M.brier_1x2(probs, outcomes),
                "accuracy": M.accuracy_1x2(probs, outcomes),
                "exact_score_ll": M.exact_score_log_loss(d["mats"], scores),
                "poisson_deviance": M.poisson_deviance(np.asarray(d["lam"]), np.asarray(d["goals"])),
                "coverage90_total_goals": M.interval_coverage(d["tot_pmf"], d["tot"], 0.9),
                "w_ens": w_ens if name == "ensemble" else np.nan,
            })

    summary = pd.DataFrame(rows)
    detail = pd.DataFrame(match_log)

    # production pool weight: refit on the full out-of-sample collection
    w_final, curve = fit_pool_weight(hist_dc_mats, hist_gbm_mats, hist_scores)
    (s.artifacts_dir / "ensemble_weight.json").write_text(
        json.dumps({"weight_dc": w_final, "grid_loss": curve.tolist()}, indent=1)
    )
    summary.to_csv(s.reports_dir / "backtest_metrics.csv", index=False)
    detail.to_csv(s.reports_dir / "backtest_by_match.csv", index=False)
    return summary, detail, w_final


def _uniform_matrix() -> np.ndarray:
    from wcforecast.models.baselines import UniformBaseline

    return UniformBaseline().predict_score_matrix()


def aggregate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Match-weighted aggregate of the per-tournament metrics by model."""
    metric_cols = [
        "log_loss_1x2", "rps", "brier", "accuracy",
        "exact_score_ll", "poisson_deviance", "coverage90_total_goals",
    ]

    def agg(g: pd.DataFrame) -> pd.Series:
        w = g["n_matches"].to_numpy(float)
        return pd.Series({c: float(np.average(g[c], weights=w)) for c in metric_cols} | {"n_matches": int(w.sum())})

    return summary.groupby("model", sort=False).apply(agg, include_groups=False).reset_index()
