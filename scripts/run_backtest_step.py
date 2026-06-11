#!/usr/bin/env python3
"""Resumable per-tournament backtest driver.

The full leave-one-tournament-out backtest (``wcf backtest``) runs all
tournaments in one process. This driver executes **one tournament per
invocation** and checkpoints partial results, which is convenient on
constrained executors (CI shards, notebook kernels with timeouts).

Usage:
    python scripts/run_backtest_step.py            # next pending tournament
    python scripts/run_backtest_step.py --status   # show progress
    python scripts/run_backtest_step.py --finalize # aggregate + pool weight
"""

from __future__ import annotations

import argparse
import sys

import joblib
import numpy as np
import pandas as pd

from wcforecast.config import get_settings
from wcforecast.data import load_results
from wcforecast.eval import metrics as M
from wcforecast.eval.backtest import TOURNAMENTS, _gbm_match_lambdas, _tournament_matches, _uniform_matrix
from wcforecast.features.build import build_fixture_features, build_training_table
from wcforecast.models.baselines import EloPoissonBaseline
from wcforecast.models.dixon_coles import DixonColesModel
from wcforecast.models.ensemble import fit_pool_weight, pool_matrices
from wcforecast.models.gbm import GbmGoalModel
from wcforecast.models.poisson_core import mixture_score_matrix, outcome_probs, totals_pmf

CKPT = get_settings().artifacts_dir / "backtest_ckpt.joblib"


def load_ckpt():
    if CKPT.exists():
        return joblib.load(CKPT)
    return {"done": [], "rows": [], "match_log": [], "dc_mats": [], "gbm_mats": [], "scores": [], "w_hist": {}}


def run_one(label: str | None = None):
    ck = load_ckpt()
    todo = [t for t in sorted(TOURNAMENTS, key=lambda t: t.start) if t.label not in ck["done"]]
    if label:
        todo = [t for t in todo if t.label == label]
    if not todo:
        print("nothing to do")
        return
    t = todo[0]
    results = load_results()
    start = pd.Timestamp(t.start)
    hist = results[results["date"] < start]
    test = _tournament_matches(results, t)
    print(f"[{t.label}] {len(test)} matches | training on {len(hist)}")

    table = build_training_table(hist)
    dc = DixonColesModel().fit(hist, cutoff=start)
    gbm = GbmGoalModel().fit(table, cutoff=start)
    gbm.rho_ = dc.rho_
    elo_base = EloPoissonBaseline().fit(table)
    feats = build_fixture_features(hist, test[["date", "home_team", "away_team", "neutral"]], t.importance)
    lam_h_gbm, lam_a_gbm = _gbm_match_lambdas(gbm, feats)

    w_ens = 0.5
    if ck["scores"]:
        w_ens, _ = fit_pool_weight(ck["dc_mats"], ck["gbm_mats"], ck["scores"])
    ck["w_hist"][t.label] = w_ens

    per_model = {name: {"probs": [], "mats": [], "lam": [], "goals": [], "tot_pmf": [], "tot": []}
                 for name in ["uniform", "elo_poisson", "dixon_coles", "gbm", "ensemble"]}

    for i, m_row in enumerate(test.itertuples(index=False)):
        xh, xa = int(m_row.home_score), int(m_row.away_score)
        neutral = bool(m_row.neutral)
        f_dc = dc.predict_match(m_row.home_team, m_row.away_team, neutral)
        mat_gbm = mixture_score_matrix(lam_h_gbm[i], lam_a_gbm[i], dc.rho_, dc.config.max_goals)
        f_elo = elo_base.predict_match(m_row.home_team, m_row.away_team,
                                       feats["elo_home"].iloc[i], feats["elo_away"].iloc[i], neutral)
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
        po = outcome_probs(mats["ensemble"])
        ck["match_log"].append({
            "tournament": t.label, "date": m_row.date,
            "home_team": m_row.home_team, "away_team": m_row.away_team,
            "home_score": xh, "away_score": xa,
            "p_home": po[0], "p_draw": po[1], "p_away": po[2],
            "lam_home": lams["ensemble"][0], "lam_away": lams["ensemble"][1],
            "w_ens": w_ens,
            "tot_pmf": totals_pmf(mats["ensemble"]).tolist(),
        })
        ck["dc_mats"].append(f_dc.matrix)
        ck["gbm_mats"].append(mat_gbm)
        ck["scores"].append((xh, xa))

    outcomes = np.array([M.outcome_index(xh, xa) for xh, xa in zip(test["home_score"], test["away_score"])])
    scores = list(zip(test["home_score"].astype(int), test["away_score"].astype(int)))
    for name, d in per_model.items():
        probs = np.asarray(d["probs"])
        ck["rows"].append({
            "tournament": t.label, "start": t.start, "model": name, "n_matches": len(test),
            "log_loss_1x2": M.log_loss_1x2(probs, outcomes),
            "rps": M.rps_1x2(probs, outcomes),
            "brier": M.brier_1x2(probs, outcomes),
            "accuracy": M.accuracy_1x2(probs, outcomes),
            "exact_score_ll": M.exact_score_log_loss(d["mats"], scores),
            "poisson_deviance": M.poisson_deviance(np.asarray(d["lam"]), np.asarray(d["goals"])),
            "coverage90_total_goals": M.interval_coverage(d["tot_pmf"], d["tot"], 0.9),
            "w_ens": w_ens if name == "ensemble" else np.nan,
        })
    ck["done"].append(t.label)
    joblib.dump(ck, CKPT)
    print(f"[{t.label}] done ({len(ck['done'])}/{len(TOURNAMENTS)})  w_ens={w_ens:.2f}")


def finalize():
    import json

    s = get_settings()
    ck = load_ckpt()
    summary = pd.DataFrame(ck["rows"])
    detail = pd.DataFrame(ck["match_log"]).drop(columns=["tot_pmf"])
    w_final, curve = fit_pool_weight(ck["dc_mats"], ck["gbm_mats"], ck["scores"])
    (s.artifacts_dir / "ensemble_weight.json").write_text(
        json.dumps({"weight_dc": w_final, "grid_loss": list(map(float, curve))}, indent=1))
    summary.to_csv(s.reports_dir / "backtest_metrics.csv", index=False)
    detail.to_csv(s.reports_dir / "backtest_by_match.csv", index=False)
    print(f"finalized: {len(ck['done'])} tournaments, {len(detail)} matches, w*={w_final:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    if args.status:
        ck = load_ckpt()
        print("done:", ck["done"])
        sys.exit(0)
    if args.finalize:
        finalize()
        sys.exit(0)
    run_one(args.label)
