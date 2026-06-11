#!/usr/bin/env python3
"""Staged production training driver (resumable).

Stages: ``dc`` → ``gbm`` → ``assemble``. Equivalent to ``wcf train`` but
checkpointed per stage for constrained executors.
"""

from __future__ import annotations

import argparse

import joblib

from wcforecast.config import get_settings
from wcforecast.data import load_results
from wcforecast.features.build import build_training_table
from wcforecast.models.dixon_coles import DixonColesModel
from wcforecast.models.gbm import GbmGoalModel

ART = get_settings().artifacts_dir


def stage_dc():
    r = load_results()
    dc = DixonColesModel().fit(r)
    joblib.dump(dc, ART / "stage_dc.joblib")
    print(f"DC fit: {len(dc.teams_)} teams, rho={dc.rho_:+.4f}±{dc.rho_se_:.4f}, gamma={dc.gamma_:+.3f}")


def stage_gbm():
    r = load_results()
    table = build_training_table(r)
    gbm = GbmGoalModel().fit(table)
    dc = joblib.load(ART / "stage_dc.joblib")
    gbm.rho_ = dc.rho_
    joblib.dump(gbm, ART / "stage_gbm.joblib")
    print(f"GBM fit: best_iter={gbm.best_iteration_}, bags={len(gbm.models_)}")


def stage_assemble():
    import json

    from wcforecast.predictor import MatchPredictor

    r = load_results()
    dc = joblib.load(ART / "stage_dc.joblib")
    gbm = joblib.load(ART / "stage_gbm.joblib")
    wfile = ART / "ensemble_weight.json"
    w = float(json.loads(wfile.read_text())["weight_dc"]) if wfile.exists() else get_settings().ensemble_weight_dc
    pred = MatchPredictor(dc=dc, gbm=gbm, weight_dc=w, results=r)
    path = pred.save()
    f = pred.forecast("France", "Brazil", True)
    print(f"saved {path} | smoke France-Brazil: λ {f.lam_home:.2f}:{f.lam_away:.2f} P={tuple(round(p, 3) for p in f.p_outcomes)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["dc", "gbm", "assemble"])
    args = ap.parse_args()
    {"dc": stage_dc, "gbm": stage_gbm, "assemble": stage_assemble}[args.stage]()
