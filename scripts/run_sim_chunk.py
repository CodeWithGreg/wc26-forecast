#!/usr/bin/env python3
"""Chunked tournament simulation: run N sims per invocation, merge into a
checkpoint. Useful on constrained executors; ``wcf simulate`` does the same
in one process.

Usage: python scripts/run_sim_chunk.py [n_sims_chunk]
"""

from __future__ import annotations

import sys

import joblib

from wcforecast.config import get_settings
from wcforecast.data import load_bracket, load_group_fixtures, load_groups
from wcforecast.predictor import MatchPredictor
from wcforecast.sim import SimResults, TournamentSimulator

ART = get_settings().artifacts_dir
CKPT = ART / "sim_ckpt.joblib"


def main(n: int = 7000):
    pred = MatchPredictor.load()
    parts: list[SimResults] = joblib.load(CKPT) if CKPT.exists() else []
    seed = 2026 + 101 * len(parts)
    sim = TournamentSimulator(
        predict_matrix=lambda h, a, nu: pred.plugin_matrix(h, a, nu),
        groups=load_groups(), fixtures=load_group_fixtures(), bracket=load_bracket(),
        rng_seed=seed,
    )
    parts.append(sim.run(n))
    joblib.dump(parts, CKPT)
    merged = SimResults.merge(parts)
    joblib.dump(merged, ART / f"sim_{merged.n_sims}.joblib")
    print(f"chunk done (seed {seed}); total sims = {merged.n_sims}")
    print(merged.champion_probs.head(8).round(4).to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 7000)
