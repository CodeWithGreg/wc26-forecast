#!/usr/bin/env python3
"""Generate and execute the showcase notebook (notebooks/01_wc2026_showcase.ipynb)."""

from __future__ import annotations

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(x):
    cells.append(nbf.v4.new_markdown_cell(x))


def code(x):
    cells.append(nbf.v4.new_code_cell(x))


md("""# World Cup 2026 — probabilistic score forecasting

**Author:** gclaus · **Package:** [`wcforecast`](https://github.com/CodeWithGreg/wc26-forecast) · compiled on tournament opening day (2026-06-11)

This notebook showcases the full pipeline:

1. fit the **Dixon-Coles + LightGBM ensemble** on 37k international matches,
2. inspect the **leave-one-tournament-out backtest** (10 majors, 535 matches) and its **calibration**,
3. forecast the **72 scheduled group games** with uncertainty,
4. explain a forecast with **SHAP**,
5. simulate the tournament (championship odds, and a forecast of **the final — before we know who plays it**),
6. derive **Mon Petit Prono optimal picks**.

Methodological background and the survey of prior art (Maher → Dixon-Coles → Karlis-Ntzoufras → Groll/Zeileis hybrids) live in [`BENCHMARK.md`](../BENCHMARK.md).""")

code("""import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import Image, display
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 30)

from wcforecast.data import load_results, load_group_fixtures
from wcforecast.predictor import MatchPredictor

results = load_results()
print(f"{len(results):,} played internationals, {results['date'].min():%Y} → {results['date'].max():%Y-%m-%d}")""")

md("""## 1 — Fit the ensemble

Two complementary scoreline models, combined by a log-linear pool (weight fit out-of-sample in the backtest):

* **Dixon-Coles** bivariate Poisson with exponential time decay (half-life 3y), ridge-MAP abilities and a **Laplace posterior** → epistemic uncertainty on every team's attack/defence;
* **LightGBM Poisson** on leak-free features (Elo, rolling form, rest, venue, importance), **Bayesian-bootstrap bagged** → epistemic spread of λ.""")

code("""pred = MatchPredictor.fit(results)   # ~15 s
print(f"pool weight on Dixon-Coles: {pred.weight_dc:.2f}")""")

code("""# Posterior team abilities (Dixon-Coles): attack + defence ± 1 sd
pred.dc.abilities().head(12).round(3)""")

md("""## 2 — Does it actually forecast? The backtest

Every model is refit with data **strictly before** each of the 10 major tournaments (WC 2006–2022, EURO 2008–2024) and scored on that tournament — the honest pre-tournament setting. Primary metric: **1X2 log-loss** (see BENCHMARK §6 for why), with RPS for literature comparability, exact-score log-loss and Poisson deviance for the goal head, and coverage for the predictive intervals.""")

code("""agg = pd.read_csv("../reports/backtest_aggregate.csv")
agg.sort_values("log_loss_1x2").round(4)""")

md("""The ensemble beats the **Elo-only Poisson baseline** (the strong simple model most public predictors amount to) on *every* proper score, and the uniform floor by a wide margin. Note the classic pattern from the literature: covariates barely move 1X2 log-loss vs Dixon-Coles alone, but consistently sharpen the **full scoreline distribution** (exact-score LL, deviance). 90% intervals on total goals cover ~95% — discrete intervals over-cover by construction.""")

code("""display(Image("../reports/figures/backtest_logloss.png"))
display(Image("../reports/figures/reliability.png"))
display(Image("../reports/figures/pit_total_goals.png"))""")

md("""## 3 — Group-stage forecasts (all 72 scheduled matches)

`wcf predict fixtures` writes the full table; below, the opening days. `λ` columns are posterior-mean expected goals, the `ci90` columns are **epistemic** 90% intervals on λ (how sure the model is about the *rate*), and `goals_90` are predictive intervals on actual goals (rate + Poisson noise).""")

code("""gp = pd.read_csv("../predictions/group_stage.csv")
gp.head(14)""")

code("""# the opening match, in full: scoreline heatmap + uncertainty
from wcforecast.viz import score_heatmap
f = pred.forecast("Mexico", "South Africa", neutral=False, date="2026-06-11")
score_heatmap(f, path="../reports/figures/opener_heatmap.png")
display(Image("../reports/figures/opener_heatmap.png"))
lo, hi = f.lam_interval("home")
print(f"Mexico λ = {f.lam_home:.2f}, 90% epistemic CI [{lo:.2f}, {hi:.2f}]")""")

md("""## 4 — Why this forecast? SHAP contributions

SHAP values on the GBM head are additive contributions to **log λ** (multiplicative on goals). The Dixon-Coles half is interpretable by construction; together they make the whole ensemble explainable.""")

code("""from wcforecast.explain import match_contributions
from wcforecast.features.build import to_side_view

feats = pred._fixture_features("France", "Brazil", True, None, "world_cup")
side = to_side_view(feats.assign(home_score=np.nan, away_score=np.nan), with_target=False)
match_contributions(pred.gbm, side.iloc[[0]]).head(10).round(3)""")

md("""## 5 — Tournament simulation

14,000 Monte-Carlo tournaments through the real 2026 format: 12 groups, best-thirds allocation by constraint matching, the full R32→final bracket, extra time + penalties, host venue advantage.""")

code("""sim_table = pd.read_csv("../predictions/simulation.csv", index_col=0)
display(Image("../reports/figures/champion_probs.png"))
(sim_table.head(12)[["R32", "R16", "QF", "SF", "FINAL", "champion"]] * 100).round(1)""")

md("""### Forecasting the **final** — participants unknown

The headline trick: a forecast for bracket slot 104 (the final, July 19) *today*. The simulation gives the pairing distribution; the unconditional scoreline forecast is the pairing-weighted mixture of conditional forecasts (`wcf predict slot 104`).""")

code("""display(pd.read_csv("../predictions/final_pairings.csv"))
pd.read_csv("../predictions/slot_forecasts.csv")""")

md("""## 6 — Mon Petit Prono: expected-points optimal picks

MPP pays odds-indexed points for the correct 1X2 plus a crowd-rarity bonus for the exact score — so the optimal pick maximises *expected points*, not probability. With model fair odds, the optimiser drifts toward draws in coin-flip matches (three-way odds ≈ 35 points) and slightly bolder scorelines for favourites (rarity bonus). Feed real MPP odds via `--odds-file` for exact optimisation.""")

code("""pd.read_csv("../predictions/mpp_picks.csv")""")

md("""## Honest limitations

* **No market/odds input** — by design (fully open data); the literature's strongest single signal (bookmaker consensus) is deliberately absent and would likely add ~0.01–0.02 log-loss.
* **No player-level data** — squad market values and plus-minus ratings (Groll/Zeileis 2026) are out of scope for the offline-reproducible core; the architecture leaves a feature slot for them.
* **Pre-tournament freeze** — forecasts condition on information at training cutoff; rerun `wcf train && wcf predict …` during the tournament for live updates.
* **Penalty shoot-outs** as a fair coin; head-to-head group tiebreakers approximated by lots.

## References

Dixon & Coles (1997) · Karlis & Ntzoufras (2003) · Groll et al. (2019) · Ley et al. (2019) · Zeileis et al. (2026) — full citations and benchmark numbers in [`BENCHMARK.md`](../BENCHMARK.md).""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}

with open("notebooks/01_wc2026_showcase.ipynb", "w") as fh:
    nbf.write(nb, fh)
print("notebook written")
