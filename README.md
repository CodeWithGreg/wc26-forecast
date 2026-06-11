# wc26-forecast

Probabilistic score forecasting for the **FIFA World Cup 2026**, built as a clean,
tested Python package (`wcforecast`) with a CLI, honest backtests, uncertainty
quantification, feature-contribution explanations, full-tournament Monte-Carlo
simulation and a [Mon Petit Prono](https://mpp.football) pick optimiser.

*Author: gclaus <https://www.linkedin.com/in/gregoireclaus/> · MIT · compiled on opening day, 2026-06-11.*

---

## What it does

Every forecast is a **full scoreline distribution** `P(home = i, away = j)` —
1X2 probabilities, exact-score picks, total-goals intervals, tournament odds and
MPP picks are all derived from that one object.

```text
$ wcf predict match France Brazil
              France vs Brazil
  λh    λa   P(H)  P(D)  P(A)  score*   p*    90% goals
 1.27  1.36   34%   27%   39%    1-1   12.6%  0-3 / 0-3
 λ 90% credible intervals — France: [1.05,1.55], Brazil: [1.15,1.62]
```

The headline trick — forecasting a match **whose participants are not known yet**
(e.g. the final, six weeks out):

```text
$ wcf predict slot 104          # match 104 = the final, July 19
 Spain – Argentina   2.4%       # most likely pairings from 14,000 simulations
 Belgium – Argentina 1.2% …
 Mixture forecast: λ = 1.04 : 1.36, top scoreline 1-1 (12.9%)
```

## Model

A two-component ensemble, following the evidence collected in
[BENCHMARK.md](BENCHMARK.md) (survey of the academic literature, Kaggle
solutions and open-source predictors):

| Component | Role | Uncertainty |
|---|---|---|
| **Dixon-Coles** bivariate Poisson, exponential time decay (3y half-life), ridge-MAP | interpretable team attack/defence abilities; the low-score ρ correction fixes draw underestimation | **Laplace posterior** over all ~480 parameters → λ draws |
| **LightGBM Poisson** on leak-free features (Elo with WC-grade K-factors, rolling form, rest days, venue, importance) | exploits covariates the ability model cannot see | **Bayesian-bootstrap bag** (8 fits) → λ spread |
| **Log-linear pool** `M ∝ M_dc^w · M_gbm^(1−w)` | sharpens where components agree | weight `w = 0.60` fit **out-of-sample** in the backtest |

Aleatoric (Poisson scoring noise) and epistemic (parameter) uncertainty are kept
separate and both propagate to every output: λ credible intervals vs goal
predictive intervals, posterior-predictive mixtures everywhere.

## Does it forecast? (leave-one-tournament-out backtest)

Each of the 10 major tournaments (WC 2006–2022, EURO 2008–2024; 535 matches) is
forecast by models trained **strictly on data before its first match** — Elo,
form, abilities and the GBM all freeze at the tournament start. The ensemble
pool weight is itself fit only on previously backtested tournaments.

| model | 1X2 log-loss ↓ | RPS ↓ | exact-score LL ↓ | Poisson deviance ↓ | accuracy ↑ | 90% coverage |
|---|---|---|---|---|---|---|
| uniform | 1.0785 | 0.2361 | 2.9155 | 1.2484 | 0.411 | 0.953 |
| Elo-only Poisson | 1.0144 | 0.2127 | 2.8314 | 1.1643 | 0.529 | 0.959 |
| GBM | 1.0052 | 0.2097 | 2.8241 | 1.1567 | 0.520 | 0.963 |
| Dixon-Coles | 0.9933 | 0.2050 | 2.8105 | 1.1459 | 0.544 | 0.955 |
| **ensemble** | **0.9929** | 0.2052 | **2.8039** | **1.1366** | 0.529 | 0.957 |

Reading: the ensemble beats the strong simple baseline (Elo-driven Poisson —
what most public predictors amount to) on every proper scoring rule, and sits
at bookmaker-level RPS (~0.205 at World Cups, see BENCHMARK §6). The classic
literature pattern shows up honestly: extra covariates barely move 1X2
log-loss vs Dixon-Coles, but consistently sharpen the **full scoreline
distribution** (exact-score LL, deviance) — which is what exact-score games
like MPP reward. Calibration: randomised-PIT flat (mean 0.514), reliability
on the diagonal, discrete 90% intervals over-cover as expected
(`reports/figures/`).

## Quickstart

```bash
git clone https://github.com/CodeWithGreg/wc26-forecast && cd wc26-forecast
pip install -e ".[explain]"     # add [dev] for tests/notebook tooling

wcf train                       # ~15 s on a laptop: fits DC + GBM on 37k matches
wcf backtest                    # ~5 min: the table above, + reports/
wcf predict fixtures            # all 72 scheduled group matches → predictions/group_stage.csv
wcf predict match Spain England --heatmap out.png
wcf simulate --sims 20000       # championship & stage probabilities
wcf predict slot 104            # the final, before we know who plays it
wcf explain France Brazil       # SHAP contributions to each side's log λ
wcf mpp picks --days 7          # expected-points-optimal MPP picks + X2 advice
wcf abilities                   # Dixon-Coles attack/defence table ± 1 sd
```

Everything runs **offline** from the committed data snapshot
(`data/results_snapshot.csv.gz`, 49k internationals 1872→present, courtesy of
[martj42/international_results](https://github.com/martj42/international_results));
`wcf data refresh` pulls the latest results — rerun `wcf train` afterwards for
live-updated forecasts during the tournament.

## Package layout

```
src/wcforecast/
├── config.py          # every tunable in one place
├── data/              # results snapshot loader · WC26 groups/fixtures/bracket
├── features/          # Elo engine (eloratings.net conventions) · leak-free feature builder
├── models/            # poisson_core (scoreline algebra) · dixon_coles · gbm · baselines · ensemble
├── eval/              # proper scoring rules, PIT/coverage · LOTO backtest
├── sim/               # Monte-Carlo tournament (groups → best-thirds matching → bracket)
├── explain/           # SHAP contributions (log-λ scale)
├── mpp/               # Mon Petit Prono expected-points optimiser
├── viz/               # heatmaps, reliability, PIT, champion bars, SHAP waterfalls
├── predictor.py       # high-level API: MatchPredictor.fit/forecast/plugin_matrix
└── cli.py             # Typer CLI (`wcf …`)
```

Separation of concerns: models only ever see feature tables; the scoreline
matrix is the single interchange format; the simulator and MPP optimiser are
generic over any `predict(home, away, neutral) → matrix` callable. 33 unit
tests cover the scoring algebra, Elo, Dixon-Coles recovery on synthetic data,
metrics propriety, bracket constraints and the MPP optimiser
(`pytest`).

## MPP (Mon Petit Prono) optimisation

MPP pays **odds-indexed points for the correct 1X2** plus a **crowd-rarity
bonus (20/30/50/70/100) for the exact score** among players with the correct
result. The optimal pick maximises expected points, not probability:

`E[pts | pick s] = P(result(s)) · pts(result(s)) + P(s) · bonus(s)`

with an explicit crowd model (forecast sharpened by β=1.3 × popularity prior
on human scorelines). Consequences the optimiser discovers by itself: draws
become attractive in coin-flip matches (three-way odds ≈ 35 pts), favourites
get slightly bolder scorelines than the modal one (rarity bonus), and the X2
booster goes on the highest-E[pts] match. Pass the app's real odds with
`--odds-file` for exact optimisation; otherwise model fair odds are used.

## Honest limitations

No bookmaker odds and no player-level data (market values, plus-minus) by
design — the package is fully reproducible from open data; the literature
suggests those signals would buy another ~0.01–0.02 log-loss (BENCHMARK §9).
Forecasts are pre-tournament freezes (re-train to update). Penalty shoot-outs
are a fair coin; group head-to-head tiebreakers are approximated by lots after
points/GD/GF. The 2026 format itself (48 teams, best-thirds) is new — no
backtest can validate format-specific dynamics.

## References

Maher (1982) · Dixon & Coles (1997) · Karlis & Ntzoufras (2003) · Ley, Van de
Wiele & Van Eetvelde (2019) · Groll, Ley, Schauberger & Van Eetvelde (2019) ·
Constantinou & Fenton (2012) / arXiv:1908.08980 (scoring rules) · Zeileis,
Groll et al. (2026). Full citations, benchmark values and design implications:
[BENCHMARK.md](BENCHMARK.md). Notebook walkthrough:
[notebooks/01_wc2026_showcase.ipynb](notebooks/01_wc2026_showcase.ipynb).
