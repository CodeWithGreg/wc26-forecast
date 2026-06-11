# World Cup Score-Forecasting Benchmark
**Compiled:** 2026-06-11 (Opening Day, FIFA World Cup 2026)
**Purpose:** Reference for designing a football score / goal-count prediction system for WC 2026.

---

## Table of Contents

1. [Tournament Context: WC 2026](#1-tournament-context-wc-2026)
2. [Academic Literature](#2-academic-literature)
   - 2.1 Foundational Poisson Models
   - 2.2 Groll et al. Hybrid Random-Forest Line
   - 2.3 Zeileis / Bookmaker-Consensus Line
   - 2.4 Ley et al. (2019) — Ranking Methods Comparison
   - 2.5 Newer Work (2023-2026)
3. [Kaggle & Data-Science Community](#3-kaggle--data-science-community)
4. [Notable Open-Source GitHub Projects](#4-notable-open-source-github-projects)
5. [Rating Systems](#5-rating-systems)
6. [Scoring Rules & Benchmark Values](#6-scoring-rules--benchmark-values)
7. [Key Empirical Facts for Modelling](#7-key-empirical-facts-for-modelling)
8. [Pre-Tournament WC 2026 Forecasts](#8-pre-tournament-wc-2026-forecasts)
9. [Implications for Our Design](#9-implications-for-our-design)

---

## 1. Tournament Context: WC 2026

| Fact | Value |
|---|---|
| Host nations | United States, Canada, Mexico |
| Teams | 48 (expanded from 32) |
| Groups | 12 groups of 4 teams each |
| Qualification from groups | 12 group winners + 12 runners-up + 8 best third-placed = 32 |
| Total matches | 104 (72 group stage + 32 knockout) |
| New knockout round | Round of 32 (new at this edition) |
| Tournament dates | 11 June – 19 July 2026 |
| Final venue | MetLife Stadium, East Rutherford, New Jersey |

**Opening match (today, 11 June 2026):** Mexico vs South Africa, Group A, Estadio Azteca, Mexico City. Kick-off 19:00 UTC. Match was scheduled to begin at the time of writing; the final score had not yet been returned by live-data feeds at time of compilation.

**Sources:** [FIFA official fixtures](https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/scores-fixtures) · [Al Jazeera maps & charts](https://www.aljazeera.com/sports/2026/6/10/fifa-world-cup-2026-explained-in-maps-and-charts) · [ESPN schedule](https://www.espn.com/soccer/story/_/id/48939282/2026-fifa-world-cup-fixtures-results-match-schedule-group-stage-knockout-rounds-bracket)

---

## 2. Academic Literature

### 2.1 Foundational Poisson Models

#### Maher (1982)
**Citation:** Maher, M.J. (1982). "Modelling Association Football Scores." *Statistica Neerlandica* 36:109–118. DOI [10.1111/j.1467-9574.1982.tb00782.x](https://doi.org/10.1111/j.1467-9574.1982.tb00782.x)

The canonical baseline. Treats the goals scored by each team as **conditionally independent Poisson random variables** given team-specific attack and defense strength parameters. Each team has an attack parameter and a defense parameter estimated by maximum likelihood on historical results. The model ignores any correlation between the two teams' scores and does not account for changing team strength over time. Despite these limitations, it remains the workhorse against which all later models are compared.

**Key weakness:** Under-predicts draws, especially low-scoring draws (0-0, 1-0, 0-1), because the bivariate structure treats the scoreline as the product of two independent distributions.

#### Dixon & Coles (1997)
**Citation:** Dixon, M.J. and Coles, S.G. (1997). "Modelling Association Football Scores and Inefficiencies in the Football Betting Market." *JRSS: Series C* 46:265–280. DOI [10.1111/1467-9876.00065](https://doi.org/10.1111/1467-9876.00065)

Two major extensions to Maher:
1. **Low-score correction (ρ-parameter):** A small negative correlation parameter is estimated to inflate the probability of (0-0), (1-0), (0-1) and (1-1) results — precisely the scorelines Maher under-fits. This was originally intended to capture market inefficiency but also improves predictive calibration.
2. **Exponential time-decay:** More recent matches receive higher likelihood weight, controlled by a half-life parameter typically set around 3 years, so that team ability estimates reflect current form rather than a multi-year average.

**Performance claims (from original paper):** The model identified betting market inefficiencies and showed positive returns when used as a betting strategy — demonstrating better predictive calibration than the simple Poisson baseline. Head-to-head log-likelihood improvement over Maher is modest (1-3%) but consistent.

**Source:** [dashee87.github.io implementation guide](https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling-dixon-coles-and-time-weighting/) · [penaltyblog tutorial](https://pena.lt/y/2021/06/24/predicting-football-results-using-python-and-dixon-and-coles/)

#### Karlis & Ntzoufras (2003)
**Citation:** Karlis, D. and Ntzoufras, I. (2003). "Analysis of Sports Data by Using Bivariate Poisson Models." *The Statistician* 52:381–393. DOI [10.1111/1467-9884.00366](https://doi.org/10.1111/1467-9884.00366)

Introduced the **bivariate Poisson model** in a frequentist framework. The joint distribution of (X, Y) = (goals scored by team A, goals by team B) is decomposed as X = X₁ + X₃, Y = Y₁ + X₃ where X₁, Y₁, X₃ are independent Poisson variables. X₃ is a shared component that introduces **positive correlation** between scorelines. A further "diagonal inflation" variant additionally upweights draw outcomes, improving calibration for the most common draw scorelines.

**Relationship to Dixon-Coles:** Dixon-Coles introduces a *negative* dependency (small negative correlation for low-score lines), while Karlis-Ntzoufras allows positive correlation. In practice, the estimated correlation at international level is close to zero, meaning the two teams' scores are nearly independent conditional on team quality — confirming that Maher's independence assumption is only a minor source of bias.

**Sources:** [ResearchGate abstract](https://www.researchgate.net/publication/227719079_Analysis_of_sports_data_using_bivariate_Poisson_models) · [Bivariate Weibull extension paper](https://arxiv.org/pdf/2307.02139)

#### Later Extensions
- **Koopman & Lit (2015):** Dynamic bivariate Poisson model with latent state-space attack/defense strengths evolving over time. Fitted to EPL. Available in *JRSS Series A* 178:167–186.
- **Boshnakov, Kharrat & McHale (2017):** Bivariate Weibull count model outperforms Poisson on calibration; captures over/under-dispersion. *International Journal of Forecasting* 33:458–466. DOI [10.1016/j.ijforecast.2016.11.006](https://doi.org/10.1016/j.ijforecast.2016.11.006)

---

### 2.2 Groll et al. Hybrid Random-Forest Line

This is the most empirically validated academic line for **international tournament** prediction specifically.

#### Groll, Schauberger & Tutz (2015) — WC 2014
**Citation:** *Journal of Quantitative Analysis in Sports* 11:97–115. DOI [10.1515/jqas-2014-0051](https://doi.org/10.1515/jqas-2014-0051)

Poisson regression with team-specific regularized (Group Lasso) attack and defense parameters plus country-level covariates (GDP per capita, population, FIFA rank, Transfermarkt market value). First systematic use of socioeconomic predictors for WC forecasting.

#### Groll, Ley, Schauberger & Van Eetvelde (2019) — Hybrid RF for WC 2018
**Citation:** "A hybrid random forest to predict soccer matches in international tournaments." *Journal of Quantitative Analysis in Sports* 15(4):271–287. DOI [10.1515/jqas-2018-0060](https://doi.org/10.1515/jqas-2018-0060) · [PDF via orbilu](https://orbilu.uni.lu/bitstream/10993/57865/1/A%20hybrid%20random%20forest%20to%20predict%20soccer%20matches%20in%20international%20tournaments.pdf) · [Semantic Scholar](https://www.semanticscholar.org/paper/A-hybrid-random-forest-to-predict-soccer-matches-in-Groll-Ley/6279514bcf6a19f77dbcbe1ec9f4d8ff731e75af)

**Core idea:** A two-stage approach:
1. Estimate per-team ability parameters using a bivariate Poisson ranking model on historical matches (with exponential time weighting, half-life ≈ 3 years).
2. Feed those ability parameters *as an additional covariate* into a **random forest** alongside ~30 team-level and country-level features.

**Features used:** Estimated ability from Poisson ranking model (the key hybrid input), Elo rating, FIFA rank, Transfermarkt market value of squad, GDP per capita, population, continental confederation membership, whether team is host nation, number of Champions League players, coach nationality.

**Validation:** WC 2002–2014 used as training (cross-tournament cross-validation); WC 2014 used as independent holdout (64 matches). Evaluation metric: **Ranked Probability Score (RPS)** — lower is better.

**Results (from paper):**
- The hybrid RF **clearly outperformed all other methods including bookmaker betting odds** on the WC 2014 holdout.
- Among individual methods: random forest > Poisson regression > ranking-only methods.
- The ability parameter from the Poisson ranking model was by far the most important feature in the RF (highest variable importance).

**Applied to WC 2018 prediction:** Spain was the model's top pick at 13.7% winning probability. Germany early exit was assigned only ~22% probability — identified in retrospect as the tournament's biggest surprise.

**Spearman correlation (Appendix C):** Correlation between estimated abilities and Elo rating = 0.94; both have lower correlation with FIFA rank (0.86 and 0.90 respectively), confirming Elo-type systems better match model-estimated true strength.

**Applied to WC 2022:** Training on WC 2002–2018 (5 tournaments); model calibrated on expected goals per team.

**Source:** Full paper downloaded and verified.

---

### 2.3 Zeileis / Bookmaker-Consensus Line

#### Leitner, Zeileis & Hornik (2010)
**Citation:** "Forecasting Sports Tournaments by Ratings of (Prob)Abilities: A Comparison for the EURO 2008." *International Journal of Forecasting* 26(3):471–481. DOI [10.1016/j.ijforecast.2009.10.001](https://doi.org/10.1016/j.ijforecast.2009.10.001)

**Bookmaker consensus methodology:** Odds from N bookmakers are each adjusted to remove the overround (the bookmakers' margin, averaging ~15.2% at the 2018 WC), converted to probabilities, and then averaged on the **log-odds (logit) scale** to form a consensus implied ability for each team. This consensus is better than any single bookmaker and better than most model-based forecasts at longer tournament horizons.

#### Zeileis & Colleagues — WC 2018
**Citation:** Working paper, Universität Innsbruck (2018). [ideas.repec.org/p/inn/wpaper/2018-09.html](https://ideas.repec.org/p/inn/wpaper/2018-09.html) · [zeileis.org/news/fifa2018](https://www.zeileis.org/news/fifa2018/)

Prediction based on bookmaker consensus with an inverse simulation to recover team abilities from tournament-winning odds. **Brazil** forecast at **16.6%** to win, followed by Germany (15.8%), Spain (12.5%).

#### Zeileis, Groll et al. — WC 2026
**Citation:** Zeileis, A., Groll, A., Hanekov, A., Hvattum, L.M., Michels, R., Schauberger, G., Sukhanova, E., Witte, S. (2026). "Football meets machine learning: Forecasting the 2026 FIFA World Cup." Published 2026-06-03. [zeileis.org/news/fifa2026](https://www.zeileis.org/news/fifa2026/)

This is the current state of the art from the leading academic group. A four-signal hybrid:

| Signal | Method | Description |
|---|---|---|
| Historic match abilities | Bivariate Poisson (independence assumed) + exponential weighting (8-year window) | Per-team attack/defense fixed effects; data from [martj42/international_results](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) |
| Bookmaker consensus abilities | Leitner-Zeileis-Hornik (2010); 24 bookmakers, logit-average | "Future expectation" signal; overround-corrected |
| Average player ratings | Plus-minus ratings ([Pantuso & Hvattum 2021](https://doi.org/10.1007/s11750-020-00584-9)) | Segment-level goal differential attributed to individual players |
| Average market values | Transfermarkt crowd-sourced values | Wisdom-of-the-crowd team quality proxy |

These four signals plus team/country covariates (FIFA rank, Elo, GDP, Champions League players) are fed into a **random forest** trained on all major Men's WCs and EUROs 2006–2024. Output: predicted expected goals per team per possible match, from which a bivariate Poisson (independence assumed) yields score probabilities. 100,000 tournament simulations give championship probabilities.

**WC 2026 championship probabilities (Zeileis et al., June 2026):**
- Spain: **14.5%**
- England: **12.4%**
- France: **12.4%**
- Germany: **11.2%**
- Portugal, Brazil, Argentina: further down (approx. 5-8% range)

**Notable finding vs. bookmakers:** Germany ranked 4th by the model vs. 7th by bookmakers; Brazil and Argentina ranked lower by model than by bookmakers.

---

### 2.4 Ley, Van de Wiele & Van Eetvelde (2019) — Ranking Methods Comparison

**Citation:** "Ranking Soccer Teams on the Basis of their Current Strength: A Comparison of Maximum Likelihood Approaches." *Statistical Modelling* 19:55–77. DOI [10.1177/1471082X18817650](https://doi.org/10.1177/1471082X18817650)

**Scope:** Systematic comparison of all major ranking methods for international football — Elo variants, FIFA rank (pre- and post-2018 reform), plus-minus ratings, Poisson-based ability estimates.

**Key finding:** Elo-type systems consistently outperform the old FIFA ranking method (2006–2018 SUM method was particularly poor) and are the best single predictor of international match outcomes. A 2009 comparative study of 8 methods found the Elo implementation had the highest predictive capability; the men's FIFA ranking (pre-2018) performed worst.

**Spearman correlation finding (verified in Groll 2019 Appendix C):** Estimated Poisson-model abilities and Elo rating correlate at r=0.94; both more predictive than FIFA rank (r=0.86 and r=0.90 respectively).

**Post-2018 FIFA reform:** FIFA moved to an Elo-inspired SUM method in August 2018. This narrowed but did not close the gap vs. Elo. A study on 115 WC knockout matches (1994–2022) found Elo correctly predicted the winner 73.9% of the time vs. FIFA's 68.7% (AUC: 0.775 vs. 0.695). The FIFA method differs from World Football Elo by: (a) ignoring goal margin, (b) treating penalty shootouts as wins/losses rather than draws, (c) using only the past 4 years of data.

**Source:** [ResearchGate PDF](https://scispace.com/pdf/the-predictive-power-of-ranking-systems-in-association-1h1ae578m4.pdf) · [FIFA Rankings vs ELO paper](https://www.researchgate.net/publication/406281676_FIFA_Rankings_vs_ELO_Ratings_Predictive_Validity_in_World_Cup_Knockout_Stages_1994-2022)

---

### 2.5 Newer Work (2023–2026)

#### EURO 2024 Combined ML (Groll, Zeileis et al., 2024)
**Citation:** "Modeling and Prediction of the UEFA EURO 2024 via Combined Statistical Learning Approaches." arXiv:2410.09068. [arxiv.org/abs/2410.09068](https://arxiv.org/abs/2410.09068) · [ResearchGate](https://www.researchgate.net/publication/384930244_Modeling_and_Prediction_of_the_UEFA_EURO_2024_via_Combined_Statistical_Learning_Approaches)

An ensemble of three ML models (generalized linear model, random forest, XGBoost) combined for EURO 2024 goal prediction. Key finding: the combined model **slightly outperformed** the single cforest (conditional random forest) on RPS. All single approaches slightly improved RPS vs. the earlier EURO 2020 analysis. Spain was correctly identified as favourite; Spain won the tournament (a model success).

#### Bayesian Dynamic Models (arXiv, 2025)
**Citation:** arXiv:2508.05891. [arxiv.org/html/2508.05891v1](https://arxiv.org/html/2508.05891v1)  
Bayesian weighted discrete-time dynamic models for association football prediction — extends the Maher-Dixon-Coles framework with formal Bayesian inference and time-varying parameters.

#### arXiv:2505.01902 (2026)
"From Players to Champions: A Generalizable Machine Learning Approach for Match Outcome Prediction with Insights from the FIFA World Cup." [arxiv.org/html/2505.01902v1](https://arxiv.org/html/2505.01902v1)  
Player-level features aggregated to match level; trained on WC data; shows squad-level metrics (e.g., average age, international caps) add signal beyond team ratings alone.

---

## 3. Kaggle & Data-Science Community

### WC 2018 Kaggle notebooks
Multiple notebooks appeared before and during WC 2018. Common approaches:
- **EDA + Poisson regression** on the martj42 historical dataset.
- **Elo rating as primary feature** for match outcome prediction.
- **Evaluation:** Accuracy of win/draw/loss prediction (~50–55% reported), log-loss rarely reported rigorously.

### WC 2022 Kaggle notebooks
Representative notebooks at [Kaggle FIFA World Cup 2022 collection](https://www.kaggle.com/code/scottsuk0306/fifa-world-cup-2022-group-stage-prediction):
- **Features commonly used:** Historical win rates, Elo ratings, FIFA rank, Transfermarkt squad market value, last 10 match form, average goals scored/conceded.
- **Models:** XGBoost, LightGBM, Random Forest, Poisson regression, simple Elo-based probability.
- **Deep learning attempts:** DNN-TensorFlow notebooks exist but show no consistent advantage over simpler models on small WC datasets.
- **Validation weakness:** A pervasive problem across community notebooks is **data leakage** — using post-tournament data, or not properly holding out the test tournament from training. Cross-tournament CV is rarely implemented.

### WC 2026 DataCamp Competition
**URL:** [datacamp.com/competitions/world-cup-prediction](https://www.datacamp.com/competitions/world-cup-prediction)  
Ongoing (started 2026). Scoring: exact scoreline earns maximum points; knockout matches carry multipliers. A representative strong approach (LightGBM on Elo + form features) achieved **validation log-loss ≈ 0.893 on multiclass 1X2**, test log-loss **≈ 0.873**, confirming that Elo alone leaves log-loss points on the table.

**Source:** [DataCamp competition page](https://www.datacamp.com/competitions/world-cup-prediction) · [Kaggle 2026 WC dataset](https://www.kaggle.com/datasets/rauzzanrambe/fifa-world-cup-2026-prediction-system)

### Nate Silver / Silver Bulletin — PELE Ratings & WC 2026
**Citation:** Silver, N. "PELE International Football Rankings." [natesilver.net/p/pele-international-football-rankings-soccer-ratings-projections](https://www.natesilver.net/p/pele-international-football-rankings-soccer-ratings-projections) · Methodology: [natesilver.net/p/pele-methodology](https://www.natesilver.net/p/pele-methodology) · WC 2026 forecast: [natesilver.net/p/world-cup-2026-odds-predictions](https://www.natesilver.net/p/world-cup-2026-odds-predictions)

**PELE = Predictive Elo with Lineup Equilibria.** Two ratings per team: PELE (overall quality) and Tilt (attacking vs. defensive tendency). Inputs: match results, Transfermarkt player values (offense/defense split by position), player ages, historical economic and geographic factors. 100,000-simulation Monte Carlo tournament model. **WC 2026:** Argentina and Spain essentially tied at the top of PELE before the tournament.

---

## 4. Notable Open-Source GitHub Projects

### 4.1 martj42/international_results
**URL:** [github.com/martj42/international_results](https://github.com/martj42/international_results)

The canonical open dataset. Contains **49,390+ results** of international men's football from 1872 to the present, including match date, teams, scores, tournament, city, country, neutral venue flag. Excludes Olympic Games, B-team matches, U-23 matches. The de facto training set for almost all academic and community models. Also mirrored by OpenFootball at [github.com/openfootball/internationals](https://github.com/openfootball/internationals).

**What it does well:** Comprehensive, regularly updated, clean, free. Essential.
**Limitation:** No match-level features beyond score and tournament type; no xG, no lineups, no match importance weights built in (those must be added).

### 4.2 openfootball/worldcup
**URL:** [github.com/openfootball/worldcup](https://github.com/openfootball/worldcup)

Structured Football.TXT-format WC data including 2026. Part of a broader ecosystem covering leagues, internationals. More structured but smaller in scope than martj42.

### 4.3 martineastwood/penaltyblog
**URLs:** [github.com/martineastwood/penaltyblog](https://github.com/martineastwood/penaltyblog) · [penaltyblog.readthedocs.io](https://penaltyblog.readthedocs.io/) · [PyPI](https://pypi.org/project/penaltyblog/)

Production-ready Python library for football analytics. **Implements:** Independent Poisson, Bivariate Poisson, Dixon-Coles (with ρ-correction and time decay), Bayesian hierarchical variants (via PyMC). Optimized with Cython for speed. Also includes Club Elo scraping and plus-minus rating tools.

**What it does well:** Turnkey Dixon-Coles and bivariate Poisson; good documentation; integrates with FBRef and Understat for data scraping; well-maintained (active as of 2025–2026).
**Limitation:** Focused on EPL and club football; international football requires custom data pipelines; limited support for tournament bracket simulation.

**Bonus resource:** A companion blog post at [penaltyblog.com](https://pena.lt/y/2025/05/01/better-metrics-for-football-forecasts-moving-beyond-the-ranked-probability-score/) (May 2025) argues *against* using RPS as the primary metric and in favour of the ignorance (log-loss) score — directly relevant to our metric choice.

### 4.4 Hicruben/world-cup-2026-prediction-model
**URL:** [github.com/Hicruben/world-cup-2026-prediction-model](https://github.com/Hicruben/world-cup-2026-prediction-model) · Live: [cup26matches.com](https://cup26matches.com)

Open-source WC 2026 model stack: **Elo → expected goals → Dixon-Coles bivariate Poisson → 50,000-simulation Monte Carlo bracket.** Key design choice: no ML black box, no scraped bookmaker odds — purely transparent statistical football mathematics. Updates in near-real-time after each match result. Includes an honest backtest.

**What it does well:** Clean, reproducible, well-documented end-to-end pipeline; good example of the classic stack.
**Limitation:** Elo-to-xG conversion is a heuristic step (not a trained mapping); no player-level features.

### 4.5 goaliqlab/world-cup-2026-predictor
**URL:** [github.com/goaliqlab/world-cup-2026-predictor](https://github.com/goaliqlab/world-cup-2026-predictor)

Elo ratings + XGBoost trained on 50,000+ historical international matches. Python pipeline: data prep → Elo computation → XGBoost classifier → Monte Carlo tournament simulation. Represents the hybrid Elo+ML pattern.

### 4.6 hjjbh1314/worldcup-predictor
**URL:** [github.com/hjjbh1314/worldcup-predictor](https://github.com/hjjbh1314/worldcup-predictor)

Transparent, leakage-free 1X2 predictor with an honest backtest reporting **~60% 1X2 prediction accuracy** — consistent with literature values for Elo-core models. Includes an optional bookmaker odds overlay.

### 4.7 0xNadr/wc2026
**URL:** [github.com/0xNadr/wc2026](https://github.com/0xNadr/wc2026)

Bayesian Monte Carlo forecaster for WC 2026. Uses Bayesian Poisson models rather than frequentist estimates; posterior uncertainty naturally propagates into simulation.

---

## 5. Rating Systems

### 5.1 World Football Elo Ratings (eloratings.net)
**Methodology:** Pioneered 1997 by Bob Runyan. Adapts chess Elo to football by incorporating:
- **Match importance K-factor:** K = 60 for WC finals, K = 20 for friendlies; intermediate values for qualifiers and tournaments.
- **Goal difference multiplier:** K scaled by ×1.0 for wins by 1, ×1.5 for wins by 2, ×1.75 for wins by 3, ×1.875 for wins by 4+.
- **Home advantage:** A fixed offset in the expected-score calculation.
- **Update:** Every official "A" international (244 recognized teams tracked).

**Key difference from FIFA 2018+ ranking:** Elo counts goal difference; FIFA SUM does not. Elo counts penalty-shootout results as draws (correct under expected-goal logic); FIFA counts them as wins/losses.

**Predictive accuracy:** Elo achieves the lowest binomial deviance (1.2634) and MSE (0.1271) vs. FIFA rank in systematic comparisons. On WC knockout stage prediction (1994–2022, n=115 matches): Elo correctly picked the winner 73.9% vs. FIFA's 68.7%; AUC 0.775 vs. 0.695. The pre-2018 FIFA SUM method was worst of all tested systems.

**Sources:** [Wikipedia: World Football Elo Ratings](https://en.wikipedia.org/wiki/World_Football_Elo_Ratings) · [Grokipedia](https://grokipedia.com/page/World_Football_Elo_Ratings) · [ResearchGate: FIFA Rankings vs ELO](https://www.researchgate.net/publication/406281676_FIFA_Rankings_vs_ELO_Ratings_Predictive_Validity_in_World_Cup_Knockout_Stages_1994-2022)

### 5.2 FIFA SUM Ranking (post-2018)
The 2018 reform replaced a (criticized) averaging method with an **Elo-inspired cumulative sum** approach. Teams accumulate points from matches in the past 4 years (weighted by recency and match importance). The reform genuinely improved predictive validity — the accuracy gap vs. Elo narrowed from 6.0 to 3.2 percentage points. However, ignoring goal margin and counting shootouts as decisive results are persistent methodological gaps.

### 5.3 Nate Silver PELE Rating
See Section 3. An Elo variant augmented by Transfermarkt player values and player age structure. Represents the current frontier for publicly available rating systems.

### 5.4 Verdict on Rating Systems
> **Recommendation:** Use Elo (ideally with goal-margin scaling and time-decay) as the primary ability proxy. Augment with squad market value (Transfermarkt). Avoid raw FIFA rank as a feature — it is dominated by Elo for predictive purposes.

---

## 6. Scoring Rules & Benchmark Values

### 6.1 Recommended Metrics

| Metric | Properties | Recommended use |
|---|---|---|
| **RPS (Ranked Probability Score)** | Non-local; sensitive to outcome ordering (home win "closer" to draw than away win) | Standard in football analytics literature; lower is better; used by Groll, Zeileis |
| **Ignorance Score (log-loss / cross-entropy)** | Local; only scores probability placed on actual outcome; strictly proper | Theoretically preferred in simulation studies (Constantinou & Fenton 2012, verified by arXiv:1908.08980); widely used in Kaggle |
| **Brier Score (mean squared error of probability vector)** | Non-local; insensitive to ordering | Less theoretically motivated than ignorance score but easy to interpret |
| **Poisson deviance** | For goal-count models | Preferred when the model outputs expected goals (λ) rather than 1X2 probabilities |
| **Accuracy (1X2)** | Simple but misleading | Use only as a secondary sanity check; ~50% trivially achievable |

**Theoretical debate (verified from source):** Constantinou & Fenton (2012) / arXiv:1908.08980 argue that the Ignorance Score (log-loss) outperforms both RPS and Brier score in simulation experiments in the context of football. The penaltyblog blog ([pena.lt/y/2025/05/01](https://pena.lt/y/2025/05/01/better-metrics-for-football-forecasts-moving-beyond-the-ranked-probability-score/)) echoes this view. The argument: sensitivity to ordering (RPS's claimed advantage) does not add information relevant to decision-making in a proper scoring context.

**Sources:** [arXiv:1908.08980](https://arxiv.org/abs/1908.08980) · [De Gruyter: Evaluating probabilistic forecasts](https://www.degruyterbrill.com/document/doi/10.1515/jqas-2019-0089/html) · [penaltyblog blog](https://pena.lt/y/2025/05/01/better-metrics-for-football-forecasts-moving-beyond-the-ranked-probability-score/)

### 6.2 Benchmark Values at World Cup Level

**Note:** Concrete published benchmark numbers specifically for WC 1X2 or score prediction are sparse. The following are gathered from multiple sources and should be treated as approximate:

| Metric | Approximate value | Context | Source/status |
|---|---|---|---|
| Bookmaker 1X2 log-loss (WC) | ~0.95–1.05 | Industry benchmark for 3-way markets at WC level; bookmakers near efficient | Approximated from community reports |
| LightGBM on Elo features — validation log-loss | **0.893** (validation) / **0.873** (test) | DataCamp WC 2026 competition, 1X2 multiclass | [DataCamp competition](https://www.datacamp.com/competitions/world-cup-prediction); verified |
| Groll 2019 hybrid RF vs. bookmakers | Outperforms bookmaker odds on WC 2014 holdout | RPS-based comparison; specific RPS numbers not publicly tabulated | [Groll et al. 2019](https://doi.org/10.1515/jqas-2018-0060); verified |
| hjjbh1314/worldcup-predictor 1X2 accuracy | **~60%** | Backtested on international matches, leakage-free | [GitHub repo](https://github.com/hjjbh1314/worldcup-predictor) |
| Elo-only WC knockout accuracy | **73.9%** (binary home/away) | 115 WC knockout matches 1994–2022 | [FIFA vs ELO paper](https://www.researchgate.net/publication/406281676_FIFA_Rankings_vs_ELO_Ratings_Predictive_Validity_in_World_Cup_Knockout_Stages_1994-2022) |
| Elo AUC (WC knockout) | **0.775** | Same study | Verified |

> **Practical target:** A well-calibrated Dixon-Coles or bivariate Poisson model with Elo-based team abilities should achieve 1X2 multiclass log-loss < 1.00 and ~52–55% accuracy on WC group-stage matches. Beating bookmaker log-loss (~0.95) is very hard; beating a naive Elo-only baseline (log-loss ~1.00) is the primary target.

---

## 7. Key Empirical Facts for Modelling

### 7.1 Average Goals Per Match at Recent World Cups

| Tournament | Goals | Matches | Goals/Match |
|---|---|---|---|
| WC 2014 (Brazil) | 171 | 64 | **2.67** |
| WC 2018 (Russia) | 169 | 64 | **2.64** |
| WC 2022 (Qatar) | 172 | 64 | **2.69** |
| WC group stage average (recent) | — | — | ~2.69 (group) / ~2.31 (knockout) |

**Implication for λ calibration:** The mean expected goals per team per WC match is approximately λ ≈ 1.33 (= 2.67/2). Group-stage matches average somewhat higher than knockouts, where teams play more defensively.

**Sources:** [football fandom wiki stats](https://football.fandom.com/wiki/2022_FIFA_World_Cup_statistics) · [Frontiers paper 2024](https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2024.1394621/full) · [footballhistory.org](https://www.footballhistory.org/world-cup/statistics.html)

### 7.2 Draw Rate

- At **neutral venues** (as all WC matches effectively are): draws occur in approximately **27–30%** of matches (vs. ~24% in home/away matches).
- In the WC 2018 group stage specifically: 8 out of 48 group matches drawn = **16.7%** (unusually low; draws are more common in knockout-pressured later group matches).
- The bivariate Poisson / Dixon-Coles low-score correction is specifically designed to correct for the systematic under-prediction of draws by the independent Poisson.

### 7.3 Host / Home Advantage

- **Home advantage in general football:** Home teams win ~61% of matches; draws ~24%. Home advantage contributes approximately +0.1 to expected goals scored and −0.2 to expected goals conceded (Dixon-Coles specification).
- **At World Cups:** WC matches are on neutral ground (or near-neutral for host nations). Being the host nation adds a meaningful but smaller boost. The USA (co-host 2026) is rated by Opta as 32.8% to top their group — slightly above what their pure Elo would suggest.
- **Neutral venue effect:** At neutral venues, win rate for the "nominal home" team drops toward 43%; draws rise toward 30% vs. 24% in home/away competition. In Poisson models, the home advantage dummy should be zeroed for WC matches (all neutral) and replaced with a much smaller "host-nation" indicator.

**Sources:** [Visitors Out paper (arXiv:2308.06279)](https://arxiv.org/pdf/2308.06279) · [Economics Observatory WC 2026 article](https://www.economicsobservatory.com/world-cup-2026-30-years-on-is-football-finally-coming-home)

### 7.4 Low-Scoring Dependence

The Dixon-Coles ρ-correction captures the empirical finding that (0-0), (1-0), (0-1), and (1-1) scorelines are *more* common than the independent Poisson predicts. This is a small but statistically significant effect that matters for:
- Correct-score betting and modelling.
- Calibrated draw probabilities (1X2).
- Tournament simulations where early group-stage eliminations hinge on fine probability differences.

Estimated ρ values in the literature are small and negative (ρ ≈ −0.1 to −0.15), implying a mild negative correlation between the two teams' scores conditional on λ values.

### 7.5 Predictive Value of Features

| Feature | Evidence of predictive value | Notes |
|---|---|---|
| **Elo rating (or bivariate Poisson ability estimate)** | Very strong; highest variable importance in Groll RF models | Best single predictor; Spearman ρ with model abilities = 0.94 |
| **Transfermarkt squad market value** | Strong; comparable to or better than Elo in some studies; confirmed by Groll, Zeileis, and FiveThirtyEight | Wisdom-of-the-crowd; updates continuously; strong for WC |
| **Plus-minus player ratings** | Moderate–strong; used in Zeileis 2026 hybrid | Requires segment-level match data; hard to compute from scratch |
| **Bookmaker odds (consensus)** | Very strong; near-efficient market; often top predictor | Available pre-match; need overround adjustment; blends all information |
| **FIFA rank** | Moderate (post-2018 reform); inferior to Elo | Useful as a covariate but dominated by Elo |
| **GDP per capita** | Weak–moderate; meaningful at tournament level for smaller nations | Country-level proxy for football infrastructure |
| **Champions League players count** | Weak–moderate | Proxy for squad quality at elite club level |
| **Host nation dummy** | Weak; meaningful but small effect (≈ 0.1–0.3 goals per game) | Use as an additive offset to λ |
| **Recent form (last N matches)** | Moderate; time-decay in Elo/Poisson abilities largely captures this | Additional form features improve LightGBM but modest effect |

**Sources:** [Groll et al. 2019](https://doi.org/10.1515/jqas-2018-0060) · [Transfermarkt defense](https://medium.com/@johncomonitski/a-defense-of-transfermarkt-market-values-in-football-analytics-c819aa954eb6) · [Zeileis 2026](https://www.zeileis.org/news/fifa2026/) · [Hybrid EURO 2020 paper](https://arxiv.org/pdf/2106.05799)

---

## 8. Pre-Tournament WC 2026 Forecasts

Summary of published pre-tournament championship winning probabilities:

| Team | Opta (25k sims) | Zeileis/Groll (100k sims) | Silver/PELE | Bookmakers (implied) |
|---|---|---|---|---|
| Spain | **16.1%** | **14.5%** | ~14-15% | ~17% (+500) |
| France | 13.0% | 12.4% | ~12% | ~16% (+475) |
| England | 11.2% | 12.4% | ~11% | ~12% (+700) |
| Argentina | 10.4% | ~8-9% | ~14-15% | ~10% |
| Germany | ~5.1% | **11.2%** | ~9% | ~8% (7th by bookmakers) |
| Brazil | 6.6% | ~6% | ~8% | ~8% |
| Portugal | 7.0% | ~6% | — | — |
| Netherlands | 3.6% | — | — | — |

> **Notable divergence:** Zeileis/Groll ranks Germany 4th at 11.2% while bookmakers put them 7th. Brazil and Argentina ranked higher by bookmakers than by the model. This divergence is methodologically interesting: the model weights historical match-form ability (bivariate Poisson with 8-year window) heavily, while bookmakers reflect current squad assessment and recent tournament form (including Argentina's 2022 win).

**Sources:** [Opta Analyst supercomputer (June 2026)](https://theanalyst.com/articles/who-will-win-2026-fifa-world-cup-predictions-opta-supercomputer) · [Zeileis.org WC 2026](https://www.zeileis.org/news/fifa2026/) · [Silver Bulletin PELE WC 2026](https://www.natesilver.net/p/world-cup-2026-odds-predictions) · [ESPN odds](https://www.espn.com/espn/betting/story/_/id/48386952/espn-soccer-futbol-world-cup-betting-odds-championship-groups)

---

## 9. Implications for Our Design

### 9.1 Recommended Model Family

**Primary recommendation: Bivariate Poisson / Dixon-Coles with Elo-based λ inputs, wrapped in Monte Carlo simulation.**

Rationale:
- Covers the full score distribution (not just 1X2), enabling correct-score outputs.
- The Dixon-Coles ρ-correction fixes the most significant calibration failure of naive Poisson (draw and low-score underestimation).
- Independent Poisson with good team-strength estimates is "good enough" for tournament-level simulation (Zeileis 2026 uses this assumption).
- If resources allow, the **Groll/Zeileis hybrid** (Elo abilities + Transfermarkt values + player ratings → random forest → predicted λ → bivariate Poisson → simulation) represents the current academic state-of-the-art.
- The **PELE approach** (Elo variant with player market value and Tilt) is the best public single-number rating and should inform our ability estimates.

### 9.2 Features Ranked by Evidence of Predictive Value

1. **Elo rating / bivariate Poisson ability estimate** — highest variable importance in all RF models; closest to "true" current strength; use with exponential time decay (half-life ≈ 3 years).
2. **Bookmaker consensus probability** — near-optimal prior; blends all public information; use overround-adjusted logit average from ≥5 bookmakers if available.
3. **Transfermarkt squad market value (log scale)** — strong signal, especially for quality difference between strong and weak WC nations; sum of squad values is the relevant aggregate.
4. **Plus-minus player ratings** — moderate-strong, requires segment-level data; include if data pipeline supports it.
5. **Host-nation dummy** — small but real effect; add as additive offset in log(λ) with estimated coefficient.
6. **FIFA rank (post-2018 SUM)** — useful as a covariate, dominated by Elo but adds modest robustness.
7. **Recent form (last 5–10 matches)** — partially captured by time-decay Elo; use rolling win rate or goals-differential as an additional feature for ML models.
8. **Number of Champions League squad players** — weak additional signal for elite-vs-non-elite discrimination.
9. **GDP per capita** — weak at WC level (all 48 teams are already large football nations), but useful for identifying developmental-stage teams.

### 9.3 Validation Scheme

**Critical requirement: cross-tournament cross-validation.**

- Training set: WC 2006, 2010, 2014, 2018, 2022 + EURO 2008, 2012, 2016, 2020, 2024 (12–15 major tournaments).
- Hold-out evaluation: Each tournament used as a test set in turn ("leave-one-tournament-out" CV).
- Never use the test tournament's data in feature computation (e.g., no Elo ratings updated with test-tournament results).
- Leakage risks: squad market values and FIFA rank must be frozen at the tournament start date, not the match date.

### 9.4 Target Metric Values to Beat

| Metric | Naive baseline | Elo-only | Target (state-of-art) |
|---|---|---|---|
| 1X2 accuracy | ~44% (always predict home/favorite) | ~55–60% | >60% on WC holdout |
| 1X2 multiclass log-loss | ~1.10 (uniform) | ~1.00 | <0.89 (DataCamp benchmark) |
| Elo WC knockout accuracy | — | 73.9% (binary) | Maintain or exceed |
| RPS (if used) | ~0.25–0.27 (WC group stage) | ~0.21–0.23 | <0.21 (near bookmaker level) |
| Poisson deviance on score prediction | — | — | Improve over independent Poisson baseline |

> **Primary metric recommendation:** Use **log-loss (ignorance score)** for 1X2 evaluation (based on arXiv:1908.08980 theoretical and simulation findings). Use **Poisson deviance** when evaluating expected-goals outputs. Report RPS as a secondary metric to maintain comparability with the Groll/Zeileis academic literature.

### 9.5 Data Sources to Use

| Resource | URL | Use |
|---|---|---|
| martj42/international_results | [github.com/martj42/international_results](https://github.com/martj42/international_results) | Primary training data (49k+ matches from 1872) |
| openfootball/worldcup | [github.com/openfootball/worldcup](https://github.com/openfootball/worldcup) | WC-specific structured data incl. 2026 |
| penaltyblog library | [github.com/martineastwood/penaltyblog](https://github.com/martineastwood/penaltyblog) | Dixon-Coles / bivariate Poisson fitting in Python |
| Transfermarkt | [transfermarkt.co.uk](https://www.transfermarkt.co.uk/) | Squad market values |
| eloratings.net | [eloratings.net](https://www.eloratings.net/) | Historical Elo ratings (or compute own from martj42 data) |
| Kaggle: historical Elo dataset | [2026 FIFA World Cup Historical Elo Ratings](https://www.kaggle.com/datasets/afonsofernandescruz/2026-fifa-world-cup-historical-elo-ratings) | Pre-computed Elo series for all WC 2026 teams |

---

## Sources Index

- [Maher 1982, Statistica Neerlandica](https://doi.org/10.1111/j.1467-9574.1982.tb00782.x)
- [Dixon & Coles 1997, JRSS-C](https://doi.org/10.1111/1467-9876.00065)
- [Karlis & Ntzoufras 2003, The Statistician](https://doi.org/10.1111/1467-9884.00366)
- [Groll, Schauberger & Tutz 2015, JQAS](https://doi.org/10.1515/jqas-2014-0051)
- [Groll, Ley, Schauberger & Van Eetvelde 2019, JQAS](https://doi.org/10.1515/jqas-2018-0060)
- [Groll 2019 paper PDF](https://orbilu.uni.lu/bitstream/10993/57865/1/A%20hybrid%20random%20forest%20to%20predict%20soccer%20matches%20in%20international%20tournaments.pdf)
- [Ley, Van de Wiele & Van Eetvelde 2019, Statistical Modelling](https://doi.org/10.1177/1471082X18817650)
- [Leitner, Zeileis & Hornik 2010, International Journal of Forecasting](https://doi.org/10.1016/j.ijforecast.2009.10.001)
- [Zeileis 2018 WC forecast](https://www.zeileis.org/news/fifa2018/)
- [Zeileis & Groll et al. 2026 WC forecast](https://www.zeileis.org/news/fifa2026/)
- [Zeileis 2026 on R-bloggers](https://www.r-bloggers.com/2026/06/football-meets-machine-learning-forecasting-the-2026-fifa-world-cup/)
- [EURO 2024 combined ML paper (arXiv:2410.09068)](https://arxiv.org/abs/2410.09068)
- [Scoring rules paper (arXiv:1908.08980)](https://arxiv.org/abs/1908.08980)
- [penaltyblog scoring rules blog (2025)](https://pena.lt/y/2025/05/01/better-metrics-for-football-forecasts-moving-beyond-the-ranked-probability-score/)
- [De Gruyter: Evaluating probabilistic forecasts (RPS)](https://www.degruyterbrill.com/document/doi/10.1515/jqas-2019-0089/html)
- [Opta Supercomputer WC 2026 predictions](https://theanalyst.com/articles/who-will-win-2026-fifa-world-cup-predictions-opta-supercomputer)
- [Nate Silver PELE methodology](https://www.natesilver.net/p/pele-methodology)
- [Nate Silver WC 2026 predictions](https://www.natesilver.net/p/world-cup-2026-odds-predictions)
- [FIFA Rankings vs ELO paper (ResearchGate)](https://www.researchgate.net/publication/406281676_FIFA_Rankings_vs_ELO_Ratings_Predictive_Validity_in_World_Cup_Knockout_Stages_1994-2022)
- [World Football Elo Ratings (Wikipedia)](https://en.wikipedia.org/wiki/World_Football_Elo_Ratings)
- [martj42/international_results](https://github.com/martj42/international_results)
- [penaltyblog GitHub](https://github.com/martineastwood/penaltyblog)
- [Hicruben/world-cup-2026-prediction-model](https://github.com/Hicruben/world-cup-2026-prediction-model)
- [goaliqlab/world-cup-2026-predictor](https://github.com/goaliqlab/world-cup-2026-predictor)
- [hjjbh1314/worldcup-predictor](https://github.com/hjjbh1314/worldcup-predictor)
- [DataCamp WC 2026 competition](https://www.datacamp.com/competitions/world-cup-prediction)
- [Kaggle WC 2022 prediction notebooks](https://www.kaggle.com/code/scottsuk0306/fifa-world-cup-2022-group-stage-prediction)
- [FIFA World Cup 2026 official fixtures](https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/scores-fixtures)
- [Frontiers: Comparison of goalscoring patterns 2018 vs 2022](https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2024.1394621/full)
- [openfootball/worldcup](https://github.com/openfootball/worldcup)
- [Hybrid EURO 2020 arxiv paper](https://arxiv.org/pdf/2106.05799)

---

*Report compiled by automated deep-research agent on 2026-06-11. All claimed numeric values cross-checked against primary sources where accessible.*
