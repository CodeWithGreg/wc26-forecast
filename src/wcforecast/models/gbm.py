"""Gradient-boosted Poisson goal model (LightGBM).

Each match contributes two team-perspective rows (team vs opponent), and the
model learns ``E[goals | features]`` with a Poisson objective, so its output
is directly an expected-goals rate λ that plugs into the shared scoreline
machinery. Features are the leak-free set from ``features.build``: Elo
levels/difference, rolling scoring form, rest days, venue and importance.

Uncertainty: a bag of ``n_bags`` models (different seeds + row subsampling,
fit on bootstrap-weighted samples) yields a spread of λ per prediction — a
pragmatic epistemic-uncertainty estimate in the spirit of bagged ensembles
(comparable in role to the Dixon-Coles Laplace draws, and combined with them
in the ensemble). Aleatoric scoreline noise stays Poisson, with the
Dixon-Coles ρ correction shared from the fitted statistical model.

Why this complements Dixon-Coles: the GBM exploits covariates the ability
model cannot see (recent form beyond the decay window, rest, importance
interactions), which is exactly the hybrid design that won the model
comparisons in Groll et al. (2019) and Zeileis et al. (2026) — see
BENCHMARK.md §2.2/§2.5.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from wcforecast.config import GbmConfig, get_settings
from wcforecast.features.build import FEATURE_COLUMNS, to_side_view
from wcforecast.models.poisson_core import MatchForecast, mixture_score_matrix


@dataclass
class GbmGoalModel:
    config: GbmConfig = field(default_factory=lambda: get_settings().gbm)
    rng_seed: int = 2026
    sample_half_life_days: float = 4380.0  # mild 12-year decay on training rows

    models_: list = field(default_factory=list, repr=False)
    rho_: float = 0.0  # injected from the Dixon-Coles fit
    best_iteration_: int = 0
    feature_names_: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))

    def fit(self, training_table: pd.DataFrame, cutoff: pd.Timestamp | str | None = None) -> "GbmGoalModel":
        cfg = self.config
        df = training_table
        cutoff = pd.Timestamp(cutoff) if cutoff is not None else df["date"].max() + pd.Timedelta(days=1)
        df = df[df["date"] < cutoff]
        side = to_side_view(df.reset_index(drop=True))
        side = side.dropna(subset=["goals"])

        age_days = (cutoff - side["date"]).dt.days.to_numpy(dtype=float)
        weights = 0.5 ** (age_days / self.sample_half_life_days)

        # time-ordered early-stopping split: last `validation_years` years
        val_start = cutoff - pd.DateOffset(years=cfg.validation_years)
        is_val = side["date"] >= val_start
        X, y = side[self.feature_names_], side["goals"].astype(float)

        params = dict(
            objective="poisson",
            num_leaves=cfg.num_leaves,
            learning_rate=cfg.learning_rate,
            n_estimators=cfg.n_estimators,
            min_child_samples=cfg.min_child_samples,
            subsample=cfg.subsample,
            subsample_freq=1,
            colsample_bytree=cfg.colsample,
            verbose=-1,
        )

        # 1) determine best_iteration on the time-ordered split
        probe = lgb.LGBMRegressor(**params, random_state=self.rng_seed)
        probe.fit(
            X[~is_val], y[~is_val],
            sample_weight=weights[~is_val.to_numpy()],
            eval_set=[(X[is_val], y[is_val])],
            eval_sample_weight=[weights[is_val.to_numpy()]],
            callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
        )
        self.best_iteration_ = int(probe.best_iteration_ or cfg.n_estimators)

        # 2) refit the bag on the full pre-cutoff sample at that size
        rng = np.random.default_rng(self.rng_seed)
        self.models_ = []
        for b in range(cfg.n_bags):
            boot = rng.exponential(1.0, size=len(side))  # Bayesian-bootstrap row weights
            m = lgb.LGBMRegressor(
                **{**params, "n_estimators": self.best_iteration_},
                random_state=self.rng_seed + 17 * b,
            )
            m.fit(X, y, sample_weight=weights * boot)
            self.models_.append(m)
        return self

    # ------------------------------------------------------------- predict
    def _lambda_draws(self, feature_rows: pd.DataFrame) -> np.ndarray:
        """(n_bags, n_rows) λ predictions."""
        X = feature_rows[self.feature_names_]
        return np.vstack([m.predict(X) for m in self.models_])

    def predict_fixends(self, fixture_features: pd.DataFrame) -> pd.DataFrame:
        """λ draws for both sides of each fixture row (match-level table in,
        side-level λ summary out)."""
        side = to_side_view(fixture_features.assign(home_score=np.nan, away_score=np.nan), with_target=False)
        lam = self._lambda_draws(side)
        side = side[["match_id", "team", "opp"]].copy()
        side["lam_mean"] = lam.mean(axis=0)
        for b in range(lam.shape[0]):
            side[f"lam_b{b}"] = lam[b]
        return side

    def predict_match_from_features(
        self, row_home: pd.Series, row_away: pd.Series, home: str, away: str, max_goals: int = 10
    ) -> MatchForecast:
        lam_h_d = np.array([m.predict(row_home[self.feature_names_].to_frame().T)[0] for m in self.models_])
        lam_a_d = np.array([m.predict(row_away[self.feature_names_].to_frame().T)[0] for m in self.models_])
        m = mixture_score_matrix(lam_h_d, lam_a_d, self.rho_, max_goals)
        return MatchForecast(home, away, float(lam_h_d.mean()), float(lam_a_d.mean()), m, lam_h_d, lam_a_d)

    def save(self, path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "GbmGoalModel":
        return joblib.load(path)
