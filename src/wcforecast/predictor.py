"""High-level match predictor: the public forecasting API.

Wraps the fitted Dixon-Coles model, GBM bag and ensemble weight behind two
methods:

* :meth:`MatchPredictor.forecast` — full posterior-predictive forecast for
  one match (epistemic λ draws + Dixon-Coles-adjusted scoreline mixture).
* :meth:`MatchPredictor.plugin_matrix` — fast MAP/plug-in scoreline matrix
  (used inside the Monte-Carlo tournament simulation, where millions of
  matrix lookups are needed and parameter uncertainty is second-order for
  pairing frequencies).

Feature snapshot: features for hypothetical/future matches are frozen as of
the end of the training data (pre-tournament protocol, identical to the
backtest), so a forecast for the final made today reflects today's
information set — by design it does not imagine intermediate results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from wcforecast.config import get_settings
from wcforecast.features.build import build_fixture_features, build_training_table
from wcforecast.models.dixon_coles import DixonColesModel
from wcforecast.models.ensemble import EnsembleModel, pool_matrices
from wcforecast.models.gbm import GbmGoalModel
from wcforecast.models.poisson_core import MatchForecast, mixture_score_matrix


@dataclass
class MatchPredictor:
    dc: DixonColesModel
    gbm: GbmGoalModel
    weight_dc: float = 0.5
    results: pd.DataFrame = field(default=None, repr=False)  # history used for features
    _plugin_cache: dict = field(default_factory=dict, repr=False)
    _gbm_lambda_cache: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------ training
    @classmethod
    def fit(cls, results: pd.DataFrame, weight_dc: float | None = None,
            cutoff: pd.Timestamp | str | None = None, verbose: bool = True) -> MatchPredictor:
        """Fit both component models on ``results`` (strictly pre-``cutoff``)."""
        s = get_settings()
        if verbose:
            print(f"[fit] Dixon-Coles on {len(results)} matches…")
        dc = DixonColesModel().fit(results, cutoff=cutoff)
        if verbose:
            print(f"[fit] ρ = {dc.rho_:+.4f} (±{dc.rho_se_:.4f}), home adv γ = {dc.gamma_:+.3f}")
            print("[fit] LightGBM Poisson bag…")
        table = build_training_table(results if cutoff is None else results[results["date"] < pd.Timestamp(cutoff)])
        gbm = GbmGoalModel().fit(table, cutoff=cutoff)
        gbm.rho_ = dc.rho_
        if weight_dc is None:
            wfile = s.artifacts_dir / "ensemble_weight.json"
            if wfile.exists():
                import json

                weight_dc = float(json.loads(wfile.read_text())["weight_dc"])
            else:
                weight_dc = s.ensemble_weight_dc
        return cls(dc=dc, gbm=gbm, weight_dc=weight_dc, results=results)

    # ---------------------------------------------------------- persistence
    def save(self, directory: Path | None = None) -> Path:
        d = directory or get_settings().artifacts_dir
        joblib.dump(
            {"dc": self.dc, "gbm": self.gbm, "weight_dc": self.weight_dc, "results": self.results},
            d / "predictor.joblib",
        )
        return d / "predictor.joblib"

    @classmethod
    def load(cls, directory: Path | None = None) -> MatchPredictor:
        d = directory or get_settings().artifacts_dir
        blob = joblib.load(d / "predictor.joblib")
        return cls(dc=blob["dc"], gbm=blob["gbm"], weight_dc=blob["weight_dc"], results=blob["results"])

    # ------------------------------------------------------------- features
    def _team_state(self):
        """Cached (EloEngine, per-team form state) replay over history."""
        if "team_state" not in self._gbm_lambda_cache:
            from wcforecast.features.build import team_state

            self._gbm_lambda_cache["team_state"] = team_state(self.results)
        return self._gbm_lambda_cache["team_state"]

    def _fixture_features(self, home: str, away: str, neutral: bool,
                          date: str | pd.Timestamp | None, importance: str) -> pd.DataFrame:
        date = pd.Timestamp(date) if date is not None else self.results["date"].max() + pd.Timedelta(days=14)
        fx = pd.DataFrame({
            "date": [date], "home_team": [home], "away_team": [away], "neutral": [neutral],
        })
        return build_fixture_features(self.results, fx, importance, state=self._team_state())

    def _gbm_lambdas(self, home: str, away: str, neutral: bool,
                     date, importance: str) -> tuple[np.ndarray, np.ndarray]:
        key = (home, away, neutral, importance)
        if key not in self._gbm_lambda_cache:
            feats = self._fixture_features(home, away, neutral, date, importance)
            side = self.gbm.predict_fixends(feats)
            bag_cols = [c for c in side.columns if c.startswith("lam_b")]
            lam_h = side.iloc[0][bag_cols].to_numpy(dtype=float)
            lam_a = side.iloc[1][bag_cols].to_numpy(dtype=float)
            self._gbm_lambda_cache[key] = (lam_h, lam_a)
        return self._gbm_lambda_cache[key]

    # ------------------------------------------------------------- forecast
    def forecast(self, home: str, away: str, neutral: bool = True,
                 date: str | pd.Timestamp | None = None,
                 importance: str = "world_cup") -> MatchForecast:
        """Full ensemble forecast with uncertainty for one match."""
        f_dc = self.dc.predict_match(home, away, neutral)
        lam_h_g, lam_a_g = self._gbm_lambdas(home, away, neutral, date, importance)
        m_gbm = mixture_score_matrix(lam_h_g, lam_a_g, self.dc.rho_, self.dc.config.max_goals)
        f_gbm = MatchForecast(home, away, float(lam_h_g.mean()), float(lam_a_g.mean()),
                              m_gbm, lam_h_g, lam_a_g)
        return EnsembleModel(weight_dc=self.weight_dc).combine(f_dc, f_gbm)

    def plugin_matrix(self, home: str, away: str, neutral: bool = True,
                      importance: str = "world_cup") -> np.ndarray:
        """Fast plug-in ensemble matrix (cached) for simulation loops."""
        key = (home, away, neutral, importance)
        if key not in self._plugin_cache:
            m_dc = self.dc.predict_score_matrix(home, away, neutral)
            lam_h_g, lam_a_g = self._gbm_lambdas(home, away, neutral, None, importance)
            from wcforecast.models.poisson_core import score_matrix

            m_gbm = score_matrix(float(lam_h_g.mean()), float(lam_a_g.mean()),
                                 self.dc.rho_, self.dc.config.max_goals)
            self._plugin_cache[key] = pool_matrices(m_dc, m_gbm, self.weight_dc)
        return self._plugin_cache[key]
