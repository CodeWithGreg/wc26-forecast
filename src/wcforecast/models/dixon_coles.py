"""Time-decayed Dixon-Coles bivariate Poisson model with Laplace posterior.

Model
-----
For match *m* between home team *h* and away team *a*::

    log λ_h = μ + att_h − def_a + γ · 1[non-neutral venue]
    log λ_a = μ + att_a − def_h

Goals are Poisson with the Dixon-Coles ``τ`` low-score adjustment (parameter
ρ). Matches are weighted by an exponential time decay
``w = 0.5^(Δt / half_life)`` (Dixon & Coles 1997), so abilities reflect
*current* strength.

Estimation
----------
1. **Poisson stage** — MAP estimate of (μ, γ, att, def) by weighted Poisson
   likelihood with a Gaussian (ridge) prior on att/def. The prior both
   regularises small-sample teams and resolves the additive identifiability
   of attack/defence (posterior mode is centred).
2. **ρ stage** — profile likelihood: ρ is fit by 1-D bounded optimisation
   given the Poisson-stage λ's. The coupling between ρ and the λ parameters
   is empirically negligible (ρ only perturbs the four low-score cells).

Uncertainty
-----------
A Laplace approximation at the MAP: the posterior covariance is the inverse
of the (sparse-assembled) Fisher information plus prior precision. Posterior
parameter draws propagate into λ draws and finally into the posterior
predictive scoreline matrix — i.e. epistemic uncertainty about team ability
is carried all the way into the match forecast, on top of aleatoric Poisson
noise. The ρ uncertainty is included via a normal approximation from the
numeric curvature of its profile likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize, minimize_scalar

from wcforecast.config import DixonColesConfig, get_settings
from wcforecast.models.poisson_core import MatchForecast, mixture_score_matrix, score_matrix


@dataclass
class DixonColesModel:
    config: DixonColesConfig = field(default_factory=lambda: get_settings().dc)
    rng_seed: int = 2026

    # learned state
    teams_: list[str] = field(default_factory=list, repr=False)
    mu_: float = 0.0
    gamma_: float = 0.0
    att_: np.ndarray = field(default=None, repr=False)
    def_: np.ndarray = field(default=None, repr=False)
    rho_: float = 0.0
    rho_se_: float = 0.0
    cov_: np.ndarray = field(default=None, repr=False)
    cutoff_: pd.Timestamp = None

    # ------------------------------------------------------------------ fit
    def fit(self, results: pd.DataFrame, cutoff: pd.Timestamp | str | None = None) -> DixonColesModel:
        """Fit on matches strictly before ``cutoff`` (default: end of data)."""
        cfg = self.config
        df = results.copy()
        cutoff = pd.Timestamp(cutoff) if cutoff is not None else df["date"].max() + pd.Timedelta(days=1)
        self.cutoff_ = cutoff
        df = df[df["date"] < cutoff]
        age = (cutoff - df["date"]).dt.days.to_numpy(dtype=float)
        df = df[age <= cfg.window_days].copy()
        age = age[age <= cfg.window_days]
        w = 0.5 ** (age / cfg.half_life_days)

        # team index: keep teams with enough weighted evidence
        cnt = pd.concat([
            pd.Series(w, index=df["home_team"].to_numpy()),
            pd.Series(w, index=df["away_team"].to_numpy()),
        ]).groupby(level=0).sum()
        keep = cnt[cnt >= cfg.min_team_matches].index
        mask = df["home_team"].isin(keep) & df["away_team"].isin(keep)
        df, w = df[mask], w[mask.to_numpy()]
        self.teams_ = sorted(keep)
        tidx = {t: i for i, t in enumerate(self.teams_)}
        T = len(self.teams_)

        h = df["home_team"].map(tidx).to_numpy()
        a = df["away_team"].map(tidx).to_numpy()
        xh = df["home_score"].to_numpy(dtype=float)
        xa = df["away_score"].to_numpy(dtype=float)
        home_ind = (~df["neutral"].to_numpy()).astype(float)

        D = 2 + 2 * T  # [mu, gamma, att..., def...]

        def unpack(theta):
            return theta[0], theta[1], theta[2 : 2 + T], theta[2 + T :]

        def lam(theta):
            mu, g, att, dfn = unpack(theta)
            lh = np.exp(mu + att[h] - dfn[a] + g * home_ind)
            la = np.exp(mu + att[a] - dfn[h])
            return lh, la

        ridge = cfg.ridge

        def nll(theta):
            mu, g, att, dfn = unpack(theta)
            lh, la = lam(theta)
            ll = w * (xh * np.log(lh) - lh + xa * np.log(la) - la)
            pen = 0.5 * ridge * (att @ att + dfn @ dfn)
            return -ll.sum() + pen

        def grad(theta):
            mu, g, att, dfn = unpack(theta)
            lh, la = lam(theta)
            rh = w * (lh - xh)  # d(-ll)/d(log lam_h) accumulated
            ra = w * (la - xa)
            g_mu = rh.sum() + ra.sum()
            g_g = (rh * home_ind).sum()
            g_att = np.bincount(h, rh, minlength=T) + np.bincount(a, ra, minlength=T) + ridge * att
            g_def = -np.bincount(a, rh, minlength=T) - np.bincount(h, ra, minlength=T) + ridge * dfn
            return np.concatenate([[g_mu, g_g], g_att, g_def])

        theta0 = np.zeros(D)
        theta0[0] = np.log(max(np.average((xh + xa) / 2, weights=w), 0.5))
        res = minimize(nll, theta0, jac=grad, method="L-BFGS-B",
                       options={"maxiter": 600, "ftol": 1e-10})
        self.mu_, self.gamma_, self.att_, self.def_ = unpack(res.x)

        # ---------------- Laplace covariance via sparse design -------------
        lh, la = lam(res.x)
        n = len(df)
        # home-goal events: d/d[mu]=1, d/d[gamma]=home_ind, +att[h], -def[a]
        r_h = np.arange(n)
        r_a = np.arange(n, 2 * n)
        data, ri, ci = [], [], []

        def add(rows_, cols_, vals_):
            ri.extend(np.asarray(rows_).tolist())
            ci.extend(np.asarray(cols_).tolist())
            data.extend(np.asarray(vals_, dtype=float).tolist())

        ones = np.ones(n)
        add(r_h, np.zeros(n, int), ones)             # mu
        add(r_h, np.ones(n, int), home_ind)          # gamma
        add(r_h, 2 + h, ones)                        # att_h
        add(r_h, 2 + T + a, -ones)                   # -def_a
        add(r_a, np.zeros(n, int), ones)             # mu
        add(r_a, 2 + a, ones)                        # att_a
        add(r_a, 2 + T + h, -ones)                   # -def_h
        A = sp.csr_matrix((data, (ri, ci)), shape=(2 * n, D))
        wl = np.concatenate([w * lh, w * la])
        H = (A.T @ sp.diags(wl) @ A).toarray()
        H[2:, 2:] += ridge * np.eye(2 * T)
        H[np.diag_indices_from(H)] += 1e-6
        self.cov_ = np.linalg.inv(H)

        # ---------------- rho by profile likelihood ------------------------
        is00 = (xh == 0) & (xa == 0)
        is01 = (xh == 0) & (xa == 1)
        is10 = (xh == 1) & (xa == 0)
        is11 = (xh == 1) & (xa == 1)

        def nll_rho(rho):
            tau = np.ones(n)
            tau[is00] = 1.0 - (lh * la * rho)[is00]
            tau[is01] = 1.0 + (lh * rho)[is01]
            tau[is10] = 1.0 + (la * rho)[is10]
            tau[is11] = 1.0 - rho
            tau = np.clip(tau, 1e-10, None)
            return -(w * np.log(tau)).sum()

        opt = minimize_scalar(nll_rho, bounds=(-0.25, 0.25), method="bounded")
        self.rho_ = float(opt.x)
        eps = 1e-4
        curv = (nll_rho(self.rho_ + eps) - 2 * nll_rho(self.rho_) + nll_rho(self.rho_ - eps)) / eps**2
        self.rho_se_ = float(1.0 / np.sqrt(max(curv, 1e-8)))
        return self

    # -------------------------------------------------------------- predict
    def _theta_draws(self, n_draws: int) -> np.ndarray:
        rng = np.random.default_rng(self.rng_seed)
        T = len(self.teams_)
        mean = np.concatenate([[self.mu_, self.gamma_], self.att_, self.def_])
        try:
            L = np.linalg.cholesky(self.cov_)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(self.cov_ + 1e-6 * np.eye(2 + 2 * T))
        z = rng.standard_normal((n_draws, len(mean)))
        return mean + z @ L.T

    def _ability(self, team: str) -> tuple[int | None, float, float]:
        if team in self.teams_:
            i = self.teams_.index(team)
            return i, self.att_[i], self.def_[i]
        return None, 0.0, 0.0  # prior-mean team (≈ average side)

    def predict_match(
        self, home: str, away: str, neutral: bool = True, n_draws: int | None = None
    ) -> MatchForecast:
        cfg = self.config
        n_draws = n_draws or cfg.n_posterior_draws
        ih, att_h, def_h = self._ability(home)
        ia, att_a, def_a = self._ability(away)
        hi = 0.0 if neutral else 1.0

        lam_h = float(np.exp(self.mu_ + att_h - def_a + self.gamma_ * hi))
        lam_a = float(np.exp(self.mu_ + att_a - def_h))

        draws = self._theta_draws(n_draws)
        T = len(self.teams_)
        mu_d, g_d = draws[:, 0], draws[:, 1]
        att_d = draws[:, 2 : 2 + T]
        def_d = draws[:, 2 + T :]
        rng = np.random.default_rng(self.rng_seed + 1)
        # unknown teams: prior N(0, 1/ridge)
        prior_sd = 1.0 / np.sqrt(cfg.ridge)
        ah = att_d[:, ih] if ih is not None else rng.normal(0, prior_sd, n_draws)
        dh = def_d[:, ih] if ih is not None else rng.normal(0, prior_sd, n_draws)
        aa = att_d[:, ia] if ia is not None else rng.normal(0, prior_sd, n_draws)
        da = def_d[:, ia] if ia is not None else rng.normal(0, prior_sd, n_draws)
        lh_d = np.exp(mu_d + ah - da + g_d * hi)
        la_d = np.exp(mu_d + aa - dh)

        m = mixture_score_matrix(lh_d, la_d, self.rho_, cfg.max_goals)
        return MatchForecast(home, away, lam_h, lam_a, m, lh_d, la_d)

    def predict_score_matrix(self, home: str, away: str, neutral: bool = True) -> np.ndarray:
        """Plug-in (MAP) scoreline matrix — used where speed matters."""
        _, att_h, def_h = self._ability(home)
        _, att_a, def_a = self._ability(away)
        hi = 0.0 if neutral else 1.0
        lam_h = np.exp(self.mu_ + att_h - def_a + self.gamma_ * hi)
        lam_a = np.exp(self.mu_ + att_a - def_h)
        return score_matrix(float(lam_h), float(lam_a), self.rho_, self.config.max_goals)

    # ---------------------------------------------------------------- intro
    def abilities(self) -> pd.DataFrame:
        """Posterior summary of team abilities (attack/defence ± 1 sd)."""
        T = len(self.teams_)
        sd = np.sqrt(np.diag(self.cov_))
        return pd.DataFrame({
            "team": self.teams_,
            "attack": self.att_,
            "attack_sd": sd[2 : 2 + T],
            "defence": self.def_,
            "defence_sd": sd[2 + T :],
            "strength": self.att_ + self.def_,
        }).sort_values("strength", ascending=False).reset_index(drop=True)

    def save(self, path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> DixonColesModel:
        return joblib.load(path)
