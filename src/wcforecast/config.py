"""Central configuration.

Every tunable of the pipeline lives here so that experiments are reproducible
and the CLI can expose overrides in one place. Defaults follow the evidence
collected in BENCHMARK.md (e.g. Elo K-factors from eloratings.net, a ~3-year
Dixon-Coles half-life as in Dixon & Coles 1997 and Groll et al. 2019).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _repo_root() -> Path:
    """Locate the repository root (directory containing ``data/``).

    Falls back to the current working directory, which allows the installed
    package to operate on a user-provided workspace.
    """
    env = os.environ.get("WCF_ROOT")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "data" / "wc2026_groups.csv").exists():
            return parent
    return Path.cwd()


@dataclass(frozen=True)
class EloConfig:
    """World-Football-Elo style rating engine parameters."""

    initial_rating: float = 1500.0
    home_advantage: float = 80.0  # Elo points added to a non-neutral home side
    # K-factor by match-importance bucket (eloratings.net convention)
    k_world_cup: float = 60.0
    k_continental_final: float = 50.0
    k_qualifier: float = 40.0
    k_nations_league: float = 30.0
    k_friendly: float = 20.0
    # Goal-difference multipliers: 1, 1.5, 1.75, then +1/8 per extra goal
    gd_multipliers: tuple[float, float, float] = (1.0, 1.5, 1.75)
    gd_extra_step: float = 0.125


@dataclass(frozen=True)
class DixonColesConfig:
    """Time-decayed Dixon-Coles bivariate Poisson configuration."""

    half_life_days: float = 1095.0  # ~3 years, literature standard
    window_days: float = 4380.0  # ignore matches older than 12 years at fit time
    min_team_matches: int = 5  # teams below this get prior (zero) abilities
    ridge: float = 5.0  # Gaussian prior precision on attack/defence (MAP shrinkage)
    max_goals: int = 10  # scoreline grid truncation (P(G>10) ~ 1e-6 at WC level)
    n_posterior_draws: int = 400  # Laplace posterior draws for epistemic UQ


@dataclass(frozen=True)
class GbmConfig:
    """LightGBM Poisson goal model configuration."""

    n_bags: int = 8  # seed-bagged models -> epistemic spread of lambda
    num_leaves: int = 31
    learning_rate: float = 0.03
    n_estimators: int = 900
    min_child_samples: int = 60
    subsample: float = 0.8
    colsample: float = 0.8
    early_stopping_rounds: int = 80
    validation_years: int = 4  # most-recent years held out for early stopping


@dataclass(frozen=True)
class Settings:
    """Aggregated package settings and canonical paths."""

    root: Path = field(default_factory=_repo_root)
    elo: EloConfig = field(default_factory=EloConfig)
    dc: DixonColesConfig = field(default_factory=DixonColesConfig)
    gbm: GbmConfig = field(default_factory=GbmConfig)

    train_start: str = "1980-01-01"  # ignore pre-1980 football entirely
    ensemble_weight_dc: float = 0.5  # default log-pool weight, refit by backtest
    rng_seed: int = 2026

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def artifacts_dir(self) -> Path:
        d = self.root / "artifacts"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def predictions_dir(self) -> Path:
        d = self.root / "predictions"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def reports_dir(self) -> Path:
        d = self.root / "reports"
        d.mkdir(parents=True, exist_ok=True)
        return d


_SETTINGS: Settings | None = None


def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` singleton."""
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = Settings()
    return _SETTINGS


#: Mapping from raw tournament names (martj42 dataset) to importance buckets.
#: Anything not matched is treated as a generic competitive match ("other").
TOURNAMENT_BUCKETS: dict[str, str] = {
    "FIFA World Cup": "world_cup",
    "FIFA World Cup qualification": "qualifier",
    "UEFA Euro": "continental",
    "Copa América": "continental",
    "African Cup of Nations": "continental",
    "AFC Asian Cup": "continental",
    "Gold Cup": "continental",
    "UEFA Euro qualification": "qualifier",
    "Copa América qualification": "qualifier",
    "African Cup of Nations qualification": "qualifier",
    "AFC Asian Cup qualification": "qualifier",
    "Gold Cup qualification": "qualifier",
    "UEFA Nations League": "nations",
    "CONCACAF Nations League": "nations",
    "Friendly": "friendly",
    "Confederations Cup": "continental",
}

IMPORTANCE_ORDER = ["friendly", "nations", "other", "qualifier", "continental", "world_cup"]
