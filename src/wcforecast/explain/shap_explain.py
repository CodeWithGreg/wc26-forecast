"""SHAP feature contributions for the GBM expected-goals head.

LightGBM's Poisson objective predicts on the log scale, so SHAP values are
additive contributions to **log λ** — exactly the right scale for a
multiplicative goals model: a contribution of +0.20 multiplies expected
goals by e^0.20 ≈ ×1.22. Contributions are averaged over the bag (the bag
spread is itself a stability diagnostic).

The Dixon-Coles half of the ensemble is interpretable by construction (its
attack/defence abilities are printed by ``wcf abilities``); SHAP covers the
machine-learning half, giving the full ensemble a complete explanation
story.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wcforecast.models.gbm import GbmGoalModel


def _explainers(gbm: GbmGoalModel):
    import shap  # optional dependency (pip install wcforecast[explain])

    return [shap.TreeExplainer(m) for m in gbm.models_]


def match_contributions(gbm: GbmGoalModel, feature_row: pd.DataFrame) -> pd.DataFrame:
    """Per-feature contributions to log λ for one team-perspective row.

    Returns a DataFrame: feature, value, contribution (mean over bag),
    contribution_sd (bag spread), sorted by |contribution|.
    """
    X = feature_row[gbm.feature_names_]
    contribs = []
    for ex in _explainers(gbm):
        sv = ex.shap_values(X)
        contribs.append(np.asarray(sv).reshape(-1))
    c = np.vstack(contribs)
    out = pd.DataFrame({
        "feature": gbm.feature_names_,
        "value": X.iloc[0].to_numpy(),
        "contribution": c.mean(axis=0),
        "contribution_sd": c.std(axis=0),
    })
    base = float(np.mean([ex.expected_value for ex in _explainers(gbm)]))
    out.attrs["base_log_lambda"] = base
    return out.reindex(out["contribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def global_importance(gbm: GbmGoalModel, X_sample: pd.DataFrame, max_rows: int = 2000) -> pd.DataFrame:
    """Mean |SHAP| global importance over a sample of training rows."""
    X = X_sample[gbm.feature_names_].sample(min(max_rows, len(X_sample)), random_state=7)
    ex = _explainers(gbm)[0]
    sv = np.asarray(ex.shap_values(X))
    return (
        pd.DataFrame({"feature": gbm.feature_names_, "mean_abs_shap": np.abs(sv).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
