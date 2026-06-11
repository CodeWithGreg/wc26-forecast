"""Publication-lean matplotlib visualisations for forecasts and diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from wcforecast.models.poisson_core import MatchForecast  # noqa: E402

_C = {"blue": "#2563eb", "red": "#dc2626", "grey": "#6b7280", "green": "#059669", "amber": "#d97706"}


def _save(fig, path: str | Path | None):
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def score_heatmap(f: MatchForecast, max_show: int = 5, path=None):
    """Joint scoreline probability heatmap with the modal score marked."""
    m = f.matrix[: max_show + 1, : max_show + 1]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(m * 100, cmap="Blues")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m[i, j] * 100:.1f}", ha="center", va="center",
                    color="white" if m[i, j] > m.max() * 0.6 else "#1f2937", fontsize=9)
    bi, bj, _ = f.most_likely_score()
    if bi <= max_show and bj <= max_show:
        ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, edgecolor=_C["red"], lw=2.5))
    ax.set_xlabel(f"{f.away_team} goals")
    ax.set_ylabel(f"{f.home_team} goals")
    ph, pd_, pa = f.p_outcomes
    ax.set_title(
        f"{f.home_team} vs {f.away_team} — λ {f.lam_home:.2f} : {f.lam_away:.2f}\n"
        f"P(H/D/A) = {ph:.0%} / {pd_:.0%} / {pa:.0%}", fontsize=11,
    )
    fig.colorbar(im, ax=ax, label="probability (%)", shrink=0.8)
    return _save(fig, path)


def backtest_metric_bars(agg: pd.DataFrame, metric: str = "log_loss_1x2", path=None):
    """Model comparison on one backtest metric (lower is better unless accuracy/coverage)."""
    order = agg.sort_values(metric, ascending=metric in ("accuracy", "coverage90_total_goals"))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    colors = [_C["green"] if m == "ensemble" else _C["blue"] if m in ("dixon_coles", "gbm") else _C["grey"]
              for m in order["model"]]
    ax.barh(order["model"], order[metric], color=colors)
    for y, v in enumerate(order[metric]):
        ax.text(v, y, f" {v:.4f}", va="center", fontsize=9)
    ax.set_xlabel(metric)
    ax.set_title(f"Leave-one-tournament-out backtest — {metric}")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, path)


def reliability_curve(detail: pd.DataFrame, n_bins: int = 8, path=None):
    """Reliability of ensemble home-win and draw probabilities."""
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    obs_h = (detail["home_score"] > detail["away_score"]).astype(float)
    obs_d = (detail["home_score"] == detail["away_score"]).astype(float)
    for p, obs, label, color in [
        (detail["p_home"], obs_h, "home win", _C["blue"]),
        (detail["p_draw"], obs_d, "draw", _C["amber"]),
    ]:
        bins = pd.qcut(p, n_bins, duplicates="drop")
        g = pd.DataFrame({"p": p, "o": obs}).groupby(bins, observed=True).mean()
        ax.plot(g["p"], g["o"], "o-", label=label, color=color)
    ax.plot([0, 1], [0, 1], "--", color=_C["grey"], lw=1)
    ax.set_xlabel("forecast probability")
    ax.set_ylabel("empirical frequency")
    ax.set_title("Reliability (backtest, ensemble)")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, path)


def pit_histogram(pit: np.ndarray, path=None):
    """Randomised PIT histogram for total-goals forecasts (flat = calibrated)."""
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.hist(pit, bins=10, range=(0, 1), color=_C["blue"], edgecolor="white", density=True)
    ax.axhline(1.0, color=_C["red"], ls="--", lw=1.2, label="perfect calibration")
    ax.set_xlabel("PIT")
    ax.set_title("Randomised PIT — total goals (backtest)")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, path)


def champion_bars(champ_probs: pd.Series, top: int = 15, path=None):
    s = champ_probs.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(6.8, 0.34 * len(s) + 1.2))
    ax.barh(s.index, s.to_numpy() * 100, color=_C["blue"])
    for y, v in enumerate(s.to_numpy() * 100):
        ax.text(v, y, f" {v:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("P(champion) %")
    ax.set_title("World Cup 2026 — championship probabilities")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, path)


def shap_waterfall(contrib: pd.DataFrame, title: str, top: int = 10, path=None):
    """Horizontal bar chart of per-feature contributions to log λ."""
    d = contrib.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(d) + 1.4))
    colors = [_C["green"] if c > 0 else _C["red"] for c in d["contribution"]]
    labels = [f"{f} = {v:.2f}" if isinstance(v, (int, float, np.floating)) else f"{f}={v}"
              for f, v in zip(d["feature"], d["value"])]
    ax.barh(labels, d["contribution"], xerr=d["contribution_sd"], color=colors, ecolor=_C["grey"])
    ax.axvline(0, color="#111", lw=0.8)
    ax.set_xlabel("contribution to log λ (mean ± bag sd)")
    ax.set_title(title, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, path)
