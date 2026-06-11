"""``wcf`` — the wcforecast command line.

Typical session::

    wcf data refresh          # update the results snapshot (optional)
    wcf train                 # fit Dixon-Coles + GBM, save artifacts
    wcf backtest              # leave-one-tournament-out evaluation
    wcf predict fixtures      # forecast all 72 scheduled group matches
    wcf predict match France Brazil --stage FINAL
    wcf simulate --sims 20000 # championship & stage probabilities
    wcf predict slot 104      # forecast the FINAL before participants known
    wcf explain France Brazil # SHAP feature contributions
    wcf mpp picks             # optimal Mon-Petit-Prono picks + X2 advice
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from wcforecast.config import get_settings

app = typer.Typer(add_completion=False, rich_markup_mode="rich", no_args_is_help=True)
data_app = typer.Typer(no_args_is_help=True, help="Dataset management.")
predict_app = typer.Typer(no_args_is_help=True, help="Match / fixture / slot forecasts.")
app.add_typer(data_app, name="data")
app.add_typer(predict_app, name="predict")

console = Console()


# ---------------------------------------------------------------- helpers
def _load_predictor():
    from wcforecast.predictor import MatchPredictor

    s = get_settings()
    path = s.artifacts_dir / "predictor.joblib"
    if not path.exists():
        console.print("[red]No trained model found — run [bold]wcf train[/bold] first.[/red]")
        raise typer.Exit(1)
    return MatchPredictor.load()


def _forecast_row(f, date="", stage="") -> dict:
    ph, pdr, pa = f.p_outcomes
    i, j, p = f.most_likely_score()
    lo_h, hi_h = f.goals_interval("home")
    lo_a, hi_a = f.goals_interval("away")
    lam_h_lo, lam_h_hi = f.lam_interval("home")
    lam_a_lo, lam_a_hi = f.lam_interval("away")
    return {
        "date": date, "stage": stage,
        "home_team": f.home_team, "away_team": f.away_team,
        "lam_home": round(f.lam_home, 3), "lam_away": round(f.lam_away, 3),
        "lam_home_ci90": f"[{lam_h_lo:.2f},{lam_h_hi:.2f}]",
        "lam_away_ci90": f"[{lam_a_lo:.2f},{lam_a_hi:.2f}]",
        "p_home": round(ph, 4), "p_draw": round(pdr, 4), "p_away": round(pa, 4),
        "most_likely_score": f"{i}-{j}", "p_most_likely": round(p, 4),
        "goals_home_90": f"{lo_h}-{hi_h}", "goals_away_90": f"{lo_a}-{hi_a}",
    }


def _print_forecasts(rows: list[dict], title: str):
    t = Table(title=title, header_style="bold cyan")
    for col in ["date", "match", "λh", "λa", "P(H)", "P(D)", "P(A)", "score*", "p*", "90% goals"]:
        t.add_column(col, justify="right" if col not in ("date", "match") else "left")
    for r in rows:
        t.add_row(
            str(r["date"])[:10], f"{r['home_team']} – {r['away_team']}",
            f"{r['lam_home']:.2f}", f"{r['lam_away']:.2f}",
            f"{r['p_home']:.0%}", f"{r['p_draw']:.0%}", f"{r['p_away']:.0%}",
            r["most_likely_score"], f"{r['p_most_likely']:.1%}",
            f"{r['goals_home_90']} / {r['goals_away_90']}",
        )
    console.print(t)


# ------------------------------------------------------------------- data
@data_app.command("refresh")
def data_refresh():
    """Update the bundled results snapshot from the upstream dataset."""
    from wcforecast.data.results import refresh_snapshot

    path = refresh_snapshot()
    console.print(f"[green]Snapshot refreshed →[/green] {path}")


@data_app.command("info")
def data_info():
    """Show dataset coverage."""
    from wcforecast.data import load_group_fixtures, load_results

    r = load_results()
    fx = load_group_fixtures()
    console.print(f"Played matches: [bold]{len(r)}[/bold] ({r['date'].min():%Y-%m-%d} → {r['date'].max():%Y-%m-%d})")
    console.print(f"WC26 group fixtures: [bold]{len(fx)}[/bold] ({fx['date'].min():%Y-%m-%d} → {fx['date'].max():%Y-%m-%d})")


# ------------------------------------------------------------------ train
@app.command()
def train(
    cutoff: str = typer.Option(None, help="Fit on matches strictly before this date (default: all)."),
):
    """Fit the Dixon-Coles + GBM ensemble and save artifacts."""
    from wcforecast.data import load_results
    from wcforecast.predictor import MatchPredictor

    results = load_results()
    pred = MatchPredictor.fit(results, cutoff=cutoff)
    path = pred.save()
    console.print(f"[green]Saved →[/green] {path} (pool weight on Dixon-Coles: {pred.weight_dc:.2f})")


# --------------------------------------------------------------- backtest
@app.command()
def backtest(
    copa: bool = typer.Option(False, help="Also include Copa América 2021/2024."),
    quick: bool = typer.Option(False, help="Reduced bags/draws (CI smoke-test mode)."),
):
    """Leave-one-tournament-out backtest (WC 2006–2022, EURO 2008–2024)."""
    from wcforecast.data import load_results
    from wcforecast.eval.backtest import aggregate_summary, run_backtest

    results = load_results()
    summary, detail, w = run_backtest(results, include_copa=copa, quick=quick)
    agg = aggregate_summary(summary)

    t = Table(title="Backtest — match-weighted averages", header_style="bold cyan")
    for col in agg.columns:
        t.add_column(col, justify="right")
    for _, r in agg.iterrows():
        style = "bold green" if r["model"] == "ensemble" else None
        t.add_row(*[f"{v:.4f}" if isinstance(v, float) else str(v) for v in r], style=style)
    console.print(t)
    console.print(f"Production pool weight (Dixon-Coles share): [bold]{w:.2f}[/bold]")
    s = get_settings()
    console.print(f"[green]Reports →[/green] {s.reports_dir}/backtest_metrics.csv, backtest_by_match.csv")


# ---------------------------------------------------------------- predict
@predict_app.command("fixtures")
def predict_fixtures(
    out: Path = typer.Option(None, help="Output CSV (default predictions/group_stage.csv)."),
):
    """Forecast all 72 scheduled WC 2026 group-stage matches."""
    from wcforecast.data import load_group_fixtures

    pred = _load_predictor()
    fx = load_group_fixtures()
    rows = []
    with console.status("[cyan]Forecasting fixtures…"):
        for r in fx.itertuples(index=False):
            f = pred.forecast(r.home_team, r.away_team, bool(r.neutral), r.date)
            rows.append(_forecast_row(f, r.date, f"group {r.group}"))
    df = pd.DataFrame(rows)
    out = out or get_settings().predictions_dir / "group_stage.csv"
    df.to_csv(out, index=False)
    _print_forecasts(rows[:20], "WC 2026 group stage — first 20 forecasts (full table in CSV)")
    console.print(f"[green]Saved {len(df)} forecasts →[/green] {out}")


@predict_app.command("match")
def predict_match(
    home: str = typer.Argument(..., help="Home/first team, e.g. 'France'."),
    away: str = typer.Argument(..., help="Away/second team."),
    neutral: bool = typer.Option(True, help="Neutral venue (host nations: use --no-neutral)."),
    date: str = typer.Option(None, help="Match date (affects rest-day features)."),
    stage: str = typer.Option("world_cup", help="Importance bucket."),
    heatmap: Path = typer.Option(None, help="Save a scoreline heatmap PNG here."),
):
    """Forecast a single (possibly hypothetical) match."""
    pred = _load_predictor()
    f = pred.forecast(home, away, neutral, date, stage)
    row = _forecast_row(f, date or "", stage)
    _print_forecasts([row], f"{home} vs {away}")
    console.print(
        f"λ 90% credible intervals — {home}: {row['lam_home_ci90']}, {away}: {row['lam_away_ci90']} "
        "(epistemic, from posterior/bag draws)"
    )
    from wcforecast.models.poisson_core import top_scorelines

    tops = top_scorelines(f.matrix, 5)
    console.print("Top scorelines: " + ", ".join(f"{i}-{j} ({p:.1%})" for i, j, p in tops))
    if heatmap:
        from wcforecast.viz import score_heatmap

        score_heatmap(f, path=heatmap)
        console.print(f"[green]Heatmap →[/green] {heatmap}")


@predict_app.command("slot")
def predict_slot(
    match_no: int = typer.Argument(..., help="Bracket match number 73–104 (104 = final)."),
    sims: int = typer.Option(10000, help="Monte-Carlo simulations."),
    top_k: int = typer.Option(8, help="Pairings kept in the mixture forecast."),
):
    """Forecast a knockout match whose participants are not yet known.

    Example: [bold]wcf predict slot 104[/bold] forecasts the FINAL today —
    pairing probabilities plus the pairing-weighted scoreline forecast.
    """
    from wcforecast.models.poisson_core import outcome_probs, top_scorelines
    from wcforecast.sim.tournament import slot_forecast

    pred = _load_predictor()
    sim = _run_or_load_sim(pred, sims)
    res = slot_forecast(sim, match_no, lambda h, a, n: pred.forecast(h, a, n), top_k)

    t = Table(title=f"Match {match_no} — most likely pairings", header_style="bold cyan")
    t.add_column("pairing")
    t.add_column("probability", justify="right")
    for r in res["pairings"].itertuples(index=False):
        t.add_row(f"{r.home} – {r.away}", f"{r.prob:.1%}")
    console.print(t)
    ph, pdr, pa = outcome_probs(res["matrix"])
    console.print(
        f"Mixture forecast: λ = {res['lam_home']:.2f} : {res['lam_away']:.2f} | "
        f"P(slot-home/draw/slot-away) = {ph:.0%}/{pdr:.0%}/{pa:.0%}"
    )
    tops = top_scorelines(res["matrix"], 5)
    console.print("Top scorelines (unconditional): " + ", ".join(f"{i}-{j} ({p:.1%})" for i, j, p in tops))


# --------------------------------------------------------------- simulate
def _run_or_load_sim(pred, sims: int):
    from wcforecast.data import load_bracket, load_group_fixtures, load_groups
    from wcforecast.sim import TournamentSimulator

    s = get_settings()
    cache = s.artifacts_dir / f"sim_{sims}.joblib"
    if cache.exists():
        return joblib.load(cache)
    simulator = TournamentSimulator(
        predict_matrix=lambda h, a, n: pred.plugin_matrix(h, a, n),
        groups=load_groups(), fixtures=load_group_fixtures(), bracket=load_bracket(),
    )
    with console.status(f"[cyan]Simulating tournament ×{sims}…"):
        res = simulator.run(sims)
    joblib.dump(res, cache)
    return res


@app.command()
def simulate(
    sims: int = typer.Option(10000, help="Number of Monte-Carlo tournaments."),
    out: Path = typer.Option(None, help="Output CSV (default predictions/simulation.csv)."),
):
    """Simulate the full tournament: stage & championship probabilities."""
    pred = _load_predictor()
    res = _run_or_load_sim(pred, sims)
    probs = res.stage_probs
    out = out or get_settings().predictions_dir / "simulation.csv"
    probs.to_csv(out)

    t = Table(title=f"WC 2026 — stage probabilities ({sims:,} sims)", header_style="bold cyan")
    t.add_column("team")
    for c in ["R32", "R16", "QF", "SF", "FINAL", "champion"]:
        t.add_column(c, justify="right")
    for team, r in probs.head(16).iterrows():
        t.add_row(team, *[f"{r[c]:.0%}" for c in ["R32", "R16", "QF", "SF", "FINAL", "champion"]])
    console.print(t)
    console.print(f"[green]Saved →[/green] {out}")


# ---------------------------------------------------------------- explain
@app.command()
def explain(
    home: str = typer.Argument(...),
    away: str = typer.Argument(...),
    neutral: bool = typer.Option(True),
    plot: Path = typer.Option(None, help="Save contribution chart PNG here."),
):
    """SHAP feature contributions to each side's expected goals (log λ)."""
    from wcforecast.explain import match_contributions
    from wcforecast.features.build import to_side_view

    pred = _load_predictor()
    feats = pred._fixture_features(home, away, neutral, None, "world_cup")
    side = to_side_view(feats.assign(home_score=np.nan, away_score=np.nan), with_target=False)
    for k, team in [(0, home), (1, away)]:
        contrib = match_contributions(pred.gbm, side.iloc[[k]])
        t = Table(title=f"log λ contributions — {team}", header_style="bold cyan")
        for col in ["feature", "value", "contribution", "± bag sd"]:
            t.add_column(col, justify="right")
        for _, r in contrib.head(10).iterrows():
            t.add_row(r["feature"], f"{r['value']:.2f}", f"{r['contribution']:+.3f}", f"{r['contribution_sd']:.3f}")
        console.print(t)
        if plot:
            from wcforecast.viz import shap_waterfall

            p = plot.with_name(plot.stem + f"_{team.replace(' ', '_')}" + plot.suffix)
            shap_waterfall(contrib, f"{team} — contributions to log λ vs {away if k == 0 else home}", path=p)
            console.print(f"[green]Plot →[/green] {p}")


@app.command()
def abilities(top: int = typer.Option(20, help="Rows to display.")):
    """Dixon-Coles posterior attack/defence abilities (current fit)."""
    pred = _load_predictor()
    df = pred.dc.abilities().head(top)
    t = Table(title="Dixon-Coles abilities (±1 sd)", header_style="bold cyan")
    for col in ["team", "attack", "defence", "strength"]:
        t.add_column(col, justify="right")
    for _, r in df.iterrows():
        t.add_row(r["team"], f"{r['attack']:+.2f}±{r['attack_sd']:.2f}",
                  f"{r['defence']:+.2f}±{r['defence_sd']:.2f}", f"{r['strength']:+.2f}")
    console.print(t)


# -------------------------------------------------------------------- mpp
mpp_app = typer.Typer(no_args_is_help=True, help="Mon Petit Prono optimisation.")
app.add_typer(mpp_app, name="mpp")


@mpp_app.command("picks")
def mpp_picks(
    days: int = typer.Option(7, help="Horizon: fixtures within N days from the first upcoming."),
    beta: float = typer.Option(1.30, help="Crowd sharpening β (higher = crowd more chalky)."),
    odds_file: Path = typer.Option(None, help="CSV with home_team,away_team,odds_h,odds_d,odds_a from the MPP app."),
    out: Path = typer.Option(None, help="Output CSV (default predictions/mpp_picks.csv)."),
):
    """Expected-points-optimal MPP picks (and X2 booster advice)."""
    from wcforecast.data import load_group_fixtures
    from wcforecast.mpp import MppRules, picks_table

    pred = _load_predictor()
    fx = load_group_fixtures().sort_values("date")
    start = fx["date"].min()
    fx = fx[fx["date"] <= start + pd.Timedelta(days=days)]
    odds = None
    if odds_file:
        o = pd.read_csv(odds_file)
        odds = {(r.home_team, r.away_team): (r.odds_h, r.odds_d, r.odds_a) for r in o.itertuples(index=False)}
    forecasts = [pred.forecast(r.home_team, r.away_team, bool(r.neutral), r.date) for r in fx.itertuples(index=False)]
    table = picks_table(forecasts, MppRules(crowd_sharpening=beta), odds)
    out = out or get_settings().predictions_dir / "mpp_picks.csv"
    table.to_csv(out, index=False)

    t = Table(title=f"MPP optimal picks (β={beta})", header_style="bold cyan")
    for col in ["match", "pick", "E[pts]", "P(result)", "result pts", "P(exact)", "bonus", "X2"]:
        t.add_column(col, justify="right")
    for _, r in table.iterrows():
        t.add_row(
            f"{r['home_team']} – {r['away_team']}", r["pick"], f"{r['expected_points']:.1f}",
            f"{r['p_outcome']:.0%}", f"{r['outcome_pts']:.0f}", f"{r['p_exact']:.1%}",
            f"{r['bonus_if_exact']:.0f}", r["x2_booster"],
        )
    console.print(t)
    console.print(
        "[dim]Points model: result points = 10 × odds (model fair odds unless --odds-file); "
        "exact-score bonus from crowd-rarity tiers 20/30/50/70/100.[/dim]"
    )
    console.print(f"[green]Saved →[/green] {out}")


def main():  # console-script entry point
    app()


if __name__ == "__main__":
    main()
