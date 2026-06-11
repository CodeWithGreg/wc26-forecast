import numpy as np
import pytest

from wcforecast.models.poisson_core import (
    dc_tau_matrix,
    mixture_score_matrix,
    outcome_probs,
    quantile_interval,
    score_matrix,
    top_scorelines,
    totals_pmf,
)


def test_score_matrix_is_distribution():
    m = score_matrix(1.4, 1.1, rho=-0.08)
    assert m.shape == (11, 11)
    assert m.min() >= 0
    assert np.isclose(m.sum(), 1.0)


def test_outcome_probs_sum_to_one():
    m = score_matrix(1.8, 0.9, rho=-0.05)
    ph, pd_, pa = outcome_probs(m)
    assert np.isclose(ph + pd_ + pa, 1.0)
    assert ph > pa  # stronger side more likely to win


def test_dc_tau_inflates_draws():
    """Negative rho must increase 0-0 and 1-1 relative to independence."""
    m0 = score_matrix(1.2, 1.2, rho=0.0)
    m1 = score_matrix(1.2, 1.2, rho=-0.1)
    assert m1[0, 0] > m0[0, 0]
    assert m1[1, 1] > m0[1, 1]
    assert m1[0, 1] < m0[0, 1]


def test_tau_matrix_cells():
    tau = dc_tau_matrix(1.0, 1.0, -0.1, 5)
    assert tau[0, 0] == pytest.approx(1.1)
    assert tau[1, 1] == pytest.approx(1.1)
    assert tau[0, 1] == pytest.approx(0.9)
    assert np.all(tau[2:, 2:] == 1.0)


def test_mixture_wider_than_plugin():
    """Posterior-predictive mixing must add dispersion vs a single lambda."""
    rng = np.random.default_rng(0)
    draws_h = np.exp(rng.normal(np.log(1.4), 0.25, 300))
    draws_a = np.exp(rng.normal(np.log(1.1), 0.25, 300))
    m_mix = mixture_score_matrix(draws_h, draws_a)
    m_plug = score_matrix(float(draws_h.mean()), float(draws_a.mean()))
    tot_mix = totals_pmf(m_mix)
    tot_plug = totals_pmf(m_plug)
    var = lambda pmf: (np.arange(len(pmf)) ** 2 * pmf).sum() - ((np.arange(len(pmf)) * pmf).sum()) ** 2  # noqa: E731
    assert var(tot_mix) > var(tot_plug)


def test_totals_and_intervals():
    m = score_matrix(1.3, 1.3)
    tot = totals_pmf(m)
    assert np.isclose(tot.sum(), 1.0)
    lo, hi = quantile_interval(tot, 0.9)
    assert lo <= 2 <= hi  # the mean total (2.6) lies inside the 90% interval


def test_top_scorelines_sorted():
    m = score_matrix(2.0, 0.7)
    tops = top_scorelines(m, 3)
    assert tops[0][2] >= tops[1][2] >= tops[2][2]
    assert tops[0][0] >= tops[0][1]  # favourite's modal score not losing
