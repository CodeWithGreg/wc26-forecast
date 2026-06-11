import numpy as np

from wcforecast.eval import metrics as M


def test_log_loss_rewards_sharp_correct_forecasts():
    sharp = np.array([[0.7, 0.2, 0.1]])
    flat = np.array([[1 / 3, 1 / 3, 1 / 3]])
    out = np.array([0])
    assert M.log_loss_1x2(sharp, out) < M.log_loss_1x2(flat, out)


def test_rps_orders_by_distance():
    """RPS must punish placing mass on the *far* outcome more than the near one."""
    out = np.array([0])  # home win
    near = np.array([[0.5, 0.5, 0.0]])  # mass on draw
    far = np.array([[0.5, 0.0, 0.5]])  # mass on away win
    assert M.rps_1x2(near, out) < M.rps_1x2(far, out)


def test_brier_perfect_zero():
    probs = np.array([[1.0, 0.0, 0.0]])
    assert M.brier_1x2(probs, np.array([0])) == 0.0


def test_poisson_deviance_minimised_at_truth():
    goals = np.array([0, 1, 2, 3, 1, 1])
    lam_true = np.full(6, goals.mean())
    assert M.poisson_deviance(lam_true, goals) < M.poisson_deviance(lam_true * 2, goals)
    assert M.poisson_deviance(lam_true, goals) < M.poisson_deviance(lam_true * 0.5, goals)


def test_pit_uniform_for_calibrated_forecasts():
    rng = np.random.default_rng(3)
    lam = 1.4
    from scipy.stats import poisson

    pmf = poisson.pmf(np.arange(15), lam)
    obs = rng.poisson(lam, 4000)
    pit = M.pit_values([pmf] * len(obs), obs.tolist(), rng)
    # calibrated -> PIT ~ U[0,1]: mean ~0.5, low KS distance
    assert abs(pit.mean() - 0.5) < 0.02
    hist, _ = np.histogram(pit, bins=10, range=(0, 1))
    assert hist.max() / hist.min() < 1.35


def test_interval_coverage_close_to_nominal():
    rng = np.random.default_rng(4)
    from scipy.stats import poisson

    lam = 2.6
    pmf = poisson.pmf(np.arange(20), lam)
    obs = rng.poisson(lam, 3000)
    cov = M.interval_coverage([pmf] * len(obs), obs.tolist(), 0.9)
    assert cov >= 0.9  # discrete intervals over-cover by construction
    assert cov < 0.99
