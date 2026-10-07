import numpy as np
import pytest
from scipy import stats

from abkit import (bootstrap_test, cuped, delta_method_ratio, mde_means, power_means, sample_size_means,
                   sample_size_proportions, srm_check, welch_ttest, ztest_proportions)


def test_sample_size_means_matches_textbook():
    # σ = 1, MDE = 0.2σ, α = 0.05, мощность 0.8 → классические ~393 на группу
    assert sample_size_means(sd=1, mde=0.2) == 393


def test_sample_size_proportions_matches_textbook():
    # конверсия 10% → 11%: ~14 750 на группу
    assert sample_size_proportions(p=0.10, mde=0.01) == pytest.approx(14_750, abs=10)


def test_mde_and_power_are_consistent():
    n = sample_size_means(sd=50, mde=5)
    assert mde_means(sd=50, n_per_group=n) <= 5
    assert power_means(sd=50, mde=5, n_per_group=n) == pytest.approx(0.8, abs=0.01)


def test_welch_matches_scipy():
    rng = np.random.default_rng(1)
    a, b = rng.normal(10, 2, 300), rng.normal(10.5, 3, 200)
    ours = welch_ttest(a, b)
    ref = stats.ttest_ind(b, a, equal_var=False)
    assert ours.p_value == pytest.approx(ref.pvalue, rel=1e-9)
    assert ours.ci_low < ours.effect < ours.ci_high


def test_ztest_proportions_detects_large_effect():
    res = ztest_proportions(1000, 10_000, 1200, 10_000)
    assert res.significant and res.effect == pytest.approx(0.02)


def test_bootstrap_ci_contains_true_difference():
    rng = np.random.default_rng(2)
    a, b = rng.exponential(100, 2000), rng.exponential(110, 2000)
    res = bootstrap_test(a, b, n_boot=2000)
    assert res.ci_low < 10 < res.ci_high


def test_delta_method_equals_ratio_of_sums():
    rng = np.random.default_rng(3)
    orders = rng.poisson(3, 1000) + 1
    revenue = orders * rng.normal(500, 50, 1000)
    res = delta_method_ratio(revenue, orders, revenue, orders)
    assert res.control == pytest.approx(revenue.sum() / orders.sum())
    assert res.effect == pytest.approx(0) and res.p_value == pytest.approx(1)


def test_cuped_reduces_variance_and_keeps_effect():
    rng = np.random.default_rng(4)
    n = 5000
    x_a, x_b = rng.normal(100, 20, n), rng.normal(100, 20, n)
    y_a = x_a + rng.normal(0, 10, n)
    y_b = x_b + rng.normal(0, 10, n) + 2
    res = cuped(y_a, x_a, y_b, x_b)
    assert res.variance_reduction > 0.7                      # корреляция ~0.9 → снижение дисперсии ~80%
    assert (res.y_b.mean() - res.y_a.mean()) == pytest.approx(2, abs=0.5)


def test_srm():
    assert srm_check(5000, 5000) > 0.9
    assert srm_check(5000, 5400) < 0.001
