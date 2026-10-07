"""Анализ результатов A/B-теста: t-тест Уэлча, z-тест для долей, бутстрап, дельта-метод, CUPED, проверка SRM."""
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class TestResult:
    control: float          # значение метрики в контроле
    treatment: float        # значение метрики в тесте
    effect: float           # абсолютная разница (тест − контроль)
    ci_low: float           # 95% ДИ для разницы
    ci_high: float
    p_value: float

    @property
    def relative_effect(self) -> float:
        return self.effect / self.control if self.control else float("nan")

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05


def _z_result(m_a: float, m_b: float, se: float, alpha: float) -> TestResult:
    diff = m_b - m_a
    z = diff / se
    q = stats.norm.ppf(1 - alpha / 2)
    return TestResult(m_a, m_b, diff, diff - q * se, diff + q * se, float(2 * stats.norm.sf(abs(z))))


def welch_ttest(a, b, alpha: float = 0.05) -> TestResult:
    """t-тест Уэлча: не требует равенства дисперсий в группах."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    dof = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    diff = b.mean() - a.mean()
    q = stats.t.ppf(1 - alpha / 2, dof)
    p = 2 * stats.t.sf(abs(diff / se), dof)
    return TestResult(a.mean(), b.mean(), diff, diff - q * se, diff + q * se, float(p))


def ztest_proportions(conv_a: int, n_a: int, conv_b: int, n_b: int, alpha: float = 0.05) -> TestResult:
    """z-тест для разницы конверсий. ДИ — по ненормированной (unpooled) ошибке, p-value — по pooled."""
    p_a, p_b = conv_a / n_a, conv_b / n_b
    p_pool = (conv_a + conv_b) / (n_a + n_b)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    se = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    res = _z_result(p_a, p_b, se, alpha)
    res.p_value = float(2 * stats.norm.sf(abs((p_b - p_a) / se_pool)))
    return res


def bootstrap_test(a, b, stat=np.mean, n_boot: int = 5000, alpha: float = 0.05, seed: int | None = 0) -> TestResult:
    """Бутстрап разницы статистик (среднее, медиана, квантиль). Без предположений о распределении."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx_a = rng.integers(0, len(a), (n_boot, len(a)))
    idx_b = rng.integers(0, len(b), (n_boot, len(b)))
    diffs = stat(b[idx_b], axis=1) - stat(a[idx_a], axis=1)
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    # p-value: удвоенная доля бутстрап-разниц по «другую сторону» от нуля
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return TestResult(float(stat(a)), float(stat(b)), float(stat(b) - stat(a)), float(lo), float(hi), float(min(p, 1.0)))


def _ratio_mean_var(num, den):
    """Оценка ratio = Σnum/Σden и её дисперсии дельта-методом (единица рандомизации — пользователь)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    n = len(num)
    mx, my = num.mean(), den.mean()
    vx, vy = num.var(ddof=1), den.var(ddof=1)
    cov = np.cov(num, den, ddof=1)[0, 1]
    var = (vx / my ** 2 - 2 * mx * cov / my ** 3 + mx ** 2 * vy / my ** 4) / n
    return mx / my, var


def delta_method_ratio(num_a, den_a, num_b, den_b, alpha: float = 0.05) -> TestResult:
    """Тест для метрики-отношения (например, средний чек = выручка / заказы), когда рандомизировали пользователей.

    Наивный t-тест по отдельным заказам занижает дисперсию: заказы одного пользователя зависимы.
    Дельта-метод корректно учитывает эту зависимость.
    """
    r_a, v_a = _ratio_mean_var(num_a, den_a)
    r_b, v_b = _ratio_mean_var(num_b, den_b)
    return _z_result(r_a, r_b, np.sqrt(v_a + v_b), alpha)


@dataclass
class CupedResult:
    y_a: np.ndarray
    y_b: np.ndarray
    theta: float
    variance_reduction: float   # доля, на которую снизилась дисперсия метрики


def cuped(y_a, x_a, y_b, x_b) -> CupedResult:
    """CUPED: Y' = Y − θ·(X − mean(X)), θ = cov(X, Y) / var(X).

    X — та же метрика до эксперимента (ковариата не должна зависеть от воздействия).
    θ и mean(X) считаются по объединённой выборке, поэтому оценка эффекта остаётся несмещённой.
    """
    y_a, x_a, y_b, x_b = (np.asarray(v, float) for v in (y_a, x_a, y_b, x_b))
    y, x = np.concatenate([y_a, y_b]), np.concatenate([x_a, x_b])
    theta = np.cov(x, y, ddof=1)[0, 1] / x.var(ddof=1)
    adj_a = y_a - theta * (x_a - x.mean())
    adj_b = y_b - theta * (x_b - x.mean())
    var_before = np.concatenate([y_a, y_b]).var(ddof=1)
    var_after = np.concatenate([adj_a, adj_b]).var(ddof=1)
    return CupedResult(adj_a, adj_b, float(theta), float(1 - var_after / var_before))


def srm_check(n_a: int, n_b: int, expected_share_a: float = 0.5) -> float:
    """Sample Ratio Mismatch: p-value хи-квадрат теста на соответствие плановому сплиту.

    p < 0.001 — сигнал, что сплит сломан (баг в рандомизации, потеря событий), и результатам теста верить нельзя.
    """
    total = n_a + n_b
    expected = [total * expected_share_a, total * (1 - expected_share_a)]
    return float(stats.chisquare([n_a, n_b], f_exp=expected).pvalue)
