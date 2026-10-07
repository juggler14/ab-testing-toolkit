"""Дизайн эксперимента: размер выборки и минимальный детектируемый эффект (MDE).

Формулы для двустороннего теста и двух групп одинакового размера:
    n на группу = (z_{1-α/2} + z_{1-β})² · (σ_A² + σ_B²) / MDE²
"""
from math import ceil, sqrt

from scipy.stats import norm


def _z_sum(alpha: float, power: float) -> float:
    return norm.ppf(1 - alpha / 2) + norm.ppf(power)


def sample_size_means(sd: float, mde: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Размер каждой группы для метрики-среднего (выручка, средний чек) при абсолютном MDE."""
    if sd <= 0 or mde <= 0:
        raise ValueError("sd и mde должны быть положительными")
    return ceil(2 * _z_sum(alpha, power) ** 2 * sd ** 2 / mde ** 2)


def sample_size_proportions(p: float, mde: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Размер каждой группы для конверсии p при абсолютном MDE (p → p + mde)."""
    p2 = p + mde
    if not (0 < p < 1 and 0 < p2 < 1):
        raise ValueError("конверсии должны лежать в интервале (0, 1)")
    return ceil(_z_sum(alpha, power) ** 2 * (p * (1 - p) + p2 * (1 - p2)) / mde ** 2)


def mde_means(sd: float, n_per_group: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Минимальный абсолютный эффект, который тест обнаружит с заданной мощностью при n на группу."""
    return _z_sum(alpha, power) * sd * sqrt(2 / n_per_group)


def power_means(sd: float, mde: float, n_per_group: int, alpha: float = 0.05) -> float:
    """Мощность теста при заданном эффекте и размере групп."""
    se = sd * sqrt(2 / n_per_group)
    return float(norm.cdf(mde / se - norm.ppf(1 - alpha / 2)))
