"""Проверка корректности abkit симуляциями и на реальных данных.

1. A/A-тесты: доля ложных срабатываний t-теста Уэлча должна быть ≈ 5%.
2. Мощность: при размере выборки из sample_size_means эмпирическая мощность должна быть ≈ 80%.
3. Peeking: если «подглядывать» и останавливать тест при p < 0.05, ложных срабатываний становится в разы больше.
4. Средний чек: наивный t-тест по заказам против дельта-метода по пользователям.
5. CUPED на реальной выручке клиентов (UCI Online Retail II): снижение дисперсии и рост мощности.

Результаты: reports/simulation_results.json и графики в reports/figures/.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from abkit import cuped, delta_method_ratio, power_means, sample_size_means, welch_ttest  # noqa: E402

FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
sns.set_theme(style="whitegrid")
NAVY, ACCENT, LIGHT = "#1f3a5f", "#e07a1f", "#9fb3c8"
results = {}


def wilson(k: int, n: int) -> list[float]:
    """95% доверительный интервал Уилсона для доли k/n."""
    p, z = k / n, 1.96
    centre = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    return [round(100 * (centre - half), 2), round(100 * (centre + half), 2)]


# --- 1. A/A-тесты -----------------------------------------------------------------
# Выручка на пользователя — логнормальная (тяжёлый правый хвост, как у реальных чеков).
N_SIM, N = 10_000, 1_000
p_aa = np.array([welch_ttest(rng.lognormal(6, 1, N), rng.lognormal(6, 1, N)).p_value for _ in range(N_SIM)])
fp = int((p_aa < 0.05).sum())
results["aa_tests"] = {"simulations": N_SIM, "n_per_group": N, "false_positive_rate_pct": round(100 * fp / N_SIM, 2),
                       "ci95_pct": wilson(fp, N_SIM)}

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.hist(p_aa, bins=20, color=NAVY, edgecolor="white")
ax.axhline(N_SIM / 20, color=ACCENT, ls="--", label="ожидание при равномерном распределении")
ax.set_title(f"A/A-тесты: p-value распределены равномерно, ложных срабатываний {100 * fp / N_SIM:.1f}%",
             loc="left", fontweight="bold", fontsize=11)
ax.set_xlabel("p-value"); ax.set_ylabel("Число тестов"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "aa_pvalues.png", dpi=150); plt.close(fig)

# --- 2. Мощность ------------------------------------------------------------------
mu, sigma = 6, 1
mean = np.exp(mu + sigma ** 2 / 2)
sd = np.sqrt((np.exp(sigma ** 2) - 1) * np.exp(2 * mu + sigma ** 2))
uplift = 0.05
n_req = sample_size_means(sd=sd, mde=uplift * mean)
N_POW = 2_000
hits = sum(welch_ttest(rng.lognormal(mu, sigma, n_req), rng.lognormal(mu, sigma, n_req) * (1 + uplift)).significant
           for _ in range(N_POW))
results["power"] = {"uplift_pct": 100 * uplift, "required_n_per_group": n_req, "designed_power_pct": 80,
                    "empirical_power_pct": round(100 * hits / N_POW, 1), "ci95_pct": wilson(hits, N_POW)}

# --- 3. Peeking -------------------------------------------------------------------
# A/A-тест на 10 000 пользователей в группе, смотрим на p-value после каждых 10% данных.
N_PEEK, N_TOTAL, LOOKS = 2_000, 10_000, 10
checkpoints = np.linspace(N_TOTAL / LOOKS, N_TOTAL, LOOKS).astype(int)
stopped_early = 0
for _ in range(N_PEEK):
    a, b = rng.lognormal(6, 1, N_TOTAL), rng.lognormal(6, 1, N_TOTAL)
    if any(welch_ttest(a[:k], b[:k]).p_value < 0.05 for k in checkpoints):
        stopped_early += 1
results["peeking"] = {"looks": LOOKS, "simulations": N_PEEK,
                      "false_positive_rate_pct": round(100 * stopped_early / N_PEEK, 1),
                      "ci95_pct": wilson(stopped_early, N_PEEK)}

fig, ax = plt.subplots(figsize=(6, 3.6))
vals = [results["aa_tests"]["false_positive_rate_pct"], results["peeking"]["false_positive_rate_pct"]]
bars = ax.bar(["Один анализ в конце", f"Подглядывание\n({LOOKS} проверок)"], vals, color=[NAVY, ACCENT], width=0.55)
ax.bar_label(bars, fmt="%.1f%%"); ax.axhline(5, color="grey", ls="--", lw=1)
ax.set_ylabel("Ложные срабатывания, %")
ax.set_title(f"Подглядывание увеличивает ложные срабатывания в {vals[1] / vals[0]:.1f} раза", loc="left",
             fontweight="bold", fontsize=11)
fig.tight_layout(); fig.savefig(FIG / "peeking.png", dpi=150); plt.close(fig)

# --- 4. Средний чек: наивный тест vs дельта-метод ------------------------------------
# У каждого пользователя свой «уровень» чека, поэтому его заказы коррелируют между собой.
N_RATIO, USERS = 2_000, 2_000
naive_fp = delta_fp = 0
for _ in range(N_RATIO):
    groups = []
    for _g in range(2):
        orders = rng.poisson(2, USERS) + 1
        user_level = rng.lognormal(6, 0.8, USERS)
        order_values = np.repeat(user_level, orders) * rng.lognormal(0, 0.3, orders.sum())
        user_revenue = np.bincount(np.repeat(np.arange(USERS), orders), weights=order_values)
        groups.append((order_values, user_revenue, orders))
    (ov_a, rev_a, ord_a), (ov_b, rev_b, ord_b) = groups
    naive_fp += welch_ttest(ov_a, ov_b).p_value < 0.05
    delta_fp += delta_method_ratio(rev_a, ord_a, rev_b, ord_b).p_value < 0.05
results["ratio_metric"] = {
    "simulations": N_RATIO,
    "naive_order_level_ttest_fpr_pct": round(100 * naive_fp / N_RATIO, 1), "naive_ci95_pct": wilson(naive_fp, N_RATIO),
    "delta_method_fpr_pct": round(100 * delta_fp / N_RATIO, 1), "delta_ci95_pct": wilson(delta_fp, N_RATIO),
}

# --- 5. CUPED на реальных данных --------------------------------------------------
# Метрика — выручка клиента за дек 2010 — ноя 2011, ковариата — его выручка за предыдущие 12 месяцев.
df = pd.read_csv(ROOT / "data" / "customer_periods.csv")
y_all, x_all = df["revenue"].to_numpy(), df["pre_revenue"].to_numpy()
n_cust = len(df)
N_CUPED, EFFECT = 2_000, 0.25
stats_rows = []
for i in range(N_CUPED):
    perm = rng.permutation(n_cust)
    ia, ib = perm[: n_cust // 2], perm[n_cust // 2:]
    for scenario, mult in (("A/A", 1.0), (f"+{EFFECT:.0%}", 1 + EFFECT)):
        ya, yb = y_all[ia], y_all[ib] * mult
        raw = welch_ttest(ya, yb)
        cu = cuped(ya, x_all[ia], yb, x_all[ib])
        adj = welch_ttest(cu.y_a, cu.y_b)
        stats_rows.append({"scenario": scenario, "raw_sig": raw.significant, "cuped_sig": adj.significant,
                           "raw_ci_width": raw.ci_high - raw.ci_low, "cuped_ci_width": adj.ci_high - adj.ci_low,
                           "var_reduction": cu.variance_reduction})
sim = pd.DataFrame(stats_rows)
aa, eff = sim[sim.scenario == "A/A"], sim[sim.scenario != "A/A"]
results["cuped_real_data"] = {
    "dataset": "UCI Online Retail II, клиенты с покупками до дек 2010",
    "customers": n_cust, "correlation_pre_post": round(float(np.corrcoef(x_all, y_all)[0, 1]), 3),
    "variance_reduction_pct": round(100 * aa.var_reduction.mean(), 1),
    "ci_width_reduction_pct": round(100 * (1 - aa.cuped_ci_width.mean() / aa.raw_ci_width.mean()), 1),
    "aa_fpr_raw_pct": round(100 * aa.raw_sig.mean(), 2), "aa_fpr_cuped_pct": round(100 * aa.cuped_sig.mean(), 2),
    "simulated_effect_pct": 100 * EFFECT,
    "power_raw_pct": round(100 * eff.raw_sig.mean(), 1), "power_cuped_pct": round(100 * eff.cuped_sig.mean(), 1),
    "splits": N_CUPED,
}
# MDE (в % от средней выручки) при мощности 80%: MDE ≈ 2.8 · SE, SE = ширина 95% ДИ / (2 · 1.96)
mean_rev = y_all.mean()
for kind in ("raw", "cuped"):
    se = sim.loc[sim.scenario == "A/A", f"{kind}_ci_width"].mean() / (2 * 1.96)
    results["cuped_real_data"][f"mde_{kind}_pct"] = round(100 * 2.8 * se / mean_rev, 1)
# сколько клиентов понадобилось бы без CUPED, чтобы достичь той же точности
results["cuped_real_data"]["equivalent_sample_size_multiplier"] = round(
    1 / (1 - results["cuped_real_data"]["variance_reduction_pct"] / 100), 1)

fig, ax = plt.subplots(figsize=(6.4, 3.6))
c = results["cuped_real_data"]
bars = ax.bar(["Обычный t-тест", "t-тест + CUPED"], [c["power_raw_pct"], c["power_cuped_pct"]],
              color=[LIGHT, NAVY], width=0.55)
ax.bar_label(bars, fmt="%.0f%%"); ax.set_ylim(0, 100); ax.set_ylabel("Мощность, %")
ax.set_title(f"CUPED на реальной выручке: дисперсия −{c['variance_reduction_pct']:.0f}%, "
             f"мощность ×{c['power_cuped_pct'] / c['power_raw_pct']:.1f}", loc="left", fontweight="bold", fontsize=11)
ax.text(0.5, -0.22, f"{c['customers']:,} клиентов, эффект +{EFFECT:.0%}, {N_CUPED:,} случайных разбиений".replace(",", " "),
        transform=ax.transAxes, ha="center", fontsize=8, color="grey")
fig.tight_layout(); fig.savefig(FIG / "cuped_power.png", dpi=150); plt.close(fig)

(ROOT / "reports" / "simulation_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
print(json.dumps(results, ensure_ascii=False, indent=2))
