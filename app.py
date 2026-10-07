"""Streamlit-приложение для дизайна и анализа A/B-тестов. Запуск: streamlit run app.py"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from abkit import (bootstrap_test, cuped, delta_method_ratio, mde_means, power_means, sample_size_means,
                   sample_size_proportions, srm_check, welch_ttest, ztest_proportions)

st.set_page_config(page_title="A/B-тест: дизайн и анализ", page_icon="🧪", layout="wide")
NAVY, LIGHT = "#1f3a5f", "#9fb3c8"
DEMO = Path(__file__).parent / "data" / "customer_periods.csv"


def fmt_result(name, r, unit=""):
    return {
        "Метод": name,
        "Контроль": f"{r.control:,.2f}{unit}", "Тест": f"{r.treatment:,.2f}{unit}",
        "Эффект": f"{r.effect:+,.2f}{unit} ({r.relative_effect:+.1%})",
        "95% ДИ": f"[{r.ci_low:+,.2f}; {r.ci_high:+,.2f}]",
        "p-value": f"{r.p_value:.4f}", "Значимо (α=0.05)": "✅ да" if r.significant else "— нет",
    }


st.title("🧪 A/B-тест: дизайн и анализ")
st.caption("Размер выборки и MDE · t-тест Уэлча · z-тест для конверсий · бутстрап · дельта-метод · CUPED · проверка SRM")

tab_design, tab_mean, tab_conv, tab_ratio = st.tabs(
    ["📐 Дизайн теста", "📊 Анализ: среднее (+CUPED)", "🎯 Анализ: конверсия", "🧾 Анализ: средний чек"])

# --- Дизайн ----------------------------------------------------------------------
with tab_design:
    c1, c2 = st.columns([1, 2])
    with c1:
        kind = st.radio("Тип метрики", ["Среднее (выручка на пользователя)", "Конверсия"])
        alpha = st.select_slider("Уровень значимости α", [0.01, 0.05, 0.1], value=0.05)
        power = st.select_slider("Мощность 1−β", [0.7, 0.8, 0.9, 0.95], value=0.8)
        traffic = st.number_input("Пользователей в день (на обе группы)", 100, 10_000_000, 5_000, step=500)
        if kind.startswith("Среднее"):
            mean = st.number_input("Текущее среднее", value=1500.0, min_value=0.01)
            sd = st.number_input("Стандартное отклонение", value=3000.0, min_value=0.01)
            mde_rel = st.number_input("MDE, % от среднего", value=5.0, min_value=0.1) / 100
            n = sample_size_means(sd, mde_rel * mean, alpha, power)
        else:
            p = st.number_input("Текущая конверсия, %", value=10.0, min_value=0.01, max_value=99.9) / 100
            mde_rel = st.number_input("MDE, % от конверсии (относительный)", value=5.0, min_value=0.1) / 100
            n = sample_size_proportions(p, p * mde_rel, alpha, power)
            sd, mean = np.sqrt(p * (1 - p)), p
    with c2:
        m1, m2, m3 = st.columns(3)
        m1.metric("На группу", f"{n:,}".replace(",", " "))
        m2.metric("Всего", f"{2 * n:,}".replace(",", " "))
        m3.metric("Длительность", f"{int(np.ceil(2 * n / traffic))} дн.")
        st.info("Округляйте длительность до целых недель, чтобы учесть недельную сезонность, "
                "и не останавливайте тест раньше срока при первом значимом результате (peeking).")
        effects = np.linspace(0.005, 3 * mde_rel, 120)
        fig = go.Figure(go.Scatter(x=100 * effects, y=[100 * power_means(sd, e * mean, n, alpha) for e in effects],
                                   line_color=NAVY))
        fig.add_hline(y=100 * power, line_dash="dash", line_color="grey")
        fig.update_layout(title="Мощность теста в зависимости от истинного эффекта", template="plotly_white",
                          xaxis_title="Истинный эффект, %", yaxis_title="Вероятность его обнаружить, %", height=380)
        st.plotly_chart(fig, use_container_width=True)

# --- Анализ среднего ---------------------------------------------------------------
with tab_mean:
    st.markdown("CSV с колонками **group** (A/B), **value** (метрика на пользователя) и, по желанию, "
                "**pre_value** (та же метрика до теста — для CUPED).")
    src = st.radio("Данные", ["Демо: реальная выручка клиентов (UCI Online Retail II)", "Загрузить CSV"],
                   horizontal=True)
    if src.startswith("Демо"):
        uplift = st.slider("Добавить эффект в группу B, %", 0, 50, 25)
        seed = st.number_input("Случайное разбиение (seed)", 0, 10_000, 7)
        d = pd.read_csv(DEMO)
        rng = np.random.default_rng(int(seed))
        d["group"] = np.where(rng.random(len(d)) < 0.5, "A", "B")
        d["value"] = d["revenue"] * np.where(d["group"] == "B", 1 + uplift / 100, 1)
        d["pre_value"] = d["pre_revenue"]
    else:
        f = st.file_uploader("CSV-файл", type="csv")
        d = pd.read_csv(f) if f else None
    if d is not None:
        a, b = d[d.group == "A"], d[d.group == "B"]
        srm_p = srm_check(len(a), len(b))
        (st.error if srm_p < 0.001 else st.success)(
            f"SRM-проверка: A = {len(a):,}, B = {len(b):,}, p-value = {srm_p:.4f}"
            + (" — сплит нарушен, результатам доверять нельзя!" if srm_p < 0.001 else " — сплит в порядке"))
        rows = [fmt_result("t-тест Уэлча", welch_ttest(a.value, b.value)),
                fmt_result("Бутстрап (среднее)", bootstrap_test(a.value, b.value, n_boot=3000)),
                fmt_result("Бутстрап (медиана)", bootstrap_test(a.value, b.value, stat=np.median, n_boot=3000))]
        if "pre_value" in d:
            cu = cuped(a.value, a.pre_value, b.value, b.pre_value)
            rows.append(fmt_result("t-тест + CUPED", welch_ttest(cu.y_a, cu.y_b)))
            st.caption(f"CUPED: θ = {cu.theta:.3f}, дисперсия метрики снизилась на {cu.variance_reduction:.1%}")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        fig = go.Figure()
        for g, color in (("A", LIGHT), ("B", NAVY)):
            fig.add_histogram(x=np.log1p(d.loc[d.group == g, "value"]), name=g, marker_color=color, opacity=0.7)
        fig.update_layout(barmode="overlay", template="plotly_white", height=320,
                          title="Распределение метрики (log(1 + x))", xaxis_title="log(1 + value)")
        st.plotly_chart(fig, use_container_width=True)

# --- Конверсия ---------------------------------------------------------------------
with tab_conv:
    c1, c2 = st.columns(2)
    n_a = c1.number_input("Пользователей в A", 1, value=20_000)
    x_a = c1.number_input("Конверсий в A", 0, value=2_000)
    n_b = c2.number_input("Пользователей в B", 1, value=20_000)
    x_b = c2.number_input("Конверсий в B", 0, value=2_150)
    if x_a <= n_a and x_b <= n_b:
        r = ztest_proportions(x_a, n_a, x_b, n_b)
        st.dataframe(pd.DataFrame([fmt_result("z-тест для долей", r)]), hide_index=True, use_container_width=True)
        st.caption(f"SRM p-value: {srm_check(n_a, n_b):.4f}")

# --- Средний чек ---------------------------------------------------------------------
with tab_ratio:
    st.markdown("Средний чек = выручка / заказы. Если рандомизировали **пользователей**, заказы одного "
                "пользователя зависимы, и t-тест по отдельным заказам занижает дисперсию. "
                "Нужен **дельта-метод**. CSV: **group**, **revenue**, **orders** (по пользователям).")
    f = st.file_uploader("CSV-файл", type="csv", key="ratio")
    if f is None and st.checkbox("Использовать демо (реальные клиенты, A/A-разбиение)", value=True):
        d = pd.read_csv(DEMO).query("orders > 0")
        d["group"] = np.where(np.random.default_rng(1).random(len(d)) < 0.5, "A", "B")
    elif f is not None:
        d = pd.read_csv(f)
    else:
        d = None
    if d is not None:
        a, b = d[d.group == "A"], d[d.group == "B"]
        r = delta_method_ratio(a.revenue, a.orders, b.revenue, b.orders)
        st.dataframe(pd.DataFrame([fmt_result("Дельта-метод", r)]), hide_index=True, use_container_width=True)

st.divider()
st.caption("Исходный код и проверка методов симуляциями: github.com/juggler14/ab-testing-toolkit")
