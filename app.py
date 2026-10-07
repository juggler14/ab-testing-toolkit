"""Streamlit-приложение для дизайна и анализа A/B-тестов. Запуск: streamlit run app.py"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

from abkit import (bootstrap_test, cuped, delta_method_ratio, mde_means, power_means, sample_size_means,
                   sample_size_proportions, srm_check, welch_ttest, ztest_proportions)

st.set_page_config(page_title="abkit — A/B test design & analysis", layout="wide")
NAVY, LIGHT, GRID = "#1f3a5f", "#9fb3c8", "#e5e9ef"
DEMO = Path(__file__).parent / "data" / "customer_periods.csv"
FONT = "IBM Plex Sans, sans-serif"

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  html, body, [class*="css"], .stMarkdown, .stText, label, p, h1, h2, h3 { font-family: 'IBM Plex Sans', sans-serif !important; }
  h1 { font-weight: 600 !important; letter-spacing: -0.5px; }
  [data-testid="stMetricValue"], [data-testid="stDataFrame"], code, .mono { font-family: 'IBM Plex Mono', monospace !important; }
  [data-testid="stMetricValue"] { font-size: 1.7rem; }
  .stTabs [data-baseweb="tab"] { font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; }
  .block-container { padding-top: 2rem; max-width: 1200px; }
</style>
""", unsafe_allow_html=True)


def fmt_result(name, r, base=None):
    """base — знаменатель относительного эффекта (для CUPED — исходное среднее контроля)."""
    rel = r.effect / base if base else r.relative_effect
    return {
        "method": name,
        "control": f"{r.control:,.3f}", "treatment": f"{r.treatment:,.3f}",
        "diff": f"{r.effect:+,.3f}", "rel. diff": f"{rel:+.2%}",
        "SE": f"{r.se:,.3f}" if np.isfinite(r.se) else "—",
        "stat": f"{r.statistic:.3f}" if np.isfinite(r.statistic) else "—",
        "df": f"{r.dof:,.0f}" if np.isfinite(r.dof) else "—",
        "95% CI": f"[{r.ci_low:+,.2f}, {r.ci_high:+,.2f}]",
        "p-value": f"{r.p_value:.4f}",
        "H0 rejected (α=0.05)": "yes" if r.significant else "no",
    }


def style_fig(fig, height=360):
    fig.update_layout(template="plotly_white", height=height, font_family=FONT, margin=dict(t=50, l=50, r=20, b=40),
                      title_font_size=14)
    fig.update_xaxes(gridcolor=GRID); fig.update_yaxes(gridcolor=GRID)
    return fig


st.title("abkit")
st.markdown("Дизайн и анализ A/B-тестов: размер выборки и MDE, t-тест Уэлча, z-тест для долей, бутстрап, "
            "дельта-метод, CUPED, проверка SRM. Каждый метод проверен симуляциями — см. README.")

tab_design, tab_mean, tab_conv, tab_ratio, tab_methods = st.tabs(
    ["Design", "Mean metric + CUPED", "Conversion", "Ratio metric (AOV)", "Methods"])

# --- Design --------------------------------------------------------------------------
with tab_design:
    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        kind = st.radio("Тип метрики", ["Среднее на пользователя", "Конверсия"], horizontal=True)
        alpha = st.select_slider("α (уровень значимости)", [0.01, 0.05, 0.1], value=0.05)
        power = st.select_slider("1 − β (мощность)", [0.7, 0.8, 0.9, 0.95], value=0.8)
        traffic = st.number_input("Пользователей в день, обе группы", 100, 10_000_000, 5_000, step=500)
        if kind.startswith("Среднее"):
            mean = st.number_input("μ — текущее среднее", value=1500.0, min_value=0.01)
            sd = st.number_input("σ — стандартное отклонение", value=3000.0, min_value=0.01)
            mde_rel = st.number_input("MDE, % от μ", value=5.0, min_value=0.1) / 100
            n = sample_size_means(sd, mde_rel * mean, alpha, power)
        else:
            p = st.number_input("p — текущая конверсия, %", value=10.0, min_value=0.01, max_value=99.9) / 100
            mde_rel = st.number_input("MDE, % от p (относительный)", value=5.0, min_value=0.1) / 100
            n = sample_size_proportions(p, p * mde_rel, alpha, power)
            sd, mean = np.sqrt(p * (1 - p)), p
    with c2:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("n на группу", f"{n:,}".replace(",", " "))
        m2.metric("n всего", f"{2 * n:,}".replace(",", " "))
        m3.metric("Дней", f"{int(np.ceil(2 * n / traffic))}")
        m4.metric("z(1−α/2) + z(1−β)", f"{norm.ppf(1 - alpha / 2) + norm.ppf(power):.3f}")
        if kind.startswith("Среднее"):
            st.latex(r"n = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2 \cdot 2\sigma^2}{\text{MDE}^2}")
        else:
            st.latex(r"n = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2 \cdot \left[p_1(1-p_1) + p_2(1-p_2)\right]}{(p_2 - p_1)^2}")
        effects = np.linspace(0.002, 3 * mde_rel, 150)
        fig = go.Figure(go.Scatter(x=100 * effects, y=[100 * power_means(sd, e * mean, n, alpha) for e in effects],
                                   line=dict(color=NAVY, width=2), name="power"))
        fig.add_hline(y=100 * power, line_dash="dot", line_color="grey")
        fig.add_vline(x=100 * mde_rel, line_dash="dot", line_color="grey")
        fig.update_layout(title=f"Кривая мощности при n = {n:,} на группу".replace(",", " "),
                          xaxis_title="истинный эффект, % от μ", yaxis_title="мощность, %")
        st.plotly_chart(style_fig(fig), use_container_width=True)
        grid = [0.01, 0.02, 0.03, 0.05, 0.1]
        st.dataframe(pd.DataFrame({"MDE, %": [f"{100 * g:.0f}" for g in grid],
                                   "n на группу": [f"{(sample_size_means(sd, g * mean, alpha, power) if kind.startswith('Среднее') else sample_size_proportions(mean, mean * g, alpha, power)):,}" for g in grid]}),
                     hide_index=True)

# --- Mean metric ----------------------------------------------------------------------
with tab_mean:
    st.markdown("CSV: `group` (A/B), `value` — метрика на пользователя, `pre_value` — та же метрика "
                "до теста (необязательно, нужна для CUPED).")
    src = st.radio("Данные", ["Демо: выручка 4 239 клиентов, UCI Online Retail II", "Загрузить CSV"],
                   horizontal=True)
    d = None
    if src.startswith("Демо"):
        c1, c2 = st.columns(2)
        uplift = c1.slider("Эффект, добавляемый в группу B, %", 0, 50, 25)
        seed = c2.number_input("seed разбиения", 0, 10_000, 7)
        d = pd.read_csv(DEMO)
        rng = np.random.default_rng(int(seed))
        d["group"] = np.where(rng.random(len(d)) < 0.5, "A", "B")
        d["value"] = d["revenue"] * np.where(d["group"] == "B", 1 + uplift / 100, 1)
        d["pre_value"] = d["pre_revenue"]
    else:
        f = st.file_uploader("CSV", type="csv")
        if f:
            d = pd.read_csv(f)
    if d is not None:
        a, b = d[d.group == "A"], d[d.group == "B"]
        srm_p = srm_check(len(a), len(b))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("n_A", f"{len(a):,}".replace(",", " "))
        m2.metric("n_B", f"{len(b):,}".replace(",", " "))
        m3.metric("SRM p-value", f"{srm_p:.4f}")
        if "pre_value" in d:
            cu = cuped(a.value, a.pre_value, b.value, b.pre_value)
            m4.metric("CUPED: Var reduction", f"{cu.variance_reduction:.1%}")
        if srm_p < 0.001:
            st.error("SRM: фактический сплит не соответствует 50/50 (p < 0.001). Результаты теста недостоверны.")
        rows = [fmt_result("Welch t-test", welch_ttest(a.value, b.value)),
                fmt_result("Bootstrap, mean", bootstrap_test(a.value, b.value, n_boot=3000)),
                fmt_result("Bootstrap, median", bootstrap_test(a.value, b.value, stat=np.median, n_boot=3000))]
        if "pre_value" in d:
            rows.append(fmt_result(f"Welch + CUPED (θ = {cu.theta:.3f})", welch_ttest(cu.y_a, cu.y_b), base=a.value.mean()))
            st.caption(f"Средняя метрика до теста: A = {a.pre_value.mean():,.1f}, B = {b.pre_value.mean():,.1f}. "
                       "Если группы различались ещё до эксперимента, обычный t-тест смещён в конкретной реализации "
                       "разбиения, а CUPED это компенсирует.")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        fig = go.Figure()
        for g, color in (("A", LIGHT), ("B", NAVY)):
            fig.add_histogram(x=np.log1p(d.loc[d.group == g, "value"]), name=g, marker_color=color, opacity=0.7,
                              nbinsx=60)
        fig.update_layout(barmode="overlay", title="Распределение log(1 + value) по группам",
                          xaxis_title="log(1 + value)", yaxis_title="пользователей")
        st.plotly_chart(style_fig(fig, 300), use_container_width=True)

# --- Conversion ----------------------------------------------------------------------
with tab_conv:
    c1, c2 = st.columns(2)
    n_a = c1.number_input("n_A", 1, value=20_000)
    x_a = c1.number_input("conversions_A", 0, value=2_000)
    n_b = c2.number_input("n_B", 1, value=20_000)
    x_b = c2.number_input("conversions_B", 0, value=2_150)
    if x_a <= n_a and x_b <= n_b:
        r = ztest_proportions(x_a, n_a, x_b, n_b)
        st.dataframe(pd.DataFrame([fmt_result("Two-proportion z-test", r)]), hide_index=True, use_container_width=True)
        st.caption(f"z-статистика по pooled-оценке p, 95% CI по unpooled SE. SRM p-value = {srm_check(n_a, n_b):.4f}")
    else:
        st.warning("Число конверсий не может превышать число пользователей.")

# --- Ratio metric ----------------------------------------------------------------------
with tab_ratio:
    st.markdown("Средний чек = Σ revenue / Σ orders. Единица рандомизации — пользователь, поэтому заказы внутри "
                "пользователя зависимы, и t-тест по заказам занижает SE. CSV: `group`, `revenue`, `orders`.")
    f = st.file_uploader("CSV", type="csv", key="ratio")
    d = None
    if f is not None:
        d = pd.read_csv(f)
    elif st.checkbox("Демо: реальные клиенты, A/A-разбиение", value=True):
        d = pd.read_csv(DEMO).query("orders > 0").copy()
        d["group"] = np.where(np.random.default_rng(1).random(len(d)) < 0.5, "A", "B")
    if d is not None:
        a, b = d[d.group == "A"], d[d.group == "B"]
        r = delta_method_ratio(a.revenue, a.orders, b.revenue, b.orders)
        st.dataframe(pd.DataFrame([fmt_result("Delta method", r)]), hide_index=True, use_container_width=True)

# --- Methods ---------------------------------------------------------------------------
with tab_methods:
    st.markdown("**Welch t-test.** Не предполагает равенства дисперсий; степени свободы — по формуле Уэлча–Саттертуэйта.")
    st.latex(r"t = \frac{\bar{Y}_B - \bar{Y}_A}{\sqrt{s_A^2/n_A + s_B^2/n_B}}")
    st.markdown("**Delta method** для отношения R = X̄ / Ȳ (числитель и знаменатель на пользователя):")
    st.latex(r"\operatorname{Var}(\hat R) \approx \frac{1}{n}\left(\frac{\sigma_X^2}{\mu_Y^2} - "
             r"\frac{2\mu_X \sigma_{XY}}{\mu_Y^3} + \frac{\mu_X^2 \sigma_Y^2}{\mu_Y^4}\right)")
    st.markdown("**CUPED.** X — значение метрики до эксперимента. θ и X̄ оцениваются по объединённой выборке, "
                "поэтому оценка эффекта несмещённая; дисперсия снижается в (1 − ρ²) раз.")
    st.latex(r"Y' = Y - \theta\,(X - \bar{X}), \qquad \theta = \frac{\operatorname{Cov}(X, Y)}{\operatorname{Var}(X)}")
    st.markdown("**SRM.** Хи-квадрат тест на соответствие фактических размеров групп плановому сплиту; "
                "при p < 0.001 результаты теста не интерпретируются.")
    st.markdown("**Bootstrap.** 3 000 ресемплингов с возвращением в каждой группе; 95% CI — перцентильный, "
                "p-value — удвоенная доля бутстрап-разниц по другую сторону от нуля.")

st.divider()
st.caption("github.com/juggler14/ab-testing-toolkit")
