"""Streamlit app for A/B test design and analysis. Run: streamlit run app.py"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

from abkit import (bootstrap_test, cuped, delta_method_ratio, power_means, sample_size_means,
                   sample_size_proportions, srm_check, welch_ttest, ztest_proportions)

st.set_page_config(page_title="A/B Testing Toolkit", layout="wide")
NAVY, LIGHT, GRID = "#1f3a5f", "#9fb3c8", "#e5e9ef"
DEMO = Path(__file__).parent / "data" / "customer_periods.csv"
FONT = "IBM Plex Sans, sans-serif"

TEXT = {
    "en": {
        "subtitle": "Sample size and MDE, Welch's t-test, two-proportion z-test, bootstrap, delta method, CUPED, "
                    "SRM check. Every method is validated with simulations — see README.",
        "tabs": ["Design", "Mean metric + CUPED", "Conversion", "Ratio metric (AOV)", "Methods"],
        "metric_type": "Metric type", "mean_metric": "Mean per user", "conversion": "Conversion",
        "alpha": "α (significance level)", "power": "1 − β (power)", "traffic": "Users per day, both groups",
        "mu": "μ — current mean", "sigma": "σ — standard deviation", "mde_mean": "MDE, % of μ",
        "p": "p — current conversion, %", "mde_p": "MDE, % of p (relative)",
        "n_group": "n per group", "n_total": "n total", "days": "Days",
        "power_curve": "Power curve at n = {n} per group", "true_effect": "true effect, % of μ", "power_axis": "power, %",
        "n_col": "n per group",
        "csv_mean": "CSV: `group` (A/B), `value` — metric per user, `pre_value` — the same metric before the test "
                    "(optional, enables CUPED).",
        "data": "Data", "demo": "Demo: revenue of 4,239 customers, UCI Online Retail II", "upload": "Upload CSV",
        "uplift": "Effect added to group B, %", "seed": "Split seed",
        "var_red": "CUPED: variance reduction",
        "srm_error": "SRM: the observed split does not match 50/50 (p < 0.001). Test results are not reliable.",
        "pre_caption": "Pre-period mean: A = {a:,.1f}, B = {b:,.1f}. If groups differed before the experiment, "
                       "the plain t-test is biased for this particular split; CUPED corrects for it.",
        "hist": "Distribution of log(1 + value) by group", "users": "users",
        "conv_error": "Conversions cannot exceed the number of users.",
        "conv_caption": "z-statistic uses pooled p; 95% CI uses unpooled SE. SRM p-value = {p:.4f}",
        "csv_ratio": "AOV = Σ revenue / Σ orders. Users are the randomization unit, so orders within a user are "
                     "correlated and an order-level t-test underestimates SE. CSV: `group`, `revenue`, `orders`.",
        "demo_ratio": "Demo: real customers, A/A split",
        "yes": "yes", "no": "no",
        "m_welch": "**Welch's t-test.** Does not assume equal variances; degrees of freedom via Welch–Satterthwaite.",
        "m_delta": "**Delta method** for the ratio R = X̄ / Ȳ (numerator and denominator per user):",
        "m_cuped": "**CUPED.** X is the metric value before the experiment. θ and X̄ are estimated on the pooled "
                   "sample, so the effect estimate stays unbiased; variance shrinks by a factor of (1 − ρ²).",
        "m_srm": "**SRM.** Chi-square goodness-of-fit test of group sizes against the planned split; "
                 "if p < 0.001, the test results are not interpreted.",
        "m_boot": "**Bootstrap.** 3,000 resamples with replacement per group; percentile 95% CI; "
                  "p-value = twice the share of bootstrap differences on the other side of zero.",
    },
    "ru": {
        "subtitle": "Размер выборки и MDE, t-тест Уэлча, z-тест для долей, бутстрап, дельта-метод, CUPED, "
                    "проверка SRM. Каждый метод проверен симуляциями — см. README.",
        "tabs": ["Дизайн", "Среднее + CUPED", "Конверсия", "Метрика-отношение (средний чек)", "Методы"],
        "metric_type": "Тип метрики", "mean_metric": "Среднее на пользователя", "conversion": "Конверсия",
        "alpha": "α (уровень значимости)", "power": "1 − β (мощность)", "traffic": "Пользователей в день, обе группы",
        "mu": "μ — текущее среднее", "sigma": "σ — стандартное отклонение", "mde_mean": "MDE, % от μ",
        "p": "p — текущая конверсия, %", "mde_p": "MDE, % от p (относительный)",
        "n_group": "n на группу", "n_total": "n всего", "days": "Дней",
        "power_curve": "Кривая мощности при n = {n} на группу", "true_effect": "истинный эффект, % от μ",
        "power_axis": "мощность, %", "n_col": "n на группу",
        "csv_mean": "CSV: `group` (A/B), `value` — метрика на пользователя, `pre_value` — та же метрика до теста "
                    "(необязательно, нужна для CUPED).",
        "data": "Данные", "demo": "Демо: выручка 4 239 клиентов, UCI Online Retail II", "upload": "Загрузить CSV",
        "uplift": "Эффект, добавляемый в группу B, %", "seed": "seed разбиения",
        "var_red": "CUPED: снижение дисперсии",
        "srm_error": "SRM: фактический сплит не соответствует 50/50 (p < 0.001). Результаты теста недостоверны.",
        "pre_caption": "Средняя метрика до теста: A = {a:,.1f}, B = {b:,.1f}. Если группы различались ещё до "
                       "эксперимента, обычный t-тест смещён в этой реализации разбиения, а CUPED это компенсирует.",
        "hist": "Распределение log(1 + value) по группам", "users": "пользователей",
        "conv_error": "Число конверсий не может превышать число пользователей.",
        "conv_caption": "z-статистика по pooled-оценке p, 95% ДИ по unpooled SE. SRM p-value = {p:.4f}",
        "csv_ratio": "Средний чек = Σ revenue / Σ orders. Единица рандомизации — пользователь, поэтому заказы "
                     "внутри пользователя зависимы, и t-тест по заказам занижает SE. CSV: `group`, `revenue`, `orders`.",
        "demo_ratio": "Демо: реальные клиенты, A/A-разбиение",
        "yes": "да", "no": "нет",
        "m_welch": "**t-тест Уэлча.** Не предполагает равенства дисперсий; степени свободы — по формуле Уэлча–Саттертуэйта.",
        "m_delta": "**Дельта-метод** для отношения R = X̄ / Ȳ (числитель и знаменатель на пользователя):",
        "m_cuped": "**CUPED.** X — значение метрики до эксперимента. θ и X̄ оцениваются по объединённой выборке, "
                   "поэтому оценка эффекта несмещённая; дисперсия снижается в (1 − ρ²) раз.",
        "m_srm": "**SRM.** Хи-квадрат тест на соответствие фактических размеров групп плановому сплиту; "
                 "при p < 0.001 результаты теста не интерпретируются.",
        "m_boot": "**Бутстрап.** 3 000 ресемплингов с возвращением в каждой группе; 95% ДИ — перцентильный, "
                  "p-value — удвоенная доля бутстрап-разниц по другую сторону от нуля.",
    },
}

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  html, body, [class*="css"], .stMarkdown, .stText, label, p, h1, h2, h3 { font-family: 'IBM Plex Sans', sans-serif !important; }
  h1 { font-weight: 600 !important; letter-spacing: -0.5px; }
  [data-testid="stMetricValue"], [data-testid="stDataFrame"], code, .mono { font-family: 'IBM Plex Mono', monospace !important; }
  [data-testid="stMetricValue"] { font-size: 1.7rem; }
  .stTabs [data-baseweb="tab"] { font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; }
  .block-container { padding-top: 3rem; max-width: 1200px; }
</style>
""", unsafe_allow_html=True)

head_l, head_r = st.columns([5, 1])
lang = head_r.segmented_control("Language", ["EN", "RU"], default="EN", label_visibility="collapsed") or "EN"
T = TEXT[lang.lower()]
head_l.title("A/B Testing Toolkit")
st.markdown(T["subtitle"])


def num(x: int) -> str:
    return f"{x:,}".replace(",", " ") if lang == "RU" else f"{x:,}"


def fmt_result(name, r, base=None):
    """base — denominator of the relative effect (for CUPED: the raw control mean)."""
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
        "H0 rejected (α=0.05)": T["yes"] if r.significant else T["no"],
    }


def style_fig(fig, height=360):
    fig.update_layout(template="plotly_white", height=height, font_family=FONT, margin=dict(t=50, l=50, r=20, b=40),
                      title_font_size=14)
    fig.update_xaxes(gridcolor=GRID); fig.update_yaxes(gridcolor=GRID)
    return fig


tab_design, tab_mean, tab_conv, tab_ratio, tab_methods = st.tabs(T["tabs"])

# --- Design --------------------------------------------------------------------------
with tab_design:
    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        is_mean = st.radio(T["metric_type"], [T["mean_metric"], T["conversion"]], horizontal=True) == T["mean_metric"]
        alpha = st.select_slider(T["alpha"], [0.01, 0.05, 0.1], value=0.05)
        power = st.select_slider(T["power"], [0.7, 0.8, 0.9, 0.95], value=0.8)
        traffic = st.number_input(T["traffic"], 100, 10_000_000, 5_000, step=500)
        if is_mean:
            mean = st.number_input(T["mu"], value=1500.0, min_value=0.01)
            sd = st.number_input(T["sigma"], value=3000.0, min_value=0.01)
            mde_rel = st.number_input(T["mde_mean"], value=5.0, min_value=0.1) / 100
            n = sample_size_means(sd, mde_rel * mean, alpha, power)
        else:
            p = st.number_input(T["p"], value=10.0, min_value=0.01, max_value=99.9) / 100
            mde_rel = st.number_input(T["mde_p"], value=5.0, min_value=0.1) / 100
            n = sample_size_proportions(p, p * mde_rel, alpha, power)
            sd, mean = np.sqrt(p * (1 - p)), p
    with c2:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(T["n_group"], num(n))
        m2.metric(T["n_total"], num(2 * n))
        m3.metric(T["days"], f"{int(np.ceil(2 * n / traffic))}")
        m4.metric("z(1−α/2) + z(1−β)", f"{norm.ppf(1 - alpha / 2) + norm.ppf(power):.3f}")
        if is_mean:
            st.latex(r"n = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2 \cdot 2\sigma^2}{\text{MDE}^2}")
        else:
            st.latex(r"n = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2 \cdot \left[p_1(1-p_1) + p_2(1-p_2)\right]}{(p_2 - p_1)^2}")
        effects = np.linspace(0.002, 3 * mde_rel, 150)
        fig = go.Figure(go.Scatter(x=100 * effects, y=[100 * power_means(sd, e * mean, n, alpha) for e in effects],
                                   line=dict(color=NAVY, width=2), name="power"))
        fig.add_hline(y=100 * power, line_dash="dot", line_color="grey")
        fig.add_vline(x=100 * mde_rel, line_dash="dot", line_color="grey")
        fig.update_layout(title=T["power_curve"].format(n=num(n)), xaxis_title=T["true_effect"],
                          yaxis_title=T["power_axis"])
        st.plotly_chart(style_fig(fig), use_container_width=True)
        grid = [0.01, 0.02, 0.03, 0.05, 0.1]
        sizes = [sample_size_means(sd, g * mean, alpha, power) if is_mean
                 else sample_size_proportions(mean, mean * g, alpha, power) for g in grid]
        st.dataframe(pd.DataFrame({"MDE, %": [f"{100 * g:.0f}" for g in grid], T["n_col"]: [num(s) for s in sizes]}),
                     hide_index=True)

# --- Mean metric ----------------------------------------------------------------------
with tab_mean:
    st.markdown(T["csv_mean"])
    use_demo = st.radio(T["data"], [T["demo"], T["upload"]], horizontal=True) == T["demo"]
    d = None
    if use_demo:
        c1, c2 = st.columns(2)
        uplift = c1.slider(T["uplift"], 0, 50, 25)
        seed = c2.number_input(T["seed"], 0, 10_000, 7)
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
        m1.metric("n_A", num(len(a)))
        m2.metric("n_B", num(len(b)))
        m3.metric("SRM p-value", f"{srm_p:.4f}")
        if "pre_value" in d:
            cu = cuped(a.value, a.pre_value, b.value, b.pre_value)
            m4.metric(T["var_red"], f"{cu.variance_reduction:.1%}")
        if srm_p < 0.001:
            st.error(T["srm_error"])
        rows = [fmt_result("Welch t-test", welch_ttest(a.value, b.value)),
                fmt_result("Bootstrap, mean", bootstrap_test(a.value, b.value, n_boot=3000)),
                fmt_result("Bootstrap, median", bootstrap_test(a.value, b.value, stat=np.median, n_boot=3000))]
        if "pre_value" in d:
            rows.append(fmt_result(f"Welch + CUPED (θ = {cu.theta:.3f})", welch_ttest(cu.y_a, cu.y_b),
                                   base=a.value.mean()))
            st.caption(T["pre_caption"].format(a=a.pre_value.mean(), b=b.pre_value.mean()))
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        fig = go.Figure()
        for g, color in (("A", LIGHT), ("B", NAVY)):
            fig.add_histogram(x=np.log1p(d.loc[d.group == g, "value"]), name=g, marker_color=color, opacity=0.7,
                              nbinsx=60)
        fig.update_layout(barmode="overlay", title=T["hist"], xaxis_title="log(1 + value)", yaxis_title=T["users"])
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
        st.caption(T["conv_caption"].format(p=srm_check(n_a, n_b)))
    else:
        st.warning(T["conv_error"])

# --- Ratio metric ----------------------------------------------------------------------
with tab_ratio:
    st.markdown(T["csv_ratio"])
    f = st.file_uploader("CSV", type="csv", key="ratio")
    d = None
    if f is not None:
        d = pd.read_csv(f)
    elif st.checkbox(T["demo_ratio"], value=True):
        d = pd.read_csv(DEMO).query("orders > 0").copy()
        d["group"] = np.where(np.random.default_rng(1).random(len(d)) < 0.5, "A", "B")
    if d is not None:
        a, b = d[d.group == "A"], d[d.group == "B"]
        r = delta_method_ratio(a.revenue, a.orders, b.revenue, b.orders)
        st.dataframe(pd.DataFrame([fmt_result("Delta method", r)]), hide_index=True, use_container_width=True)

# --- Methods ---------------------------------------------------------------------------
with tab_methods:
    st.markdown(T["m_welch"])
    st.latex(r"t = \frac{\bar{Y}_B - \bar{Y}_A}{\sqrt{s_A^2/n_A + s_B^2/n_B}}")
    st.markdown(T["m_delta"])
    st.latex(r"\operatorname{Var}(\hat R) \approx \frac{1}{n}\left(\frac{\sigma_X^2}{\mu_Y^2} - "
             r"\frac{2\mu_X \sigma_{XY}}{\mu_Y^3} + \frac{\mu_X^2 \sigma_Y^2}{\mu_Y^4}\right)")
    st.markdown(T["m_cuped"])
    st.latex(r"Y' = Y - \theta\,(X - \bar{X}), \qquad \theta = \frac{\operatorname{Cov}(X, Y)}{\operatorname{Var}(X)}")
    st.markdown(T["m_srm"])
    st.markdown(T["m_boot"])

st.divider()
st.caption("github.com/juggler14/ab-testing-toolkit")
