# A/B Testing Toolkit

**English** · [Русский](README.ru.md)

**Live app:** [ab-testing-toolkit-kozulin.streamlit.app](https://ab-testing-toolkit-kozulin.streamlit.app)

A Python library (`abkit`) and a Streamlit app for A/B test design and analysis: sample size and MDE, Welch's t-test, two-proportion z-test, bootstrap, delta method for ratio metrics, CUPED, and SRM checks. Every method is validated with simulations (A/A tests, power, peeking); CUPED is also evaluated on real revenue data for 4,239 customers from UCI Online Retail II.

**Stack:** Python · NumPy · SciPy · pandas · Streamlit · Plotly · pytest

![Analysis](reports/figures/app_analysis.jpg)

---

## Validation results

| Check | Result | Interpretation |
|---|---|---|
| 10,000 A/A tests | false positive rate **4.9%** (95% CI 4.5–5.3%) | Welch's t-test holds the nominal α = 5% even on log-normal revenue |
| Power at the computed n | **78%** vs. 80% planned | the sample size formula works; the small gap comes from the heavy right tail |
| Peeking (10 interim looks) | false positive rate **18.8%** instead of 5% | stopping at the first p < 0.05 almost quadruples the risk of a false conclusion |
| AOV: order-level t-test | false positive rate **27.9%** | orders of the same user are correlated, so the naive test underestimates variance |
| AOV: delta method | false positive rate **4.7%** | correct variance for a ratio metric randomized by user |
| CUPED on real data | variance **−75%**, MDE **44% → 22%**, power **29% → 80%** | equivalent to a 4x larger sample without a single extra customer |

### Peeking

![Peeking](reports/figures/peeking.png)

### CUPED on real revenue

Metric: customer revenue for Dec 2010 – Nov 2011. Covariate: the same customer's revenue for the previous 12 months (correlation 0.87). 2,000 random splits into groups, with a +25% effect added to group B.

![CUPED](reports/figures/cuped_power.png)

**A single split (seed = 7, screenshot above).** Pre-period revenue differs between groups by chance: £2,212 in A vs. £1,713 in B. With a true effect of +25%, Welch's t-test estimates +2.4% (SE = 266, p = 0.87). With CUPED (θ = 0.873) the estimate is +27.2% (SE = 141, p = 0.0007): the pre-period adjustment removes the group imbalance and halves the confidence interval.

### A/A tests

![A/A](reports/figures/aa_pvalues.png)

### Test design

![Design](reports/figures/app_design.jpg)

## Project structure

| Module | Contents |
|---|---|
| `abkit/design.py` | `sample_size_means`, `sample_size_proportions`, `mde_means`, `power_means` |
| `abkit/analysis.py` | `welch_ttest`, `ztest_proportions`, `bootstrap_test`, `delta_method_ratio`, `cuped`, `srm_check` |
| `app.py` | Streamlit app (EN/RU): test design, mean metric with CUPED, conversion, AOV, methods reference |
| `simulations/run_simulations.py` | all checks from the table above → `reports/simulation_results.json` |
| `tests/` | 9 unit tests: cross-checked against SciPy and textbook sample sizes |
| `data/customer_periods.csv` | revenue and orders of 4,239 customers before and during the "experiment" (exported with `data/customer_periods.sql` from [retail-rfm-cohort-analysis](https://github.com/juggler14/retail-rfm-cohort-analysis)) |

### Usage

```python
from abkit import sample_size_means, welch_ttest, cuped

n = sample_size_means(sd=3000, mde=75)            # 25,117 users per group
res = welch_ttest(control, treatment)             # effect, 95% CI, p-value, SE, t, df
cu = cuped(y_a, x_a, y_b, x_b)                    # x — the same metric before the test
print(cu.variance_reduction, welch_ttest(cu.y_a, cu.y_b).p_value)
```

## Methods

- **Sample size:** n = (z₁₋α/₂ + z₁₋β)² · 2σ² / MDE² per group.
- **Welch's t-test** instead of Student's: does not assume equal variances.
- **Delta method:** for ratio metrics (AOV = Σrevenue / Σorders) randomized by user; the variance of the ratio is linearized through the variances and covariance of the numerator and denominator.
- **CUPED:** Y′ = Y − θ(X − X̄), θ = cov(X, Y) / var(X); θ and X̄ are estimated on the pooled sample, so the effect estimate stays unbiased. Variance reduction ≈ ρ².
- **SRM:** chi-square goodness-of-fit test of the observed split against the planned one; if p < 0.001, test results should not be trusted.

## Running locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                                   # unit tests
python simulations/run_simulations.py    # simulations (~1 min)
streamlit run app.py                     # app at http://localhost:8501
```

## Limitations

- Frequentist methods only; Bayesian testing and sequential testing (the correct alternative to peeking) are not implemented.
- Offline retail, where stores rather than customers are randomized, needs additional methods (stratification, difference-in-differences) — a natural next step.
- In the CUPED experiment the effect is injected artificially (by scaling group B revenue): this validates the method, it is not a real test.

Data: [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) (CC BY 4.0).
