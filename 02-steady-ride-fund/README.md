# Steady-Ride Fund Test

**Client:** The product team at an ETF issuer
**Business problem:** Cautious investors want S&P 500 exposure but abandon it in crashes. The team is testing a fund that holds less stock when markets swing.
**Question:** Would a simple volatility rule give a smoother ride, and what does it cost?
**Data:** SPY, VIX and the 13-week T-bill yield, daily, 2010–2024 (Yahoo Finance via `yfinance`)
**Methods:** Product trade-off analysis · train/test backtest design · trading costs · forecasting with regression

## Key findings
- On unseen 2018–2024 data the worst drop fell from −33.7% to −19.3%. The cost was 2.9 points of growth a year.
- **Multiple regression:** VIX and this month's swings explain 40% of next month's volatility. Risk persists, which is why the rule works.
- **Poisson regression:** 10 more VIX points multiplies the expected number of −2% days by 2.5.
- **Binomial regression (decision):** predicting a 5%+ monthly fall scored AUC 0.62 on unseen years. Weaker than the simple rule, so the product should use the rule and market it as 'shallower drops', not 'better returns'.

## What, why, how, when, where
- **What:** see Key findings.
- **Why:** Market swings come in clusters: a volatile month is usually followed by another. Holding less stock when swings are high avoids much of the worst stretch, while calm periods keep full exposure.
- **How:** Each Friday, exposure = target volatility ÷ recent volatility, capped at 100%; the rest earns T-bill interest. Settings were chosen on 2010–2017 and judged only on 2018–2024, after trading costs.
- **When:** Rebalanced weekly. The rule steps back when swings spike and returns as markets calm. It helps most in fast crashes like 2020 and less in slow declines like 2022.
- **Where:** As a separate 'steady' S&P 500 fund for cautious investors. Not for investors who want the full index return and can sit through full drops.

## How we know: testing the why
- **Supported:** Swings come in clusters, so recent volatility warns of more. *Test:* Correlation between one month's volatility and the next month's, 2010–2024. *Result:* Correlation 0.50: a stormy month is usually followed by another.
- **Supported:** It is the timing that protects, not just owning less stock. *Test:* Compared the rule with a fixed 87% in stocks (same average), and with the rule's own weekly exposures shuffled into random order 500 times. *Result:* Worst drop: rule -19.3%, fixed 87% mix -29.9%. Random timing did worse than the rule in 100% of 500 shuffles.

## Models
| Model | Outcome type | Used for |
|---|---|---|
| Multiple (OLS) regression | Continuous | Explaining or forecasting a size (return, volatility) |
| Poisson regression | Count | Expected number of bad days, shocks or breaches |
| Binomial (logistic) regression | Yes/No | The business decision, tested on unseen years wherever the data allows |

Running the script saves the fitted coefficients to `model_coefficients.csv`. On the dashboard, the Trust Check card shows how reliable each model is, and a slider lets you run the decision model yourself.

## Run it
```bash
pip install -r ../requirements.txt
python analysis.py
```
This writes the project's page into `../docs/` along with the CSV results. Then run `python ../build_site.py` to refresh the overview page. Prices are saved in `data/`; delete that folder if you want fresh data.

Not financial advice.
