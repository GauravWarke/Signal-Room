# Sector Risk Budget

**Client:** The risk manager of a mid-size equity fund
**Business problem:** The fund sets daily loss limits per sector with a bell-curve (normal) model. If the model understates losses, limits break more often than planned.
**Question:** How much can each sector lose on a bad day, and when does the risk model fail?
**Data:** 9 SPDR sector ETFs and VIX, daily, 2010–2024 (Yahoo Finance via `yfinance`)
**Methods:** Model validation (VaR backtesting) · early-warning classifier · risk reporting in dollars

## Key findings
- The bell-curve 99% limit was broken on about 2.4% of days against the promised 1%. Energy lost about $6,660 per $100k on its worst 1% of days.
- **Multiple regression:** this month's volatility and VIX explain 40% of next month's sector volatility, so limits should be reset monthly.
- **Poisson regression:** breaches pile up when recent swings run above the past year (×1.7 at a ratio of 1.5).
- **Binomial regression (decision):** a 'tighten tomorrow's limit' flag scored AUC 0.80 on unseen 2018–2024 data. Ready to pilot as an early warning.

## What, why, how, when, where
- **What:** see Key findings.
- **Why:** Daily losses have 'fat tails': big down days happen far more often than a bell curve assumes. Swapping in a fat-tailed curve fixes most of the gap; making the model react faster helps only a little.
- **How:** Each day set the bell-curve 99% limit from the past year only, counted how often real losses broke it, and modelled breaches from recent-vs-yearly swings and the week's VIX change, tested on unseen years.
- **When:** Check every night. Breaches bunch up when recent swings run above the past year and after VIX jumps, so tighten the next day's limit then.
- **Where:** All nine US sectors, most of all Energy and Financials, inside the fund's daily loss-limit process.

## How we know: testing the why
- **Partly supported:** The model breaks because it reacts too slowly to new swings. *Test:* Replaced the 1-year average with a fast-reacting average (weights recent days most, like RiskMetrics) and re-ran the breach test. *Result:* Breach rate 2.37% → 2.15%. Faster reaction helps only a little.
- **Supported:** The model breaks because real losses have fat tails. *Test:* Kept the 1-year window but swapped the bell curve for a fat-tailed curve (Student-t, 4 degrees of freedom). *Result:* Breach rate 2.37% → 1.50%, and 1.40% with both fixes (target 1.00%). Fat tails are the main cause.

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
