# Bank Earnings Shock Monitor

**Client:** Investor relations and trading-risk teams at a large US bank
**Business problem:** Results days drive a bank's biggest share-price moves. IR needs to prepare; the desk needs to size positions.
**Question:** How hard do results days hit the big six US banks beyond the market's own move?
**Data:** JPM, BAC, C, WFC, GS, MS, SPY and VIX; 60 quarterly reports per bank, 2010–2024 (Yahoo Finance via `yfinance`)
**Methods:** Event study · market-adjusted (abnormal) returns · clustered standard errors

## Key findings
- A typical results day moves a bank about 2.4× a normal day after removing the market's move. The next week's direction is a coin flip.
- **Multiple regression:** market fear (VIX) makes moves bigger but explains only 7% of their size.
- **Poisson regression:** 10 more VIX points roughly doubles the yearly count of 3%+ shocks.
- **Binomial regression (decision):** predicting a shock scored AUC 0.58 on unseen years. Weak: use it to rank upcoming reports, not to trade.

## What, why, how, when, where
- **What:** see Key findings.
- **Why:** Results reveal loan losses and trading income in one release, so the share price resets in a day. Fear (high VIX) makes every day more volatile, but the tests show results days are not extra sensitive to fear.
- **How:** Removed the market's own move using each bank's beta from the year before, compared results-day moves with normal days, and modelled shock size, shock counts and the chance of a 3%+ shock.
- **When:** On results day itself, from the open. Risk is highest in fearful years (2011, 2020, 2022). The following week shows no reliable follow-through.
- **Where:** The six largest US banks; GS and MS react slightly more. Used in investor-relations prep and in the trading desk's position sizing before results.

## How we know: testing the why
- **Supported:** Results days move banks more than ordinary days. *Test:* Placebo test: drew 2,000 sets of 312 random non-results days and compared their average move with the real results days. *Result:* Real results days: 2.36% average move. Random days: 0.88% on average and never above 1.08%.
- **Not supported:** Fear makes results-day reactions extra sharp. *Test:* Regressed move size (vs each bank's normal) on VIX, a results-day flag and their interaction, errors clustered by bank. *Result:* Higher VIX enlarges moves on all days (p < 0.001). The extra effect on results days is +0.022 per VIX point (p = 0.406).

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
