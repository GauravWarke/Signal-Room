# Hold-ability Score

**Client:** A retail investing app (product, compliance and app users)
**Business problem:** App users buy famous stocks, then panic-sell during drops. The product team wants a warning on stock pages that sets honest expectations before someone buys.
**Question:** Which big-name stocks could an ordinary investor actually hold, and when should the app warn a buyer?
**Data:** The 10 largest US companies on 1 Jan 2015 (AAPL, XOM, MSFT, BRK-B, GOOGL, JNJ, WFC, WMT, GE, JPM) and SPY, daily, 2015–2024 (Yahoo Finance via `yfinance`)
**Methods:** Product analytics · designing an in-app warning from data · panel data (950 stock-months) · out-of-sample testing of a decision model

## Key findings
- Only MSFT and AAPL gave a better growth-for-stress trade than the S&P 500. GE spent 71% of days more than 10% under water.
- **Multiple regression:** the signals a buyer sees explain only 18% of next-year returns. Being far below the peak did not predict a rebound.
- **Poisson regression:** stocks already in a slump keep having bad days (×1.27 more −3% days per extra 10% below peak).
- **Binomial regression (decision):** a 'you may spend most of next year under water' warning scored AUC 0.77 on unseen 2021–2023 data. A 'you will lose money' warning failed (0.46). Recommendation: ship the slump warning, not the loss warning.

## What, why, how, when, where
- **What:** see Key findings.
- **Why:** Investors judge a stock by its long-run return but live through it day by day. Long stretches below the peak are what trigger panic selling, and a stock already in a slump tends to keep having bad days.
- **How:** Scored each stock on time spent more than 10% below its peak and on return per unit of pain. A binomial model uses one thing a buyer can see, distance below the peak, to flag likely long slumps. It was tested on years it never saw.
- **When:** At the moment someone buys, and again whenever a holding falls more than 10% below its peak.
- **Where:** On single-stock pages in the app, for large US companies. Not tested on small companies or crypto.

## How we know: testing the why
- **Supported:** Being in a slump brings more bad days, not just being a 'bad stock'. *Test:* Poisson model with a separate baseline for every stock, so each stock is compared only with itself at other times. *Result:* Within the same stock, each extra 10% below the peak still means ×1.31 more −3% days (p < 0.001).
- **Supported:** The pattern is stable, not a one-period accident. *Test:* Fitted the same model separately on 2015–2019 and 2020–2023. *Result:* 2015–2019: ×1.28 (p < 0.001) · 2020–2023: ×1.31 (p < 0.001)

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
