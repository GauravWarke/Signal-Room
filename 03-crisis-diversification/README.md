# Crisis Diversification Check

**Client:** A superannuation fund investment committee
**Business problem:** Funds diversify so one asset holds up when another falls. That promise is only tested in a panic.
**Question:** Which assets actually protected a share portfolio in a panic, and has that changed?
**Data:** SPY, EFA, EEM, VNQ, GLD, TLT, IEF and VIX, daily, 2010–2024 (Yahoo Finance via `yfinance`)
**Methods:** Regime analysis · interaction terms in regression · portfolio backtesting with rebalancing and costs

## Key findings
- Only bonds and gold held steady on crash days. Foreign shares and property fell with US shares, more so in panics.
- **Multiple regression:** bonds moved −0.16% per 1% move in shares before 2022. From 2022 that cushion mostly disappeared.
- **Poisson regression:** days when shares and bonds fall together are 5× as common since 2022, at the same VIX.
- **Binomial regression (decision):** bonds rose on about 89% of crash days before 2022, and about 44% since. The committee should add a second hedge such as gold.

## What, why, how, when, where
- **What:** see Key findings.
- **Why:** In a panic, investors sell every risky asset at once, so foreign shares and property fall with US shares. Bonds cushion shares only while interest rates are falling or stable. When rates rise, bond prices fall at the same time as shares, as in 2013 and from 2022. The data points to rising rates, not inflation fears as such.
- **How:** Compared each asset's link to US shares on calm and panic days, measured returns on the 5% worst days, and modelled the chance bonds rise on a crash day before and after 2022.
- **When:** When VIX is above 25, and above all in inflation-driven sell-offs like 2022.
- **Where:** In the fund's asset mix: keep bonds but add a second hedge such as gold. Don't count on international shares or property as crash insurance.

## How we know: testing the why
- **Supported:** Bonds stop cushioning shares when interest rates are rising. *Test:* Regressed the 1-year stock–bond correlation on the 1-year change in the 10-year Treasury yield, monthly, with Newey-West errors. *Result:* Each 1-point rise in yields lifts the correlation by +0.22 (p < 0.001). It also holds using only pre-2022 data (+0.17, p < 0.001), e.g. the 2013 'taper tantrum'.
- **Not supported:** Inflation fears, on their own, broke the link. *Test:* Added an inflation-expectations proxy to the same model: inflation-linked bond (TIP) return minus normal bond (IEF) return. *Result:* Effect -0.37 (p = 0.554). Once rate changes are known, inflation expectations add no clear information.

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
