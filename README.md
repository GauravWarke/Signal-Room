# Signal Room

**Five Python dashboards that turn 15 years of market data into business decisions.**

### [▶ Open the live site](https://gauravwarke.github.io/Signal-Room/)

Each project starts from a decision someone in a business actually has to make, not from a dataset. Should an investing app warn a buyer? Should a fund step out of the market? Can a super fund rely on bonds in a crash? Should a trading desk hedge before results day? Should a risk team tighten tomorrow's loss limit?

Every answer comes with the numbers behind it, a chart that shows the pattern, a recommendation for each person involved, and an honest score for how far the prediction can be trusted.

---

## The five dashboards

| Dashboard | Who it's for | The question | What the data says |
|---|---|---|---|
| [**Hold-ability Score**](https://gauravwarke.github.io/Signal-Room/hold-ability.html) | A retail investing app | Which big-name stocks can an ordinary investor actually hold, and when should the app warn them? | Only 2 of the 10 biggest US companies of 2015 gave more growth for less stress than the S&P 500. A "you may spend most of next year in a slump" warning works on unseen years (77%); a "you'll lose money" warning doesn't (46%). |
| [**Steady-Ride Fund**](https://gauravwarke.github.io/Signal-Room/steady-ride-fund.html) | The product team at an ETF issuer | Can a simple rule make the S&P 500 a smoother ride, and what does it cost? | On years the rule never saw (2018–2024), its biggest fall was −19.3% vs −33.7% for the index. The price was 2.9 points of growth a year. |
| [**Crisis Check**](https://gauravwarke.github.io/Signal-Room/crisis-diversification.html) | A superannuation fund's investment committee | Which assets really protect a share portfolio in a panic? | Only bonds and gold held steady on crash days. Since 2022 bonds rose on about 44% of crash days, down from about 89%. |
| [**Bank Earnings**](https://gauravwarke.github.io/Signal-Room/bank-earnings-risk.html) | Investor relations and trading-risk teams at a large US bank | How hard do results days hit the big six US banks? | A typical results day moves a bank 2.4× an ordinary day, even after removing market-wide moves. The following week is a coin flip. |
| [**Sector Risk**](https://gauravwarke.github.io/Signal-Room/sector-risk-budget.html) | The risk manager of an equity fund | How much can each sector lose on a bad day, and when does the risk model fail? | The standard risk model's daily limit broke on 2.4% of days, not the 1% it promises. A "tighten tomorrow" warning flag scored 80% on unseen years. |

---

## How each project works

1. **Start with the business problem.** Who decides what, and what would change their mind?
2. **Collect and clean the data.** Daily prices from Yahoo Finance, with duplicates removed, gaps filled and every model input lagged so it was known at the time.
3. **Explore it.** Key numbers, rankings and a sortable table of every series.
4. **Show it.** One main chart and a few supporting ones, each titled with what it shows. Some are interactive (hover for values), most are still.
5. **Turn it into a recommendation**, and test whether the explanation behind it holds up.

### Three models, used the same way every time

| Model | What it answers | Example |
|---|---|---|
| **Multiple regression** | How big? | How volatile will next month be? |
| **Poisson regression** | How many? | How many −3% days should a buyer expect? |
| **Binomial regression** | Yes or no? | Should the desk hedge before this report? |

The yes/no models are fitted on earlier years and scored only on later years they never saw. The dashboards say plainly when a model is weak, as in Bank Earnings (58%), rather than hiding it.

### Checking the "why"

A pattern isn't an explanation. Each project tests its explanation against the obvious alternative. Two examples:
- **Steady-Ride:** is it the timing that helps, or just holding less stock? A fixed mix with the same average did much worse (−29.9% vs −19.3%), and shuffling the timing made it worse every time.
- **Crisis Check:** did bonds stop protecting because of inflation fears or rising interest rates? Rising rates explain it, including back in 2013. Inflation fears add nothing once rates are known.

---

## What's on each dashboard

- **Key numbers** with a clear "favourable" or "watch" tag
- **A main chart** with supporting charts beneath it
- **A decision slider** that runs the yes/no model live
- **Recommended actions** for each person involved
- **A reliability ring** showing how often the decision model gets it right
- **A Trust Check card** with where the data came from, how it was cleaned, how it was measured and its limits
- Dark mode

---

## Run it yourself

```bash
pip install -r requirements.txt
python run_all.py      # builds all five dashboards and the overview into docs/, then checks them
python serve.py        # opens http://localhost:8123
```

The price data is saved in each project's `data/` folder, so the numbers match the site exactly. To download fresh data instead, run `python run_all.py --refresh`. Yahoo Finance sometimes revises old prices, so results can move slightly.

## Project layout

```
Signal-Room/
├── 01-hold-ability/              analysis.py · README.md · data/
├── 02-steady-ride-fund/
├── 03-crisis-diversification/
├── 04-bank-earnings-risk/
├── 05-sector-risk-budget/
├── docs/assets/                  site styling and scripts
├── report.py                     shared dashboard builder
├── build_site.py                 overview page
├── run_all.py                    one command to build and check everything
├── serve.py                      local preview
└── .github/workflows/pages.yml   builds and publishes the site on every push
```

## Built with

Python · pandas · NumPy · SciPy · statsmodels · matplotlib · yfinance · Chart.js · GitHub Actions · GitHub Pages

## Limits

- Daily prices only; no taxes, fund fees or intraday moves.
- One market (US) and one period (2010–2024). The next crisis may behave differently.
- The tests show the evidence supports each explanation. They aren't controlled experiments and don't prove cause.

*Not financial advice.*
