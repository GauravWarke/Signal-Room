"""Project 1 - Hold-ability Score
Client: a retail investing app. Problem: users panic-sell big-name stocks during drops.
Question: which of the 10 largest US companies (as ranked on 1 Jan 2015) could an ordinary investor actually hold?
Universe is fixed in advance to avoid hindsight bias (no picking winners after the fact)."""
import os, json
import numpy as np, pandas as pd, yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # shared report.py at repo root
import report as R

TICKERS = ["AAPL", "XOM", "MSFT", "BRK-B", "GOOGL", "JNJ", "WFC", "WMT", "GE", "JPM"]
BENCH = "SPY"
START, END = "2015-01-01", "2024-12-31"
HOLE = -0.10          # "in the hole" = more than 10% below the previous peak
ANN = 252

def load_prices(tickers, start, end, cache="data/prices.csv"):
    """Download adjusted closes and volumes once with yfinance, then reuse the local copy."""
    if os.path.exists(cache):
        df = pd.read_csv(cache, index_col=0, parse_dates=True, header=[0, 1])
    else:
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, threads=False)
        df = raw[["Close", "Volume"]]
        os.makedirs("data", exist_ok=True); df.to_csv(cache)
    df = df[~df.index.duplicated(keep="first")]          # remove duplicate trading days
    return df["Close"][tickers].dropna(how="all"), df["Volume"][tickers]

def metrics(r):
    cum = (1 + r).cumprod(); dd = cum / cum.cummax() - 1
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    cagr = cum.iloc[-1] ** (1 / yrs) - 1
    ulcer = np.sqrt((dd ** 2).mean())                       # depth AND length of drawdowns
    hole = (dd < HOLE).astype(int)
    longest = hole.groupby((hole != hole.shift()).cumsum()).sum().max()
    one_yr = cum.pct_change(ANN).dropna()                   # every possible 1-year holding period
    return dict(cagr=cagr, max_dd=dd.min(), pain_to_gain=cagr / ulcer, time_in_hole=(dd < HOLE).mean(),
                longest_hole_days=int(longest), loss_1y_chance=(one_yr < 0).mean(),
                big_loss_1y_chance=(one_yr < -0.2).mean()), dd


def build_panel(px, rets):
    """One row per stock per month-end: what an app user could see on the day they buy."""
    rows = []
    for t in TICKERS:
        p, r = px[t], rets[t]
        vol = r.rolling(ANN).std() * np.sqrt(ANN)
        below = p / p.cummax() - 1
        mom = p.pct_change(ANN)
        fwd = p.shift(-ANN) / p - 1
        hole_next = (below < HOLE).astype(float)[::-1].rolling(ANN).mean()[::-1].shift(-1)   # share of NEXT year in the hole
        big_down = (r < -0.03).astype(int)[::-1].rolling(63).sum()[::-1].shift(-1)   # count in the NEXT 3 months
        me = p.groupby(p.index.to_period("M")).tail(1).index
        df = pd.DataFrame({"stock": t, "vol": vol, "below_peak": below, "momentum": mom, "fwd_1y": fwd, "big_down_3m": big_down, "hole_next": hole_next}).loc[me]
        rows.append(df)
    return pd.concat(rows).dropna()

def run_models(panel):
    X = sm.add_constant(panel[["vol", "below_peak", "momentum"]])
    labels = {"vol": "Past-year volatility (1.0 = 100%)", "below_peak": "Distance below peak (−0.2 = 20% below)", "momentum": "Past-year return (0.1 = 10%)"}
    ols = sm.OLS(panel.fwd_1y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 12})        # overlapping 1-year windows
    poi = sm.GLM(panel.big_down_3m, X, family=sm.families.Poisson()).fit(cov_type="HC0")
    train = panel.index < "2021-01-01"                  # fit on 2015-2020, judge on 2021-2023
    lose = (panel.fwd_1y < 0).astype(int)                # tried first: "will next year lose money?"
    lose_auc = R.auc(lose[~train], sm.GLM(lose[train], X[train], family=sm.families.Binomial()).fit().predict(X[~train]))
    y = (panel.hole_next > 0.5).astype(int)             # decision target: "most of next year >10% under water"
    # One driver: adding volatility gave a confusing negative sign (famous high-volatility winners) for almost no gain.
    Xd = sm.add_constant(panel[["below_peak"]])
    logit = sm.GLM(y[train], Xd[train], family=sm.families.Binomial()).fit()
    test_auc = R.auc(y[~train], logit.predict(Xd[~train]))
    full = sm.GLM(y, Xd, family=sm.families.Binomial()).fit(cov_type="HC0")
    return labels, ols, poi, full, test_auc, lose_auc, y

def evidence(panel):
    """Test the 'why': do slumps themselves bring more bad days, or is it just that some stocks are always bad?"""
    X = pd.get_dummies(panel.stock, drop_first=True, dtype=float)       # one dummy per stock = compare a stock only with itself
    X["below_peak"], X["vol"] = panel.below_peak, panel.vol; X = sm.add_constant(X)
    fe = sm.GLM(panel.big_down_3m, X, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": panel.stock.astype("category").cat.codes})
    rr = np.exp(-0.1 * fe.params["below_peak"])
    halves = []
    for lab, s in [("2015–2019", panel.index < "2020-01-01"), ("2020–2023", panel.index >= "2020-01-01")]:
        q = sm.GLM(panel.big_down_3m[s], X[s], family=sm.families.Poisson()).fit(cov_type="HC0")
        halves.append((lab, np.exp(-0.1 * q.params["below_peak"]), q.pvalues["below_peak"]))
    return [
        dict(claim="Being in a slump brings more bad days, not just being a 'bad stock'.",
             test="Poisson model with a separate baseline for every stock, so each stock is compared only with itself at other times.",
             result=f"Within the same stock, each extra 10% below the peak still means ×{rr:.2f} more −3% days ({R.pt(fe.pvalues['below_peak'])}).",
             verdict="Supported" if fe.pvalues["below_peak"] < 0.05 else "Not supported"),
        dict(claim="The pattern is stable, not a one-period accident.",
             test="Fitted the same model separately on 2015–2019 and 2020–2023.",
             result=" · ".join(f"{l}: ×{v:.2f} ({R.pt(p)})" for l, v, p in halves),
             verdict="Supported" if all(p < 0.05 for _, _, p in halves) else "Partly supported")]

def main():
    R.style()
    px, _ = load_prices(TICKERS + [BENCH], START, END)
    rets = px.pct_change().dropna()
    rows, dds = {}, {}
    for t in TICKERS + [BENCH]:
        rows[t], dds[t] = metrics(rets[t])
    m = pd.DataFrame(rows).T
    m["beats_spy_on_pain"] = m.pain_to_gain > m.loc[BENCH, "pain_to_gain"]
    m.round(4).to_csv("results.csv")
    stocks = m.drop(BENCH).sort_values("pain_to_gain", ascending=False)
    best, worst = stocks.index[0], stocks.index[-1]
    n_beat = int(stocks.beats_spy_on_pain.sum())
    S = m.loc[BENCH]

    # Chart 1 (interactive, Chart.js): reward-for-stress ranking with the S&P 500 as its own bar. Hover for the score.
    rank = pd.concat([stocks.pain_to_gain, pd.Series({"S&P 500": S.pain_to_gain})]).sort_values(ascending=False)
    c1 = R.js_chart(dict(type="bar", horizontal=True, fmt_x="num1", x_title="Reward-for-stress score (higher = more growth for less time in losses)",
        labels=list(rank.index), datasets=[dict(label="Reward-for-stress score", data=rank.round(2).tolist(),
        colors=[R.INK if t == "S&P 500" else (R.ACC if t in (best, worst) else R.GREY) for t in rank.index])]))

    # Chart 2 (interactive, Chart.js): return vs time in the hole. Hover a dot to see the stock.
    c2 = R.js_chart(dict(type="scatter", fmt_x="pct0", fmt_y="pct0", x_title="Share of days spent 10%+ below the previous high", y_title="Average growth per year",
        datasets=[dict(label="Stocks", color=R.ACC, points=[dict(x=round(float(m.loc[t, "time_in_hole"]), 4), y=round(float(m.loc[t, "cagr"]), 4), name=t) for t in TICKERS]),
                  dict(label="S&P 500", color=R.INK, points=[dict(x=round(float(S.time_in_hole), 4), y=round(float(S.cagr), 4), name="S&P 500")])]))

    # Chart 3: one panel per series (small multiples) instead of three overlapping lines
    fig, axes = plt.subplots(3, 1, figsize=(9, 6.4), sharex=True, sharey=True)
    for ax, (t, col, name) in zip(axes, [(BENCH, R.INK, "S&P 500 (whole market)"), (best, R.GOOD, f"{best} (easiest to hold)"), (worst, R.BAD, f"{worst} (hardest to hold)")]):
        d = dds[t] * 100
        ax.fill_between(d.index, d, 0, color=col, alpha=.45, lw=0); ax.plot(d.index, d, color=col, lw=1)
        ax.axhline(HOLE * 100, color=R.MUTE, lw=.9, ls=":")
        ax.text(0.99, 0.08, f"{name} · deepest fall {d.min():.0f}%", transform=ax.transAxes, fontsize=12, color=R.INK, ha="right", bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=3))
        ax.set_ylim(-85, 3); ax.set_yticks([0, -20, -40, -60, -80])
    axes[1].set_ylabel("% below previous high")
    c3 = R.fig_to_b64(fig)

    # Chart 4: chance a random 1-year hold loses money
    order = m.sort_values("loss_1y_chance").index
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar(order, m.loc[order, "loss_1y_chance"] * 100, color=[R.INK if t == BENCH else R.ACC for t in order], alpha=.85, label="Any loss")
    ax.bar(order, m.loc[order, "big_loss_1y_chance"] * 100, color=R.BAD, label="Loss worse than 20%")
    ax.set_ylabel("Chance of a losing year (%)"); ax.legend()
    c4 = R.fig_to_b64(fig)

    pct = lambda x: f"{x*100:.0f}%"
    tbl = m[["cagr", "max_dd", "pain_to_gain", "time_in_hole", "longest_hole_days", "loss_1y_chance"]].copy()
    tbl["longest_hole_days"] = tbl["longest_hole_days"].astype(int)
    tbl.columns = ["Growth per year", "Biggest fall", "Reward-for-stress score", "Days 10%+ below high", "Longest slump (days)", "Chance of losing over 1 year"]
    for c in ["Growth per year", "Biggest fall", "Days 10%+ below high", "Chance of losing over 1 year"]: tbl[c] = tbl[c].map(pct)
    tbl["Reward-for-stress score"] = tbl["Reward-for-stress score"].map(lambda x: f"{x:.2f}")

    panel = build_panel(px, rets)
    labels, ols, poi, full, test_auc, lose_auc, y = run_models(panel)
    pd.DataFrame({'ols': ols.params, 'poisson': poi.params, 'logit': full.params}).round(4).to_csv('model_coefficients.csv')
    rr = np.exp(poi.params["below_peak"] * -0.10)
    models = [
        ("Multiple regression · continuous", "Does what a buyer sees predict the next year's return?",
         f"Only partly. The signals explain {ols.rsquared*100:.0f}% of next-year returns, and only volatility matters ({R.pt(ols.pvalues['vol'])}). Being far below the peak did not predict a rebound, so the app should never hint a stock is 'due'.",
         f"OLS on {len(panel):,} stock-months, Newey-West errors for overlapping years. R² = {ols.rsquared:.3f}.", R.coef_table(ols, labels, "ols")),
        ("Poisson regression · count", "How many −3% days should a buyer expect in the next 3 months?",
         f"Stocks already in a slump keep having bad days. Each extra 10% below the peak multiplies the expected number of −3% days by {rr:.2f}. Past volatility adds little once that is known.",
         f"Poisson GLM, robust errors. Average count: {panel.big_down_3m.mean():.1f} days per 3 months.", R.coef_table(poi, labels, "poisson")),
        ("Binomial regression · decision", "Should the app warn 'you may spend most of next year under water'?",
         f"Yes. Trained on 2015–2020 and tested on 2021–2023, it ranked a long-slump case above a normal one {test_auc*100:.0f}% of the time (50% = guessing). A 'will you lose money?' warning failed the same test ({lose_auc*100:.0f}%), so the app should warn about slumps, not losses.",
         f"Logistic GLM. Outcome: more than half of next year spent 10%+ below the peak ({y.mean()*100:.0f}% of cases). Test AUC {test_auc:.2f}.", R.coef_table(full, {"below_peak": labels["below_peak"]}, "logit"))]
    b = full.params
    sim = R.sim_html("Try it: should the app warn this buyer?", "Move the slider to how far the stock is below its previous high today. The model gives the chance the buyer spends most of the next year more than 10% down.",
        float(b["const"]), [("below_peak", "How far below its previous high today", -0.50, 0, 0.01, -0.05, float(b["below_peak"]), 2, "")],
        35, "Show warning", "No warning needed")

    R.build("dashboard.html", "Hold-ability Score", "Retail investing app · 10 largest US stocks of Jan 2015",
        "Which big-name stocks could an ordinary investor actually hold?",
        f"Only {n_beat} of the 10 biggest companies gave more growth for less stress than simply owning the whole market (S&P 500). "
        f"{best} was the easiest to hold. {worst} spent {pct(m.loc[worst,'time_in_hole'])} of days more than 10% below its previous high.",
        [("Better than the index", f"{n_beat} of 10", "big stocks that gave more growth for less stress than the S&P 500", "good" if n_beat > 5 else "bad"),
         ("Easiest to hold", best, f"score {m.loc[best,'pain_to_gain']:.1f} vs {S.pain_to_gain:.1f} for the S&P 500", "good"),
         ("Hardest to hold", worst, f"longest slump: {int(m.loc[worst,'longest_hole_days'])} trading days (about {int(m.loc[worst,'longest_hole_days'])/252:.1f} years)", "bad"),
         ("Chance of a losing year", pct(S.loss_1y_chance), f"for the S&P 500, vs {pct(m.drop(BENCH).loss_1y_chance.mean())} for a single big stock", "")],
        [("App product team", "Add a warning on stock pages: 'this stock has spent X% of the last 10 years more than 10% below its high'. People who expect the dips are less likely to panic-sell."),
         ("Investor (app user)", "Only 2 of the 10 biggest companies beat simply owning the whole market. One company is a bigger bet than it looks."),
         ("Compliance / risk", "Show 'chance of losing money over a year' in risk warnings. People understand it better than technical risk scores.")],
        [(f"Ranking: {best} gave the most growth for the least stress; {worst} the least", "Score = yearly growth divided by how deep and how long the falls were. The dashed line is the whole market (S&P 500).", c1),
         ("Growth vs stress: Faster-growing stocks often spent long stretches in the red", "Each dot is a company. Further right = more days spent 10% or more below its previous high. Hover a dot for the name.", c2),
         (f"Slumps over time: {worst} spent years far below its high; the market recovered quickly", "One panel each, on the same scale. The shaded area shows how far each sat below its previous high; the dotted line marks a 10% fall.", c3),
         ("Losing years: Owning one company for a year lost money more often than owning the market", "Out of every possible 1-year period: how often you ended with any loss (blue) or a loss worse than 20% (red).", c4)],
        ["Prices: Yahoo Finance adjusted closes (dividends included), daily, Jan 2015 to Dec 2024.",
         "Pain-to-Gain = compound yearly growth ÷ Ulcer Index (root-mean-square of % below peak).",
         "Time in the hole = share of days more than 10% below the previous high.",
         "Chance of 1-year loss = share of all rolling 252-day windows with a negative return.",
         "Models use month-end snapshots per stock: past-year volatility, distance below peak and past-year return as drivers."],
         ["The list is the 10 largest companies on 1 Jan 2015, chosen in advance. This removes picking-winners bias, but one decade is still one market cycle.",
          "No taxes or trading costs. Buy-and-hold only.",
          "Past hold-ability does not predict future hold-ability."],
        tbl.to_html(border=0, classes="sort"), "Data: Yahoo Finance via yfinance", models, sim, evidence=evidence(panel))
    print(m.round(3))

if __name__ == "__main__":
    main()
