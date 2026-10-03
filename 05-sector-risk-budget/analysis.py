"""Project 5 - Sector Risk Budget
Client: the risk manager of a mid-size equity fund that sets a daily loss limit per sector.
Question: how much can each US sector lose on a bad day, and does the standard 'bell-curve' risk model understate it?
Fixes a weakness of the original repo: VaR was computed but never tested. Here the model is backtested:
if it works, losses should exceed the 99% limit on about 1% of days."""
import os
import numpy as np, pandas as pd, yfinance as yf
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # shared report.py at repo root
import report as R

SECTORS = {"XLK": "Technology", "XLF": "Financials", "XLE": "Energy", "XLV": "Health care", "XLY": "Consumer discretionary",
           "XLP": "Consumer staples", "XLU": "Utilities", "XLI": "Industrials", "XLB": "Materials"}
START, END = "2010-01-01", "2024-12-31"
CONF = 0.99
POSITION = 100_000      # dollar loss figures are per $100,000 held
WINDOW = 252            # the model uses the past year only (no look-ahead)

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

def backtest_normal_var(r):
    """Each day, set a 99% limit from the past year's mean and spread (bell-curve model). Count days the loss beat it."""
    z = stats.norm.ppf(1 - CONF)
    limit = (r.rolling(WINDOW).mean() + z * r.rolling(WINDOW).std()).shift(1)
    valid = limit.dropna(); hit = r.loc[valid.index] < valid
    return hit.mean(), hit


def run_models(rets, hits, vix):
    """Stack all sectors. Drivers are known at the close before the day being predicted."""
    rows = []
    for t in rets.columns:
        r = rets[t]; h = hits[t].astype(int)
        ratio = (r.rolling(20).std() / r.rolling(WINDOW).std()).shift(1)     # recent swings vs the year the model uses
        dvix = vix.reindex(r.index).ffill().pct_change(5).shift(1)          # VIX change over the past week
        rows.append(pd.DataFrame({"sector": SECTORS[t], "hit": h, "ratio": ratio.reindex(h.index), "dvix": dvix.reindex(h.index)}))
    d = pd.concat(rows).dropna()
    X = sm.add_constant(d[["ratio", "dvix"]]); lab = {"ratio": "Recent swings ÷ past-year swings (1 = same)", "dvix": "VIX change over past week (0.1 = +10%)"}
    # continuous: next-month volatility per sector-month
    mrows = []
    for t in rets.columns:
        r = rets[t]; g = r.groupby(r.index.to_period("M"))
        mv = pd.DataFrame({"vol": g.std() * np.sqrt(252), "vix": vix.reindex(r.index).ffill().groupby(r.index.to_period("M")).last()})
        mv["next_vol"] = mv.vol.shift(-1); mrows.append(mv)
    mv = pd.concat(mrows).dropna()
    ols = sm.OLS(mv.next_vol, sm.add_constant(mv[["vol", "vix"]])).fit(cov_type="HC0")
    lab_o = {"vol": "This month's volatility (0.01 = 1 point)", "vix": "VIX at month-end (1 point)"}
    # count: breaches per sector-month
    d["m"] = d.index.to_period("M")
    cm = d.groupby(["sector", "m"]).agg(breaches=("hit", "sum"), days=("hit", "size"), ratio=("ratio", "first"), dvix=("dvix", "first")).reset_index()
    poi = sm.GLM(cm.breaches, sm.add_constant(cm[["ratio", "dvix"]]), family=sm.families.Poisson(), exposure=cm.days).fit(cov_type="HC0")
    lab_p = {"ratio": "Swings ratio at start of month (1 = same)", "dvix": "VIX change the week before (0.1 = +10%)"}
    train = d.index < "2018-01-01"
    logit = sm.GLM(d.hit[train], X[train], family=sm.families.Binomial()).fit()
    test_auc = R.auc(d.hit[~train], logit.predict(X[~train]))
    return d, (ols, lab_o, mv), (poi, lab_p, cm), (logit, lab, test_auc)

def evidence(rets):
    """Test the 'why' for model breaches: slow reaction, or fat tails?"""
    z, tq = stats.norm.ppf(1 - CONF), stats.t.ppf(1 - CONF, 4) * np.sqrt(2 / 4)   # t with 4 degrees of freedom, scaled to the same spread
    def rate(r, limit): v = limit.dropna(); return (r.loc[v.index] < v).mean()
    res = {k: [] for k in ["bell_1y", "bell_fast", "fat_1y", "fat_fast"]}
    for t in rets.columns:
        r = rets[t]; sd_fast = np.sqrt((r ** 2).ewm(alpha=0.06).mean()).shift(1)
        res["bell_1y"].append(rate(r, (r.rolling(WINDOW).mean() + z * r.rolling(WINDOW).std()).shift(1)))
        res["bell_fast"].append(rate(r, z * sd_fast))
        res["fat_1y"].append(rate(r, (r.rolling(WINDOW).mean() + tq * r.rolling(WINDOW).std()).shift(1)))
        res["fat_fast"].append(rate(r, tq * sd_fast))
    a = {k: np.mean(v) for k, v in res.items()}
    return [
        dict(claim="The model breaks because it reacts too slowly to new swings.",
             test="Replaced the 1-year average with a fast-reacting average (weights recent days most, like RiskMetrics) and re-ran the breach test.",
             result=f"Breach rate {a['bell_1y']:.2%} → {a['bell_fast']:.2%}. Faster reaction helps only a little.",
             verdict="Partly supported" if a["bell_fast"] < a["bell_1y"] else "Not supported"),
        dict(claim="The model breaks because real losses have fat tails.",
             test="Kept the 1-year window but swapped the bell curve for a fat-tailed curve (Student-t, 4 degrees of freedom).",
             result=f"Breach rate {a['bell_1y']:.2%} → {a['fat_1y']:.2%}, and {a['fat_fast']:.2%} with both fixes (target 1.00%). Fat tails are the main cause.",
             verdict="Supported" if a["fat_1y"] < a["bell_fast"] else "Partly supported")]

def main():
    R.style()
    tick = list(SECTORS)
    px, _ = load_prices(tick + ["SPY", "^VIX"], START, END)
    vix = px["^VIX"].ffill()
    rets = px[tick].pct_change().dropna()
    rows, hits = {}, {}
    for t in tick:
        r = rets[t]; q = r.quantile(1 - CONF)
        z = stats.norm.ppf(1 - CONF)
        normal_var = r.mean() + z * r.std()
        normal_cvar = r.mean() - r.std() * stats.norm.pdf(z) / (1 - CONF)   # bell-curve average loss on worst 1% days
        rate, hit = backtest_normal_var(r); hits[t] = hit
        rows[SECTORS[t]] = dict(var99=-q * POSITION, cvar99=-r[r <= q].mean() * POSITION, normal_var99=-normal_var * POSITION, normal_cvar99=-normal_cvar * POSITION,
                                worst_day=-r.min() * POSITION, breach_rate=rate, fat_tail=stats.kurtosis(r))
    m = pd.DataFrame(rows); m = m.T
    m["understatement"] = m.cvar99 / m.normal_cvar99 - 1   # like-for-like: actual vs bell-curve worst-1% average
    m.round(4).to_csv("results.csv")
    o = m.sort_values("cvar99")
    worst, safest = o.index[-1], o.index[0]
    avg_breach = m.breach_rate.mean(); ratio = avg_breach / (1 - CONF)
    H = pd.DataFrame(hits); H.columns = [SECTORS[c] for c in H.columns]
    by_year = H.groupby(H.index.year).sum().sum(axis=1)

    # Chart 1 (interactive, Chart.js): real bad-day loss vs what the standard model expects, per sector. Hover for dollars.
    oo = o.sort_values("cvar99", ascending=False)
    c1 = R.js_chart(dict(type="bar", horizontal=True, fmt_x="usd", x_title=f"Loss on a ${POSITION:,.0f} investment on the worst 1-in-100 days",
        labels=list(oo.index), datasets=[dict(label="Real average loss", data=oo.cvar99.round(0).tolist(), colors=[R.BAD if s == worst else R.ACC for s in oo.index]),
                                         dict(label="What the standard model expects", data=oo.normal_cvar99.round(0).tolist(), color=R.ORANGE)]))

    ob = m.breach_rate.sort_values()
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.barh(ob.index, ob * 100, color=[R.BAD if v > (1 - CONF) * 1.5 else R.GREY for v in ob])
    ax.axvline((1 - CONF) * 100, color=R.INK, ls="--", lw=1); ax.text((1 - CONF) * 100, -.9, " model promises 1%", fontsize=9)
    ax.set_xlabel("% of days the loss was bigger than the limit (should be 1%)"); c2 = R.fig_to_b64(fig)

    # Chart 3 (interactive, Chart.js): breaches per year vs what the model expects. Hover a bar for the count.
    expected = round(len(SECTORS) * 252 * (1 - CONF), 1)
    c3 = R.js_chart(dict(type="bar", fmt_y="int", y_title="Days the loss limit was broken (all sectors)", labels=[str(y) for y in by_year.index],
        datasets=[dict(label="Actual limit breaks", color=R.ACC, data=[int(v) for v in by_year]),
                  dict(label="Expected by the model", color=R.INK, data=[expected] * len(by_year), kind="line")]))

    wt = [k for k, v in SECTORS.items() if v == worst][0]; r = rets[wt] * 100
    fig, ax = plt.subplots(figsize=(9, 3.4))
    bins = np.linspace(r.min(), -1, 40); ax.hist(r[r < -1], bins=bins, color=R.BAD, alpha=.7, label="Actual losing days")
    x = np.linspace(r.min(), -1, 200); ax.plot(x, stats.norm.pdf(x, r.mean(), r.std()) * len(r) * (bins[1] - bins[0]), color=R.INK, label="What the standard model expects")
    ax.set_yscale("log"); ax.set_ylim(0.5, None); ax.set_xlabel(f"{worst}: daily loss (%)"); ax.set_ylabel("Number of days (log scale)"); ax.legend(); c4 = R.fig_to_b64(fig)

    usd = lambda x: f"${x:,.0f}"
    t = m[["var99", "cvar99", "normal_var99", "worst_day", "breach_rate", "understatement"]].copy()
    t.columns = ["1-in-100-day loss", "Average loss on those days", "Standard model limit", "Worst single day", "Limit broken (% of days)", "Model understates by"]
    for c in t.columns[:4]: t[c] = t[c].map(usd)
    t["Limit broken (% of days)"] = t["Limit broken (% of days)"].map(lambda x: f"{x*100:.2f}%"); t["Model understates by"] = t["Model understates by"].map(lambda x: f"{x*100:.0f}%")

    d, (ols, lo, mv), (poi, lp, cm), (logit, ll, test_auc) = run_models(rets, pd.DataFrame(hits), vix)
    pd.DataFrame({"ols": ols.params, "poisson": poi.params, "logit": logit.params}).round(5).to_csv("model_coefficients.csv")
    bl = logit.params
    models = [
        ("Multiple regression · continuous", "Can we forecast next month's risk for each sector?",
         f"Partly. This month's volatility and VIX explain {ols.rsquared*100:.0f}% of next month's sector volatility. Risk limits should be reset monthly, not yearly.",
         f"OLS on {len(mv):,} sector-months, robust errors. R² = {ols.rsquared:.2f}.", R.coef_table(ols, lo, "ols")),
        ("Poisson regression · count", "When do limit breaches pile up?",
         f"When recent swings run above the past year. A swings ratio of 1.5 instead of 1.0 multiplies expected breaches by {np.exp(poi.params['ratio']*0.5):.1f}. That is the bell-curve model reacting too slowly.",
         f"Poisson GLM on {len(cm):,} sector-months, trading days as exposure.", R.coef_table(poi, lp, "poisson")),
        ("Binomial regression · decision", "Should the risk team tighten tomorrow's limit?",
         f"Fitted on 2010–2017 and tested on 2018–2024, it ranked a breach day above a normal day {test_auc*100:.0f}% of the time (50% = guessing). {'Good enough for an early-warning flag.' if test_auc >= 0.65 else 'Only a weak flag.'}",
         f"Logistic GLM on {len(d):,} sector-days. Outcome: loss beat the 99% limit ({d.hit.mean()*100:.1f}% of days). Test AUC {test_auc:.2f}.", R.coef_table(logit, ll, "logit"))]
    sim = R.sim_html("Try it: tighten a sector's limit tomorrow?", "Describe the market tonight. The model gives the chance tomorrow's loss breaks the standard daily loss limit (normally a 1-in-100 event).",
        float(bl["const"]), [("ratio", "Recent swings vs the past year (1 = normal, 2 = twice as stormy)", 0.5, 3.0, 0.05, 1.0, float(bl["ratio"]), 2, ""),
                             ("dvix", "Change in the market fear index this week (0.2 = up 20%)", -0.3, 1.0, 0.05, 0.0, float(bl["dvix"]), 2, "")],
        5, "Tighten limit", "Keep limit")

    R.build("dashboard.html", "Sector Risk Budget", "Equity fund risk manager · 9 US sectors, 2010–2024",
        "How much can each sector lose on a bad day?",
        f"On its worst 1-in-100 days, {worst} lost about {usd(m.loc[worst,'cvar99'])} per $100,000 invested, vs {usd(m.loc[safest,'cvar99'])} for {safest}. "
        f"The standard risk model's loss limit was broken on {avg_breach*100:.1f}% of days, {ratio:.1f}× the 1% it promises.",
        [("Riskiest sector", worst, f"loses about {usd(m.loc[worst,'cvar99'])} per $100k on its worst 1-in-100 days", "bad"),
         ("Safest sector", safest, f"about {usd(m.loc[safest,'cvar99'])} per $100k on those days", "good"),
         ("Loss limit broken", f"{avg_breach*100:.1f}% of days", "the model promises 1%: about 1 day in 100", "bad" if ratio > 1.3 else "good"),
         ("Bad days are bigger than expected", f"+{m.understatement.mean()*100:.0f}%", "real worst-day losses vs what the standard model predicts", "bad")],
        [("Risk manager", f"Set loss limits from real past losses, or raise the standard model's estimates by about {m.understatement.mean()*100:.0f}%."),
         ("Portfolio manager", f"Hold less in {worst} and {o.index[-2]} than in {safest} or {o.index[1]} to keep the same level of daily risk."),
         ("Board / CFO", "Limit breaks come in bunches during stressful years. Plan for several at once, not one at a time.")],
        [(f"Bad-day loss: {worst} loses the most on a bad day", f"Bar = real average loss on the worst 1-in-100 days, per ${POSITION:,.0f} invested. Black mark = what the standard risk model expects.", c1),
         ("Model check: The standard risk model was wrong more often than it promised, in every sector" if (m.breach_rate > 1 - CONF).all() else "Model check: The standard risk model was wrong more often than it promised, in most sectors",
          "How often the daily loss was bigger than the model's limit. The dashed line is the 1% the model promises.", c2),
         ("By year: Limit breaks bunch up in stressful years", "Days the loss limit was broken, all 9 sectors added up. Dashed line = what the model expects. Hover a bar for the count.", c3),
         (f"Big losses: {worst} had far more big losing days than the standard model expects", "Count of losing days by size (bars) vs the model's expectation (line). Far left = the biggest losses.", c4)],
        ["Sector ETFs (SPDR Select Sector funds) as stand-ins. Yahoo Finance adjusted closes, 2010–2024.",
         "1-in-100-day loss = a loss bigger than on 99 of every 100 days. Bad-day loss = the average loss on those worst days.",
         "Backtest: each day the model limit uses only the previous 252 days. A good model is breached on about 1% of days.",
         "Newer sectors (Real estate, Communication services) are excluded because they lack a full 15-year history."],
        ["Losses assume the position cannot be sold during the day. Real intraday losses can be larger.",
         "One-day horizon only. Multi-day crashes (2020) add up to much bigger losses.",
         "Historical figures describe the past 15 years, which may not include the next kind of crisis."],
        t.to_html(border=0, classes="sort"), "Data: Yahoo Finance via yfinance", models, sim, evidence=evidence(rets))
    print(m.round(3))

if __name__ == "__main__":
    main()
