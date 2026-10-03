"""Project 2 - Steady-Ride Fund Test (volatility targeting)
Client: the product team at an ETF issuer planning a 'smoother S&P 500' fund for cautious investors.
Rule: each week, hold less of the S&P 500 when recent swings are large, and park the rest in cash (T-bills).
Question: would that rule have given a smoother ride without giving up too much return?
Fixes weaknesses of the original repo's portfolio test: settings are chosen on 2010-2017 only and judged on
unseen 2018-2024 data, trading costs are charged, cash earns the real T-bill rate, and there is no leverage."""
import os
import numpy as np, pandas as pd, yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # shared report.py at repo root
import report as R

START, END = "2010-01-01", "2024-12-31"
TRAIN_END, TEST_START = "2017-12-31", "2018-01-01"
TARGETS = [0.10, 0.12, 0.15]      # target yearly volatility options tested on training years
LOOKBACKS = [20, 60]              # days of history used to measure recent swings
MAX_EXPOSURE = 1.0                # no borrowing
COST = 0.0005                     # 0.05% per dollar traded

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

def strategy(spy, cash, target, lookback):
    realised = spy.rolling(lookback).std() * np.sqrt(252)
    fridays = spy.index[spy.index.weekday == 4]
    want = (target / realised).clip(upper=MAX_EXPOSURE)
    exposure = want.where(spy.index.isin(fridays)).ffill().shift(1)   # decided at Friday close, applied from next day
    exposure = exposure.fillna(0)
    trade_cost = exposure.diff().abs().fillna(0) * COST
    return exposure * spy + (1 - exposure) * cash - trade_cost, exposure

def stats(r):
    cum = (1 + r).cumprod(); dd = cum / cum.cummax() - 1; yrs = (r.index[-1] - r.index[0]).days / 365.25
    monthly = (1 + r).resample("ME").prod() - 1; vol = r.std() * np.sqrt(252)
    return dict(cagr=cum.iloc[-1] ** (1 / yrs) - 1, vol=vol, max_dd=dd.min(), worst_month=monthly.min(),
                sharpe=r.mean() * 252 / vol, final_10k=10_000 * cum.iloc[-1]), cum, dd


def monthly_panel(spy, vix):
    """One row per month: what the product team knows at month-end, and what happened next month."""
    m = spy.index.to_period("M")
    g = spy.groupby(m)
    cum = (1 + spy).cumprod()
    def month_dd(x):
        c = (1 + x).cumprod(); c = pd.concat([pd.Series([1.0]), c.reset_index(drop=True)])
        return (c / c.cummax() - 1).min()
    df = pd.DataFrame({"rv": g.std() * np.sqrt(252), "vix": vix.groupby(vix.index.to_period("M")).last(),
                       "ret": g.apply(lambda x: (1 + x).prod() - 1), "down2": g.apply(lambda x: int((x < -0.02).sum())),
                       "dd": g.apply(month_dd), "days": g.size()})
    nxt = df[["rv", "down2", "dd", "days"]].shift(-1).add_prefix("next_")
    return pd.concat([df, nxt], axis=1).dropna()

def run_models(mp):
    X = sm.add_constant(mp[["vix", "rv"]]); labels = {"vix": "VIX at month-end (1 point)", "rv": "This month's volatility (0.01 = 1 point)"}
    ols = sm.OLS(mp.next_rv, X).fit(cov_type="HC0")
    poi = sm.GLM(mp.next_down2, X, family=sm.families.Poisson(), exposure=mp.next_days).fit(cov_type="HC0")
    y = (mp.next_dd < -0.05).astype(int)
    train = mp.index <= pd.Period(TRAIN_END, "M")
    # VIX and realised volatility overlap heavily; together they give a confusing negative VIX sign.
    # One driver keeps the decision model readable and tests just as well out of sample.
    Xd = sm.add_constant(mp[["rv"]])
    logit = sm.GLM(y[train], Xd[train], family=sm.families.Binomial()).fit()
    test_auc = R.auc(y[~train], logit.predict(Xd[~train]))
    return labels, ols, poi, logit, test_auc, y, train

def evidence(spy, cash, expo):
    """Test the 'why': is it the timing of exposure that helps, or just holding less stock on average?"""
    rng = np.random.default_rng(7)
    def perf(e):
        r = (e * spy + (1 - e) * cash - e.diff().abs().fillna(0) * COST).loc[TEST_START:]
        c = (1 + r).cumprod(); return (c / c.cummax() - 1).min()
    real = perf(expo); avg = expo.loc[TEST_START:].mean()
    static = perf(pd.Series(avg, index=spy.index))
    wk = expo.loc[TEST_START:].resample("W-FRI").last()
    shuffled = [perf(pd.Series(rng.permutation(wk.values), index=wk.index).reindex(spy.index, method="ffill").fillna(avg)) for _ in range(500)]
    beat = np.mean(np.array(shuffled) < real)
    m = spy.groupby(spy.index.to_period("M")).std(); ac = m.autocorr(1)
    return [
        dict(claim="Swings come in clusters, so recent volatility warns of more.",
             test="Correlation between one month's volatility and the next month's, 2010–2024.",
             result=f"Correlation {ac:.2f}: a stormy month is usually followed by another.", verdict="Supported" if ac > 0.3 else "Partly supported"),
        dict(claim="It is the timing that protects, not just owning less stock.",
             test=f"Compared the rule with a fixed {avg:.0%} in stocks (same average), and with the rule's own weekly exposures shuffled into random order 500 times.",
             result=f"Worst drop: rule {real:.1%}, fixed {avg:.0%} mix {static:.1%}. Random timing did worse than the rule in {beat:.0%} of 500 shuffles.",
             verdict="Supported" if beat > 0.95 and real > static else "Partly supported")]

def main():
    R.style()
    px, _ = load_prices(["SPY", "^VIX", "^IRX"], START, END)
    spy = px.SPY.pct_change().dropna()
    cash = (px["^IRX"].ffill() / 100 / 252).reindex(spy.index).fillna(0)

    # 1) choose settings on training years only
    grid = []
    for tg in TARGETS:
        for lb in LOOKBACKS:
            r, _ = strategy(spy, cash, tg, lb)
            s = stats(r.loc[:TRAIN_END].iloc[max(LOOKBACKS):])[0]
            grid.append(dict(target=tg, lookback=lb, **s))
    grid = pd.DataFrame(grid); grid.round(4).to_csv("training_grid.csv", index=False)
    best = grid.sort_values("sharpe", ascending=False).iloc[0]
    tg, lb = best.target, int(best.lookback)

    # 2) judge on unseen years
    r, expo = strategy(spy, cash, tg, lb)
    test = slice(TEST_START, END)
    S, cS, dS = stats(r.loc[test]); B, cB, dB = stats(spy.loc[test])
    pd.DataFrame({"Steady-Ride": S, "S&P 500": B}).round(4).to_csv("results.csv")
    e = expo.loc[test]; rv = (r.loc[test].rolling(63).std() * np.sqrt(252)); bv = (spy.loc[test].rolling(63).std() * np.sqrt(252))

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(cB.index, cB * 10_000, color=R.GREY, lw=1.6, label="S&P 500"); ax.plot(cS.index, cS * 10_000, color=R.ACC, lw=2, label="Steady-Ride fund")
    ax.set_ylabel("Value of $10,000"); ax.legend(loc="upper left"); c1 = R.fig_to_b64(fig)

    # Chart 2 (interactive, Chart.js): worst fall in each year, fund vs S&P 500 side by side. Hover a bar for the value.
    yrs = sorted(set(dS.index.year))
    c2 = R.js_chart(dict(type="bar", fmt_y="pct0", y_title="Worst fall within the year", labels=[str(y) for y in yrs],
        datasets=[dict(label="S&P 500", color=R.GREY, data=[round(float(dB.loc[str(y)].min()), 4) for y in yrs]),
                  dict(label="Steady-Ride fund", color=R.ACC, data=[round(float(dS.loc[str(y)].min()), 4) for y in yrs])]))

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.fill_between(e.index, e * 100, 0, color=R.ACC, alpha=.35); ax.set_ylim(0, 105); ax.set_ylabel("% held in shares"); c3 = R.fig_to_b64(fig)

    # Chart 4: average swing size per year as paired bars, with the target as a line
    ys = pd.DataFrame({"S&P 500": spy.loc[TEST_START:].groupby(spy.loc[TEST_START:].index.year).std() * np.sqrt(252) * 100,
                       "Steady-Ride fund": r.loc[TEST_START:].groupby(r.loc[TEST_START:].index.year).std() * np.sqrt(252) * 100})
    fig, ax = plt.subplots(figsize=(9, 4)); x = np.arange(len(ys)); w = .38
    for i, (n, col) in enumerate(zip(ys.columns, [R.GREY, R.ACC])):
        bars = ax.bar(x + (i - .5) * w, ys[n], w, color=col, label=n); ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=10, color=R.INK)
    ax.axhline(tg * 100, color=R.INK, ls="--", lw=1.2); ax.text(len(ys) - .5, tg * 100 + .8, f"target {tg*100:.0f}%", ha="right", color=R.INK, fontsize=11)
    ax.set_xticks(x, ys.index); ax.set_ylabel("Size of price swings (%, yearly rate)"); ax.legend(loc="upper left"); c4 = R.fig_to_b64(fig)

    pc = lambda x: f"{x*100:.1f}%"
    dd_in = lambda dd, y: dd.loc[y].min()
    gave_up = B["cagr"] - S["cagr"]
    t = pd.DataFrame({"Steady-Ride": S, "S&P 500": B}).T
    t = pd.DataFrame({"Growth/yr": t.cagr.map(pc), "Volatility": t.vol.map(pc), "Worst drop": t.max_dd.map(pc), "Worst month": t.worst_month.map(pc),
                      "Return per unit of risk": t.sharpe.map(lambda x: f"{x:.2f}"), "$10k became": t.final_10k.map(lambda x: f"${x:,.0f}")})
    g = grid.copy(); g["target"] = g.target.map(pc); g = g[["target", "lookback", "cagr", "vol", "max_dd", "sharpe"]].round(3)

    vix = px["^VIX"].reindex(spy.index).ffill()
    mp = monthly_panel(spy, vix)
    labels, ols, poi, logit, test_auc, y, train = run_models(mp)
    pd.DataFrame({"ols": ols.params, "poisson": poi.params, "logit": logit.params}).round(5).to_csv("model_coefficients.csv")
    r10 = np.exp(poi.params["vix"] * 10)
    models = [
        ("Multiple regression · continuous", "Can we forecast next month's volatility?",
         f"Partly. VIX and this month's swings explain {ols.rsquared*100:.0f}% of next month's volatility, with VIX doing most of the work. Calm and storms tend to persist, which is why a volatility rule can work at all.",
         f"OLS on {len(mp)} months, robust errors. R² = {ols.rsquared:.2f}.", R.coef_table(ols, labels, "ols")),
        ("Poisson regression · count", "How many −2% days should we expect next month?",
         f"VIX drives it. Ten more VIX points multiplies the expected count of −2% days by {r10:.1f}. At VIX 15 expect about {np.exp(poi.params['const'] + poi.params['vix']*15 + poi.params['rv']*0.12)*21:.1f}; at VIX 30 about {np.exp(poi.params['const'] + poi.params['vix']*30 + poi.params['rv']*0.25)*21:.1f}.",
         "Poisson GLM with trading days as exposure, robust errors.", R.coef_table(poi, labels, "poisson")),
        ("Binomial regression · decision", "Should the fund cut exposure this month?",
         f"Somewhat. Fitted on 2010–2017 only, it ranked a month with a 5%+ fall above a calm month {test_auc*100:.0f}% of the time on 2018–2024 (50% = guessing). Better than nothing, but not reliable enough to act on alone. The simpler volatility rule above is easier to defend.",
         f"Logistic GLM. Outcome: next month has a peak-to-trough fall above 5% ({y.mean()*100:.0f}% of months). Test AUC {test_auc:.2f}.", R.coef_table(logit, {"rv": labels["rv"]}, "logit"))]
    b = logit.params
    sim = R.sim_html("Try it: should the fund cut exposure next month?", "Move the slider to how much the market is swinging this month. The model gives the chance of a 5%+ fall next month.",
        float(b["const"]), [("rv", "Market swings this month (0.12 = calm, 0.40 = stormy)", 0.05, 0.60, 0.01, 0.12, float(b["rv"]), 2, "")],
        30, "Cut exposure", "Stay fully invested")

    R.build("dashboard.html", "Steady-Ride Fund Test", "ETF product team · S&P 500 with a volatility rule, tested 2018–2024",
        "Can a simple rule make the S&P 500 a smoother ride?",
        f"From 2018 to 2024 the fund's biggest fall was {pc(S['max_dd'])} vs {pc(B['max_dd'])} for the S&P 500, and its worst month {pc(S['worst_month'])} vs "
        f"{pc(B['worst_month'])}. The cost: {pc(gave_up)} less growth a year.",
        [("Biggest fall", pc(S["max_dd"]), f"vs {pc(B['max_dd'])} for the S&P 500", "good"),
         ("Worst month", pc(S["worst_month"]), f"vs {pc(B['worst_month'])} for the S&P 500", "good"),
         ("Growth given up", f"{pc(gave_up)} a year", f"{pc(S['cagr'])} vs {pc(B['cagr'])} a year", "bad"),
         ("$10,000 became", f"${S['final_10k']:,.0f}", f"vs ${B['final_10k']:,.0f} in the S&P 500 (2018–2024)", "")],
        [("Product team", f"Sell it as 'smaller falls', not 'higher returns'. It gave up about {pc(gave_up)} a year of growth to roughly halve the biggest fall."),
         ("Cautious investor", f"In the 2020 crash it fell {pc(-dd_in(dS,'2020'))} vs {pc(-dd_in(dB,'2020'))} for the market. In the slower 2022 fall it helped less ({pc(-dd_in(dS,'2022'))} vs {pc(-dd_in(dB,'2022'))}), and it misses part of quick rebounds."),
         ("Compliance", "The fund's settings were fixed using 2010–2017 and then tested on 2018–2024, so the results were not tuned to look good. Past results still don't guarantee future ones.")],
        [("Growth: The fund grew more slowly, but with smaller dips", "What $10,000 invested in January 2018 became.", c1),
         ("Falls: Its falls were shallower, most of all in the 2020 crash", "The deepest point each year, measured from the previous high. Shorter bars = smaller falls. Hover a bar for the value.", c2),
         ("Invested share: The fund moved into cash when markets got stormy", "Share of the fund held in shares each day; the rest sat in cash earning interest.", c3),
         ("Swings: The fund's ups and downs stayed near its target", f"Average size of price swings in each year. Dashed line = the fund's {pc(tg)} target. Lower = a calmer ride.", c4)],
        [f"Settings tested on 2010–2017: targets {', '.join(pc(x) for x in TARGETS)} × lookbacks {LOOKBACKS} days. Chosen by best return per unit of risk: {pc(tg)} target, {lb}-day lookback.",
         "Each Friday: exposure = target ÷ recent volatility, capped at 100% (no borrowing). Applied from the next trading day.",
         "Cash earns the 13-week T-bill yield. Every change in exposure costs 0.05% of the amount traded.",
         "Return per unit of risk = average yearly return ÷ yearly volatility (a simplified Sharpe ratio)."],
        ["Seven test years include only one fast crash (2020) and one slow bear market (2022). Performance in other crises may differ.",
         "The rule is late by design: it reacts after swings rise, so it cannot avoid the first days of a crash.",
         "Taxes and fund fees are not included. Real fund fees would lower both lines."],
        t.to_html(border=0, classes="sort") + "<p><b>Training-period grid (2010–2017)</b></p>" + g.to_html(border=0, index=False), "Data: Yahoo Finance via yfinance", models, sim, evidence=evidence(spy, cash, expo),
        series=dict(dates=[d.strftime("%Y-%m-%d") for d in cS.resample("W-FRI").last().index],
                    rule=(cS.resample("W-FRI").last() * 10_000).round(0).tolist(), spy=(cB.resample("W-FRI").last() * 10_000).round(0).tolist(),
                    rule_dd=(dS.resample("W-FRI").min() * 100).round(1).tolist(), spy_dd=(dB.resample("W-FRI").min() * 100).round(1).tolist()))
    print(grid.round(3)); print(pd.DataFrame({"rule": S, "spy": B}).round(3))

if __name__ == "__main__":
    main()
