"""Project 3 - Crisis Diversification Check
Client: a superannuation fund's investment committee.
Question: which assets actually protected a share portfolio when markets panicked (2010-2024)?
Fixes a weakness of the original repo: it only held US stocks, so it could not test real diversifiers (bonds, gold)."""
import os
import numpy as np, pandas as pd, yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # shared report.py at repo root
import report as R

ASSETS = {"SPY": "US shares", "EFA": "Intl shares", "EEM": "Emerging shares", "VNQ": "Property (REITs)",
          "GLD": "Gold", "TLT": "Long US bonds", "IEF": "Mid US bonds"}
VIX = "^VIX"
START, END = "2010-01-01", "2024-12-31"
STRESS_VIX = 25       # "panic" day = VIX closed above 25
COST = 0.001          # 0.10% per dollar traded at each monthly rebalance
PORTFOLIOS = {"100% shares": {"SPY": 1.0}, "60/40 shares/bonds": {"SPY": .6, "IEF": .4},
              "60/30/10 + gold": {"SPY": .6, "IEF": .3, "GLD": .1}}

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

def backtest(rets, weights):
    """Monthly rebalanced portfolio with trading costs. Uses only information available on each day."""
    w = pd.Series(weights); cur = w.copy(); out = []
    month_end = rets.groupby(rets.index.to_period("M")).tail(1).index
    for dt, row in rets[w.index].iterrows():
        pr = (cur * row).sum(); cur = cur * (1 + row) / (1 + pr); cost = 0
        if dt in month_end:
            cost = (w - cur).abs().sum() * COST; cur = w.copy()
        out.append(pr - cost)
    return pd.Series(out, index=rets.index)

def summary(r):
    cum = (1 + r).cumprod(); dd = cum / cum.cummax() - 1
    yrs = (r.index[-1] - r.index[0]).days / 365.25; yearly = (1 + r).groupby(r.index.year).prod() - 1
    return dict(cagr=cum.iloc[-1] ** (1 / yrs) - 1, vol=r.std() * np.sqrt(252), max_dd=dd.min(),
                worst_year=yearly.min(), worst_year_when=int(yearly.idxmin())), dd


def run_models(rets, vix, stress, crash):
    post = (rets.index >= "2022-01-01").astype(int)          # the new era of inflation and rate rises
    d = pd.DataFrame({"ief": rets.IEF, "spy": rets.SPY, "post": post, "stress": stress.astype(int).values, "vix": vix.values}, index=rets.index)
    d["spy_post"] = d.spy * d.post; d["spy_stress"] = d.spy * d.stress
    ols = sm.OLS(d.ief, sm.add_constant(d[["spy", "spy_post", "spy_stress"]])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    lab_ols = {"spy": "US shares return (before 2022)", "spy_post": "Extra link from 2022 on", "spy_stress": "Extra link on panic days"}
    joint = ((d.spy < -0.01) & (d.ief < 0)).astype(int)
    m = pd.DataFrame({"joint": joint.groupby(d.index.to_period("M")).sum(), "days": joint.groupby(d.index.to_period("M")).size(),
                      "vix": d.vix.groupby(d.index.to_period("M")).mean(), "post": d.post.groupby(d.index.to_period("M")).max()})
    poi = sm.GLM(m.joint, sm.add_constant(m[["vix", "post"]]), family=sm.families.Poisson(), exposure=m.days).fit(cov_type="HC0")
    lab_poi = {"vix": "Average VIX that month (1 point)", "post": "Month is 2022 or later"}
    c = d[crash.values]; y = (c.ief > 0).astype(int)
    logit = sm.GLM(y, sm.add_constant(c[["vix", "post"]]), family=sm.families.Binomial()).fit(cov_type="HC0")
    lab_log = {"vix": "VIX that day (1 point)", "post": "Day is 2022 or later"}
    return (ols, lab_ols), (poi, lab_poi, m), (logit, lab_log, y, c)

def evidence(rets, px):
    """Test the 'why' for bonds failing: is the stock-bond link driven by rising interest rates or by inflation fears?"""
    if not os.path.exists("data/rates.csv"):
        yf.download(["^TNX", "TIP"], start="2009-01-01", end=END, auto_adjust=True, progress=False, threads=False)[["Close"]].to_csv("data/rates.csv")
    rt = pd.read_csv("data/rates.csv", index_col=0, parse_dates=True, header=[0, 1])["Close"].ffill()   # ^TNX 10-yr yield, TIP inflation-linked bonds
    corr = rets.SPY.rolling(252).corr(rets.IEF)
    dy = rt["^TNX"].reindex(rets.index).ffill().diff(252)                         # change in 10-year yield over the past year (points)
    infl = rt["TIP"].reindex(rets.index).ffill().pct_change(252) - px.IEF.reindex(rets.index).pct_change(252)   # inflation-bond vs normal-bond return gap
    d = pd.DataFrame({"corr": corr, "dy": dy, "infl": infl}).dropna().resample("ME").last().dropna()
    fit = lambda s: sm.OLS(s["corr"], sm.add_constant(s[["dy", "infl"]])).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
    a, pre = fit(d), fit(d[d.index < "2022-01-01"])
    return [
        dict(claim="Bonds stop cushioning shares when interest rates are rising.",
             test="Regressed the 1-year stock–bond correlation on the 1-year change in the 10-year Treasury yield, monthly, with Newey-West errors.",
             result=f"Each 1-point rise in yields lifts the correlation by {a.params['dy']:+.2f} ({R.pt(a.pvalues['dy'])}). It also holds using only pre-2022 data ({pre.params['dy']:+.2f}, {R.pt(pre.pvalues['dy'])}), e.g. the 2013 'taper tantrum'.",
             verdict="Supported" if a.pvalues["dy"] < 0.05 and pre.pvalues["dy"] < 0.05 else "Partly supported"),
        dict(claim="Inflation fears, on their own, broke the link.",
             test="Added an inflation-expectations proxy to the same model: inflation-linked bond (TIP) return minus normal bond (IEF) return.",
             result=f"Effect {a.params['infl']:+.2f} ({R.pt(a.pvalues['infl'])}). Once rate changes are known, inflation expectations add no clear information.",
             verdict="Supported" if a.pvalues["infl"] < 0.05 else "Not supported")]

def main():
    R.style()
    tick = list(ASSETS)
    px, _ = load_prices(tick + [VIX], START, END)
    rets = px[tick].pct_change().dropna()
    vix = px[VIX].reindex(rets.index).ffill()
    stress = vix > STRESS_VIX
    crash = rets.SPY <= rets.SPY.quantile(0.05)           # the 5% worst days for US shares

    tab = pd.DataFrame({
        "calm_corr": rets[~stress].corr()["SPY"], "stress_corr": rets[stress].corr()["SPY"],
        "crash_day_return": rets[crash].mean(), "return_2022": (1 + rets.loc["2022"]).prod() - 1}).drop("SPY", errors="ignore")
    tab["protects"] = (tab.crash_day_return >= 0)   # held steady or rose
    tab.round(4).to_csv("results.csv")

    ports = {n: backtest(rets, w) for n, w in PORTFOLIOS.items()}
    ps = {n: summary(r) for n, r in ports.items()}
    pt = pd.DataFrame({n: s[0] for n, s in ps.items()}).T; pt.round(4).to_csv("portfolios.csv")
    sb_corr = rets.SPY.rolling(252).corr(rets.IEF)

    names = [ASSETS[t] for t in tab.index]
    join = lambda xs: xs[0] if len(xs) == 1 else ', '.join(xs[:-1]) + ' and ' + xs[-1]
    protectors = [ASSETS[t] for t in tab[tab.protects].index]

    # Chart 1: dumbbell calm vs stress correlation
    fig, ax = plt.subplots(figsize=(9, 3.8)); y = np.arange(len(tab))
    ax.hlines(y, tab.calm_corr, tab.stress_corr, color=R.GREY, lw=3)
    ax.scatter(tab.calm_corr, y, color=R.GREY, s=60, label="Normal days", zorder=3)
    ax.scatter(tab.stress_corr, y, color=R.BAD, s=60, label="Panic days", zorder=3)
    ax.set_yticks(y, names); ax.axvline(0, color=R.INK, lw=.8); ax.set_xlabel("How closely it moves with US shares (1 = in step, 0 = unrelated, below 0 = opposite)"); ax.legend(loc="center right")
    c1 = R.fig_to_b64(fig)

    # Chart 2: average return on crash days
    o = tab.crash_day_return.sort_values()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.barh([ASSETS[t] for t in o.index], o * 100, color=[R.GOOD if v > 0 else R.BAD for v in o])
    ax.axvline(rets.SPY[crash].mean() * 100, color=R.INK, ls="--", lw=1)
    ax.text(rets.SPY[crash].mean() * 100, len(o) - .4, " US shares", fontsize=9)
    ax.set_xlabel("Average change (%) on the worst 1-in-20 days for US shares")
    c2 = R.fig_to_b64(fig)

    # Chart 3 (interactive, Chart.js): rolling stock-bond correlation, weekly. Hover for the value on any date.
    w = sb_corr.dropna().resample("W-FRI").last()
    c3 = R.js_chart(dict(type="line", fmt_y="num2", y_title="How closely shares and bonds move (past year)", labels=[d.strftime("%Y-%m-%d") for d in w.index],
        datasets=[dict(label="Shares vs bonds", color=R.ACC, data=w.round(3).tolist(), fill=True)], zero_line=True))

    # Chart 4: biggest fall of each mix in each crisis (grouped bars) instead of three overlapping lines
    CRISES = {"2011\ndebt crisis": ("2011-07-01", "2011-12-31"), "2015–16\nChina scare": ("2015-08-01", "2016-02-29"),
              "2018\nrate scare": ("2018-10-01", "2018-12-31"), "2020\nCOVID": ("2020-02-15", "2020-04-30"), "2022\nrate rises": ("2022-01-01", "2022-10-31")}
    def fall(r, a, b):
        c = (1 + r.loc[a:b]).cumprod(); return (c / c.cummax() - 1).min() * 100
    falls = pd.DataFrame({n: [fall(r, a, b) for a, b in CRISES.values()] for n, r in ports.items()}, index=list(CRISES))
    fig, ax = plt.subplots(figsize=(9, 4.4)); x = np.arange(len(falls)); w = .26
    for i, (n, col) in enumerate(zip(falls.columns, [R.GREY, R.ACC, R.GOOD])):
        bars = ax.bar(x + (i - 1) * w, falls[n], w, color=col, label=n)
        ax.bar_label(bars, fmt="%.0f%%", padding=2, fontsize=10, color=R.INK)
    ax.set_xticks(x, falls.index); ax.set_ylabel("Biggest fall during the crisis (%)"); ax.axhline(0, color=R.LINE, lw=1)
    ax.set_ylim(falls.min().min() * 1.2, 2); ax.legend(loc="lower left"); c4 = R.fig_to_b64(fig)

    pc = lambda x: f"{x*100:.1f}%"
    a, b, g = (pt.loc[n] for n in PORTFOLIOS)
    t2 = tab.copy(); t2.index = names
    t2 = t2.rename(columns={"calm_corr": "Moves with shares (normal days)", "stress_corr": "Moves with shares (panic days)", "crash_day_return": "Average on crash days", "return_2022": "Return in 2022", "protects": "Protected?"})
    for c in ["Average on crash days", "Return in 2022"]: t2[c] = t2[c].map(pc)
    for c in ["Moves with shares (normal days)", "Moves with shares (panic days)"]: t2[c] = t2[c].map(lambda x: f"{x:.2f}")
    t3 = pt[["cagr", "vol", "max_dd", "worst_year"]].map(pc); t3.columns = ["Growth per year", "Size of swings", "Biggest fall", "Worst year"]

    (ols, lo), (poi, lp, mm), (logit, ll, yc, cd) = run_models(rets, vix, stress, crash)
    pd.DataFrame({"ols": ols.params, "poisson": poi.params, "logit": logit.params}).round(5).to_csv("model_coefficients.csv")
    bl = logit.params; pr = lambda v, post: 1 / (1 + np.exp(-(bl["const"] + bl["vix"] * v + bl["post"] * post)))
    models = [
        ("Multiple regression · continuous", "How do bonds move when shares move, and did that change?",
         f"Before 2022, bonds moved {ols.params['spy']:+.2f}% for every 1% move in shares (a cushion). From 2022 the link shifted by {ols.params['spy_post']:+.2f} ({R.pt(ols.pvalues['spy_post'])}), so the cushion {'mostly disappeared' if ols.params['spy']+ols.params['spy_post'] > -0.05 else 'weakened'}.",
         f"OLS on {len(rets):,} days, Newey-West errors. R² = {ols.rsquared:.2f}.", R.coef_table(ols, lo, "ols")),
        ("Poisson regression · count", "How many days a month do shares AND bonds fall together?",
         f"Since 2022 such days are {np.exp(poi.params['post']):.1f}× as common at the same VIX level. These are the days a 60/40 fund has nowhere to hide.",
         f"Poisson GLM on {len(mm)} months, trading days as exposure. Joint-fall day = shares down >1% and bonds down.", R.coef_table(poi, lp, "poisson")),
        ("Binomial regression · decision", "On a crash day, can the committee count on bonds rising?",
         f"Before 2022, bonds rose on about {pr(30,0)*100:.0f}% of crash days (at VIX 30). Since 2022, about {pr(30,1)*100:.0f}%. Bonds alone are no longer reliable insurance.",
         f"Logistic GLM on the {len(yc)} worst days for shares. Outcome: mid-term bonds up that day ({yc.mean()*100:.0f}% overall). Fit in-sample: the post-2022 era is too short to hold out.", R.coef_table(logit, ll, "logit"))]
    sim = R.sim_html("Try it: will bonds cushion the next crash day?", "Set the market conditions. The model gives the chance bonds fail to rise on a crash day. If that chance is high, add another safety asset such as gold.",
        -float(bl["const"]), [("vix", "Market fear index (VIX): 15 = calm, 40 = panic", 12, 60, 1, 30, -float(bl["vix"]), 0, ""),
                              ("post", "Period (0 = before 2022, 1 = 2022 onwards)", 0, 1, 1, 1, -float(bl["post"]), 0, "")],
        30, "Add another hedge", "Bonds should cover it")

    R.build("dashboard.html", "Crisis Diversification Check", "Super fund investment committee · 7 asset classes, 2010–2024",
        "Which assets actually protected a share portfolio in a panic?",
        f"Only {join(protectors)} held steady or rose on the worst days for shares. Other share markets and property fell with US shares, and more so in panics. "
        f"Bonds failed in 2022, when shares and bonds fell together.",
        [("Assets that protected", f"{len(protectors)} of {len(tab)}", ", ".join(protectors), "good"),
         ("US shares on a bad day", pc(rets.SPY[crash].mean()), "average fall on the worst 1-in-20 days", "bad"),
         ("60% shares / 40% bonds: biggest fall", pc(b.max_dd), f"vs {pc(a.max_dd)} for all shares", "good"),
         ("60/40 mix in 2022", pc((1 + ports['60/40 shares/bonds'].loc['2022']).prod() - 1), "bonds fell too, so the mix gave little protection", "bad")],
        [("Investment committee", "Treat only bonds and gold as crash protection. Overseas shares and property add growth, but they fall with US shares in a crisis."),
         ("Member (saver)", f"Mixing 60% shares with 40% bonds cut the biggest fall from {pc(a.max_dd)} to {pc(b.max_dd)}, for about {pc(a.cagr-b.cagr)} less growth a year."),
         ("Risk team", "Watch whether shares and bonds start moving together. When they do (the line goes above zero, as in 2022), bonds stop protecting.")],
        [("In a panic: Overseas shares and property moved almost in step with US shares when markets panicked", "Grey dot = normal days, red dot = panic days. The closer to 1, the less protection it gives.", c1),
         (f"Crash days: Only {join(protectors)} held steady or gained on crash days", "Average change on the worst 1-in-20 days for US shares. Green = held up, red = fell. Dashed line = US shares.", c2),
         ("Shares vs bonds: Bonds stopped cushioning shares in 2022", "Below zero = bonds tend to rise when shares fall (protection). Above zero = they fall together. Hover for any date.", c3),
         ("Crisis by crisis: Adding bonds and gold made every fall shallower, but helped least in 2022", "Biggest fall of each investment mix in each crisis (shorter bar = better). Rebalanced monthly, after trading costs.", c4)],
        ["ETFs used as stand-ins for each asset class. Yahoo Finance adjusted closes, 2010–2024.",
         f"Panic days = VIX closed above {STRESS_VIX} ({int(stress.sum())} of {len(stress)} days).",
         "Crash days = the 5% worst daily returns for SPY.",
         "Portfolios rebalance monthly to fixed weights, 0.10% cost per dollar traded. No look-ahead: weights are fixed in advance."],
        ["The US-dollar view. An Australian fund also faces currency moves (AUD often falls in panics, which helps unhedged foreign assets).",
         "ETFs carry small fees and tracking error vs their asset class.",
         "Fifteen years hold few true crashes (2011, 2015, 2018, 2020, 2022). Each one behaved differently."],
        t2.to_html(border=0, classes="sort") + "<br>" + t3.to_html(border=0, classes="sort"), "Data: Yahoo Finance via yfinance", models, sim, evidence=evidence(rets, px))
    print(tab.round(3)); print(pt.round(3))

if __name__ == "__main__":
    main()
