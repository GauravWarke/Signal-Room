"""Project 4 - Bank Earnings Shock Monitor
Client: investor-relations and trading-risk teams at a large US bank.
Question: how much do the six big US banks move on results day, beyond what the market did,
and does the move keep going afterwards?
Fixes two weaknesses of the original repo: (1) moves are market-adjusted, so a market-wide sell-off on results
day is not blamed on earnings; (2) all 60 quarterly reports from 2010-2024 are covered, not just recent ones."""
import os, json
import numpy as np, pandas as pd, yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # shared report.py at repo root
import report as R

BANKS = ["JPM", "BAC", "C", "WFC", "GS", "MS"]
BENCH = "SPY"
START, END = "2010-01-01", "2024-12-31"
BETA_WINDOW = 252      # beta estimated on the year BEFORE each report (no look-ahead)
DRIFT_DAYS = 5
BIG = 0.03             # a "shock" = market-adjusted move bigger than 3%

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

def load_earnings(path="data/earnings_dates.json"):
    if os.path.exists(path):
        return {k: pd.to_datetime(v) for k, v in json.load(open(path)).items()}
    out = {}
    for t in BANKS:
        d = yf.Ticker(t).get_earnings_dates(limit=80).index.tz_localize(None).normalize()
        out[t] = sorted(set(d[(d >= START) & (d <= END)]))
    json.dump({k: [str(x.date()) for x in v] for k, v in out.items()}, open(path, "w"))
    return {k: pd.to_datetime(v) for k, v in out.items()}

def event_table(rets, vol, t, dates):
    """US banks report before the market opens, so the reaction happens on the report date itself."""
    rows = []
    for d in dates:
        i = rets.index.searchsorted(d)
        if i < BETA_WINDOW or i + DRIFT_DAYS >= len(rets): continue
        hist = rets.iloc[i - BETA_WINDOW:i]
        beta = np.cov(hist[t], hist[BENCH])[0, 1] / hist[BENCH].var()
        ab = rets[t] - beta * rets[BENCH]
        day0 = ab.iloc[i]; after = ab.iloc[i + 1:i + 1 + DRIFT_DAYS].sum()
        typical = ab.iloc[i - BETA_WINDOW:i].abs().mean()
        vmult = vol[t].iloc[i] / vol[t].iloc[i - 20:i].mean()
        rows.append(dict(bank=t, date=rets.index[i], move=day0, abs_move=abs(day0), typical=typical,
                         shock_multiple=abs(day0) / typical, after_5d=after, volume_multiple=vmult))
    return pd.DataFrame(rows)


def run_models(ev, vix):
    ev = ev.sort_values(["bank", "date"]).copy()
    ev["vix"] = vix.shift(1).reindex(ev.date, method="ffill").values        # use previous close: known before the report
    ev["prev_abs"] = ev.groupby("bank").abs_move.shift(1)                   # how big last quarter's reaction was
    ev["inv_bank"] = ev.bank.isin(["GS", "MS"]).astype(int)
    ev["shock"] = (ev.abs_move > BIG).astype(int)
    d = ev.dropna(subset=["prev_abs"])
    X = sm.add_constant(d[["vix", "prev_abs", "inv_bank"]])
    lab = {"vix": "VIX the day before (1 point)", "prev_abs": "Last quarter's move (0.01 = 1%)", "inv_bank": "Investment bank (GS, MS)"}
    ols = sm.OLS(d.abs_move, X).fit(cov_type="cluster", cov_kwds={"groups": d.bank.astype("category").cat.codes})
    yr = ev.assign(year=ev.date.dt.year).groupby(["bank", "year"]).agg(shocks=("shock", "sum"), reports=("shock", "size"), vix=("vix", "mean")).reset_index()
    yr["inv_bank"] = yr.bank.isin(["GS", "MS"]).astype(int)
    poi = sm.GLM(yr.shocks, sm.add_constant(yr[["vix", "inv_bank"]]), family=sm.families.Poisson(), exposure=yr.reports).fit(cov_type="HC0")
    lab_p = {"vix": "Average VIX on report days (1 point)", "inv_bank": "Investment bank (GS, MS)"}
    train = d.date < "2019-01-01"
    logit = sm.GLM(d.shock[train], X[train], family=sm.families.Binomial()).fit()
    test_auc = R.auc(d.shock[~train], logit.predict(X[~train]))
    return d, (ols, lab), (poi, lab_p, yr), (logit, lab, test_auc)

def evidence(rets, vix, ev):
    """Test the 'why': are results days really special, and does fear make them extra sensitive?"""
    rng = np.random.default_rng(3); rows = []
    for t in BANKS:
        beta = (rets[t].rolling(BETA_WINDOW).cov(rets[BENCH]) / rets[BENCH].rolling(BETA_WINDOW).var()).shift(1)
        ab = rets[t] - beta * rets[BENCH]; typical = ab.abs().rolling(BETA_WINDOW).mean().shift(1)
        d = pd.DataFrame({"abs": ab.abs(), "mult": ab.abs() / typical, "vix": vix.shift(1).reindex(rets.index), "bank": t}).dropna()
        d["event"] = d.index.isin(ev[ev.bank == t].date).astype(int); rows.append(d)
    D = pd.concat(rows); E, N = D[D.event == 1], D[D.event == 0]
    fake = np.array([N["abs"].sample(len(E), random_state=int(s)).mean() for s in rng.integers(0, 1_000_000, 2000)])
    D["vix_x_event"] = D.vix * D.event
    m = sm.OLS(D["mult"], sm.add_constant(D[["vix", "event", "vix_x_event"]])).fit(cov_type="cluster", cov_kwds={"groups": D.bank.astype("category").cat.codes})
    return [
        dict(claim="Results days move banks more than ordinary days.",
             test=f"Placebo test: drew 2,000 sets of {len(E)} random non-results days and compared their average move with the real results days.",
             result=f"Real results days: {E['abs'].mean():.2%} average move. Random days: {fake.mean():.2%} on average and never above {fake.max():.2%}.",
             verdict="Supported" if (fake >= E["abs"].mean()).mean() < 0.01 else "Not supported"),
        dict(claim="Fear makes results-day reactions extra sharp.",
             test="Regressed move size (vs each bank's normal) on VIX, a results-day flag and their interaction, errors clustered by bank.",
             result=f"Higher VIX enlarges moves on all days ({R.pt(m.pvalues['vix'])}). The extra effect on results days is {m.params['vix_x_event']:+.3f} per VIX point ({R.pt(m.pvalues['vix_x_event'])}).",
             verdict="Supported" if m.pvalues["vix_x_event"] < 0.05 else "Not supported")]

def main():
    R.style()
    px, vol = load_prices(BANKS + [BENCH, "^VIX"], START, END)
    vix = px["^VIX"].ffill(); px = px[BANKS + [BENCH]]; vol = vol[BANKS + [BENCH]]
    rets = px.pct_change().dropna(); vol = vol.reindex(rets.index)
    earn = load_earnings()
    ev = pd.concat([event_table(rets, vol, t, earn[t]) for t in BANKS], ignore_index=True)
    ev.to_csv("events.csv", index=False)

    g = ev.groupby("bank")
    s = pd.DataFrame({"reports": g.size(), "avg_move": g.abs_move.mean(), "shock_multiple": g.shock_multiple.median(),
                      "shock_rate": g.abs_move.apply(lambda x: (x > BIG).mean()), "volume_multiple": g.volume_multiple.median()}).loc[BANKS]
    big = ev[ev.abs_move > BIG].copy(); big["continued"] = np.sign(big.move) == np.sign(big.after_5d)
    s["continued_rate"] = big.groupby("bank").continued.mean().reindex(BANKS)
    s.round(4).to_csv("results.csv")
    top, low = s.shock_multiple.idxmax(), s.shock_multiple.idxmin()
    cont = big.continued.mean(); corr = ev.move.corr(ev.after_5d)

    # Chart 1: median shock multiple
    o = s.shock_multiple.sort_values()
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.barh(o.index, o, color=R.highlight(o.index, [top]))
    ax.axvline(1, color=R.INK, lw=.8, ls="--"); ax.text(1, -.8, " normal day", fontsize=9)
    ax.set_xlabel("Results-day move compared with a normal day (1× = normal)"); c1 = R.fig_to_b64(fig)

    # Chart 2 (interactive, Chart.js): every results day. Hover a dot for bank, date and move.
    rng = np.random.default_rng(0)
    pts = lambda sel: [dict(x=round(BANKS.index(r.bank) + rng.uniform(-.2, .2), 3), y=round(float(r.move), 4), name=f"{r.bank} · {r.date:%d %b %Y}") for r in ev[sel].itertuples()]
    c2 = R.js_chart(dict(type="scatter", fmt_y="pct1", y_title="Share-price move on results day (market move removed)", x_ticks=BANKS,
        datasets=[dict(label=f"Move under {BIG:.0%}", color=R.GREY, points=pts(ev.abs_move <= BIG)),
                  dict(label=f"Move of {BIG:.0%} or more", color=R.BAD, points=pts(ev.abs_move > BIG))]))

    # Chart 3: shock rate per year (is it getting calmer?)
    ev["year"] = ev.date.dt.year; yr = ev.groupby("year").abs_move.apply(lambda x: (x > BIG).mean())
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.bar(yr.index, yr * 100, color=R.ACC); ax.set_ylabel(f"% of results days with a {BIG:.0%}+ move"); c3 = R.fig_to_b64(fig)

    # Chart 4: day-0 move vs next 5 days
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(ev.move * 100, ev.after_5d * 100, s=16, color=R.ACC, alpha=.6)
    ax.axhline(0, color=R.INK, lw=.8); ax.axvline(0, color=R.INK, lw=.8)
    ax.set_xlabel("Move on results day (%)"); ax.set_ylabel(f"Move over the next {DRIFT_DAYS} trading days (%)"); c4 = R.fig_to_b64(fig)

    pc = lambda x: f"{x*100:.0f}%"
    t = s.copy(); t.columns = ["Results days", "Average move", "Times a normal day", f"Share with {BIG:.0%}+ move", "Trading vs normal", "Big moves that kept going"]
    for c in ["Average move", f"Share with {BIG:.0%}+ move", "Big moves that kept going"]: t[c] = t[c].map(lambda x: f"{x*100:.1f}%")
    for c in ["Times a normal day", "Trading vs normal"]: t[c] = t[c].map(lambda x: f"{x:.1f}×")

    d, (ols, lab), (poi, lab_p, byr), (logit, lab_l, test_auc) = run_models(ev, vix)
    pd.DataFrame({"ols": ols.params, "poisson": poi.params, "logit": logit.params}).round(5).to_csv("model_coefficients.csv")
    bl = logit.params
    models = [
        ("Multiple regression · continuous", "What makes a results-day move bigger?",
         f"Market fear matters, but explains little ({ols.rsquared*100:.0f}% of the variation). Each extra 10 VIX points adds about {ols.params['vix']*10*100:.1f} points to the size of the move ({R.pt(ols.pvalues['vix'])}). Last quarter's reaction adds {'little' if ols.pvalues['prev_abs'] > 0.05 else 'some'} information ({R.pt(ols.pvalues['prev_abs'])}).",
         f"OLS on {len(d)} results days, errors clustered by bank. R² = {ols.rsquared:.2f}.", R.coef_table(ols, lab, "ols")),
        ("Poisson regression · count", "How many 3%+ shocks should a bank expect in a year?",
         f"Ten more VIX points multiplies the yearly shock count by {np.exp(poi.params['vix']*10):.1f}. Investment banks: ×{np.exp(poi.params['inv_bank']):.2f} vs commercial banks at the same VIX.",
         f"Poisson GLM on {len(byr)} bank-years, reports per year as exposure.", R.coef_table(poi, lab_p, "poisson")),
        ("Binomial regression · decision", "Should the desk hedge before this report?",
         f"Fitted on 2010–2018 and tested on 2019–2024, it ranked a 3%+ shock above a quiet report {test_auc*100:.0f}% of the time (50% = guessing). {'Useful as a hedging trigger.' if test_auc >= 0.65 else 'A weak signal: use it to rank reports, not to trade on alone.'}",
         f"Logistic GLM. Outcome: market-adjusted move above {BIG:.0%} ({d.shock.mean()*100:.0f}% of reports). Test AUC {test_auc:.2f}.", R.coef_table(logit, lab_l, "logit"))]
    sim = R.sim_html("Try it: hedge before this bank's results?", "Describe the upcoming results day. The model gives the chance of a 3% or bigger move.",
        float(bl["const"]), [("vix", "Market fear index (VIX) the day before: 15 = calm, 40 = panic", 10, 50, 1, 18, float(bl["vix"]), 0, ""),
                             ("prev_abs", "Size of last quarter's results-day move (0.02 = 2%)", 0, 0.10, 0.005, 0.02, float(bl["prev_abs"]), 3, ""),
                             ("inv_bank", "Investment bank like GS or MS? (0 = no, 1 = yes)", 0, 1, 1, 0, float(bl["inv_bank"]), 0, "")],
        40, "Hedge before results", "No hedge needed")

    R.build("dashboard.html", "Bank Earnings Shock Monitor", f"Bank IR & trading risk · {len(ev)} results days, 2010–2024",
        "How hard do results days hit the big US banks?",
        f"On results day a big bank's shares typically move {s.shock_multiple.median():.1f}× as much as on an ordinary day, even after removing market-wide moves. "
        f"{top} reacts most. Big moves did not reliably continue: only {pc(cont)} kept going the same way the next week.",
        [("Results-day move", f"{s.shock_multiple.median():.1f}× normal", "a typical results day vs an ordinary day", "bad"),
         ("Most reactive bank", top, f"{s.loc[top,'shock_multiple']:.1f}× a normal day · {low} least at {s.loc[low,'shock_multiple']:.1f}×", ""),
         (f"Results days with a {BIG:.0%}+ move", pc((ev.abs_move > BIG).mean()), "about 1 in 3, not counting market-wide moves", ""),
         ("Big moves that kept going", pc(cont), "about a coin flip: no reliable follow-through", "")],
        [("Investor relations", f"Expect about {s.volume_multiple.median():.1f}× the usual trading on results day. Have answers ready for the first hour of trading, when most of the move happens."),
         ("Trading risk desk", f"Reduce or protect bank holdings before results. A move of {BIG:.0%} or more happens on about {pc((ev.abs_move > BIG).mean())} of results days."),
         ("Investor", "Buying or selling after a big results-day move did not pay: the next week went either way.")],
        [(f"By bank: {top} moves the most on results day", "How big a typical results-day move is compared with an ordinary day for that bank. 1× = an ordinary day.", c1),
         ("Every report: Most reports are quiet; a few are large shocks", f"Each dot is one results day (2011–2024). Red = a move of more than {BIG:.0%}; grey = smaller. Hover a dot for the bank and date.", c2),
         (f"By year: Shocks were most common in {', '.join(str(y) for y in sorted(yr.nlargest(3).index))} and rarest in {yr.idxmin()}", f"Share of all results days each year with a move of more than {BIG:.0%}. Busy years line up with market stress (2011, 2020, 2022).", c3),
         ("Next week: A big results-day move told you nothing about the next week", "Each dot is one results day. Dots spread evenly in all four corners: no pattern to trade on.", c4)],
        ["Report dates: Yahoo Finance earnings calendar (60 quarterly reports per bank, 2010–2024).",
         f"Market-adjusted move = bank return − beta × S&P 500 return. Beta uses the {BETA_WINDOW} trading days before each report only.",
         "Shock multiple = |results-day move| ÷ average |daily move| over the prior year.",
         "Volume multiple = results-day volume ÷ average of the previous 20 days."],
        ["Banks report before the US open; the model assumes the reaction is on the report date. A late-dated report would be mis-measured.",
         "Earnings surprise (actual vs forecast) is not included, so the analysis measures size of reaction, not its cause.",
         "Six banks only. Regional banks behave differently (e.g. the 2023 crisis)."],
        t.to_html(border=0, classes="sort"), "Data: Yahoo Finance via yfinance", models, sim, evidence=evidence(rets, vix, ev))
    print(s.round(3)); print("continued", cont, "corr", corr)

if __name__ == "__main__":
    main()
