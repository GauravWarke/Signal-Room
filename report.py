"""Shared builder for the Decision Analytics site (GitHub Pages).
Pages are built with a fixed working method (business problem → data collection & cleaning → EDA → visualisation →
insights & recommendations). The method shapes the work; the page itself leads with the decision and keeps details collapsible.
Charts are drawn in Python (matplotlib) in the site's dark theme; light vanilla JavaScript handles stage tabs,
chart tabs, sortable tables and the decision simulator. Each page also writes a JSON summary used by the landing page."""
import base64, io, html, json, os, hashlib, datetime
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Dark mauve palette (matches the site CSS)
INK, MUTE, LINE, ACC, GOOD, BAD = "#1E2A5A", "#64748B", "#E6EAF2", "#2F5BEA", "#10B981", "#EF4444"
GREY = "#C3CCDB"
ORANGE, SKY, NAVY = "#F5A04A", "#5BB6F4", "#1B237A"
_HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.join(_HERE, "docs") if os.path.isdir(os.path.join(_HERE, "docs")) else os.path.join(_HERE, "..", "docs")

NAV = [("index", "Overview", "grid"), ("hold-ability", "Hold-ability", "shield"), ("steady-ride-fund", "Steady-Ride Fund", "wave"),
       ("crisis-diversification", "Crisis Check", "umbrella"), ("bank-earnings-risk", "Bank Earnings", "bolt"), ("sector-risk-budget", "Sector Risk", "layers")]
FOLDER_TO_SLUG = {"01-hold-ability": "hold-ability", "02-steady-ride-fund": "steady-ride-fund", "03-crisis-diversification": "crisis-diversification",
                  "04-bank-earnings-risk": "bank-earnings-risk", "05-sector-risk-budget": "sector-risk-budget"}

META = {
 "hold-ability": dict(five_w={'why': 'Investors judge a stock by its long-run return but live through it day by day. Long stretches below the peak are what trigger panic selling, and a stock already in a slump tends to keep having bad days.', 'how': 'Scored each stock on time spent more than 10% below its peak and on return per unit of pain. A binomial model uses one thing a buyer can see, distance below the peak, to flag likely long slumps. It was tested on years it never saw.', 'when': 'At the moment someone buys, and again whenever a holding falls more than 10% below its peak.', 'where': 'On single-stock pages in the app, for large US companies. Not tested on small companies or crypto.'},
   tier="Product decisions", short="Hold-ability",
   client="A retail investing app: product team, compliance, app users",
   problem="App users buy famous stocks, then panic-sell during drops. The product team wants a warning on stock pages that sets honest expectations before someone buys.",
   decision="Should the app show a 'you may spend most of next year under water' warning on a stock page?",
   data="The 10 largest US companies on 1 Jan 2015 (AAPL, XOM, MSFT, BRK-B, GOOGL, JNJ, WFC, WMT, GE, JPM) plus SPY as the benchmark. Daily, 2015–2024.",
   cleaning=["Used dividend-adjusted closing prices so returns include dividends.",
             "Fixed the stock list in advance (largest companies on 1 Jan 2015) to avoid picking winners with hindsight.",
             "Converted prices to daily returns and dropped the first day (no prior price).",
             "Built month-end snapshots per stock for the models; dropped months without a full year of history before or after."]),
 "steady-ride-fund": dict(five_w={'why': 'Market swings come in clusters: a volatile month is usually followed by another. Holding less stock when swings are high avoids much of the worst stretch, while calm periods keep full exposure.', 'how': 'Each Friday, exposure = target volatility ÷ recent volatility, capped at 100%; the rest earns T-bill interest. Settings were chosen on 2010–2017 and judged only on 2018–2024, after trading costs.', 'when': 'Rebalanced weekly. The rule steps back when swings spike and returns as markets calm. It helps most in fast crashes like 2020 and less in slow declines like 2022.', 'where': "As a separate 'steady' S&P 500 fund for cautious investors. Not for investors who want the full index return and can sit through full drops."},
   tier="Product decisions", short="Steady-Ride Fund",
   client="The product team at an ETF issuer",
   problem="Cautious investors want S&P 500 exposure but abandon it in crashes. The team is testing a fund that holds less stock when markets swing, and parks the rest in T-bills.",
   decision="Should the fund cut its stock exposure next month?",
   data="SPY (S&P 500 fund), the VIX fear index and the 13-week T-bill yield (^IRX). Daily, 2010–2024.",
   cleaning=["Used dividend-adjusted SPY prices.", "Converted the T-bill yield from yearly percent to a daily rate.",
             "Filled 2 missing T-bill days with the previous day's value.",
             "Split time: settings chosen on 2010–2017 only, results judged on 2018–2024 only."]),
 "crisis-diversification": dict(five_w={'why': 'In a panic, investors sell every risky asset at once, so foreign shares and property fall with US shares. Bonds cushion shares only while interest rates are falling or stable. When rates rise, bond prices fall at the same time as shares, as in 2013 and from 2022. The data points to rising rates, not inflation fears as such.', 'how': "Compared each asset's link to US shares on calm and panic days, measured returns on the 5% worst days, and modelled the chance bonds rise on a crash day before and after 2022.", 'when': 'When VIX is above 25, and above all in inflation-driven sell-offs like 2022.', 'where': "In the fund's asset mix: keep bonds but add a second hedge such as gold. Don't count on international shares or property as crash insurance."},
   tier="Risk analytics", short="Crisis Check",
   client="A superannuation fund investment committee",
   problem="Funds diversify so one asset holds up when another falls. That promise is only tested in a panic, when many assets can fall together.",
   decision="On a crash day, can the committee count on bonds, or should it add a second hedge?",
   data="ETFs for US shares (SPY), international shares (EFA), emerging markets (EEM), property (VNQ), gold (GLD), long and mid US bonds (TLT, IEF), plus VIX. Daily, 2010–2024.",
   cleaning=["Used dividend-adjusted prices.", "Aligned VIX to trading days and filled gaps forward.",
             "Defined panic days (VIX above 25) and crash days (5% worst days for SPY) before any analysis.",
             "Portfolios rebalance monthly with a 0.10% trading cost and fixed weights (no look-ahead)."]),
 "bank-earnings-risk": dict(five_w={'why': 'Results reveal loan losses and trading income in one release, so the share price resets in a day. Fear (high VIX) makes every day more volatile, but the tests show results days are not extra sensitive to fear.', 'how': "Removed the market's own move using each bank's beta from the year before, compared results-day moves with normal days, and modelled shock size, shock counts and the chance of a 3%+ shock.", 'when': 'On results day itself, from the open. Risk is highest in fearful years (2011, 2020, 2022). The following week shows no reliable follow-through.', 'where': "The six largest US banks; GS and MS react slightly more. Used in investor-relations prep and in the trading desk's position sizing before results."},
   tier="Risk analytics", short="Bank Earnings",
   client="Investor relations and trading-risk teams at a large US bank",
   problem="Results days drive a bank's biggest share-price moves. IR needs to prepare, and the trading desk needs to size positions ahead of them.",
   decision="Should the desk hedge before this bank's results?",
   data="JPM, BAC, C, WFC, GS, MS with SPY and VIX. Daily prices and volumes 2010–2024, plus 60 quarterly report dates per bank from the Yahoo earnings calendar.",
   cleaning=["Matched each report date to the next trading day if it fell on a non-trading day.",
             "Removed the market's move: bank return minus beta × S&P 500 return, beta from the year before each report only.",
             "Dropped the first year of reports per bank (no prior year to estimate beta).",
             "Used VIX from the previous close so every driver was known before the report."]),
 "sector-risk-budget": dict(five_w={'why': "Daily losses have 'fat tails': big down days happen far more often than a bell curve assumes. Swapping in a fat-tailed curve fixes most of the gap; making the model react faster helps only a little.", 'how': "Each day set the bell-curve 99% limit from the past year only, counted how often real losses broke it, and modelled breaches from recent-vs-yearly swings and the week's VIX change, tested on unseen years.", 'when': "Check every night. Breaches bunch up when recent swings run above the past year and after VIX jumps, so tighten the next day's limit then.", 'where': "All nine US sectors, most of all Energy and Financials, inside the fund's daily loss-limit process."},
   tier="Risk analytics", short="Sector Risk",
   client="The risk manager of a mid-size equity fund",
   problem="The fund sets daily loss limits per sector using a bell-curve (normal) model. If the model understates losses, limits break more often than planned.",
   decision="Should the risk team tighten a sector's loss limit tomorrow?",
   data="Nine SPDR sector ETFs (XLK, XLF, XLE, XLV, XLY, XLP, XLU, XLI, XLB) with SPY and VIX. Daily, 2010–2024.",
   cleaning=["Used dividend-adjusted prices.", "Excluded newer sectors (Real estate, Communication services) that lack 15 years of history.",
             "Each day's model limit uses only the previous 252 trading days (no look-ahead).",
             "Lagged every model driver by one day so it was known the night before."]),
}

def style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 170, "font.size": 13, "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12, "font.family": "DejaVu Sans",
        "text.color": INK, "axes.edgecolor": LINE, "axes.labelcolor": MUTE, "axes.titlecolor": INK, "axes.facecolor": "white",
        "figure.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": LINE, "grid.linewidth": .7, "xtick.color": MUTE, "ytick.color": MUTE,
        "legend.frameon": False, "lines.linewidth": 1.8, "legend.labelcolor": INK,
        "axes.prop_cycle": plt.cycler(color=[ACC, ORANGE, GOOD, BAD, SKY, NAVY]),
    })

def fig_to_b64(fig):
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig); return base64.b64encode(buf.getvalue()).decode()

def js_chart(spec):
    """Mark a chart to be drawn in the browser with Chart.js instead of as a matplotlib image.
    spec: type ('line'|'bar'|'scatter'), datasets [{label, color, data | points[{x,y,name}], fill?, kind?}],
    labels, fmt_x/fmt_y ('pct0','pct1','num2','int'), x_title, y_title, x_ticks, zero_line."""
    return {"js": spec}

def highlight(labels, focus):
    return [ACC if l in focus else GREY for l in labels]

def fp(p):
    """p-value for tables: very small values shown as <0.001."""
    return "<0.001" if p < 0.001 else f"{p:.3f}"

def coef_table(res, labels, kind):
    """Plain-English coefficient table for a statsmodels result.
    kind: 'ols' (effect in outcome units), 'poisson' (x rate), 'logit' (x odds)."""
    import numpy as np
    rows = []
    for k, lab in labels.items():
        b, p = res.params[k], res.pvalues[k]
        eff = f"{b:+.3g}" if kind == "ols" else f"×{np.exp(b):.2f}"
        sig = "clear" if p < 0.01 else ("likely" if p < 0.05 else "not clear")
        rows.append(f"<tr><td>{html.escape(lab)}</td><td>{eff}</td><td>{fp(p)}</td><td>{sig}</td></tr>")
    head = {"ols": "Effect (per 1 unit)", "poisson": "Change in count rate", "logit": "Change in odds"}[kind]
    return f'<table class="sort"><thead><tr><th>Driver</th><th>{head}</th><th>p-value</th><th>Evidence</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'


def sim_html(title, caption, b0, sliders, threshold, yes, no):
    """sliders: [(key, label, min, max, step, value, coef, decimals, unit)]"""
    import json
    cfg = dict(b0=b0, coefs={s[0]: s[6] for s in sliders}, dec={s[0]: s[7] for s in sliders}, unit={s[0]: s[8] for s in sliders}, yes=yes, no=no)
    rows = "".join(f'<label class="sl"><span>{html.escape(s[1])}</span><input type="range" data-k="{s[0]}" min="{s[2]}" max="{s[3]}" step="{s[4]}" value="{s[5]}"><span class="v"></span></label>' for s in sliders)
    rows += f'<label class="sl"><span>Act when chance is above</span><input class="thr" type="range" min="5" max="95" step="5" value="{threshold}"><span class="v"></span></label>'
    return (f'<div class="sim" data-cfg=\'{json.dumps(cfg)}\'><h2>{html.escape(title)}</h2><p class="fit">{html.escape(caption)}</p>{rows}'
            f'<div class="out"><div><div class="kl">Model\'s chance</div><div class="pv"></div></div><div class="dec"></div></div></div>')


import numpy as np
def auc(y, p):
    """Area under the ROC curve via rank statistic: chance a random 'yes' case gets a higher score than a random 'no'."""
    from scipy.stats import rankdata
    y = np.asarray(y); r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def pt(p):
    """p-value phrase for sentences: 'p < 0.001' or 'p = 0.006'."""
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"

ICONS = {  # simple inline line icons
 "grid": '<path d="M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z"/>',
 "shield": '<path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/>',
 "wave": '<path d="M3 15c3 0 3-6 6-6s3 6 6 6 3-6 6-6"/>',
 "umbrella": '<path d="M3 12a9 9 0 0118 0zM12 12v7a2 2 0 01-4 0"/>',
 "bolt": '<path d="M13 3L5 14h6l-1 7 8-11h-6z"/>',
 "layers": '<path d="M12 3l9 5-9 5-9-5zM3 13l9 5 9-5"/>',
 "gh": '<path d="M9 19c-4 1.5-4-2-6-2m12 4v-3.5c0-1 .1-1.4-.5-2 2.8-.3 5.5-1.4 5.5-6a4.6 4.6 0 00-1.3-3.2 4.2 4.2 0 00-.1-3.2s-1-.3-3.4 1.3a11.6 11.6 0 00-6 0C6.8 2.8 5.8 3.1 5.8 3.1a4.2 4.2 0 00-.1 3.2A4.6 4.6 0 004.4 9.5c0 4.6 2.7 5.7 5.5 6-.6.6-.6 1.2-.5 2V21"/>',
 "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>',
 "trend": '<path d="M3 17l6-6 4 4 8-8M15 7h6v6"/>', "target": '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3"/>',
 "alert": '<path d="M12 3l9 16H3zM12 10v4M12 17h.01"/>', "coin": '<circle cx="12" cy="12" r="8"/><path d="M12 8v8M9.5 10.5h4a1.5 1.5 0 010 3h-3a1.5 1.5 0 000 3h4"/>',
 "moon": '<path d="M20 14.5A8 8 0 019.5 4 8 8 0 1020 14.5z"/>',
}
def icon(name):
    return f'<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{ICONS[name]}</svg>'

REPO = "https://github.com/GauravWarke/Signal-Room"   # link used by the sidebar "Source code" button

def asset_version():
    """Short hash of the CSS/JS so browsers fetch the new files after every change (cache busting)."""
    h = hashlib.sha1()
    for f in ("site.css", "site.js", "landing.js"):
        p = os.path.join(SITE, "assets", f)
        if os.path.exists(p): h.update(open(p, "rb").read())
    return h.hexdigest()[:8]

def shell(active, title, heading, sub, body, extra_head=""):
    """Sidebar + header frame shared by every page of the site."""
    e = html.escape
    nav = "".join(f'<a href="{s}.html" class="{"on" if s == active else ""}">{icon(i)}<span>{e(n)}</span></a>'
                  + ('<div class="navlab">Product decisions</div>' if s == "index" else "") + ('<div class="navlab">Risk analytics</div>' if s == "steady-ride-fund" else "")
                  for s, n, i in NAV)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(title)}</title><link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="assets/site.css?v={asset_version()}">{extra_head}
<script>try{{if(localStorage.getItem("theme")==="dark")document.documentElement.setAttribute("data-theme","dark")}}catch(e){{}}</script></head><body><div class="app">
<aside class="side"><a class="brand" href="index.html"><span class="logo">SR</span><span>Signal Room</span></a><nav>{nav}</nav>
<div class="sidefoot"><button class="themet" type="button" aria-pressed="false">{icon("moon")}<span>Dark mode</span><span class="sw"><i></i></span></button><a href="{REPO}" target="_blank" rel="noopener">{icon("gh")}<span>Source code</span></a></div></aside>
<main><header class="top"><div><h1>{heading}</h1><p class="sub">{e(sub)}</p></div></header>
{body}<footer class="foot">Not financial advice. · Version {asset_version()}</footer></main></div>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js" integrity="sha384-9nhczxUqK87bcKHh20fSQcTGD4qq5GhayNYSYWqwBkINBhOfQLg/P5HG5lF1urn4" crossorigin="anonymous"></script><script src="assets/site.js?v={asset_version()}"></script></body></html>"""


def data_profile():
    """Facts about the raw data file, for the Data Collection stage."""
    try:
        df = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True, header=[0, 1])["Close"]
        return [("Trading days", f"{len(df):,}"), ("Series", str(df.shape[1])),
                ("Period", f"{df.index.min():%b %Y} – {df.index.max():%b %Y}"), ("Missing values filled", f"{int(df.isna().sum().sum()):,}"), ("Duplicate days removed", f"{int(df.index.duplicated().sum()):,}")]
    except Exception:
        return []

KPI_ICONS = ["trend", "target", "alert", "coin"]
def kpi_card(label, value, sub, cls, i=0):
    """KPI tile: icon badge, big number, status pill (favourable / watch) and comparison line."""
    e = html.escape
    pill = {"good": '<span class="pill-s up">▲ Favourable</span>', "bad": '<span class="pill-s down">▼ Watch</span>'}.get(cls, '<span class="pill-s flat">● Context</span>')
    return (f'<div class="card kpi"><div class="kpi-h"><span class="kic k{i % 4}">{icon(KPI_ICONS[i % 4])}</span><span class="kl">{e(label)}</span></div>'
            f'<div class="kv">{e(value)}</div><div class="kpi-f">{pill}<span class="ks">{e(sub)}</span></div></div>')

def ring(pct, label, sub):
    """Donut gauge (SVG) like a progress ring. pct in 0..100."""
    r, c = 52, 2 * 3.14159 * 52; off = c * (1 - max(0, min(pct, 100)) / 100)
    return (f'<div class="ring"><svg viewBox="0 0 130 130" width="130" height="130" role="img" aria-label="{html.escape(label)} {pct:.0f}%">'
            f'<circle cx="65" cy="65" r="{r}" class="ring-bg"/><circle cx="65" cy="65" r="{r}" class="ring-fg" stroke-dasharray="{c:.1f}" stroke-dashoffset="{off:.1f}" transform="rotate(-90 65 65)"/>'
            f'<text x="65" y="70" text-anchor="middle" class="ring-t">{pct:.0f}%</text></svg><div><b>{html.escape(label)}</b><p class="ks">{html.escape(sub)}</p></div></div>')

def build(path, title, tag, question, answer, kpis, stakeholders, charts, method, limits, table_html="", source="", models=None, sim="", series=None, evidence=None):
    """Write docs/<slug>.html (five-stage page) and docs/data/<slug>.json (landing-page summary)."""
    e = html.escape
    slug = FOLDER_TO_SLUG[os.path.basename(os.getcwd())]; m = META[slug]
    k = "".join(kpi_card(l, v, s, c, i) for i, (l, v, s, c) in enumerate(kpis))
    roles = "".join(f"<li>{e(r)}</li>" for r, _ in stakeholders)
    recs = "".join(f'<div class="card rec"><div class="kl">{e(r)}</div><p>{e(t)}</p></div>' for r, t in stakeholders)
    prof = "".join(f'<div class="card kpi sm"><div class="kl">{e(a)}</div><div class="kv">{e(b)}</div></div>' for a, b in data_profile())
    clean = "".join(f"<li>{e(x)}</li>" for x in m["cleaning"]); meth = "".join(f"<li>{e(x)}</li>" for x in method)
    lim = "".join(f"<li>{e(x)}</li>" for x in limits)
    ctabs = "".join(f'<button class="chip" aria-selected="{str(i==0).lower()}">{e(h.split(":")[0])}</button>' for i, (h, p, b) in enumerate(charts))
    def visual(h, b):
        if isinstance(b, dict):   # interactive chart drawn by assets/site.js
            return f'<div class="jsbox"><canvas role="img" aria-label="{e(h)}" data-chart="{e(json.dumps(b["js"]))}"></canvas></div>'
        return f'<img alt="{e(h)}" src="data:image/png;base64,{b}">'
    cpanes = "".join(f'<div class="cpane"{"" if i==0 else " hidden"}><h3>{e(h.split(":",1)[-1].strip())}</h3><p class="muted">{e(p)}</p>{visual(h, b)}</div>' for i, (h, p, b) in enumerate(charts))
    # Layout follows the working method (problem → data → EDA → visuals → insight & action) without using it as headings:
    # the decision comes first, the evidence follows, and the data/method details sit in collapsible panels.
    fw = [("What", answer)] + [(w.title(), m["five_w"][w]) for w in ("why", "how", "when", "where")]
    wrows = "".join(f'<div class="w5r"><span class="w5k">{e(a)}</span><p>{e(b)}</p></div>' for a, b in fw)
    # Industry-style layout: KPI strip → main chart + side panel (decision tool, actions) → supporting charts →
    # detail table → "Trust Check" card (data source, cleaning, method, limits and prediction reliability in one place). The why/what narrative lives in each project's BRIEF.md.
    def chart_card(h, p, b, big=False):
        title = h.split(":", 1)[-1].strip()
        return (f'<div class="card chart{" main" if big else ""}"><div class="ch-h"><h3>{e(title)}</h3>'
                f'<span class="info" tabindex="0" aria-label="{e(p)}">i<span class="tip">{e(p)}</span></span></div>{visual(h, b)}</div>')
    main_c = chart_card(*charts[0], big=True) if charts else ""
    rest_c = "".join(chart_card(*c) for c in charts[1:])
    acts = "".join(f'<li><b>{e(r)}</b><span>{e(x)}</span></li>' for r, x in stakeholders)
    def plain_fit(fit):
        import re as _re
        a = _re.search(r"AUC ([0-9]+\.[0-9]+)", fit); r2 = _re.search(r"R² = ([0-9.]+[0-9])", fit)
        if a:
            v = float(a.group(1)); tag = "Reliable" if v >= .7 else ("Fair" if v >= .6 else "Weak")
            return f"{tag}: picks the riskier case {v*10:.1f} times out of 10 (5 = a coin flip)"
        if r2: return f"Explains {float(r2.group(1))*100:.0f}% of the ups and downs"
        if "Poisson" in fit: return "Estimates how many bad days to expect"
        return "Estimates the chance, using past years"
    kind = {"Multiple": "Size", "Poisson": "Count", "Binomial": "Yes / no decision"}
    score = "".join(f'<tr><td>{e(kind.get(ty.split(" ")[0], ty))}</td><td>{e(q)}</td><td>{e(plain_fit(fit))}</td></tr>'
                    for ty, q, res, fit, tbl in (models or []))
    import re as _re
    gauge = ""
    for ty, q, res, fit, tbl in (models or []):
        a = _re.search(r"AUC ([0-9]+\.[0-9]+)", fit)
        if ty.startswith("Binomial") and a:
            v = float(a.group(1)); tag = "Reliable" if v >= .7 else ("Fair" if v >= .6 else "Weak")
            gauge = f'<div class="card"><div class="kl">Decision model reliability</div>{ring(v * 100, tag, f"Picks the riskier case {v*10:.1f} times out of 10 on years it never saw. 50% = a coin flip.")}</div>'
    body = f"""<div class="kpis">{k}</div>
<div class="dgrid"><div class="dmain">{main_c}{rest_c}</div>
<aside class="dside">{gauge}{sim}<div class="card"><div class="kl">Recommended actions</div><ul class="acts">{acts}</ul></div></aside></div>
<div class="stack"><div class="card"><div class="row"><div class="kl">Detail table</div><span class="ks">click a column to sort</span></div><div class="tw">{table_html}</div></div>
<div class="card trust" id="trust"><div class="row"><div><h3 class="trust-h">Trust Check</h3><p class="ks">Before you act on these numbers: where they came from, how they were cleaned, and how far each prediction can be trusted.</p></div><span class="pill-s flat">Data & reliability</span></div>
<div class="kpis mini-k">{prof}</div>
<div class="kl" style="margin-top:6px">How reliable are the predictions?</div><div class="tw"><table class="score"><thead><tr><th>Type</th><th>What we predicted</th><th>How good it is</th></tr></thead><tbody>{score}</tbody></table></div>
<div class="grid2" style="margin-top:14px"><div><div class="kl">Where the data comes from</div><p>{e(m['data'])} Source: Yahoo Finance.</p><div class="kl" style="margin-top:12px">How it was cleaned</div><ul class="tight">{clean}</ul></div>
<div><div class="kl">How it was measured</div><ul class="tight">{meth}</ul><div class="kl" style="margin-top:12px">Limits to keep in mind</div><ul class="tight">{lim}</ul></div></div></div></div>"""
    os.makedirs(os.path.join(SITE, "data"), exist_ok=True)
    open(os.path.join(SITE, f"{slug}.html"), "w").write(shell(slug, f"{title} · Signal Room", e(title), f"{m['client']} · {question}", body))
    summary = dict(evidence=evidence or [], five_w=m["five_w"], problem=m["problem"], decision=m["decision"], data=m["data"], actions=[list(x) for x in stakeholders], limits=limits, slug=slug, title=title, short=m["short"], tier=m["tier"], client=m["client"], question=question, answer=answer,
                   kpis=[list(x) for x in kpis], models=[dict(type=ty, question=q, fit=fit) for ty, q, res, fit, tbl in (models or [])])
    if series: summary["series"] = series
    json.dump(summary, open(os.path.join(SITE, "data", f"{slug}.json"), "w"))
