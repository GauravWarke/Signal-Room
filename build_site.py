"""Builds docs/index.html (the landing page) from the JSON summaries each project writes.
Run the five project scripts first, then: python build_site.py"""
import json, os, re, html
import report as R

ORDER = ["hold-ability", "steady-ride-fund", "crisis-diversification", "bank-earnings-risk", "sector-risk-budget"]
HEADLINE_KPI = {"sector-risk-budget": 2}          # which KPI represents each project on the landing page (default: first)
e = html.escape

def main():
    S = {s: json.load(open(f"docs/data/{s}.json")) for s in ORDER}
    sr = S["steady-ride-fund"]; k0 = sr["kpis"][0]

    rows = ""
    for s in ORDER:
        d = S[s]; l, v, sub, c = d["kpis"][HEADLINE_KPI.get(s, 0)]
        rows += (f'<a href="{s}.html" data-tier="{d["tier"]}" data-text="{e((d["title"]+" "+d["question"]+" "+d["answer"]+" "+d["client"]).lower())}">'
                 f'<div><b>{e(d["short"])}</b><small>{e(l)}</small></div><div class="v"><b class="{c}">{e(v)}</b><small>{e(sub)}</small></div></a>')

    minis = ""
    for s in ORDER:
        for m in S[s]["models"]:
            a = re.search(r"Test AUC ([0-9]+\.[0-9]+)", m["fit"])
            if m["type"].startswith("Binomial") and a:
                v = float(a.group(1)); c = "good" if v >= .7 else ("" if v >= .6 else "bad")
                verdict = "Ready to use" if v >= .7 else ("Use with care" if v >= .6 else "Weak signal")
                minis += f'<a class="mini" href="{s}.html"><div class="row"><b>{e(S[s]["short"])}</b><span class="kv {c}" style="font-size:20px">{v*100:.0f}%</span></div><div class="bar"><i style="width:{v*100:.0f}%"></i></div><small class="ks">{e(verdict)}</small></a>'

    cards = "".join(f'<a class="card proj" href="{s}.html" data-tier="{S[s]["tier"]}" data-text="{e((S[s]["title"]+" "+S[s]["question"]+" "+S[s]["answer"]).lower())}" style="text-decoration:none">'
                    f'<div class="kl">{e(S[s]["client"])}</div><h3 class="q" style="font-size:19px">{e(S[s]["question"])}</h3>'
                    f'<p class="muted">{e(S[s]["answer"])}</p><span class="btn" style="margin-top:8px">See the full story →</span></a>' for s in ORDER)

    body = f"""<div class="row"><div class="pills" role="tablist">
<button class="pill" data-f="all" aria-selected="true">All projects</button><button class="pill" data-f="Product decisions" aria-selected="false">Product decisions</button><button class="pill" data-f="Risk analytics" aria-selected="false">Risk analytics</button></div>
<label class="search"><svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.7"><circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/></svg>
<input id="q" type="search" placeholder="Search findings: bonds, VIX, warning…" aria-label="Search findings"></label></div>
<div class="dash">
 <div class="col">
  <div class="card glow hero"><div class="row"><span class="kl">{e(k0[0])} · Steady-Ride</span><span class="chip" aria-selected="true">2018–24</span></div>
   <div class="kv good">{e(k0[1])}</div><div class="ks">{e(k0[2])}, with no borrowing and after trading costs</div></div>
  <div class="card promo"><h3 style="margin:0;font-size:20px;font-weight:600">Decisions powered by data</h3>
   <p class="muted">Each project ends in a yes/no business decision, tested on years the model never saw.</p><a class="btn" href="hold-ability.html">See the full story →</a></div>
 </div>
 <div class="card"><div class="row"><h3 style="margin:0;font-weight:600">Key findings</h3><span class="ks" id="count"></span></div><div class="list" id="list">{rows}</div><div class="empty" id="empty" hidden>No project matches that search.</div></div>
 <div class="card"><div class="row"><h3 style="margin:0;font-weight:600">Decision models</h3><span class="ks">reliability on unseen years</span></div>
  <div class="minis">{minis}</div><p class="fit" style="margin-top:10px">How often each decision model picks the riskier case correctly. 50% = a coin flip.</p></div>
</div>
<div class="card"><div class="row"><h3 style="margin:0;font-weight:600">Steady-Ride rule vs S&amp;P 500 · value of $10,000</h3>
 <div class="chips" style="margin:0"><button class="chip" data-v="growth" aria-selected="true">Growth</button><button class="chip" data-v="dd" aria-selected="false">Drawdown</button>
 <span style="width:10px"></span><button class="chip" data-r="52" aria-selected="false">1Y</button><button class="chip" data-r="156" aria-selected="false">3Y</button><button class="chip" data-r="0" aria-selected="true">All</button></div></div>
 <div class="chartbox"><canvas id="perf" aria-label="Growth of $10,000: Steady-Ride rule vs S&P 500"></canvas></div></div>
<div class="kl sec">All projects</div><div class="grid3" id="cards">{cards}</div>
<script id="series" type="application/json">{json.dumps(sr["series"])}</script>
<script src="assets/landing.js?v={R.asset_version()}"></script>"""
    open("docs/index.html", "w").write(R.shell("index", "Signal Room", "<em>Signal</em> Room",
                                               "Fifteen years of market data, turned into five decisions you can defend", body))
    print("wrote docs/index.html")

if __name__ == "__main__":
    main()
