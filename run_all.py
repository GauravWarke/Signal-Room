"""One command to rebuild everything from the frozen data snapshot:
runs the five project analyses, rebuilds the landing page, writes each BRIEF.md, then checks the output.
Usage: python run_all.py            (uses cached data in each project's data/ folder)
       python run_all.py --refresh  (deletes caches and re-downloads from Yahoo Finance first)"""
import json, os, shutil, subprocess, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECTS = ["01-hold-ability", "02-steady-ride-fund", "03-crisis-diversification", "04-bank-earnings-risk", "05-sector-risk-budget"]
PAGES = ["index", "hold-ability", "steady-ride-fund", "crisis-diversification", "bank-earnings-risk", "sector-risk-budget"]

def run(cmd, cwd):
    print(f"→ {' '.join(cmd)}  ({os.path.relpath(cwd, ROOT) or '.'})")
    subprocess.run(cmd, cwd=cwd, check=True, stdout=subprocess.DEVNULL)

def check():
    problems = []
    for p in PAGES:
        f = os.path.join(ROOT, "docs", f"{p}.html")
        if not os.path.exists(f) or os.path.getsize(f) < 5_000: problems.append(f"missing or empty page: {p}.html")
    for p in PAGES[1:]:
        d = json.load(open(os.path.join(ROOT, "docs", "data", f"{p}.json")))
        for label, value, *_ in d["kpis"]:
            if str(value).lower() in ("nan", "nan%", "inf", ""): problems.append(f"{p}: KPI '{label}' has no value")
        if len(d["models"]) != 3: problems.append(f"{p}: expected 3 models, found {len(d['models'])}")
    return problems

def main():
    if "--refresh" in sys.argv:
        for p in PROJECTS:
            shutil.rmtree(os.path.join(ROOT, p, "data"), ignore_errors=True)
    for p in PROJECTS:
        run([sys.executable, "analysis.py"], os.path.join(ROOT, p))
    run([sys.executable, "build_site.py"], ROOT)
    run([sys.executable, "make_briefs.py"], ROOT)
    problems = check()
    if problems:
        print("\nChecks failed:\n  " + "\n  ".join(problems)); sys.exit(1)
    print("\nAll 6 pages built and checked. Preview with: python serve.py")

if __name__ == "__main__":
    main()
