"""Writes briefs/<project>.md (kept out of git): a plain-English explainer to read before presenting the project.
Built from the JSON each analysis writes, so the numbers always match the dashboard. Run after the analyses."""
import json, os

PROJECTS = {"01-hold-ability": "hold-ability", "02-steady-ride-fund": "steady-ride-fund", "03-crisis-diversification": "crisis-diversification",
            "04-bank-earnings-risk": "bank-earnings-risk", "05-sector-risk-budget": "sector-risk-budget"}

def brief(d):
    w = d["five_w"]; k = d["kpis"]
    kp = "\n".join(f"| {l} | **{v}** | {s} |" for l, v, s, _ in k)
    ev = "\n".join(f"- **{x['verdict']}:** {x['claim']}\n  - Test: {x['test']}\n  - Result: {x['result']}" for x in d["evidence"])
    acts = "\n".join(f"- **{r}:** {t}" for r, t in d["actions"])
    mods = "\n".join(f"- **{m['type']}:** {m['question']} ({m['fit']})" for m in d["models"])
    lim = "\n".join(f"- {x}" for x in d["limits"])
    return f"""# {d['title']}: project brief

*Read this before you present the project. The dashboard has the numbers; this explains what they mean.*

## The five questions
| | |
|---|---|
| **What** | {d['answer']} |
| **Why** | {w['why']} |
| **How** | {w['how']} |
| **When** | {w['when']} |
| **Where** | {w['where']} |

## In plain words
- **What it does:** answers "{d['question']}"
- **Why it exists:** {d['problem']}
- **Who it's for:** {d['client']}
- **When to use it:** {w['when']}
- **The decision it supports:** {d['decision']}

## Explain it in 30 seconds
1. **Client:** {d['client']}.
2. **Problem:** {d['problem']}
3. **Question:** {d['question']}
4. **Finding:** {d['answer']}
5. **Action:** {d['actions'][0][1]}

## Key numbers
| Measure | Value | Context |
|---|---|---|
{kp}

## How we know the "why" (evidence, not just a story)
{ev}

Say "the evidence supports", not "proves". These tests rule out the obvious alternatives, but they are not a controlled experiment.

## The three models, in one line each
{mods}

## Recommended actions
{acts}

## Limits to raise before someone else does
{lim}
"""

def main():
    for folder, slug in PROJECTS.items():
        d = json.load(open(os.path.join("docs", "data", f"{slug}.json")))
        os.makedirs("briefs", exist_ok=True)
        open(os.path.join("briefs", f"{folder}.md"), "w").write(brief(d))
    print("wrote 5 briefs to briefs/")

if __name__ == "__main__":
    main()
