"""Render markdown reports from templates so every number is computed, not typed.

Templates live in reports/templates/*.md.tmpl (-> reports/) and docs_templates/*.md.tmpl (-> repo root). Inside a
template, `{{ expression }}` is evaluated against:
    km      - outputs/tables/key_metrics.json
    t(name) - a pandas DataFrame from outputs/tables/<name>.csv
    pct, num, f2, f3, md - formatting helpers
Rendering fails loudly if any expression cannot be evaluated.
"""
from __future__ import annotations

import json
import re

import pandas as pd

from . import config

TEMPLATE_DIR = config.REPORTS_DIR / "templates"
PATTERN = re.compile(r"\{\{(.+?)\}\}", re.S)


def pct(x, d=1):
    return f"{100 * float(x):.{d}f}%"


def num(x):
    x = float(x)
    return f"{int(round(x)):,}" if abs(x - round(x)) < 1e-9 or abs(x) >= 1000 else f"{x:,.2f}"


def f2(x):
    return f"{float(x):.2f}"


def f3(x):
    return f"{float(x):.3f}"


def md(df: pd.DataFrame, cols=None, fmt=None, index=False) -> str:
    d = df.copy() if cols is None else df[cols].copy()
    for c, f in (fmt or {}).items():
        d[c] = d[c].map(lambda v, f=f: "" if pd.isna(v) else f(v))
    d = d.astype(object).where(d.notna(), "")
    return d.to_markdown(index=index)


def case_md(c: dict) -> str:
    """One case study as markdown (all values from case_studies.json)."""
    lab = "labelled (AMLSim typology participant)" if c["label_is_suspicious"] else "not labelled"
    fp, bt, tp, nw = c["full_period"], c["baseline_before_test"], c["test_period"], c["network_full_period"]
    contrib = "\n".join(f"   - {x['feature']} = {x['value']:.4g} (contribution {x['contribution']:+.3f})"
                         for x in c["top_contributions_peak_week"]) or "   - none"
    weeks = ", ".join(f"w{w}: {s:.1f}" for w, s in c["weekly_scores"].items())
    return f"""### {c['case']} - account {c['account_id']}

*Category:* {c['category']} · *Selection rule:* {c['selection_rule']}

1. **Account profile** - initial balance {c['initial_balance']:.2f}; full period {fp['transactions']} transfers
   ({fp['in_count']} in / {fp['out_count']} out), value {fp['total_value']:,.2f}, {fp['unique_counterparties']} counterparties,
   median amount {fp['median_amount']:.2f}.
2. **Behavioural baseline (before the test period)** - {bt['transactions']} transfers over {bt['weeks_observed']} weeks,
   median amount {bt['median_amount'] if bt['median_amount'] is not None else 'n/a'}.
3. **Test-period activity** - {tp['transactions']} transfers worth {tp['value']:,.2f} across {tp['active_weeks']} active weeks.
4. **Incoming relationships** - {tp['senders']} distinct senders in test weeks.
5. **Outgoing relationships** - {tp['receivers']} distinct receivers in test weeks.
6. **Network position (full period, descriptive)** - in-degree {nw['in_degree']:.0f}, out-degree {nw['out_degree']:.0f},
   PageRank {nw['pagerank']:.6f}, sampled betweenness {nw['betweenness']:.6f}.
7. **Signals fired (any test week)** - {', '.join(c['signals_fired_any_test_week']) or 'none'}.
8. **Risk** - max score {c['max_risk_score']:.2f} in week {c['peak_week']}; priority **{c['priority']}**; alerted in
   {c['weeks_alerted']} of {tp['active_weeks']} active weeks; amount+velocity baseline alerted in {c['b2_amount_velocity_alert_weeks']} week(s).
   Weekly scores: {weeks}.
9. **Evidence (top contributions, peak week)**
{contrib}
10. **Interpretation** - {c['interpretation']}
11. **Ground truth and limitation** - the account is {lab}. The data contain no KYC, account purpose, device or
    location information, so the analytical reading cannot be corroborated as an investigator would.
"""


def _table(name: str) -> pd.DataFrame:
    return pd.read_csv(config.TABLES_DIR / f"{name}.csv")


def render(text: str, env: dict) -> str:
    def sub(m):
        expr = m.group(1).strip()
        try:
            safe = {"len": len, "round": round, "min": min, "max": max, "abs": abs, "sorted": sorted,
                    "list": list, "int": int, "float": float, "str": str, "sum": sum, "dict": dict}
            return str(eval(expr, {**env, "__builtins__": safe}))
        except Exception as exc:  # pragma: no cover - surfaced to the user
            raise RuntimeError(f"template expression failed: {expr!r}: {exc}") from exc
    return PATTERN.sub(sub, text)


def render_all() -> list[str]:
    km = json.loads((config.TABLES_DIR / "key_metrics.json").read_text())
    cases = json.loads((config.TABLES_DIR / "case_studies.json").read_text())
    env = {"km": km, "t": _table, "pct": pct, "num": num, "f2": f2, "f3": f3, "md": md, "pd": pd,
           "cases": cases, "cfg": config, "case_md": case_md, "B": km["baseline"], "R": km["ranking"], "H": km["hypotheses"],
           # shorthands for the most-cited approaches
           "I": km["baseline"]["B5b Integrated risk score (top 1%)"], "RI": km["ranking"]["B5b Integrated risk score (top 1%)"],
           "V": km["baseline"]["B2 Amount + velocity"], "RV": km["ranking"]["B2 Amount + velocity"],
           "A": km["baseline"]["B1 Amount threshold"], "RA": km["ranking"]["B1 Amount threshold"],
           "IR": km["baseline"]["B5a Integrated rules (>=2 families)"], "ML": km["ranking"]["ML1 Logistic regression (unconstrained)"],
           "MLB": km["baseline"]["ML1 Logistic regression (unconstrained)"], "AN": km["artifact_neutralised"]}
    written = []
    for tmpl in sorted(TEMPLATE_DIR.glob("*.md.tmpl")):
        out = config.REPORTS_DIR / tmpl.name.replace(".md.tmpl", ".md")
        out.write_text(render(tmpl.read_text(), env))
        written.append(str(out))
    for tmpl in sorted((config.ROOT / "docs_templates").glob("*.md.tmpl")):
        out = config.ROOT / tmpl.name.replace(".md.tmpl", ".md")
        out.write_text(render(tmpl.read_text(), env))
        written.append(out.name)
    print(f"reports rendered: {len(written)}", flush=True)
    return written
