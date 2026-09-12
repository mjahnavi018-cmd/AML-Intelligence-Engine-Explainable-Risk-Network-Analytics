"""Programmatic case-study selection (held-out accounts, test weeks).

Cases are SELECTED BY RULE from the data (never hand-picked or invented) so
that every number is reproducible:
  1. highest-priority labelled account                    -> strong detection example
  2. labelled potential mule-account candidate            -> mule example
  3. labelled account on a time-respecting cycle          -> AML-style pattern
  4. unlabelled account with the highest weekly velocity  -> 'unusual is not suspicious'
  5. account whose best weekly score sits closest above the alert threshold -> borderline
  6. labelled account with the lowest score among labelled accounts active >= 4 test weeks -> missed case
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config
from .alert_prioritization import FEATURE_LABEL, assign_tier
from .network_analysis import ego_edges
from .risk_scoring import FRAMEWORK_SIGNALS, SIGNAL_BY_ID


def _profile(ctx, acc_id: int) -> dict:
    run, tx, aw = ctx["run"], ctx["tx"], ctx["aw"]
    m = run.masks["heldout_test"] & (aw["account_id"] == acc_id)
    rows = aw[m].sort_values("week")
    sc = run.score[m]
    best = sc.idxmax()
    contrib = ctx["contrib"].loc[best].sort_values(ascending=False)
    fired = [s for s in FRAMEWORK_SIGNALS if run.sig.at[best, s]]
    fired_any = sorted({s for i in rows.index for s in FRAMEWORK_SIGNALS if run.sig.at[i, s]})
    pre = tx[(tx["step"] < config.TEST_WEEKS[0] * config.WEEK_LEN - 6) & ((tx["src"] == acc_id) | (tx["dst"] == acc_id))]
    during = tx[tx["week"].isin(config.TEST_WEEKS) & ((tx["src"] == acc_id) | (tx["dst"] == acc_id))]
    prof = ctx["prof"].set_index("account_id").loc[acc_id]
    cent = ctx["cent"].set_index("account_id")
    cut = ctx["km"]["tier_cutoffs"]
    tier = assign_tier(pd.Series([sc.max()]), cut).iloc[0] if sc.max() >= config.ALERT_PERCENTILE else "no alert"
    return {
        "account_id": int(acc_id),
        "label_is_suspicious": int(prof["is_suspicious"]),
        "initial_balance": float(prof["init_balance"]),
        "full_period": {"transactions": int(prof["tx_count"]), "total_value": round(float(prof["total_value"]), 2),
                        "in_count": int(prof["in_count"]), "out_count": int(prof["out_count"]),
                        "unique_counterparties": int(prof["unique_counterparties"]) if pd.notna(prof["unique_counterparties"]) else 0,
                        "median_amount": round(float(prof["median_amount"]), 2)},
        "baseline_before_test": {"transactions": len(pre),
                                 "weeks_observed": int(pre["week"].nunique()),
                                 "median_amount": round(float(pre["amount"].median()), 2) if len(pre) else None},
        "test_period": {"transactions": len(during), "value": round(float(during["amount"].sum()), 2),
                        "senders": int(during.loc[during["dst"] == acc_id, "src"].nunique()),
                        "receivers": int(during.loc[during["src"] == acc_id, "dst"].nunique()),
                        "active_weeks": int(rows.shape[0])},
        "network_full_period": {k: (round(float(cent.at[acc_id, k]), 6) if acc_id in cent.index else None)
                                for k in ("in_degree", "out_degree", "pagerank", "betweenness")},
        "max_risk_score": round(float(sc.max()), 2),
        "peak_week": int(aw.at[best, "week"]),
        "priority": tier,
        "weeks_alerted": int((sc >= config.ALERT_PERCENTILE).sum()),
        "signals_fired_peak_week": [f"{s} {SIGNAL_BY_ID[s]['name']}" for s in fired],
        "signals_fired_any_test_week": [f"{s} {SIGNAL_BY_ID[s]['name']}" for s in fired_any],
        "top_contributions_peak_week": [{"feature": FEATURE_LABEL[f], "value": round(float(aw.at[best, f]), 4),
                                         "contribution": round(float(v), 3)} for f, v in contrib.head(4).items()],
        "weekly_scores": {int(w): round(float(s), 2) for w, s in zip(rows["week"], sc.loc[rows.index])},
        "b2_amount_velocity_alert_weeks": int((run.sig.loc[rows.index, "TXN-01"] | run.sig.loc[rows.index, "VEL-01"]).sum()),
    }


def build(ctx: dict) -> dict:
    run, aw, km = ctx["run"], ctx["aw"], ctx["km"]
    m = run.masks["heldout_test"]
    h = aw[m].assign(score=run.score[m])
    acc = h.groupby("account_id").agg(score=("score", "max"), y=("is_suspicious", "first"),
                                      weeks=("week", "nunique"), max_ntx=("n_tx", "max"),
                                      cyc=("cycle_count", "max"))
    cent = ctx["cent"].assign(deg=lambda d: d["in_degree"] + d["out_degree"])
    hubs = set(cent.loc[cent["deg"] >= cent["deg"].quantile(0.99), "account_id"])  # top-1% degree accounts
    acc["hub"] = acc.index.isin(hubs)
    cases, used = [], set()

    def pick(series, title, kind, rationale):
        for a in series.index:
            if a not in used:
                used.add(a)
                cases.append({"case": title, "category": kind, "selection_rule": rationale, **_profile(ctx, a)})
                return

    pick(acc[acc.y == 1].sort_values("score", ascending=False), "Case 1 - highest-priority labelled account",
         "strong detection", "labelled held-out account with the highest weekly risk score in the test weeks")
    mule = ctx["mule"]
    mule_lab = mule[mule["is_suspicious"] == 1].merge(acc, left_on="account_id", right_index=True)
    mule_lab = mule_lab[~mule_lab["account_id"].isin(hubs)]
    mule_lab["cf"] = mule_lab["pattern"].eq("collector-forwarder")
    mule_lab = mule_lab.sort_values(["cf", "distinct_mule_signals", "score"], ascending=False).set_index("account_id")
    pick(mule_lab, "Case 2 - potential mule-account candidate", "mule",
         "labelled, non-hub (below top-1% degree) potential mule candidate; collector-forwarder pattern preferred, "
         "then most distinct mule signals, then highest score")
    pick(acc[(acc.y == 1) & (acc.cyc >= 1) & ~acc.hub].sort_values("score", ascending=False), "Case 3 - circular flow",
         "AML pattern", "labelled non-hub account on >= 1 time-respecting cycle in a test week, highest score")
    pick(acc[acc.y == 0].sort_values(["max_ntx", "score"], ascending=False),
         "Case 4 - unusual but not labelled", "legitimate unusual / false positive",
         "unlabelled account with the highest weekly transaction count in the test weeks")
    above = acc[acc.score >= config.ALERT_PERCENTILE].sort_values("score")
    pick(above, "Case 5 - borderline alert", "borderline",
         "account whose best weekly score is the lowest score that still produced an alert")
    pick(acc[(acc.y == 1) & (acc.weeks >= 4)].sort_values("score"), "Case 6 - missed labelled account",
         "false negative", "labelled account active >= 4 test weeks with the lowest risk score")

    for c in cases:
        c["interpretation"] = interpret(c)
        edges = ego_edges(ctx["tx"], c["account_id"], config.TEST_WEEKS[0] * 7 - 6, config.TEST_WEEKS[-1] * 7,
                          radius=1, max_edges=60)
        c["ego_edges_test_period"] = len(edges)
    with open(config.TABLES_DIR / "case_studies.json", "w") as fh:
        json.dump(cases, fh, indent=2)
    km["case_accounts"] = [c["account_id"] for c in cases]
    ctx["cases"] = cases
    return {"cases": cases}


def interpret(c: dict) -> str:
    lab = "labelled as a typology participant" if c["label_is_suspicious"] else "NOT labelled"
    sigs = ", ".join(s.split(" ", 1)[1] for s in c["signals_fired_any_test_week"]) or "no framework signal"
    top = c["top_contributions_peak_week"][0]["feature"] if c["top_contributions_peak_week"] else "n/a"
    if c["category"] == "legitimate unusual / false positive":
        return (f"This account is {lab} yet is among the busiest in the test weeks "
                f"({c['test_period']['transactions']} transfers with {c['test_period']['senders']} senders and "
                f"{c['test_period']['receivers']} receivers). The amount+velocity baseline alerted it in "
                f"{c['b2_amount_velocity_alert_weeks']} week(s); the integrated framework gave priority "
                f"'{c['priority']}' (max score {c['max_risk_score']}). High activity alone is weak evidence: "
                f"in a real bank this profile could be a business or collection account, and an analyst would "
                f"close it after checking account purpose - a field this dataset does not have.")
    if c["category"] == "false negative":
        return (f"This account is {lab} but its best weekly score was only {c['max_risk_score']} "
                f"(signals: {sigs}). Its weekly behaviour is indistinguishable from background accounts "
                f"with the available fields - typology members that transact rarely leave little evidence. "
                f"This illustrates the recall ceiling of behaviour-only monitoring.")
    return (f"Account is {lab}. Peak week {c['peak_week']} scored {c['max_risk_score']} (priority "
            f"'{c['priority']}'), alerted in {c['weeks_alerted']} of {c['test_period']['active_weeks']} active test weeks. "
            f"Signals fired: {sigs}. Largest contribution: {top}. Test-week flows: "
            f"{c['test_period']['senders']} distinct senders, {c['test_period']['receivers']} distinct receivers, "
            f"{c['test_period']['transactions']} transfers. The evidence describes a pattern worth review; it is "
            f"not proof of laundering or mule activity.")
