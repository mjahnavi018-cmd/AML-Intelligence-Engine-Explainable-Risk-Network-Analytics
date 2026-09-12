"""Alert generation, priority tiers, and analyst-readable explanations.

Risk (how concerning) and evidence strength (how much independent support) are
reported separately. Priority tiers are cut on the development-period alert
score distribution and validated on the held-out test period.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .risk_scoring import FRAMEWORK_SIGNALS, SIGNAL_BY_ID, evidence_strength

FEATURE_LABEL = {
    "max_amt": "largest transfer", "n_tx": "weekly transaction count", "velocity_ratio": "velocity vs own history",
    "new_cp": "new counterparties this week", "amount_log_ratio": "amount vs own historical median",
    "rapid_movement": "rapid pass-through of funds", "cycle_count": "circular-flow participation",
    "fan_in_28": "distinct senders (28d)", "fan_out_28": "distinct receivers (28d)",
    "pagerank_28": "flow centrality (28d PageRank)",
}
FAMILY_CATEGORY = {"transaction": "Transaction anomaly", "velocity": "Velocity anomaly",
                   "behavioral": "Behavioural change", "aml_pattern": "AML pattern",
                   "network": "Network / mule pattern"}
TIERS = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
# shares of development-period alerts assigned to each tier (top 10% CRITICAL, next 20% HIGH, ...)
TIER_SHARES = {"CRITICAL": 0.10, "HIGH": 0.20, "MEDIUM": 0.30, "LOW": 0.40}


def tier_cutoffs(dev_alert_scores: pd.Series) -> dict[str, float]:
    s = dev_alert_scores.sort_values(ascending=False).to_numpy()
    cum, cuts = 0.0, {}
    for t in TIERS[:-1]:
        cum += TIER_SHARES[t]
        cuts[t] = float(s[min(len(s) - 1, int(np.ceil(cum * len(s))) - 1)])
    cuts["LOW"] = float(s[-1])
    return cuts


def assign_tier(score: pd.Series, cuts: dict[str, float]) -> pd.Series:
    return pd.Series(np.select([score >= cuts["CRITICAL"], score >= cuts["HIGH"], score >= cuts["MEDIUM"]],
                               ["CRITICAL", "HIGH", "MEDIUM"], "LOW"), index=score.index)


def explain(aw: pd.DataFrame, sig: pd.DataFrame, contrib: pd.DataFrame, top_n: int = 4) -> pd.DataFrame:
    """Per row: fired signals, primary signal/category, ranked evidence sentences."""
    fired_ids = sig[FRAMEWORK_SIGNALS]
    rows = []
    for idx in aw.index:
        fired = [s for s in FRAMEWORK_SIGNALS if fired_ids.at[idx, s]]
        c = contrib.loc[idx].sort_values(ascending=False)
        c = c[c > 0].head(top_n)
        feat_to_sig = {SIGNAL_BY_ID[s]["feature"]: s for s in FRAMEWORK_SIGNALS}
        evidence = []
        for feat, val in c.items():
            sid = feat_to_sig.get(feat)
            tag = f" [{sid} fired]" if sid in fired else ""
            v = aw.at[idx, feat]
            evidence.append(f"{FEATURE_LABEL[feat]} = {v:.4g} (contribution {val:+.2f}){tag}")
        # primary: the fired signal with the largest contribution; else top contributor
        fired_by_contrib = [feat_to_sig[f] for f in contrib.loc[idx].sort_values(ascending=False).index
                            if feat_to_sig.get(f) in fired]
        primary = fired_by_contrib[0] if fired_by_contrib else (feat_to_sig.get(c.index[0]) if len(c) else None)
        rows.append({
            "fired_signals": ", ".join(fired),
            "primary_signal": f"{primary} {SIGNAL_BY_ID[primary]['name']}" if primary else "",
            "risk_category": FAMILY_CATEGORY[SIGNAL_BY_ID[primary]["family"]] if primary else "",
            "supporting_signals": ", ".join(f"{s} {SIGNAL_BY_ID[s]['name']}" for s in fired if s != primary),
            "evidence": " | ".join(evidence),
        })
    return pd.DataFrame(rows, index=aw.index)


def alert_persistence(aw: pd.DataFrame, alert: pd.Series) -> pd.Series:
    """Point-in-time: number of monitoring weeks (up to and including this one) in which the
    account has been alerted. Uses only current and past runs."""
    tmp = pd.DataFrame({"account_id": aw["account_id"], "week": aw["week"], "a": alert.astype(int)})
    tmp = tmp.sort_values(["account_id", "week"])
    return tmp.groupby("account_id")["a"].cumsum().reindex(aw.index)


def build_alert_queue(aw: pd.DataFrame, sig: pd.DataFrame, contrib: pd.DataFrame, fam: pd.DataFrame,
                      score: pd.Series, alert_mask: pd.Series, cuts: dict[str, float],
                      evidence_count: pd.Series) -> pd.DataFrame:
    a = aw[alert_mask].copy()
    a["risk_score"] = score[alert_mask].round(3)
    a["risk_level"] = assign_tier(a["risk_score"], cuts)
    a["families_fired"] = fam[alert_mask].sum(axis=1)
    a["alert_weeks_to_date"] = evidence_count[alert_mask]
    a["evidence_strength"] = evidence_strength(a["alert_weeks_to_date"])
    a = a.join(explain(a, sig.loc[a.index], contrib.loc[a.index]))
    a["alert_id"] = [f"ALT-W{w:02d}-{acc:05d}" for w, acc in zip(a["week"], a["account_id"])]
    a["transaction_value"] = a["amt_in"] + a["amt_out"]
    a["transaction_count"] = a["n_tx"]
    a["status"] = "New"
    order = pd.Categorical(a["risk_level"], TIERS, ordered=True)
    a = a.assign(_o=order).sort_values(["week", "_o", "risk_score"], ascending=[True, True, False]).drop(columns="_o")
    cols = ["alert_id", "week", "run_step", "account_id", "risk_level", "risk_score", "evidence_strength",
            "families_fired", "alert_weeks_to_date", "risk_category", "primary_signal", "supporting_signals", "fired_signals", "evidence",
            "transaction_value", "transaction_count", "status"]
    return a[cols]
