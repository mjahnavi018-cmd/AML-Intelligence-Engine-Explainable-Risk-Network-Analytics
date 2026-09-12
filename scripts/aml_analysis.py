"""AML pattern analysis: rule catalogue, structuring-style analysis, simulator-artifact audit.

All AML signals are analytical risk indicators. They are not evidence of money
laundering and carry no legal meaning.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .risk_scoring import SIGNALS

FEATURE_SOURCES = {
    "max_amt": "amount", "n_tx": "sender, receiver, step", "velocity_ratio": "sender, receiver, step (history)",
    "new_cp": "sender, receiver, step (history)", "amount_log_ratio": "amount, step (history)",
    "is_reactivated": "step (history)", "rapid_movement": "sender, receiver, amount, step",
    "max_repeat_same_cp": "sender, receiver, step", "cycle_count": "sender, receiver, step",
    "fan_in_28": "sender, receiver, step", "fan_out_28": "sender, receiver, step",
    "pagerank_28": "sender, receiver, amount, step",
}

LIMITATIONS = {
    "TXN-01": "AMLSim caps amounts near 600, so 'large' has a narrow meaning; typology transfers are not larger.",
    "VEL-01": "Absolute counts penalise legitimately busy accounts; simulation volume varies by week.",
    "BEH-01": "Needs history; unstable for accounts with very few past transactions.",
    "BEH-02": "Almost every AMLSim background transfer goes to a new counterparty, so novelty is common.",
    "BEH-03": "Partly driven by the sub-100 typology amounts (artifact A1) inflating/deflating baselines.",
    "BEH-04": "Dormancy is mostly a simulator start/end effect here.",
    "AML-01": "Step granularity is one day; intra-day ordering of in/out flows is unknown.",
    "AML-02": "Simulator artifact (A2/A3): background accounts never repeat a receiver within a week.",
    "AML-03": "Random background activity also forms time-respecting cycles; cycles are rare overall.",
    "NET-01": "Legitimate collection accounts (e.g. merchants) would also trigger; no account-type field exists.",
    "NET-02": "Legitimate payout accounts (e.g. payroll) would also trigger; no account-type field exists.",
    "NET-03": "Highly correlated with fan-in and inflow value; adds little independent evidence.",
}


def artifact_audit(accounts: pd.DataFrame, tx: pd.DataFrame, aw: pd.DataFrame) -> pd.DataFrame:
    lab = set(accounts.loc[accounts["is_suspicious"] == 1, "account_id"])
    both = tx["src"].isin(lab) & tx["dst"].isin(lab)
    rows = []
    lt = tx["amount"] < 100
    rows.append({"artifact": "A1 sub-100 amounts", "observation": f"{int(lt.sum())} transfers below 100",
                 "share_between_labelled_accounts": both[lt].mean(),
                 "base_share": both.mean(),
                 "consequence": "Any amount rule tuned to small values would reproduce the label; not used."})
    dup = tx["is_repeat_copy"] == 1
    rows.append({"artifact": "A2 exact duplicate transfers", "observation": f"{int(dup.sum())} extra identical copies",
                 "share_between_labelled_accounts": both[dup].mean(), "base_share": both.mean(),
                 "consequence": "Repeated-transfer rule AML-02 excluded from the framework."})
    pair_n = tx[tx["is_self_transfer"] == 0].groupby(["src", "dst"]).size()
    rep_pairs = pair_n[pair_n > 1].reset_index()
    rp_both = rep_pairs["src"].isin(lab) & rep_pairs["dst"].isin(lab)
    rows.append({"artifact": "A3 counterparty recurrence",
                 "observation": f"{len(rep_pairs)} of {len(pair_n)} directed pairs transact more than once",
                 "share_between_labelled_accounts": rp_both.mean(), "base_share": both.mean(),
                 "consequence": "Background relationships are one-off; models rewarding 'few new counterparties' "
                                "exploit the simulator and would not transfer to real banking."})
    d = aw[aw["period"] == "dev"].copy()
    d["recurring"] = (d["n_cp"] - d["new_cp"]) > 0
    lab_w = d["account_id"].isin(lab)
    rows.append({"artifact": "A3b recurring counterparty in an account-week (dev)",
                 "observation": f"labelled {d.loc[lab_w, 'recurring'].mean():.4f} vs unlabelled "
                                f"{d.loc[~lab_w, 'recurring'].mean():.4f}",
                 "share_between_labelled_accounts": np.nan, "base_share": np.nan,
                 "consequence": "Same artifact measured on monitoring features."})
    return pd.DataFrame(rows)


def structuring_analysis(accounts: pd.DataFrame, tx: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Data-driven look for split/repeated transfers. No legal reporting threshold is imported:
    AMLSim amounts are unit-less simulation values, so an INR/USD threshold would be meaningless."""
    lab = set(accounts.loc[accounts["is_suspicious"] == 1, "account_id"])
    t = tx[tx["is_self_transfer"] == 0]
    g = t.groupby(["src", "dst", "week"])["amount"].agg(["size", "mean", "std", "sum"]).reset_index()
    g = g[g["size"] >= 2].copy()
    g["cv"] = (g["std"] / g["mean"]).fillna(0)
    g["both_labelled"] = g["src"].isin(lab) & g["dst"].isin(lab)
    summary = pd.DataFrame([{
        "sender-receiver-week groups with >=2 transfers": len(g),
        "share with near-identical amounts (CV < 5%)": (g["cv"] < 0.05).mean() if len(g) else np.nan,
        "share between two labelled accounts": g["both_labelled"].mean() if len(g) else np.nan,
        "median transfers per group": g["size"].median() if len(g) else np.nan,
        "median amount per transfer": g["mean"].median() if len(g) else np.nan,
    }]).T.reset_index()
    summary.columns = ["metric", "value"]
    # amount heaping just below round hundreds (would indicate threshold avoidance)
    rows = []
    for x in range(200, 601, 100):
        below = t["amount"].between(x * 0.95, x - 0.01).sum()
        above = t["amount"].between(x, x * 1.05 - 0.01).sum()
        rows.append({"round_value": x, "tx_in_5pct_below": int(below), "tx_in_5pct_above": int(above),
                     "below_above_ratio": below / above if above else np.nan})
    return {"summary": summary, "heaping": pd.DataFrame(rows), "groups": g}


def rule_catalogue(th: dict, sig: pd.DataFrame, frame: pd.DataFrame, dev_mask: pd.Series,
                   test_mask: pd.Series) -> pd.DataFrame:
    rows = []
    base_dev = frame.loc[dev_mask, "is_suspicious"].mean()
    base_test = frame.loc[test_mask, "is_suspicious"].mean()
    for s in SIGNALS:
        hit = sig[s["id"]]
        dv, tv = hit & dev_mask, hit & test_mask
        if s["rule"] == "quantile_tx":
            just = f"{s['q']:.0%} quantile of development-period transfer amounts (label-free)"
        elif s["rule"] == "quantile":
            just = f"{s['q']:.0%} quantile of development account-weeks (label-free)"
        elif s["rule"] == "fixed":
            just = "structural definition (presence of the pattern)"
        else:
            just = "binary flag"
        prec_dev = frame.loc[dv, "is_suspicious"].mean() if dv.any() else np.nan
        prec_test = frame.loc[tv, "is_suspicious"].mean() if tv.any() else np.nan
        rows.append({
            "rule_id": s["id"], "rule_name": s["name"], "family": s["family"],
            "business_rationale": s["rationale"], "data_required": FEATURE_SOURCES[s["feature"]],
            "logic": f"{s['feature']} >= {th[s['id']]:.4g}" + (" and history exists" if s.get("requires_history") else ""),
            "threshold": th[s["id"]], "threshold_justification": just,
            "alerts_dev": int(dv.sum()), "pct_dev_account_weeks": dv.sum() / dev_mask.sum(),
            "alerts_test": int(tv.sum()), "pct_test_account_weeks": tv.sum() / test_mask.sum(),
            "label_precision_dev": prec_dev, "label_precision_test": prec_test,
            "lift_test": prec_test / base_test if base_test else np.nan,
            "in_framework": s["in_framework"], "limitations": LIMITATIONS[s["id"]],
        })
    out = pd.DataFrame(rows)
    out.attrs["base_dev"], out.attrs["base_test"] = base_dev, base_test
    return out
