"""Account behavioural profiling, activity segmentation and concentration.

Two different views are kept deliberately separate:
* `account_profiles` - FULL-PERIOD descriptive statistics for EDA and for the
  investigation page. Never used as detection features (would be look-ahead).
* `activity_segment` - point-in-time segment of an account at a monitoring run,
  based only on its history before that run (used for segment thresholds).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def account_profiles(accounts: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    t = tx[tx["is_self_transfer"] == 0]
    out_g = t.groupby("src")
    in_g = t.groupby("dst")
    p = pd.DataFrame({
        "out_count": out_g.size(), "out_amount": out_g["amount"].sum(),
        "out_median_amount": out_g["amount"].median(), "out_counterparties": out_g["dst"].nunique(),
        "in_count": in_g.size(), "in_amount": in_g["amount"].sum(),
        "in_median_amount": in_g["amount"].median(), "in_counterparties": in_g["src"].nunique(),
    })
    p.index.name = "account_id"
    p = accounts.set_index("account_id").join(p, how="left")
    cnt_cols = ["out_count", "in_count", "out_amount", "in_amount", "out_counterparties", "in_counterparties"]
    p[cnt_cols] = p[cnt_cols].fillna(0)
    p["tx_count"] = p["out_count"] + p["in_count"]
    p["total_value"] = p["out_amount"] + p["in_amount"]
    legs = pd.concat([t[["src", "amount", "step"]].rename(columns={"src": "a"}),
                      t[["dst", "amount", "step"]].rename(columns={"dst": "a"})])
    lg = legs.groupby("a")
    p["mean_amount"] = lg["amount"].mean()
    p["median_amount"] = lg["amount"].median()
    p["amount_std"] = lg["amount"].std()
    p["max_amount"] = lg["amount"].max()
    p["active_weeks"] = legs.assign(w=(legs["step"] - 1) // config.WEEK_LEN + 1).groupby("a")["w"].nunique()
    p["inflow_outflow_ratio"] = np.where(p["out_amount"] > 0, p["in_amount"] / p["out_amount"], np.nan)
    p["net_flow"] = p["in_amount"] - p["out_amount"]
    cps = pd.concat([t[["src", "dst"]].rename(columns={"src": "a", "dst": "c"}),
                     t[["dst", "src"]].rename(columns={"dst": "a", "src": "c"})])
    p["unique_counterparties"] = cps.groupby("a")["c"].nunique()
    p["tx_per_counterparty"] = p["tx_count"] / p["unique_counterparties"]
    return p.reset_index()


def add_full_period_segment(profiles: pd.DataFrame) -> pd.DataFrame:
    """Descriptive activity tiers by full-period transaction count (EDA only)."""
    p = profiles.copy()
    q = p.loc[p["tx_count"] > 0, "tx_count"].quantile([0.5, 0.9, 0.99]).to_list()
    bins = [-1, 0, q[0], q[1], q[2], np.inf]
    labels = ["inactive", "low (<= median)", "medium (median-p90)", "high (p90-p99)", "very high (> p99)"]
    p["activity_tier"] = pd.cut(p["tx_count"], bins=bins, labels=labels).astype(str)
    return p


def activity_segment(aw: pd.DataFrame, cuts: tuple[float, float] | None = None) -> tuple[pd.Series, tuple]:
    """Point-in-time segment from historical weekly mean (history only).

    Cut-points are terciles of `hist_weekly_mean` in the development period.
    """
    if cuts is None:
        dev = aw.loc[aw["period"] == "dev", "hist_weekly_mean"].dropna()
        cuts = tuple(dev.quantile([1 / 3, 2 / 3]).to_list())
    hm = aw["hist_weekly_mean"]
    seg = np.select([hm.isna(), hm <= cuts[0], hm <= cuts[1]], ["no history", "low", "medium"], "high")
    return pd.Series(seg, index=aw.index, name="segment"), cuts


def lorenz(values: pd.Series) -> pd.DataFrame:
    """Concentration curve: cumulative share of value vs cumulative share of accounts (desc)."""
    v = np.sort(values.fillna(0).to_numpy())[::-1]
    cum = np.cumsum(v) / v.sum()
    share = np.arange(1, len(v) + 1) / len(v)
    return pd.DataFrame({"account_share": share, "value_share": cum})


def gini(values: pd.Series) -> float:
    v = np.sort(values.fillna(0).to_numpy())
    n = len(v)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * v.sum()))
