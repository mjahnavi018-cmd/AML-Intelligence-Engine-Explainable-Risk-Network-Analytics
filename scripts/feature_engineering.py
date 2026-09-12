"""Point-in-time account-week features (NO LOOK-AHEAD).

For a monitoring run at the end of week w (end step E = 7w):

    current window  C = steps [E-6, E]         (this week's behaviour)
    network window  N = steps [E-27, E]        (28-day flow / network context)
    history         H = steps <  E-6           (behavioural baseline)

Every feature for run w is computed only from transactions with step <= E.
Nothing uses labels. Self-transfers are excluded (they have no counterparty).
Within one step the ordering of transactions is unknown, so an inflow and an
outflow on the same step are treated as "same day" (hold time 0).
"""
from __future__ import annotations

import bisect
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd

from . import config


def make_legs(tx: pd.DataFrame) -> pd.DataFrame:
    """One row per (account, counterparty, direction) leg of each transfer."""
    t = tx[tx["is_self_transfer"] == 0]
    out_legs = pd.DataFrame({"account_id": t["src"].values, "cp": t["dst"].values, "direction": "out",
                             "amount": t["amount"].values, "step": t["step"].values, "tx_id": t["tx_id"].values})
    in_legs = pd.DataFrame({"account_id": t["dst"].values, "cp": t["src"].values, "direction": "in",
                            "amount": t["amount"].values, "step": t["step"].values, "tx_id": t["tx_id"].values})
    legs = pd.concat([out_legs, in_legs], ignore_index=True)
    legs["week"] = (legs["step"] - 1) // config.WEEK_LEN + 1
    return legs


def _hold_times(legs_upto: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    """For each outflow in [lo, hi], days since the most recent prior-or-same-step inflow."""
    outs = legs_upto[(legs_upto["direction"] == "out") & legs_upto["step"].between(lo, hi)][["account_id", "step", "amount"]]
    ins = legs_upto[legs_upto["direction"] == "in"][["account_id", "step"]].rename(columns={"step": "in_step"})
    if outs.empty:
        return pd.DataFrame(columns=["account_id", "fwd_share", "median_hold_days"])
    outs = outs.sort_values("step")
    ins = ins.sort_values("in_step").drop_duplicates()
    m = pd.merge_asof(outs, ins, left_on="step", right_on="in_step", by="account_id", direction="backward")
    m["hold"] = m["step"] - m["in_step"]
    m["fwd_amt"] = np.where(m["hold"].le(config.FORWARD_DAYS), m["amount"], 0.0)
    g = m.groupby("account_id")
    return pd.DataFrame({"fwd_share": g["fwd_amt"].sum() / g["amount"].sum(),
                         "median_hold_days": g["hold"].median()}).reset_index()


def temporal_cycles(edges, max_len: int = config.CYCLE_MAX_LEN) -> set[tuple]:
    """Time-respecting directed cycles (length 3..max_len).

    A cycle A->B->...->A counts only if each hop happens on the same step as, or
    after, the previous hop, i.e. funds could plausibly have travelled round the
    loop and returned to the originator. `edges` is an iterable of (src, dst, step).
    Cycles are de-duplicated by rotating to the smallest account id.
    """
    adj: dict = defaultdict(list)
    for u, v, s in edges:
        adj[u].append((s, v))
    for u in adj:
        adj[u].sort()
    found: set[tuple] = set()
    for s0 in list(adj):
        stack = [(s0, -1, (s0,))]
        while stack:
            node, tmin, path = stack.pop()
            lst = adj.get(node, [])
            for st, v in lst[bisect.bisect_left(lst, (tmin, -1)):]:
                if v == s0 and len(path) >= 3:
                    k = path.index(min(path))
                    found.add(path[k:] + path[:k])
                elif v not in path and len(path) < max_len:
                    stack.append((v, st, path + (v,)))
    return found


def _cycle_counts(win: pd.DataFrame) -> pd.Series:
    """Number of distinct time-respecting cycles through each account in the window."""
    e = win.loc[win["direction"] == "out", ["account_id", "cp", "step"]].drop_duplicates()
    counts: dict[int, int] = defaultdict(int)
    for cyc in temporal_cycles(e.itertuples(index=False, name=None)):
        for n in cyc:
            counts[n] += 1
    return pd.Series(dict(counts), name="cycle_count", dtype=float)


def _pagerank(win: pd.DataFrame) -> pd.Series:
    e = win.loc[win["direction"] == "out"].groupby(["account_id", "cp"])["amount"].sum()
    G = nx.DiGraph()
    G.add_weighted_edges_from(((a, b, w) for (a, b), w in e.items()))
    pr = nx.pagerank(G, weight="weight")
    return pd.Series(pr, name="pagerank_28")


def features_for_week(legs: pd.DataFrame, w: int) -> pd.DataFrame:
    E = w * config.WEEK_LEN
    c_lo, n_lo = E - config.WEEK_LEN + 1, E - config.NETWORK_WINDOW_DAYS + 1
    upto = legs[legs["step"] <= E]
    cur = upto[upto["step"] >= c_lo]
    hist = upto[upto["step"] < c_lo]
    net = upto[upto["step"] >= n_lo]

    g = cur.groupby("account_id")
    f = pd.DataFrame({
        "n_in": cur[cur["direction"] == "in"].groupby("account_id").size(),
        "n_out": cur[cur["direction"] == "out"].groupby("account_id").size(),
        "amt_in": cur[cur["direction"] == "in"].groupby("account_id")["amount"].sum(),
        "amt_out": cur[cur["direction"] == "out"].groupby("account_id")["amount"].sum(),
        "max_amt": g["amount"].max(),
        "mean_amt": g["amount"].mean(),
        "n_cp": g["cp"].nunique(),
        "n_cp_in": cur[cur["direction"] == "in"].groupby("account_id")["cp"].nunique(),
        "n_cp_out": cur[cur["direction"] == "out"].groupby("account_id")["cp"].nunique(),
    })
    cols0 = ["n_in", "n_out", "amt_in", "amt_out", "n_cp_in", "n_cp_out"]
    f[cols0] = f[cols0].fillna(0)
    f["n_tx"] = f["n_in"] + f["n_out"]

    # repeated transfers to the same receiver inside the week
    rep = cur[cur["direction"] == "out"].groupby(["account_id", "cp"]).size()
    f["max_repeat_same_cp"] = rep.groupby(level=0).max()
    f["max_repeat_same_cp"] = f["max_repeat_same_cp"].fillna(0)

    # new counterparties: counterparties this week never seen before this week
    seen = hist[["account_id", "cp"]].drop_duplicates()
    cur_pairs = cur[["account_id", "cp"]].drop_duplicates().merge(seen, how="left", indicator=True)
    f["new_cp"] = cur_pairs[cur_pairs["_merge"] == "left_only"].groupby("account_id").size()
    f["new_cp"] = f["new_cp"].fillna(0)
    f["new_cp_share"] = f["new_cp"] / f["n_cp"]

    # historical baseline (strictly before this week)
    hg = hist.groupby("account_id")
    first_week = hg["week"].min()
    last_week = hg["week"].max()
    hb = pd.DataFrame({
        "hist_n_tx": hg.size(),
        "hist_amt_mean": hg["amount"].mean(),
        "hist_amt_median": hg["amount"].median(),
        "hist_max_amt": hg["amount"].max(),
        "hist_n_cp": hg["cp"].nunique(),
        "hist_weeks": (w - 1) - first_week + 1,
        "weeks_since_last": w - last_week,
    })
    f = f.join(hb, how="left")
    f["hist_n_tx"] = f["hist_n_tx"].fillna(0)
    f["hist_weekly_mean"] = f["hist_n_tx"] / f["hist_weeks"]
    f["velocity_ratio"] = (f["n_tx"] + 1) / (f["hist_weekly_mean"].fillna(0) + 1)
    # amount deviation from the account's own historical median (log-ratio: robust to the
    # near-zero standard deviations that make z-scores explode for repetitive accounts)
    enough = f["hist_n_tx"] >= config.MIN_HISTORY_TX
    # signed: >0 means larger-than-usual amounts this week, <0 smaller-than-usual
    f["amount_log_ratio"] = np.where(enough, np.log(f["mean_amt"] / f["hist_amt_median"]), np.nan)
    f["is_new_account"] = f["hist_n_tx"].eq(0).astype(int)
    f["is_reactivated"] = (f["weeks_since_last"] > config.DORMANCY_WEEKS).astype(int)

    # 28-day network / flow window
    ng = net.groupby("account_id")
    nf = pd.DataFrame({
        "fan_in_28": net[net["direction"] == "in"].groupby("account_id")["cp"].nunique(),
        "fan_out_28": net[net["direction"] == "out"].groupby("account_id")["cp"].nunique(),
        "in_amt_28": net[net["direction"] == "in"].groupby("account_id")["amount"].sum(),
        "out_amt_28": net[net["direction"] == "out"].groupby("account_id")["amount"].sum(),
        "n_tx_28": ng.size(),
    }).fillna(0)
    f = f.join(nf, how="left")
    both = (f["in_amt_28"] > 0) & (f["out_amt_28"] > 0)
    f["passthrough_ratio"] = np.where(both, np.minimum(f["in_amt_28"], f["out_amt_28"]) /
                                      np.maximum(f["in_amt_28"], f["out_amt_28"]), 0.0)
    ht = _hold_times(upto, n_lo, E).set_index("account_id")
    f = f.join(ht, how="left")
    f["fwd_share"] = f["fwd_share"].fillna(0)
    f["rapid_movement"] = np.minimum(f["passthrough_ratio"], f["fwd_share"])
    f = f.join(_cycle_counts(net), how="left")
    f["cycle_count"] = f["cycle_count"].fillna(0)
    f = f.join(_pagerank(net), how="left")

    f.index.name = "account_id"
    f = f.reset_index()
    f.insert(1, "week", w)
    f.insert(2, "run_step", E)
    return f


def build_account_week_features(tx: pd.DataFrame, weeks: list[int] | None = None) -> pd.DataFrame:
    weeks = weeks or (config.DEV_WEEKS + config.TEST_WEEKS)
    legs = make_legs(tx)
    frames = [features_for_week(legs, w) for w in weeks]
    out = pd.concat(frames, ignore_index=True)
    out["period"] = np.where(out["week"].isin(config.DEV_WEEKS), "dev", "test")
    return out
