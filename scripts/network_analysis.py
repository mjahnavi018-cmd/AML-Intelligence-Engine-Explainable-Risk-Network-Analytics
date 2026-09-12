"""Transaction-network analytics: nodes = accounts, edges = aggregated transfers.

Full-period graph metrics here are DESCRIPTIVE (they answer "what does the
network look like?"). Detection only ever uses the point-in-time 28-day
window features built in feature_engineering.py.
"""
from __future__ import annotations

from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd

from . import config
from .feature_engineering import temporal_cycles

MULE_SIGNALS = ["NET-01", "NET-02", "AML-01", "AML-03", "BEH-02"]


def build_graph(tx: pd.DataFrame, lo: int | None = None, hi: int | None = None) -> nx.DiGraph:
    t = tx[tx["is_self_transfer"] == 0]
    if lo is not None:
        t = t[t["step"] >= lo]
    if hi is not None:
        t = t[t["step"] <= hi]
    e = t.groupby(["src", "dst"]).agg(weight=("amount", "sum"), count=("amount", "size"),
                                      first_step=("step", "min"), last_step=("step", "max")).reset_index()
    G = nx.DiGraph()
    G.add_edges_from((r.src, r.dst, {"weight": r.weight, "count": r.count, "first_step": r.first_step,
                                     "last_step": r.last_step}) for r in e.itertuples(index=False))
    return G


def graph_summary(G: nx.DiGraph, labelled: set) -> pd.DataFrame:
    wcc = sorted((len(c) for c in nx.weakly_connected_components(G)), reverse=True)
    scc = sorted((len(c) for c in nx.strongly_connected_components(G)), reverse=True)
    indeg = np.array([d for _, d in G.in_degree()])
    outdeg = np.array([d for _, d in G.out_degree()])
    rows = [
        ("nodes (accounts with >=1 non-self transfer)", G.number_of_nodes()),
        ("edges (distinct directed account pairs)", G.number_of_edges()),
        ("density", nx.density(G)),
        ("weakly connected components", len(wcc)),
        ("largest weakly connected component (nodes)", wcc[0]),
        ("largest strongly connected component (nodes)", scc[0]),
        ("reciprocity (share of edges with reverse edge)", nx.reciprocity(G)),
        ("in-degree mean / max", f"{indeg.mean():.2f} / {indeg.max()}"),
        ("out-degree mean / max", f"{outdeg.mean():.2f} / {outdeg.max()}"),
        ("labelled accounts in graph", len(labelled & set(G.nodes))),
    ]
    sub = G.subgraph(labelled & set(G.nodes))
    lw = sorted((len(c) for c in nx.weakly_connected_components(sub)), reverse=True)
    rows += [("edges among labelled accounts only", sub.number_of_edges()),
             ("largest weakly connected component of labelled-only subgraph", lw[0] if lw else 0)]
    return pd.DataFrame(rows, columns=["metric", "value"])


def centrality_table(G: nx.DiGraph, k_betweenness: int = 300, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Degree, weighted degree, PageRank and sampled betweenness (k source nodes)."""
    c = pd.DataFrame({
        "in_degree": dict(G.in_degree()), "out_degree": dict(G.out_degree()),
        "in_value": dict(G.in_degree(weight="weight")), "out_value": dict(G.out_degree(weight="weight")),
        "pagerank": nx.pagerank(G, weight="weight"),
        "betweenness": nx.betweenness_centrality(G, k=min(k_betweenness, G.number_of_nodes()), seed=seed),
    })
    c.index.name = "account_id"
    return c.reset_index()


def centrality_lift(cent: pd.DataFrame, labels: pd.Series, top: float = 0.01) -> pd.DataFrame:
    """For each metric: labelled share among the top `top` fraction vs the base rate."""
    c = cent.merge(labels.rename("y"), left_on="account_id", right_index=True, how="left").fillna({"y": 0})
    base = c["y"].mean()
    rows = []
    for m in ["in_degree", "out_degree", "in_value", "out_value", "pagerank", "betweenness"]:
        k = max(1, int(len(c) * top))
        topk = c.nlargest(k, m)
        rows.append({"metric": m, "top_share": top, "accounts": k, "labelled_rate_top": topk["y"].mean(),
                     "base_rate": base, "lift": topk["y"].mean() / base})
    return pd.DataFrame(rows)


def full_period_cycles(tx: pd.DataFrame, window: int = config.NETWORK_WINDOW_DAYS) -> pd.DataFrame:
    """Time-respecting cycles found in consecutive non-overlapping 28-day windows."""
    t = tx[tx["is_self_transfer"] == 0]
    rows = []
    for lo in range(1, int(t["step"].max()) + 1, window):
        e = t[(t["step"] >= lo) & (t["step"] < lo + window)][["src", "dst", "step"]].drop_duplicates()
        for cyc in temporal_cycles(e.itertuples(index=False, name=None)):
            rows.append({"window_start": lo, "length": len(cyc), "accounts": "-".join(map(str, cyc))})
    return pd.DataFrame(rows)


def cycle_participation(cycles: pd.DataFrame) -> Counter:
    cnt: Counter = Counter()
    for acc in cycles["accounts"]:
        for a in acc.split("-"):
            cnt[int(a)] += 1
    return cnt


def mule_candidates(aw_period: pd.DataFrame, sig_period: pd.DataFrame, min_signals: int = 2) -> pd.DataFrame:
    """Potential mule-account candidates: >= `min_signals` DISTINCT mule-relevant signals
    (collector, distributor, pass-through, circular flow, new-counterparty burst) within the period.
    These are candidates for review, not confirmed mules."""
    s = sig_period[MULE_SIGNALS].copy()
    s["account_id"] = aw_period["account_id"].values
    per_acc = s.groupby("account_id")[MULE_SIGNALS].any()
    weeks = s.groupby("account_id")[MULE_SIGNALS].sum().sum(axis=1).rename("mule_signal_weeks")
    per_acc["distinct_mule_signals"] = per_acc[MULE_SIGNALS].sum(axis=1)
    per_acc = per_acc.join(weeks)
    per_acc["pattern"] = np.select(
        [per_acc["NET-01"] & (per_acc["AML-01"] | per_acc["NET-02"]),
         per_acc["NET-01"], per_acc["AML-03"], per_acc["AML-01"]],
        ["collector-forwarder", "collector", "circular-flow member", "pass-through"], "other combination")
    cand = per_acc[per_acc["distinct_mule_signals"] >= min_signals].reset_index()
    return cand


def ego_edges(tx: pd.DataFrame, account: int, lo: int, hi: int, radius: int = 2, max_edges: int = 120) -> pd.DataFrame:
    """Edges within `radius` hops of an account in [lo, hi], strongest first (for plots)."""
    t = tx[(tx["step"].between(lo, hi)) & (tx["is_self_transfer"] == 0)]
    frontier, seen = {account}, {account}
    keep = []
    for _ in range(radius):
        e = t[t["src"].isin(frontier) | t["dst"].isin(frontier)]
        keep.append(e)
        new = (set(e["src"]) | set(e["dst"])) - seen
        seen |= new
        frontier = new
        if len(seen) > max_edges:
            break
    if not keep:
        return pd.DataFrame(columns=["src", "dst", "amount", "count", "first_step", "last_step"])
    e = pd.concat(keep).drop_duplicates("tx_id")
    agg = e.groupby(["src", "dst"]).agg(amount=("amount", "sum"), count=("amount", "size"),
                                        first_step=("step", "min"), last_step=("step", "max")).reset_index()
    direct = agg["src"].eq(account) | agg["dst"].eq(account)
    agg = pd.concat([agg[direct], agg[~direct].nlargest(max(0, max_edges - int(direct.sum())), "amount")])
    return agg.reset_index(drop=True)
