"""Figures for reports, README and notebooks. Every figure answers one question
(stated in its title) and carries a data note. Palette: validated categorical
slots (blue = unlabelled/baseline, orange = labelled/integrated), status colours
only for priority tiers, always paired with text labels.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from . import config
from .network_analysis import ego_edges

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = C
STATUS = {"CRITICAL": "#d03b3b", "HIGH": "#ec835a", "MEDIUM": "#fab219", "LOW": "#898781"}
NOTE = "Data: IBM AMLSim public sample (synthetic, 20K accounts). Not real bank data."


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 10,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "font.family": "DejaVu Sans", "legend.frameon": False,
        "legend.fontsize": 9, "lines.linewidth": 2, "figure.dpi": 110,
    })


def _finish(fig, name: str, note: str = NOTE) -> str:
    fig.text(0.01, 0.005, note, fontsize=7.5, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = config.FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def _shade_periods(ax):
    ax.axvspan(config.DEV_WEEKS[0] - 0.5, config.DEV_WEEKS[-1] + 0.5, color=BLUE, alpha=0.06, lw=0)
    ax.axvspan(config.TEST_WEEKS[0] - 0.5, config.TEST_WEEKS[-1] + 0.5, color=ORANGE, alpha=0.06, lw=0)


def fig_weekly_activity(weekly: pd.DataFrame) -> str:
    w = weekly[weekly["complete_week"]]
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True)
    for ax in axes:
        _shade_periods(ax)
    axes[0].bar(w["week"], w["transactions"], color=BLUE, width=0.8)
    axes[0].set_ylabel("transfers per week")
    axes[0].set_title("Volume ramps up and down: absolute velocity thresholds are unstable over time")
    axes[1].plot(w["week"], 100 * w["proxy_typology_share"], color=ORANGE, marker="o", ms=4)
    axes[1].set_ylabel("% transfers between\ntwo labelled accounts")
    axes[1].set_xlabel("simulation week (7 steps)")
    axes[1].set_title("Typology share rises as background volume falls", fontsize=11)
    axes[0].text(config.DEV_WEEKS[0], axes[0].get_ylim()[1] * 0.92, "development (calibration)", color=INK2, fontsize=8)
    axes[0].text(config.TEST_WEEKS[0], axes[0].get_ylim()[1] * 0.92, "held-out test", color=INK2, fontsize=8)
    return _finish(fig, "fig01_weekly_activity")


def fig_amounts(tx: pd.DataFrame, proxy: pd.Series) -> str:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bins = np.arange(0, 610, 10)
    ax.hist(tx.loc[proxy == 0, "amount"], bins=bins, color=BLUE, alpha=0.9, label="other transfers",
            density=True, histtype="stepfilled")
    ax.hist(tx.loc[proxy == 1, "amount"], bins=bins, color=ORANGE, label="between two labelled accounts",
            density=True, histtype="step", lw=2)
    ax.axvline(100, color=INK2, lw=1, ls="--")
    ax.annotate("every transfer below 100 is between two labelled\naccounts: a generator artifact, not a usable rule",
                xy=(100, ax.get_ylim()[1] * 0.8), xytext=(160, ax.get_ylim()[1] * 0.85), fontsize=8.5, color=INK2,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.set_xlabel("transfer amount (simulation currency units)")
    ax.set_ylabel("density")
    ax.set_title("Typology transfers are not larger - a 'large amount' rule has little to find")
    ax.legend(loc="upper right")
    return _finish(fig, "fig02_amount_distribution")


def fig_lorenz(lor: pd.DataFrame, gini: float) -> str:
    fig, ax = plt.subplots(figsize=(5.4, 5))
    ax.plot(100 * lor["account_share"], 100 * lor["value_share"], color=BLUE)
    ax.plot([0, 100], [0, 100], color=MUTED, lw=1, ls="--")
    top10 = lor.loc[lor["account_share"] <= 0.10, "value_share"].max()
    ax.scatter([10], [100 * top10], color=BLUE, s=40, zorder=3)
    ax.annotate(f"top 10% of accounts carry {100 * top10:.0f}% of value", (10, 100 * top10), (20, 100 * top10 - 12),
                fontsize=9, color=INK2)
    ax.set_xlabel("% of active accounts (most active first)")
    ax.set_ylabel("cumulative % of transfer value")
    ax.set_title(f"How concentrated is value? Gini = {gini:.2f}")
    return _finish(fig, "fig03_value_concentration")


def fig_fingerprint(gc: pd.DataFrame) -> str:
    g = gc.copy()
    g["ratio"] = g["labelled_mean"] / g["unlabelled_mean"]
    g = g.sort_values("ratio")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.hlines(range(len(g)), 1, g["ratio"], color=GRID, lw=2)
    ax.scatter(g["ratio"], range(len(g)), color=np.where(g["ratio"] >= 1, ORANGE, BLUE), s=60, zorder=3)
    ax.axvline(1, color=INK2, lw=1)
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels(g["metric"].str.replace("_", " "))
    ax.set_xscale("log")
    for i, r in enumerate(g["ratio"]):
        ax.text(r * (1.06 if r >= 1 else 0.94), i, f"{r:.2f}x", va="center", ha="left" if r >= 1 else "right",
                fontsize=8.5, color=INK2)
    ax.set_xlabel("labelled mean / unlabelled mean (log scale; full period, descriptive)")
    ax.set_title("Behavioural fingerprint: labelled accounts are busier, not bigger")
    return _finish(fig, "fig04_behavioural_fingerprint")


def fig_signal_quality(rc: pd.DataFrame, base: float) -> str:
    r = rc.sort_values("label_precision_test")
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [MUTED if not f else BLUE for f in r["in_framework"]]
    ax.barh(r["rule_id"] + " " + r["rule_name"], r["label_precision_test"], color=colors, height=0.65)
    ax.axvline(base, color=ORANGE, lw=2)
    ax.text(base + 0.01, -0.9, f"base rate {base:.1%}", color=INK2, fontsize=8.5)
    for i, (p, n) in enumerate(zip(r["label_precision_test"], r["alerts_test"])):
        ax.text((p if pd.notna(p) else 0) + 0.01, i, f"{p:.0%}  ({n} alerts)" if pd.notna(p) else "no alerts",
                va="center", fontsize=8, color=INK2)
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("share of alerts on labelled accounts (held-out accounts, test weeks)")
    ax.set_title("Which single signals carry information? (grey = excluded / quarantined)")
    return _finish(fig, "fig05_signal_precision")


def fig_signal_overlap(sig: pd.DataFrame, ids: list[str]) -> str:
    s = sig[ids].astype(bool)
    n = len(ids)
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            a, b = s.iloc[:, i], s.iloc[:, j]
            u = (a | b).sum()
            J[i, j] = (a & b).sum() / u if u else 0
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(J, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(ids, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(ids)
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            if i != j and J[i, j] >= 0.05:
                ax.text(j, i, f"{J[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if J[i, j] > 0.5 else INK)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Jaccard overlap of alerts")
    ax.set_title("Do signals repeat each other? (all monitoring weeks)")
    return _finish(fig, "fig06_signal_overlap")


def fig_degree(G: nx.DiGraph, labelled: set) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, kind in zip(axes, ("in", "out")):
        deg = dict(G.in_degree() if kind == "in" else G.out_degree())
        for lab, col, name in ((0, BLUE, "unlabelled"), (1, ORANGE, "labelled")):
            v = np.sort([d for n, d in deg.items() if (n in labelled) == bool(lab)])
            ccdf = 1 - np.arange(len(v)) / len(v)
            ax.loglog(v, ccdf, color=col, label=name, lw=2)
        ax.set_xlabel(f"{kind}-degree (distinct counterparties, full period)")
        ax.set_ylabel("share of accounts with degree >= x")
        ax.set_title(f"{kind.capitalize()}-degree tail")
        ax.legend()
    fig.suptitle("Labelled accounts sit in the heavy tail - but so do many unlabelled ones",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    return _finish(fig, "fig07_degree_distribution")


def fig_centrality_lift(lift: pd.DataFrame) -> str:
    l = lift.sort_values("lift")
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.barh(l["metric"].str.replace("_", " "), l["lift"], color=BLUE, height=0.6)
    for i, (v, r) in enumerate(zip(l["lift"], l["labelled_rate_top"])):
        ax.text(v + 0.08, i, f"{v:.1f}x  ({r:.0%} labelled)", va="center", fontsize=8.5, color=INK2)
    ax.axvline(1, color=INK2, lw=1)
    ax.set_xlim(0, l["lift"].max() * 1.35)
    ax.set_xlabel("lift over base rate among top 1% of accounts (full-period graph, descriptive)")
    ax.set_title("Which network position concentrates labelled accounts?")
    return _finish(fig, "fig08_centrality_lift")


def fig_tradeoff(bc: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for i, r in bc.reset_index(drop=True).iterrows():
        ml = r["approach"].startswith("ML")
        col = MUTED if ml else (ORANGE if r["approach"].startswith("B5") else BLUE)
        ax.scatter(r["alerts_per_week"], r["alert_precision"], s=60 + 900 * r["account_recall"], color=col,
                   alpha=0.85, edgecolor=SURFACE, lw=2, zorder=3, marker="s" if ml else "o")
        off = (8, -22) if r["approach"].startswith("ML3") else (8, -4)
        ax.annotate(f"{r['approach']}\nrecall {r['account_recall']:.0%}", (r["alerts_per_week"], r["alert_precision"]),
                    xytext=off, textcoords="offset points", fontsize=7.5, color=INK2)
    ax.set_xlabel("alerts per week (held-out half of accounts)")
    ax.set_ylabel("alert precision (share on labelled accounts)")
    ax.set_ylim(0, 1.08)
    ax.set_title("Detection vs workload: bubble size = account recall (grey squares = ML, see caveats)")
    return _finish(fig, "fig09_detection_workload_tradeoff")


def fig_topk(curves: pd.DataFrame) -> str:
    keep = {"B1 Amount threshold": MUTED, "B2 Amount + velocity": BLUE,
            "B5b Integrated risk score (top 1%)": ORANGE, "ML1 Logistic regression (unconstrained)": VIOLET}
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for name, col in keep.items():
        c = curves[curves["approach"] == name]
        ax.plot(100 * c["share_reviewed"], 100 * c["recall"], color=col, label=name,
                ls="--" if name.startswith("ML") else "-")
    ax.plot([0, 20], [0, 20], color=GRID, lw=1)
    ax.set_xlabel("% of active accounts an analyst team can review (ranked by worst week)")
    ax.set_ylabel("% of labelled accounts captured")
    ax.set_title("If analysts review only the top k%, how much is captured?")
    ax.legend(loc="lower right")
    return _finish(fig, "fig10_topk_capture")


def fig_sensitivity(sens: pd.DataFrame) -> str:
    s = sens[sens["parameter"] == "rule quantile"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    for k, (ax, metric, lab) in enumerate(zip(axes, ("alerts", "alert_precision", "account_recall"),
                                              ("alerts (test weeks)", "alert precision", "account recall"))):
        for j, (name, g) in enumerate(s.groupby("approach")):
            ax.plot(100 * g["value"], g[metric], marker="o", ms=4, color=C[j], label=name)
        ax.set_xlabel("rule threshold quantile (%)")
        ax.set_title(lab, fontsize=11)
    axes[0].set_yscale("log")
    axes[2].legend(fontsize=7.5, loc="upper right")
    fig.suptitle("Threshold sensitivity: stricter thresholds mean fewer alerts, higher precision, lower recall",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    return _finish(fig, "fig11_threshold_sensitivity")


def fig_prevalence(prev: pd.DataFrame) -> str:
    keep = {"B2 Amount + velocity": BLUE, "B5a Integrated rules (>=2 families)": AQUA,
            "B5b Integrated risk score (top 1%)": ORANGE}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, col in keep.items():
        g = prev[prev["approach"] == name]
        ax.plot(100 * g["prevalence"], 100 * g["projected_precision"], marker="o", color=col, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("assumed prevalence of typology accounts (%) - log scale")
    ax.set_ylabel("projected alert precision (%)")
    ax.set_title("At realistic prevalence, most alerts would be false positives")
    ax.legend()
    return _finish(fig, "fig12_prevalence_projection",
                   NOTE + " Projection assumes recall and false-positive rate transfer unchanged.")


def fig_stability(stab: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, lab in ((axes[0], "alerts", "alerts per week"), (axes[1], "precision", "alert precision")):
        _shade_periods(ax)
        for name, col in (("B2 Amount + velocity", BLUE), ("B5b Integrated risk score (top 1%)", ORANGE)):
            g = stab[stab["approach"] == name]
            ax.plot(g["week"], g[metric], marker="o", ms=4, color=col, label=name)
        ax.set_xlabel("week")
        ax.set_title(lab, fontsize=11)
    axes[1].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8)
    fig.suptitle("Is performance stable week to week? (held-out accounts)", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    return _finish(fig, "fig13_weekly_stability")


def fig_tiers(tv: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].bar(tv["risk_level"], tv["alerts_per_week"], color=[STATUS[t] for t in tv["risk_level"]])
    axes[0].set_title("alerts per week by tier", fontsize=11)
    axes[1].bar(tv["risk_level"], tv["precision"], color=[STATUS[t] for t in tv["risk_level"]])
    for i, (p, n) in enumerate(zip(tv["precision"], tv["alerts"])):
        axes[1].text(i, p + 0.02, f"{p:.0%}\n(n={n})", ha="center", fontsize=8.5, color=INK2)
    axes[1].set_ylim(0, 1.2)
    axes[1].set_title("precision by tier", fontsize=11)
    fig.suptitle("Do priority tiers order the queue? (held-out accounts, test weeks)", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    return _finish(fig, "fig14_priority_tiers")


def fig_deciles(dec: pd.DataFrame, base: float) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(dec["decile"].astype(int), 100 * dec["labelled_rate"], color=BLUE, width=0.7)
    ax.axhline(100 * base, color=ORANGE, lw=2)
    ax.text(10.4, 100 * base, f"base {base:.1%}", color=INK2, fontsize=8.5, va="bottom", ha="right")
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("risk-score decile (1 = highest)")
    ax.set_ylabel("% labelled accounts")
    ax.set_title(f"Top decile holds {dec.loc[0, 'share_of_all_labelled']:.0%} of labelled accounts")
    return _finish(fig, "fig15_risk_deciles")


def fig_coefficients(coefs: pd.DataFrame) -> str:
    c = coefs.set_index("feature")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    axes[0].barh(c.index, c["integrated_nonneg_weight"], color=ORANGE)
    axes[0].set_title("integrated model: weights >= 0", fontsize=11)
    cols = [BLUE if v >= 0 else RED for v in c["unconstrained_lr_coef"]]
    axes[1].barh(c.index, c["unconstrained_lr_coef"], color=cols)
    axes[1].axvline(0, color=INK2, lw=1)
    axes[1].set_title("unconstrained logistic regression", fontsize=11)
    i = list(c.index).index("new_cp")
    axes[1].annotate("large negative weight on new counterparties\n= exploits artifact A3 (recurrence)",
                     (c.loc["new_cp", "unconstrained_lr_coef"] * 0.5, i + 0.3), (c.loc["new_cp", "unconstrained_lr_coef"], i + 2.3),
                     fontsize=8, color=INK2, arrowprops=dict(arrowstyle="->", color=MUTED), bbox=dict(fc=SURFACE, ec="none", alpha=0.9), zorder=5)
    fig.suptitle("Why not just use ML? The better-scoring model learns the simulator", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    return _finish(fig, "fig16_why_not_ml")


def fig_artifacts(art: pd.DataFrame) -> str:
    a = art.dropna(subset=["share_between_labelled_accounts"])
    fig, ax = plt.subplots(figsize=(8, 3.3))
    ax.barh(a["artifact"], 100 * a["share_between_labelled_accounts"], color=ORANGE, height=0.55)
    ax.axvline(100 * a["base_share"].iloc[0], color=BLUE, lw=2)
    ax.text(100 * a["base_share"].iloc[0] + 1, 2.45, f"all transfers: {a['base_share'].iloc[0]:.1%}", fontsize=8.5,
            color=INK2)
    ax.set_xlabel("% of these transfers / pairs that are between two labelled accounts")
    ax.set_xlim(0, 110)
    ax.set_title("Simulator artifacts that would make a model look better than it is")
    return _finish(fig, "fig17_simulator_artifacts")


def fig_case_network(tx: pd.DataFrame, case: dict, labelled: set, alerted: set, name: str) -> str:
    acc = case["account_id"]
    e = ego_edges(tx, acc, config.TEST_WEEKS[0] * 7 - 6, config.TEST_WEEKS[-1] * 7, radius=1, max_edges=60)
    G = nx.DiGraph()
    for r in e.itertuples():
        G.add_edge(r.src, r.dst, w=r.amount)
    fig, ax = plt.subplots(figsize=(7.2, 6))
    pos = nx.spring_layout(G, seed=7, k=0.6)
    senders = [n for n in G if G.has_edge(n, acc)]
    receivers = [n for n in G if G.has_edge(acc, n)]
    colors = [ORANGE if n == acc else (AQUA if n in senders and n in receivers else (BLUE if n in senders else VIOLET))
              for n in G]
    widths = [0.6 + 3 * d["w"] / max(1, max(x["w"] for *_, x in G.edges(data=True))) for *_, d in G.edges(data=True)]
    nx.draw_networkx_edges(G, pos, ax=ax, width=widths, edge_color=MUTED, arrowsize=9, alpha=0.8)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=[380 if n == acc else 120 for n in G],
                           edgecolors=[INK if n in alerted else SURFACE for n in G], linewidths=1.5)
    nx.draw_networkx_labels(G, pos, labels={acc: str(acc)}, font_size=9, font_color=INK, ax=ax)
    ax.set_axis_off()
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color=c, label=l, ms=8) for c, l in
               ((ORANGE, "account under review"), (BLUE, "sends to it"), (VIOLET, "receives from it"),
                (AQUA, "both directions"))]
    handles.append(Line2D([], [], marker="o", ls="", mfc=SURFACE, mec=INK, label="alerted in a test week", ms=8))
    ax.legend(handles=handles, loc="lower left", fontsize=8)
    ax.set_title(f"{case['case']}\naccount {acc} - direct counterparties, test weeks", fontsize=11)
    return _finish(fig, name, NOTE + " Edge width = transferred value.")


def fig_case_timelines(cases: list[dict]) -> str:
    n = len(cases)
    fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(11, 2.2 * ((n + 1) // 2)), sharex=True, sharey=True)
    for ax, c in zip(axes.flat, cases):
        w = list(map(int, c["weekly_scores"].keys()))
        s = list(c["weekly_scores"].values())
        ax.axhline(config.ALERT_PERCENTILE, color=RED, lw=1, ls="--")
        ax.plot(w, s, marker="o", ms=4, color=ORANGE if c["label_is_suspicious"] else BLUE)
        ax.set_title(f"{c['case'].split(' - ')[0]} | acct {c['account_id']} | "
                     f"{'labelled' if c['label_is_suspicious'] else 'unlabelled'}", fontsize=9.5)
        ax.set_ylim(0, 102)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("test week")
    fig.suptitle("Case-study risk timelines (dashed = alert threshold, score = percentile vs development weeks)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    return _finish(fig, "fig18_case_timelines")


def make_all(ctx: dict) -> list[str]:
    style()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    T = config.TABLES_DIR
    run = ctx["run"]
    km = ctx["km"]
    labelled = set(ctx["accounts"].loc[ctx["accounts"]["is_suspicious"] == 1, "account_id"])
    proxy = ctx["tx"]["src"].isin(labelled) & ctx["tx"]["dst"].isin(labelled)
    out = [
        fig_weekly_activity(ctx["weekly"]),
        fig_amounts(ctx["tx"], proxy.astype(int)),
        fig_lorenz(ctx["lor"], km["gini_value"]),
        fig_fingerprint(pd.read_csv(T / "labelled_vs_unlabelled_profiles.csv")),
        fig_signal_quality(ctx["rc"], km["heldout_test_base_rate"]),
        fig_signal_overlap(run.sig, [s for s in ctx["rc"]["rule_id"]]),
        fig_degree(ctx["G"], labelled),
        fig_centrality_lift(pd.read_csv(T / "centrality_lift_top1pct.csv")),
        fig_tradeoff(pd.read_csv(T / "baseline_comparison.csv")),
        fig_topk(ctx["curves"]),
        fig_sensitivity(ctx["sens"]),
        fig_prevalence(ctx["prev"]),
        fig_stability(ctx["stab"]),
        fig_tiers(ctx["tv"]),
        fig_deciles(ctx["dec"], km["heldout_test_base_rate"]),
        fig_coefficients(pd.read_csv(T / "model_coefficients.csv")),
        fig_artifacts(pd.read_csv(T / "artifact_audit.csv")),
    ]
    alerted = set(ctx["queue"].loc[ctx["queue"]["period"] == "test", "account_id"])
    cases = ctx.get("cases") or json.load(open(T / "case_studies.json"))
    for i, c in enumerate(cases[:3], start=1):
        out.append(fig_case_network(ctx["tx"], c, labelled, alerted, f"fig19_case{i}_network"))
    out.append(fig_case_timelines(cases))
    return out
