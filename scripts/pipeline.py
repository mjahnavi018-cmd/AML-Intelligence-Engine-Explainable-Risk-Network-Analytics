"""End-to-end analytical pipeline. Entry point: `python run_pipeline.py`.

Stages write every table to outputs/tables, figures to outputs/figures, models
to outputs/models and analytical tables to data/processed/banking_risk.db.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from . import (aml_analysis, behavioral_analysis, config, data_validation, evaluation, feature_engineering,
               fraud_analysis, ingestion, network_analysis, preprocessing, risk_scoring, stats_tests)
from .alert_prioritization import TIERS, alert_persistence, assign_tier, build_alert_queue, tier_cutoffs

T = config.TABLES_DIR
LOG: list[str] = []


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    LOG.append(line)
    print(line, flush=True)


def save(df: pd.DataFrame, name: str, index: bool = False) -> None:
    df.to_csv(T / f"{name}.csv", index=index)


def assign_groups(accounts: pd.DataFrame) -> pd.Series:
    """Seeded random 50/50 account split: group 0 = model fitting, group 1 = held-out evaluation."""
    ids = np.sort(accounts["account_id"].to_numpy())
    rng = np.random.default_rng(config.RANDOM_SEED)
    return pd.Series(rng.integers(0, 2, len(ids)), index=ids, name="group")


# ---------------------------------------------------------------------------
# Framework (reused for main run, artifact-neutralised scenario and other datasets)
# ---------------------------------------------------------------------------
@dataclass
class Run:
    accounts: pd.DataFrame
    tx: pd.DataFrame
    aw: pd.DataFrame
    groups: pd.Series
    th: dict
    sig: pd.DataFrame
    fam: pd.DataFrame
    model: risk_scoring.IntegratedRiskModel
    score: pd.Series
    alert: pd.Series
    masks: dict = field(default_factory=dict)


def run_framework(accounts: pd.DataFrame, tx: pd.DataFrame, aw: pd.DataFrame | None = None,
                  features: list[str] | None = None, signal_q: float | None = None,
                  alert_pct: float = config.ALERT_PERCENTILE) -> Run:
    if aw is None:
        aw = feature_engineering.build_account_week_features(tx)
    aw = aw.drop(columns=[c for c in ("is_suspicious", "group") if c in aw.columns])
    aw = aw.merge(accounts[["account_id", "is_suspicious"]], on="account_id", how="left")
    groups = assign_groups(accounts)
    aw["group"] = aw["account_id"].map(groups)
    dev = aw["period"] == "dev"
    test = aw["period"] == "test"
    masks = {"dev": dev, "test": test, "fit": dev & (aw["group"] == 0), "val": dev & (aw["group"] == 1),
             "heldout_test": test & (aw["group"] == 1)}
    th = risk_scoring.calibrate_thresholds(aw[dev], tx[tx["week"].isin(config.DEV_WEEKS)], q=signal_q)
    sig = risk_scoring.evaluate_signals(aw, th)
    fam = risk_scoring.family_hits(sig)
    labels = accounts.set_index("account_id")["is_suspicious"]
    model = risk_scoring.IntegratedRiskModel(features).fit(aw[dev], masks["fit"][dev], labels)
    score = pd.Series(model.risk_score(aw), index=aw.index, name="risk_score")
    alert = score >= alert_pct
    return Run(accounts, tx, aw, groups, th, sig, fam, model, score, alert, masks)


def n_tx_in(run: Run, weeks: list[int], group: int | None = 1) -> int:
    t = run.tx[run.tx["week"].isin(weeks)]
    if group is None:
        return len(t)
    ids = set(run.groups[run.groups == group].index)
    return int((t["src"].isin(ids) | t["dst"].isin(ids)).sum())


def baseline_alerts(run: Run) -> dict[str, pd.Series]:
    s = run.sig
    return {
        "B1 Amount threshold": s["TXN-01"],
        "B2 Amount + velocity": s["TXN-01"] | s["VEL-01"],
        "B3 Behavioural anomaly": s[["BEH-01", "BEH-02"]].any(axis=1),
        "B4 Network / AML patterns": s[["AML-01", "AML-03", "NET-01", "NET-02", "NET-03"]].any(axis=1),
        "B5a Integrated rules (>=2 families)": run.fam.sum(axis=1) >= 2,
        "B5b Integrated risk score (top 1%)": run.alert,
    }


def evaluate_run(run: Run, extra_scores: dict[str, pd.Series] | None = None) -> dict[str, pd.DataFrame]:
    """Held-out accounts (group 1), test weeks: alert metrics + ranking metrics."""
    m = run.masks["heldout_test"]
    frame = run.aw[m]
    ntx = n_tx_in(run, config.TEST_WEEKS, group=1)
    rows = [evaluation.alert_metrics(frame, a[m], name, ntx, len(config.TEST_WEEKS))
            for name, a in baseline_alerts(run).items()]
    rank_scores = {
        "B1 Amount threshold": run.aw["max_amt"],
        "B2 Amount + velocity": run.aw["n_tx"] + run.aw["max_amt"] / 1e4,
        "B5a Integrated rules (>=2 families)": run.fam.sum(axis=1) + run.score / 1000,
        "B5b Integrated risk score (top 1%)": run.score,
    }
    for name, sc in (extra_scores or {}).items():
        cut = sc[run.masks["dev"]].quantile(config.ALERT_PERCENTILE / 100)
        rows.append(evaluation.alert_metrics(frame, (sc >= cut)[m], name, ntx, len(config.TEST_WEEKS)))
        rank_scores[name] = sc
    ranking = [evaluation.ranking_metrics(evaluation.account_scores(frame, sc[m]), name)
               for name, sc in rank_scores.items()]
    return {"alerts": pd.DataFrame(rows), "ranking": pd.DataFrame(ranking), "rank_scores": rank_scores}


def ml_comparators(run: Run) -> dict[str, pd.Series]:
    """Supporting ML models, same features, same fit/validation discipline."""
    feats = run.model.features
    Z = run.model.standardise(run.aw)
    fit = run.masks["fit"]
    labels = run.accounts.set_index("account_id")["is_suspicious"]
    A = Z[fit].assign(account_id=run.aw.loc[fit, "account_id"].values).groupby("account_id")[feats].max()
    y = labels.reindex(A.index).to_numpy()
    lr = LogisticRegression(max_iter=2000).fit(A.to_numpy(), y)
    gbm = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05,
                                         random_state=config.RANDOM_SEED).fit(A.to_numpy(), y)
    iso = fraud_analysis.isolation_forest_scores(run.aw[run.masks["dev"]], run.aw, feats)
    out = {
        "ML1 Logistic regression (unconstrained)": pd.Series(lr.decision_function(Z.to_numpy()), index=run.aw.index),
        "ML2 Gradient boosting": pd.Series(gbm.predict_proba(Z.to_numpy())[:, 1], index=run.aw.index),
        "ML3 Isolation Forest (unsupervised)": pd.Series(iso, index=run.aw.index),
    }
    coefs = pd.DataFrame({"feature": feats, "unconstrained_lr_coef": lr.coef_[0],
                          "integrated_nonneg_weight": run.model.weights})
    out["_coefs"] = coefs
    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main(skip_robustness: bool = False) -> dict:
    t0 = time.time()
    for d in (config.TABLES_DIR, config.FIGURES_DIR, config.MODELS_DIR, config.PROCESSED_DIR):
        d.mkdir(parents=True, exist_ok=True)
    km: dict = {}  # key metrics used by reports / README / dashboard

    # ---------------- Phase 1: ingestion + audit ----------------
    ingestion.verify_checksums()
    log("raw checksums verified")
    audits, raw = [], {}
    for name in config.DATASETS:
        nodes, tx_raw, meta = ingestion.load_raw(name)
        rep = data_validation.audit(nodes, tx_raw, meta, name)
        data_validation.assert_no_failures(rep)
        audits.append(rep)
        raw[name] = (nodes, tx_raw)
    dq = pd.concat(audits, ignore_index=True)
    save(dq, "data_quality_report")
    log(f"data audit passed for {len(raw)} datasets ({int((dq.status == 'WARN').sum())} warnings)")

    nodes, tx_raw = raw[config.PRIMARY_DATASET]
    accounts, tx = preprocessing.clean(nodes, tx_raw)
    km["n_accounts"] = len(accounts)
    km["n_tx"] = len(tx)
    km["n_labelled"] = int(accounts["is_suspicious"].sum())
    km["labelled_share"] = accounts["is_suspicious"].mean()
    km["total_value"] = float(tx["amount"].sum())
    km["n_steps"] = int(tx["step"].max())
    km["n_weeks_full"] = int(tx["step"].max() // config.WEEK_LEN)
    km["amount_min"], km["amount_max"] = float(tx["amount"].min()), float(tx["amount"].max())
    km["n_dup_extra"] = int(tx["is_repeat_copy"].sum())
    km["n_self"] = int(tx["is_self_transfer"].sum())
    km["n_inactive_accounts"] = int(accounts["first_step"].isna().sum())
    km["metadata_labelled"] = 1803
    ev_labels = preprocessing.evaluation_labels(accounts, tx)
    km["n_proxy_typology_tx"] = int(ev_labels["proxy_typology_tx"].sum())

    # ---------------- Phase 2: features ----------------
    aw = feature_engineering.build_account_week_features(tx)
    log(f"point-in-time features: {len(aw):,} account-weeks")

    # ---------------- Phase 3/4: EDA + profiling ----------------
    prof = behavioral_analysis.add_full_period_segment(behavioral_analysis.account_profiles(accounts, tx))
    save(prof, "account_profiles")
    weekly = tx.assign(proxy=ev_labels["proxy_typology_tx"].values).groupby("week").agg(
        transactions=("tx_id", "size"), value=("amount", "sum"), proxy_typology_share=("proxy", "mean"),
        senders=("src", "nunique"), receivers=("dst", "nunique")).reset_index()
    weekly["complete_week"] = weekly["week"] <= km["n_weeks_full"]
    save(weekly, "weekly_activity")
    seg = prof.groupby("activity_tier").agg(accounts=("account_id", "size"), labelled_rate=("is_suspicious", "mean"),
                                           median_tx=("tx_count", "median"),
                                           median_counterparties=("unique_counterparties", "median")).reset_index()
    save(seg, "activity_tier_summary")
    gc = fraud_analysis.group_comparisons(prof)
    save(gc, "labelled_vs_unlabelled_profiles")
    vp = fraud_analysis.value_proxies(accounts, tx)
    save(vp, "value_proxies")
    act = prof[prof["tx_count"] > 0]
    km["gini_tx_count"] = behavioral_analysis.gini(act["tx_count"])
    km["gini_value"] = behavioral_analysis.gini(act["total_value"])
    lor = behavioral_analysis.lorenz(act["total_value"])
    km["top1pct_value_share"] = float(lor.loc[lor["account_share"] <= 0.01, "value_share"].max())
    km["top10pct_value_share"] = float(lor.loc[lor["account_share"] <= 0.10, "value_share"].max())
    # amount distribution by proxy label
    amt = tx.assign(proxy=ev_labels["proxy_typology_tx"].values)
    save(amt.groupby("proxy")["amount"].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).reset_index(),
         "amount_distribution_by_proxy_label")
    log("EDA and profiling tables written")

    # ---------------- Phase 5-11: framework ----------------
    run = run_framework(accounts, tx, aw=aw)
    aw = run.aw
    km["dev_account_weeks"] = int(run.masks["dev"].sum())
    km["test_account_weeks"] = int(run.masks["test"].sum())
    km["heldout_test_account_weeks"] = int(run.masks["heldout_test"].sum())
    km["heldout_test_accounts"] = int(aw.loc[run.masks["heldout_test"], "account_id"].nunique())
    km["heldout_test_labelled"] = int(aw[run.masks["heldout_test"]].groupby("account_id")["is_suspicious"].first().sum())
    km["heldout_test_base_rate"] = km["heldout_test_labelled"] / km["heldout_test_accounts"]
    km["thresholds"] = run.th
    run.model.save(config.MODELS_DIR / "integrated_risk_model.json")
    with open(config.MODELS_DIR / "signal_thresholds.json", "w") as fh:
        json.dump({"thresholds": run.th, "calibration": "development weeks 5-12, label-free quantiles",
                   "quantile": config.SIGNAL_QUANTILE}, fh, indent=2)
    km["model_weights"] = run.model.to_dict()["weights"]

    quar = risk_scoring.artifact_quarantine(run.sig, aw["is_suspicious"].astype(bool), run.masks["dev"])
    save(quar, "artifact_quarantine")
    if not quar["consistent"].all():
        log("WARNING: a framework signal meets the artifact-quarantine rule: "
            + ", ".join(quar.loc[~quar["consistent"], "signal"]))
    km["quarantined"] = quar.loc[quar["quarantine_flag"], "signal"].tolist()
    rc = aml_analysis.rule_catalogue(run.th, run.sig, aw, run.masks["dev"], run.masks["heldout_test"])
    save(rc, "aml_rule_catalogue")
    km["base_rate_dev_weeks"] = rc.attrs["base_dev"]
    km["base_rate_test_weeks"] = rc.attrs["base_test"]

    art = aml_analysis.artifact_audit(accounts, tx, aw)
    save(art, "artifact_audit")
    km["artifact_lt100_share"] = float(art.loc[0, "share_between_labelled_accounts"])
    km["artifact_dup_share"] = float(art.loc[1, "share_between_labelled_accounts"])
    d = aw[run.masks["dev"]]
    rec = (d["n_cp"] - d["new_cp"]) > 0
    km["recurring_cp_rate_labelled"] = float(rec[d["is_suspicious"] == 1].mean())
    km["recurring_cp_rate_unlabelled"] = float(rec[d["is_suspicious"] == 0].mean())
    st = aml_analysis.structuring_analysis(accounts, tx)
    save(st["summary"], "structuring_summary")
    save(st["heaping"], "amount_heaping_near_round_values")

    # model selection on development VALIDATION accounts only (group 1, dev weeks)
    ml = ml_comparators(run)
    coefs = ml.pop("_coefs")
    save(coefs, "model_coefficients")
    km["lr_coefs"] = dict(zip(coefs["feature"], coefs["unconstrained_lr_coef"]))
    cand = {"M0 Velocity only (weekly count)": aw["n_tx"] + aw["max_amt"] / 1e4,
            "M1 Evidence count (families fired)": run.fam.sum(axis=1) + run.score / 1000,
            "M2 Integrated non-negative logistic": run.score, **ml}
    sel = []
    val = run.masks["val"]
    for name, sc in cand.items():
        r = evaluation.ranking_metrics(evaluation.account_scores(aw[val], sc[val]), name)
        r["transferable_by_design"] = name.startswith(("M0", "M1", "M2"))
        sel.append(r)
    sel = pd.DataFrame(sel)
    save(sel, "model_selection_dev_validation")
    transferable = sel[sel["transferable_by_design"]].sort_values("pr_auc", ascending=False)
    km["selected_method"] = transferable.iloc[0]["approach"]
    km["selection_val_pr_auc"] = dict(zip(sel["approach"], sel["pr_auc"]))
    log(f"model selection (dev validation): best transferable = {km['selected_method']}")

    # evaluation on held-out accounts x test weeks
    ev = evaluate_run(run, extra_scores=ml)
    save(ev["alerts"], "baseline_comparison")
    save(ev["ranking"], "ranking_metrics")
    km["baseline"] = ev["alerts"].set_index("approach").to_dict(orient="index")
    km["ranking"] = ev["ranking"].set_index("approach").to_dict(orient="index")
    hm = run.masks["heldout_test"]
    curves = pd.concat([evaluation.topk_curve(evaluation.account_scores(aw[hm], sc[hm]), n)
                        for n, sc in ev["rank_scores"].items()])
    save(curves, "topk_capture_curves")
    dec = evaluation.decile_table(evaluation.account_scores(aw[hm], run.score[hm]))
    save(dec, "risk_decile_performance")
    km["top_decile_capture"] = float(dec.loc[0, "share_of_all_labelled"])
    km["top_decile_rate"] = float(dec.loc[0, "labelled_rate"])

    # alert queue (all accounts, all monitoring weeks) + priority tiers
    contrib = run.model.contributions(aw)
    cuts = tier_cutoffs(run.score[run.alert & run.masks["dev"]])
    km["tier_cutoffs"] = cuts
    # evidence-strength definition chosen on DEVELOPMENT alerts of fitting accounts (group 0):
    # breadth (distinct signal families) vs persistence (alerted weeks to date)
    persist = alert_persistence(aw, run.alert)
    breadth = run.fam.sum(axis=1)
    dm = run.alert & run.masks["fit"]
    rows = []
    for nm, cnt in (("breadth: signal families fired", breadth), ("persistence: alerted weeks to date", persist)):
        lvl = risk_scoring.evidence_strength(cnt[dm])
        for L in risk_scoring.EVIDENCE_LEVELS:
            sel_ = lvl == L
            rows.append({"definition": nm, "level": L, "dev_alerts": int(sel_.sum()),
                         "dev_precision": float(aw.loc[dm, "is_suspicious"][sel_].mean()) if sel_.any() else np.nan})
    evdef = pd.DataFrame(rows)
    spread = evdef.pivot(index="definition", columns="level", values="dev_precision")
    spread["strong_minus_limited"] = spread["Strong"] - spread["Limited"]
    chosen = spread["strong_minus_limited"].idxmax()
    evdef["chosen"] = evdef["definition"] == chosen
    save(evdef, "evidence_definition_selection")
    km["evidence_definition"] = chosen
    ev_count = persist if chosen.startswith("persistence") else breadth
    queue = build_alert_queue(aw, run.sig, contrib, run.fam, run.score, run.alert, cuts, ev_count)
    queue["period"] = np.where(queue["week"].isin(config.DEV_WEEKS), "dev", "test")
    save(queue, "alert_queue")
    # tier validation (held-out, test)
    qv = queue.merge(aw[["account_id", "week", "is_suspicious", "group"]], on=["account_id", "week"])
    qv = qv[(qv["period"] == "test") & (qv["group"] == 1)]
    tv = qv.groupby("risk_level").agg(alerts=("alert_id", "size"), precision=("is_suspicious", "mean"),
                                      accounts=("account_id", "nunique")).reindex(TIERS).reset_index()
    tv["share_of_alerts"] = tv["alerts"] / tv["alerts"].sum()
    tv["alerts_per_week"] = tv["alerts"] / len(config.TEST_WEEKS)
    save(tv, "priority_tier_validation")
    ev_str = qv.groupby("evidence_strength").agg(alerts=("alert_id", "size"),
                                                 precision=("is_suspicious", "mean")).reset_index()
    save(ev_str, "evidence_strength_validation")
    cross = pd.crosstab(qv["risk_level"], qv["evidence_strength"]).reindex(TIERS).fillna(0).astype(int)
    save(cross.reset_index(), "risk_vs_evidence_crosstab")
    km["tiers"] = tv.set_index("risk_level").to_dict(orient="index")
    km["evidence_strength"] = ev_str.set_index("evidence_strength").to_dict(orient="index")
    km["alerts_total_all_accounts_test"] = int((queue["period"] == "test").sum())
    km["alerts_per_week_all_accounts_test"] = km["alerts_total_all_accounts_test"] / len(config.TEST_WEEKS)
    km["test_tx_all"] = n_tx_in(run, config.TEST_WEEKS, group=None)
    log(f"alert queue: {len(queue):,} alerts ({km['alerts_total_all_accounts_test']:,} in test weeks)")

    # ---------------- threshold sensitivity ----------------
    rows = []
    ntx = n_tx_in(run, config.TEST_WEEKS, group=1)
    frame = aw[hm]
    for q in config.SENSITIVITY_QUANTILES:
        th_q = risk_scoring.calibrate_thresholds(aw[run.masks["dev"]], tx[tx["week"].isin(config.DEV_WEEKS)], q=q)
        sq = risk_scoring.evaluate_signals(aw, th_q)
        fq = risk_scoring.family_hits(sq)
        for name, a in {"B2 Amount + velocity": sq["TXN-01"] | sq["VEL-01"],
                        "B3 Behavioural anomaly": sq[["BEH-01", "BEH-02"]].any(axis=1),
                        "B4 Network / AML patterns": sq[["AML-01", "AML-03", "NET-01", "NET-02", "NET-03"]].any(axis=1),
                        "B5a Integrated rules (>=2 families)": fq.sum(axis=1) >= 2}.items():
            r = evaluation.alert_metrics(frame, a[hm], name, ntx, len(config.TEST_WEEKS))
            r.update(parameter="rule quantile", value=q)
            rows.append(r)
    for p in config.ALERT_PERCENTILE_GRID:
        r = evaluation.alert_metrics(frame, (run.score >= p)[hm], "B5b Integrated risk score", ntx, len(config.TEST_WEEKS))
        r.update(parameter="score percentile", value=p)
        rows.append(r)
    sens = pd.DataFrame(rows)
    save(sens, "threshold_sensitivity")

    # ---------------- segment-specific thresholds ----------------
    segs, cuts_seg = behavioral_analysis.activity_segment(aw)
    aw["segment"] = segs
    km["segment_cuts"] = cuts_seg
    rows = []
    glob = run.sig["VEL-01"]
    dev = run.masks["dev"]
    seg_th = {s: aw.loc[dev & (aw["segment"] == s), "n_tx"].quantile(config.SIGNAL_QUANTILE) for s in aw["segment"].unique()}
    segv = aw["n_tx"] >= aw["segment"].map(seg_th)
    for s in ["no history", "low", "medium", "high"]:
        mm = hm & (aw["segment"] == s)
        for kind, a in (("global threshold", glob), ("segment threshold", segv)):
            hits = a & mm
            rows.append({"segment": s, "threshold_type": kind,
                         "threshold": run.th["VEL-01"] if kind == "global threshold" else seg_th[s],
                         "account_weeks": int(mm.sum()), "alerts": int(hits.sum()),
                         "alert_rate": hits.sum() / mm.sum() if mm.sum() else np.nan,
                         "precision": aw.loc[hits, "is_suspicious"].mean() if hits.any() else np.nan,
                         "segment_base_rate": aw.loc[mm, "is_suspicious"].mean()})
    segtab = pd.DataFrame(rows)
    save(segtab, "segment_threshold_comparison")
    tot = {k: evaluation.alert_metrics(frame, a[hm], k, ntx, len(config.TEST_WEEKS))
           for k, a in (("VEL-01 global", glob), ("VEL-01 segment-specific", segv))}
    save(pd.DataFrame(tot.values()), "segment_threshold_overall")
    km["segment_overall"] = {k: {kk: v[kk] for kk in ("alerts", "alert_precision", "account_recall")} for k, v in tot.items()}

    # ---------------- prevalence projection ----------------
    rows = []
    for _, r in ev["alerts"].iterrows():
        for p in config.REALISTIC_PREVALENCES:
            rows.append({"approach": r["approach"], "prevalence": p, "recall": r["account_recall"],
                         "fpr": r["fpr_accounts"],
                         "projected_precision": evaluation.precision_at_prevalence(r["account_recall"], r["fpr_accounts"], p),
                         "alerted_accounts_per_10k": 10000 * (r["account_recall"] * p + r["fpr_accounts"] * (1 - p))})
    prev = pd.DataFrame(rows)
    save(prev, "prevalence_projection")

    # ---------------- temporal stability / drift ----------------
    rows = []
    for w in config.DEV_WEEKS + config.TEST_WEEKS:
        mm = (aw["week"] == w) & (aw["group"] == 1)
        fw = aw[mm]
        for name, a in (("B2 Amount + velocity", baseline_alerts(run)["B2 Amount + velocity"]),
                        ("B5b Integrated risk score (top 1%)", run.alert)):
            hits = a[mm]
            accs = fw.groupby("account_id")["is_suspicious"].first()
            hit_acc = hits.groupby(fw["account_id"]).any()
            rows.append({"week": w, "period": "dev" if w in config.DEV_WEEKS else "test", "approach": name,
                         "alerts": int(hits.sum()),
                         "precision": fw.loc[hits, "is_suspicious"].mean() if hits.any() else np.nan,
                         "weekly_recall": (hit_acc & (accs == 1)).sum() / max(1, (accs == 1).sum()),
                         "active_accounts": len(accs), "labelled_active_share": accs.mean()})
    stab = pd.DataFrame(rows)
    save(stab, "weekly_stability")
    psi_rows = []
    for c in risk_scoring.SCORE_FEATURES + ["n_cp", "hist_n_tx"]:
        psi_rows.append({"feature": c, "psi_dev_vs_test": stats_tests.psi(aw.loc[dev, c], aw.loc[run.masks["test"], c])})
    psi_rows.append({"feature": "risk_score", "psi_dev_vs_test": stats_tests.psi(run.score[dev], run.score[run.masks["test"]])})
    psit = pd.DataFrame(psi_rows)
    save(psit, "feature_drift_psi")
    km["psi_max"] = float(psit["psi_dev_vs_test"].max())
    km["psi_max_feature"] = psit.loc[psit["psi_dev_vs_test"].idxmax(), "feature"]
    km["psi_risk_score"] = float(psit.loc[psit.feature == "risk_score", "psi_dev_vs_test"].iloc[0])
    t_stab = stab[(stab["period"] == "test")]
    km["weekly_precision_range"] = {n: [float(g["precision"].min()), float(g["precision"].max())]
                                    for n, g in t_stab.groupby("approach")}
    log("sensitivity, segment, prevalence and drift analyses written")

    # ---------------- network analytics ----------------
    labelled = set(accounts.loc[accounts["is_suspicious"] == 1, "account_id"])
    G = network_analysis.build_graph(tx)
    gs = network_analysis.graph_summary(G, labelled)
    save(gs, "network_summary")
    cent = network_analysis.centrality_table(G)
    save(cent, "network_centrality_full_period")
    lift = network_analysis.centrality_lift(cent, accounts.set_index("account_id")["is_suspicious"])
    save(lift, "centrality_lift_top1pct")
    cyc = network_analysis.full_period_cycles(tx)
    save(cyc, "temporal_cycles_full_period")
    part = network_analysis.cycle_participation(cyc)
    act_ids = set(prof.loc[prof["tx_count"] > 0, "account_id"])
    k1 = sum(1 for a in part if a in labelled)
    k2 = sum(1 for a in part if a not in labelled)
    n1 = len(labelled & act_ids)
    n2 = len(act_ids - labelled)
    km["cycles_found"] = len(cyc)
    km["cycle_members_labelled"], km["cycle_members_unlabelled"] = k1, k2
    km["cycle_rate_labelled"], km["cycle_rate_unlabelled"] = k1 / n1, k2 / n2
    deg = pd.DataFrame({"in_degree": dict(G.in_degree()), "out_degree": dict(G.out_degree())})
    deg = deg.join(accounts.set_index("account_id")["is_suspicious"])
    save(deg.groupby("is_suspicious").describe().T.reset_index(), "degree_by_label")

    msig = run.sig.loc[hm]
    mule = network_analysis.mule_candidates(aw[hm], msig)
    mule = mule.merge(accounts[["account_id", "is_suspicious"]], on="account_id")
    save(mule, "mule_candidates_heldout_test")
    km["mule_candidates"] = len(mule)
    km["mule_candidates_labelled_rate"] = float(mule["is_suspicious"].mean()) if len(mule) else float("nan")
    km["mule_patterns"] = mule.groupby("pattern").agg(n=("account_id", "size"), labelled_rate=("is_suspicious", "mean")).to_dict(orient="index")
    degsum = cent.set_index("account_id")[["in_degree", "out_degree"]].sum(axis=1)
    hubs = set(degsum[degsum >= degsum.quantile(0.99)].index)
    cf = mule[mule["pattern"] == "collector-forwarder"]
    km["collector_forwarder_n"] = len(cf)
    km["collector_forwarder_hub_share"] = float(cf["account_id"].isin(hubs).mean()) if len(cf) else float("nan")
    mule_all = network_analysis.mule_candidates(aw[run.masks["test"]], run.sig.loc[run.masks["test"]])
    save(mule_all, "mule_candidates_all_test")   # no label column: operational view
    log(f"network analytics written; {len(mule)} potential mule-account candidates (held-out)")

    # ---------------- hypothesis tests ----------------
    dprof = behavioral_analysis.account_profiles(accounts, tx[tx["week"].isin(config.DEV_WEEKS)])
    dprof = dprof[dprof["tx_count"] > 0].copy()
    dprof["weekly_tx"] = dprof["tx_count"] / dprof["active_weeks"]
    hyp = fraud_analysis.hypothesis_h1_h2(dprof)
    dmax = aw[dev].groupby("account_id").agg(fan_in=("fan_in_28", "max"), fan_out=("fan_out_28", "max"),
                                             y=("is_suspicious", "first"))
    for col, desc in (("fan_in", "distinct senders"), ("fan_out", "distinct receivers")):
        r = stats_tests.mann_whitney(dmax.loc[dmax.y == 1, col], dmax.loc[dmax.y == 0, col], alternative="greater")
        r.update(hypothesis="H3a" if col == "fan_in" else "H3b",
                 description=f"Labelled accounts have more {desc} in their busiest 28-day window",
                 test="Mann-Whitney U (one-sided, greater)", unit="account", metric=f"max 28-day {desc}, dev period")
        hyp.append(r)
    r = stats_tests.two_proportions(k1, n1, k2, n2)
    r.update(hypothesis="H3c", description="Labelled accounts participate in time-respecting cycles more often",
             test="Fisher exact (two-sided)", unit="account", metric="share of accounts on >=1 cycle, full period (descriptive)")
    hyp.append(r)
    # H4: integrated detects labelled accounts missed by the amount rule (held-out, test)
    acc_lab = aw[hm].groupby("account_id")["is_suspicious"].first()
    det_b1 = baseline_alerts(run)["B1 Amount threshold"][hm].groupby(aw.loc[hm, "account_id"]).any()
    det_b5 = run.alert[hm].groupby(aw.loc[hm, "account_id"]).any()
    pos = acc_lab[acc_lab == 1].index
    r = stats_tests.mcnemar(det_b5.reindex(pos), det_b1.reindex(pos))
    r.update(hypothesis="H4", description="Among labelled accounts, the integrated score detects cases the amount rule misses",
             test="Exact McNemar (paired, labelled held-out accounts)", unit="labelled account",
             metric="detected in test weeks")
    hyp.append(r)
    km["h4_only_integrated"] = int((det_b5.reindex(pos) & ~det_b1.reindex(pos)).sum())
    km["h4_only_amount"] = int((~det_b5.reindex(pos) & det_b1.reindex(pos)).sum())
    # H5: integrated vs amount+velocity ranking, paired bootstrap on PR-AUC
    a5 = evaluation.account_scores(aw[hm], run.score[hm])
    a2 = evaluation.account_scores(aw[hm], ev["rank_scores"]["B2 Amount + velocity"][hm])
    diff, lo, hi = evaluation.bootstrap_diff(a5, a2)
    hyp.append({"hypothesis": "H5", "description": "Integrated score ranks labelled accounts better than amount+velocity (PR-AUC)",
                "test": "Paired account bootstrap (1,000 resamples) of PR-AUC difference", "unit": "held-out account",
                "metric": "PR-AUC, test weeks", "n_group1": len(a5), "n_group2": len(a2), "statistic": diff,
                "p_value": float("nan"), "effect_size": diff, "effect_size_name": "PR-AUC difference",
                "estimate": diff, "ci_low": lo, "ci_high": hi, "estimate_name": "PR-AUC(integrated) - PR-AUC(B2)"})
    hyp = pd.DataFrame(hyp)
    ok = hyp["p_value"].notna()
    hyp.loc[ok, "p_holm"] = stats_tests.holm(hyp.loc[ok, "p_value"].tolist())
    cols = ["hypothesis", "description", "test", "unit", "metric", "n_group1", "n_group2", "median_group1",
            "median_group2", "rate_group1", "rate_group2", "statistic", "p_value", "p_holm", "effect_size",
            "effect_size_name", "estimate", "estimate_name", "ci_low", "ci_high"]
    hyp = hyp.reindex(columns=cols)
    save(hyp, "hypothesis_tests")
    km["hypotheses"] = hyp.set_index("hypothesis").to_dict(orient="index")
    log("hypothesis tests written")

    # ---------------- persist analytical tables to SQLite ----------------
    scores_tbl = aw[["account_id", "week", "run_step", "period", "group", "segment"]].copy()
    scores_tbl["risk_score"] = run.score.round(3)
    scores_tbl["alert"] = run.alert.astype(int)
    for s in risk_scoring.SIGNALS:
        scores_tbl["sig_" + s["id"].replace("-", "_")] = run.sig[s["id"]].astype(int)
    for c in contrib.columns:
        scores_tbl["contrib_" + c] = contrib[c].round(4)
    feat_cols = [c for c in aw.columns if c not in ("is_suspicious", "group", "segment", "period")]
    preprocessing.write_sqlite({
        "accounts": accounts, "transactions": tx, "evaluation_labels": ev_labels,
        "account_week_features": aw[feat_cols + ["period"]], "account_week_scores": scores_tbl,
        "account_groups": run.groups.rename("group").rename_axis("account_id").reset_index(),
        "account_profiles": prof, "alert_queue": queue,
        "signal_catalogue": rc, "network_centrality": cent,
        "mule_candidates": mule_all,
    })
    log(f"SQLite database written: {config.DB_PATH.relative_to(config.ROOT)}")

    # stash objects for later stages
    ctx = dict(run=run, accounts=accounts, tx=tx, aw=aw, prof=prof, queue=queue, G=G, cent=cent, cyc=cyc,
               mule=mule, ev=ev, contrib=contrib, weekly=weekly, lor=lor, sens=sens, prev=prev, stab=stab,
               curves=curves, dec=dec, sel=sel, raw=raw, km=km, hyp=hyp, rc=rc, tv=tv, segtab=segtab)
    if not skip_robustness:
        from . import robustness
        robustness.run_all(ctx)
    from . import case_studies, leakage_audit
    lk = leakage_audit.run(ctx)
    km["leakage_checks_passed"] = int((lk["result"] == "PASS").sum())
    km["leakage_checks_total"] = len(lk)
    log(f"leakage audit: {km['leakage_checks_passed']}/{km['leakage_checks_total']} checks passed")
    case_studies.build(ctx)
    km["runtime_seconds"] = time.time() - t0
    with open(T / "key_metrics.json", "w") as fh:
        json.dump(km, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    (config.TABLES_DIR / "pipeline_log.txt").write_text("\n".join(LOG))
    return ctx
