"""Robustness analyses: feature-family removal, weight perturbation, artifact
neutralisation, alternative typology datasets, and early vs late test weeks."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, evaluation, preprocessing, risk_scoring
from .pipeline import baseline_alerts, evaluate_run, log, n_tx_in, run_framework, save


def _summary(run, name: str) -> dict:
    m = run.masks["heldout_test"]
    frame = run.aw[m]
    ntx = n_tx_in(run, config.TEST_WEEKS, group=1)
    a = evaluation.alert_metrics(frame, run.alert[m], name, ntx, len(config.TEST_WEEKS))
    r = evaluation.ranking_metrics(evaluation.account_scores(frame, run.score[m]), name)
    return {"scenario": name, "alerts": a["alerts"], "alert_precision": a["alert_precision"],
            "account_recall": a["account_recall"], "pr_auc": r["pr_auc"], "roc_auc": r["roc_auc"],
            "precision_top5pct": r["precision_top5pct"], "base_rate": r["base_rate"]}


def run_all(ctx: dict) -> None:
    run, accounts, tx, aw, km = ctx["run"], ctx["accounts"], ctx["tx"], ctx["aw"], ctx["km"]
    base_feats = run.model.features

    # 1) feature-family removal (refit weights each time)
    rows = [_summary(run, "full framework")]
    fam_feats = {}
    for sid in risk_scoring.FRAMEWORK_SIGNALS:
        s = risk_scoring.SIGNAL_BY_ID[sid]
        fam_feats.setdefault(s["family"], set()).add(s["feature"])
    for fam, feats in fam_feats.items():
        keep = [f for f in base_feats if f not in feats]
        r = run_framework(accounts, tx, aw=aw, features=keep)
        rows.append(_summary(r, f"without {fam} features"))
    fr = pd.DataFrame(rows)
    save(fr, "robustness_feature_removal")
    km["robust_feature_removal"] = fr.set_index("scenario")[["pr_auc", "alert_precision", "account_recall"]].to_dict(orient="index")
    log("robustness: feature-family removal done")

    # 2) weight perturbation (no refit): equal weights and random +/-50% multiplicative noise
    m = run.masks["heldout_test"]
    Z = run.model.standardise(aw)
    w = run.model.weights.copy()
    rows = []
    rng = np.random.default_rng(config.RANDOM_SEED)
    variants = {"fitted weights": w, "equal weights (mean of fitted, all features)": np.full_like(w, w[w > 0].mean())}
    for i in range(200):
        variants[f"random_{i:03d}"] = w * rng.uniform(0.5, 1.5, len(w))
    for name, wv in variants.items():
        sc = pd.Series(Z.to_numpy() @ wv, index=aw.index)
        r = evaluation.ranking_metrics(evaluation.account_scores(aw[m], sc[m]), name)
        rows.append({"variant": name, "pr_auc": r["pr_auc"], "roc_auc": r["roc_auc"],
                     "precision_top5pct": r["precision_top5pct"]})
    wp = pd.DataFrame(rows)
    save(wp, "robustness_weight_perturbation")
    rnd = wp[wp["variant"].str.startswith("random_")]
    km["weight_perturb_pr_auc"] = {"fitted": float(wp.iloc[0]["pr_auc"]), "equal": float(wp.iloc[1]["pr_auc"]),
                                   "random_p05": float(rnd["pr_auc"].quantile(0.05)),
                                   "random_p95": float(rnd["pr_auc"].quantile(0.95))}
    log("robustness: weight perturbation done")

    # 3) artifact-neutralised scenario: remove exact duplicate copies, floor amounts at 100
    tx_n = tx[tx["is_repeat_copy"] == 0].copy()
    tx_n["amount"] = tx_n["amount"].clip(lower=100)
    run_n = run_framework(accounts, tx_n)
    ev_n = evaluate_run(run_n)
    ev_n["alerts"].insert(0, "scenario", "artifact-neutralised")
    ev_n["ranking"].insert(0, "scenario", "artifact-neutralised")
    save(ev_n["alerts"], "robustness_artifact_neutralised_alerts")
    save(ev_n["ranking"], "robustness_artifact_neutralised_ranking")
    km["artifact_neutralised"] = {
        "alerts": ev_n["alerts"].set_index("approach")[["alerts", "alert_precision", "account_recall"]].to_dict(orient="index"),
        "ranking": ev_n["ranking"].set_index("approach")[["pr_auc", "roc_auc"]].to_dict(orient="index"),
        "weights": run_n.model.to_dict()["weights"]}
    log("robustness: artifact-neutralised scenario done")

    # 4) alternative typology datasets (same code, own calibration)
    rows = [dict(dataset="combined (primary)", typologies=config.DATASETS["combined"]["typologies"],
                 **{k: v for k, v in _summary(run, "integrated").items() if k != "scenario"},
                 b2_alert_precision=evaluation.alert_metrics(aw[m], baseline_alerts(run)["B2 Amount + velocity"][m], "",
                                                             1, 1)["alert_precision"])]
    for name in ("fanin", "cycle"):
        nodes, tx_raw = ctx["raw"][name]
        acc_d, tx_d = preprocessing.clean(nodes, tx_raw)
        r = run_framework(acc_d, tx_d)
        mm = r.masks["heldout_test"]
        s = _summary(r, "integrated")
        s.pop("scenario")
        rows.append(dict(dataset=name, typologies=config.DATASETS[name]["typologies"], **s,
                         b2_alert_precision=evaluation.alert_metrics(r.aw[mm], baseline_alerts(r)["B2 Amount + velocity"][mm],
                                                                     "", 1, 1)["alert_precision"],
                         weights=str({k: round(v, 3) for k, v in r.model.to_dict()["weights"].items()})))
    ds = pd.DataFrame(rows)
    save(ds, "robustness_datasets")
    km["robust_datasets"] = ds.set_index("dataset")[["pr_auc", "alert_precision", "account_recall", "base_rate"]].to_dict(orient="index")
    log("robustness: alternative datasets done")

    # 5) early vs late test weeks
    rows = []
    ntx_all = n_tx_in(run, config.TEST_WEEKS, group=1)
    for label, weeks in (("early test (weeks 13-16)", range(13, 17)), ("late test (weeks 17-21)", range(17, 22))):
        mm = m & aw["week"].isin(weeks)
        for name, a in baseline_alerts(run).items():
            r = evaluation.alert_metrics(aw[mm], a[mm], name, ntx_all, len(weeks))
            r["window"] = label
            r["base_rate"] = aw[mm].groupby("account_id")["is_suspicious"].first().mean()
            rows.append(r)
    et = pd.DataFrame(rows)
    save(et, "robustness_early_vs_late")
    log("robustness: early vs late test windows done")
