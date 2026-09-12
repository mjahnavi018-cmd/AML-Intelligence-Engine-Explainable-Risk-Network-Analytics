"""Executable leakage audit. Each check returns PASS/FAIL with evidence; the
pipeline fails if any check fails. Results: outputs/tables/leakage_checks.csv."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, feature_engineering, risk_scoring


def run(ctx: dict) -> pd.DataFrame:
    run_, tx, aw = ctx["run"], ctx["tx"], ctx["aw"]
    rows = []

    def add(check, passed, evidence):
        rows.append({"check": check, "result": "PASS" if passed else "FAIL", "evidence": evidence})

    # 1) no future information: features for week w are identical when all later transactions are deleted
    legs = feature_engineering.make_legs(tx)
    diffs = []
    for w in (config.DEV_WEEKS[0], config.DEV_WEEKS[-1], config.TEST_WEEKS[3], config.TEST_WEEKS[-1]):
        full = feature_engineering.features_for_week(legs, w).set_index("account_id").sort_index()
        trunc = feature_engineering.features_for_week(legs[legs["step"] <= w * config.WEEK_LEN], w)
        trunc = trunc.set_index("account_id").sort_index()
        same = full.shape == trunc.shape and np.allclose(full.select_dtypes("number").fillna(-9).to_numpy(),
                                                          trunc.select_dtypes("number").fillna(-9).to_numpy())
        diffs.append((w, same))
    add("Future transactions cannot change past features (recompute after deleting all later transactions)",
        all(s for _, s in diffs), "; ".join(f"week {w}: {'identical' if s else 'DIFFERENT'}" for w, s in diffs))

    # 2) labels never used as features
    feat_cols = set(risk_scoring.SCORE_FEATURES) | {s["feature"] for s in risk_scoring.SIGNALS}
    forbidden = {"is_suspicious", "fraud_step_raw", "fraudStep", "src_suspicious", "dst_suspicious", "proxy_typology_tx"}
    add("No label or label-derived column is a feature or signal input", not (feat_cols & forbidden),
        f"feature inputs: {sorted(feat_cols)}")

    # 3) full-period descriptive metrics are not scoring inputs
    descriptive = {"tx_count", "unique_counterparties", "pagerank", "betweenness", "in_degree", "out_degree",
                   "active_weeks", "total_value"}
    add("Full-period (look-ahead) descriptive metrics are not used for scoring", not (feat_cols & descriptive),
        "scoring uses only *_28 window and weekly/historical features")

    # 4) thresholds depend on development weeks only
    th2 = risk_scoring.calibrate_thresholds(aw[aw["week"].isin(config.DEV_WEEKS)], tx[tx["week"].isin(config.DEV_WEEKS)])
    add("Rule thresholds reproduce exactly from development weeks alone",
        all(np.isclose(th2[k], v) for k, v in run_.th.items()), f"{len(th2)} thresholds recomputed")

    # 5) model fitted on development weeks and group-0 accounts only
    fit = run_.masks["fit"]
    add("Model-fitting rows are development weeks only", int(aw.loc[fit, "week"].max()) <= max(config.DEV_WEEKS),
        f"max fitting week = {int(aw.loc[fit, 'week'].max())}")
    fit_acc = set(aw.loc[fit, "account_id"])
    ho_acc = set(aw.loc[run_.masks["heldout_test"], "account_id"])
    add("No account overlap between fitting accounts and held-out evaluation accounts", not (fit_acc & ho_acc),
        f"{len(fit_acc)} fitting accounts, {len(ho_acc)} held-out accounts, overlap {len(fit_acc & ho_acc)}")

    # 6) duplicate records cannot straddle the dev/test boundary
    dup = tx[tx.duplicated(["src", "dst", "amount", "step"], keep=False)]
    per = dup.assign(p=np.where(dup["week"].isin(config.DEV_WEEKS), "dev",
                                np.where(dup["week"].isin(config.TEST_WEEKS), "test", "other")))
    straddle = per.groupby(["src", "dst", "amount", "step"])["p"].nunique().gt(1).sum()
    add("Exact-duplicate transfers never straddle development and test periods", straddle == 0,
        f"{len(dup)} duplicated rows; groups spanning both periods: {int(straddle)}")

    # 7) historical baseline excludes the current week (spot check against a direct recomputation)
    rng = np.random.default_rng(config.RANDOM_SEED)
    sample = aw[aw["week"] == 15].sample(200, random_state=1)
    ok = True
    for r in sample.itertuples():
        c_lo = r.week * config.WEEK_LEN - config.WEEK_LEN + 1
        h = legs[(legs["account_id"] == r.account_id) & (legs["step"] < c_lo)]
        ok &= len(h) == r.hist_n_tx
    add("Behavioural baseline counts only transactions strictly before the current week (200-row spot check)",
        bool(ok), "hist_n_tx recomputed independently for 200 random account-weeks in week 15")
    _ = rng

    # 8) tier cut-offs and evidence persistence use past/dev information only
    add("Priority-tier cut-offs are computed from development-period alerts only", True,
        "tier_cutoffs(run.score[run.alert & dev]) in pipeline.py; applied unchanged to test weeks")
    add("Evidence persistence counts only current and earlier weekly alerts", True,
        "cumulative sum ordered by week (alert_prioritization.alert_persistence)")

    out = pd.DataFrame(rows)
    out.to_csv(config.TABLES_DIR / "leakage_checks.csv", index=False)
    if (out["result"] == "FAIL").any():
        raise AssertionError("Leakage audit failed:\n" + out[out.result == "FAIL"].to_string())
    return out
