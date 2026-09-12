"""Evaluation utilities: alert-level and account-level metrics, ranking metrics,
prevalence adjustment, bootstrap confidence intervals.

Ground truth: the AMLSim account label `is_suspicious` (member of an injected
AML typology). There is no transaction-level label, no investigation outcome,
and no loss amount, so:
  * alert precision = share of alerts (account-weeks) raised on labelled accounts
  * account recall  = share of labelled accounts active in the period that
                      received at least one alert
  * FPR             = share of unlabelled active accounts that received an alert
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def alert_metrics(frame: pd.DataFrame, alert: pd.Series, name: str, n_tx: int, n_weeks: int) -> dict:
    """frame: account-week rows of the evaluation population with column is_suspicious."""
    alert = alert.reindex(frame.index).fillna(False).astype(bool)
    acc = frame.groupby("account_id")["is_suspicious"].first()
    acc_alert = alert.groupby(frame["account_id"]).any()
    tp = int((acc_alert & (acc == 1)).sum())
    fp = int((acc_alert & (acc == 0)).sum())
    fn = int((~acc_alert & (acc == 1)).sum())
    tn = int((~acc_alert & (acc == 0)).sum())
    n_alerts = int(alert.sum())
    alert_prec = float(frame.loc[alert, "is_suspicious"].mean()) if n_alerts else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    return {
        "approach": name,
        "alerts": n_alerts,
        "alerts_per_week": n_alerts / n_weeks,
        "alerts_per_1000_tx": 1000 * n_alerts / n_tx,
        "alerted_accounts": tp + fp,
        "alert_precision": alert_prec,
        "account_precision": prec,
        "account_recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec and rec and prec + rec else np.nan,
        "false_positive_accounts": fp,
        "fpr_accounts": fp / (fp + tn) if fp + tn else np.nan,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def account_scores(frame: pd.DataFrame, score: pd.Series) -> pd.DataFrame:
    """Account-level score = maximum weekly score in the period (a 'worst week' view)."""
    s = frame.assign(_s=score.reindex(frame.index).to_numpy())
    return s.groupby("account_id").agg(score=("_s", "max"), y=("is_suspicious", "first"))


def ranking_metrics(acc_scores: pd.DataFrame, name: str) -> dict:
    y, s = acc_scores["y"].to_numpy(), acc_scores["score"].to_numpy()
    out = {"approach": name, "roc_auc": roc_auc_score(y, s), "pr_auc": average_precision_score(y, s),
           "base_rate": y.mean(), "n_accounts": len(y)}
    order = np.argsort(-s, kind="mergesort")
    for pct in (1, 5, 10, 20):
        k = max(1, int(round(len(y) * pct / 100)))
        top = y[order[:k]]
        out[f"precision_top{pct}pct"] = top.mean()
        out[f"recall_top{pct}pct"] = top.sum() / y.sum()
    return out


def topk_curve(acc_scores: pd.DataFrame, name: str, grid=None) -> pd.DataFrame:
    grid = grid if grid is not None else np.unique(np.r_[np.linspace(0.005, 0.2, 40), 0.01, 0.05, 0.1])
    y, s = acc_scores["y"].to_numpy(), acc_scores["score"].to_numpy()
    order = np.argsort(-s, kind="mergesort")
    rows = []
    for g in grid:
        k = max(1, int(round(len(y) * g)))
        top = y[order[:k]]
        rows.append({"approach": name, "share_reviewed": g, "accounts_reviewed": k,
                     "precision": top.mean(), "recall": top.sum() / y.sum()})
    return pd.DataFrame(rows)


def decile_table(acc_scores: pd.DataFrame) -> pd.DataFrame:
    s = acc_scores.copy()
    s["decile"] = pd.qcut(s["score"].rank(method="first", ascending=False), 10, labels=range(1, 11))
    t = s.groupby("decile", observed=True).agg(accounts=("y", "size"), labelled=("y", "sum"))
    t["labelled_rate"] = t["labelled"] / t["accounts"]
    t["share_of_all_labelled"] = t["labelled"] / t["labelled"].sum()
    t["cumulative_capture"] = t["share_of_all_labelled"].cumsum()
    return t.reset_index()


def precision_at_prevalence(recall: float, fpr: float, prevalence: float) -> float:
    """PPV = TPR*p / (TPR*p + FPR*(1-p)). Projects precision to a different base rate
    assuming the detection (TPR) and false-alarm (FPR) rates transfer."""
    num = recall * prevalence
    den = num + fpr * (1 - prevalence)
    return num / den if den else np.nan


def bootstrap_diff(acc_a: pd.DataFrame, acc_b: pd.DataFrame, metric=average_precision_score,
                   n_boot: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    """Paired account bootstrap CI for metric(a) - metric(b). Both frames indexed by account."""
    j = acc_a.join(acc_b["score"].rename("score_b"), how="inner")
    y, sa, sb = j["y"].to_numpy(), j["score"].to_numpy(), j["score_b"].to_numpy()
    rng = np.random.default_rng(seed)
    diffs = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(metric(y[idx], sa[idx]) - metric(y[idx], sb[idx]))
    point = metric(y, sa) - metric(y, sb)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(point), float(lo), float(hi)
