"""Labelled suspicious-activity analytics (the "fraud analytics" phase).

IMPORTANT SCOPE NOTE: the AMLSim sample has NO payment-fraud label, no merchant,
card, device, channel or loss field. Its only label (`isFraud` in the raw file,
`is_suspicious` here) marks accounts that participate in an injected AML
typology. This module therefore applies fraud-analytics METHODS (group
comparison, anomaly detection, value-at-risk proxies) to that label. It does not
claim supervised payment-fraud detection and never estimates monetary loss.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from . import config, stats_tests


def group_comparisons(profiles: pd.DataFrame) -> pd.DataFrame:
    """Descriptive comparison of full-period account behaviour (EDA, not detection)."""
    p = profiles[profiles["tx_count"] > 0]
    cols = ["tx_count", "median_amount", "max_amount", "unique_counterparties", "tx_per_counterparty",
            "in_counterparties", "out_counterparties", "active_weeks", "total_value"]
    rows = []
    for c in cols:
        a = p.loc[p["is_suspicious"] == 1, c]
        b = p.loc[p["is_suspicious"] == 0, c]
        rows.append({"metric": c, "labelled_median": a.median(), "unlabelled_median": b.median(),
                     "labelled_mean": a.mean(), "unlabelled_mean": b.mean(),
                     "labelled_p90": a.quantile(0.9), "unlabelled_p90": b.quantile(0.9)})
    return pd.DataFrame(rows)


def value_proxies(accounts: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    """Measurable value proxies (NOT losses)."""
    lab = set(accounts.loc[accounts["is_suspicious"] == 1, "account_id"])
    both = tx["src"].isin(lab) & tx["dst"].isin(lab)
    touch = tx["src"].isin(lab) | tx["dst"].isin(lab)
    total = tx["amount"].sum()
    rows = [
        ("total transfer value", total, 1.0),
        ("value between two labelled accounts (typology proxy)", tx.loc[both, "amount"].sum(),
         tx.loc[both, "amount"].sum() / total),
        ("value touching >= 1 labelled account", tx.loc[touch, "amount"].sum(), tx.loc[touch, "amount"].sum() / total),
        ("transfers between two labelled accounts", int(both.sum()), both.mean()),
        ("mean amount, typology-proxy transfers", tx.loc[both, "amount"].mean(), np.nan),
        ("mean amount, other transfers", tx.loc[~both, "amount"].mean(), np.nan),
    ]
    return pd.DataFrame(rows, columns=["measure", "value", "share_of_total"])


def isolation_forest_scores(train: pd.DataFrame, apply_to: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Unsupervised anomaly score (higher = more anomalous). Trained on development weeks only."""
    def prep(df):
        return np.log1p(df[features].fillna(0).clip(lower=0)).to_numpy()
    iso = IsolationForest(n_estimators=300, random_state=config.RANDOM_SEED).fit(prep(train))
    return -iso.score_samples(prep(apply_to))


def hypothesis_h1_h2(profiles_dev: pd.DataFrame) -> list[dict]:
    """H1 amounts, H2 velocity - account level, development period only."""
    lab = profiles_dev["is_suspicious"] == 1
    out = []
    r = stats_tests.mann_whitney(profiles_dev.loc[lab, "median_amount"], profiles_dev.loc[~lab, "median_amount"])
    r.update(hypothesis="H1", description="Labelled accounts' typical (median) transfer amount differs from unlabelled accounts'",
             test="Mann-Whitney U (two-sided)", unit="account", metric="median transfer amount, dev period")
    out.append(r)
    r = stats_tests.mann_whitney(profiles_dev.loc[lab, "weekly_tx"], profiles_dev.loc[~lab, "weekly_tx"],
                                 alternative="greater")
    r.update(hypothesis="H2", description="Labelled accounts transact at higher velocity (transfers per active week)",
             test="Mann-Whitney U (one-sided, greater)", unit="account", metric="transfers per active week, dev period")
    out.append(r)
    return out
