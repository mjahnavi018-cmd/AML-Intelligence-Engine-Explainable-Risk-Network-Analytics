"""Explainable risk-signal engine and integrated risk score.

Design
------
1. Signal catalogue: each signal is a transparent rule on one point-in-time
   feature. Thresholds are EMPIRICAL QUANTILES of the development period
   (weeks 5-12) computed WITHOUT labels, or a structural threshold (e.g. "at
   least one time-respecting cycle"). Thresholds are frozen and applied
   unchanged to the held-out test period (weeks 13-21).
2. Standardisation: features are log-transformed (counts) and standardised
   against the frozen development distribution.
3. Integrated score: a logistic model whose weights are constrained to be
   NON-NEGATIVE (more extreme evidence can never lower risk; this also stops
   the model from exploiting the simulator's "recurring counterparty" artifact
   through negative weights). Weights are fitted on development-period data of
   a random half of the accounts (group A) only. Per-feature contributions
   (weight x standardised value) give the ranked "why flagged" evidence. The
   0-100 risk score is the percentile of the weekly logit among development
   account-weeks.
4. Evidence strength is reported separately from risk: the number of
   independent signal FAMILIES that fired.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import config

# ---------------------------------------------------------------------------
# Signal catalogue
# ---------------------------------------------------------------------------
SIGNALS = [
    dict(id="TXN-01", name="Large transaction", family="transaction", feature="max_amt",
         rule="quantile_tx", q=config.SIGNAL_QUANTILE, in_framework=True,
         rationale="Largest single transfer this week is in the top 1% of all development-period transfer amounts."),
    dict(id="VEL-01", name="High absolute velocity", family="velocity", feature="n_tx",
         rule="quantile", q=config.SIGNAL_QUANTILE, in_framework=True,
         rationale="Weekly transaction count in the top 1% of development account-weeks."),
    dict(id="BEH-01", name="Velocity spike vs own baseline", family="behavioral", feature="velocity_ratio",
         rule="quantile", q=config.SIGNAL_QUANTILE, in_framework=True, requires_history=True,
         rationale="(this week's count + 1) / (historical weekly mean + 1) in the top 1%; account must have history."),
    dict(id="BEH-02", name="New-counterparty burst", family="behavioral", feature="new_cp",
         rule="quantile", q=config.SIGNAL_QUANTILE, in_framework=True,
         rationale="Number of counterparties never seen before this week in the top 1%."),
    dict(id="BEH-03", name="Amount escalation vs own baseline", family="behavioral", feature="amount_log_ratio",
         rule="quantile", q=config.SIGNAL_QUANTILE, in_framework=False, artifact=True,
         rationale="log(mean amount this week / historical median amount) in the top 1% (larger than usual). "
                   "Requires >= 5 historical transactions. QUARANTINED: perfect development precision traced to "
                   "artifact A1 (tiny typology amounts deflate the account's own baseline, so ordinary amounts "
                   "look like escalation)."),
    dict(id="BEH-04", name="Dormant-account reactivation", family="behavioral", feature="is_reactivated",
         rule="binary", in_framework=False,
         rationale="Activity after > 4 empty weeks. Evaluated but EXCLUDED: precision below the base rate in development."),
    dict(id="AML-01", name="Rapid pass-through", family="aml_pattern", feature="rapid_movement",
         rule="quantile", q=config.SIGNAL_QUANTILE, in_framework=True,
         rationale="min(in/out balance of 28-day flows, share of outflow sent within 3 days of an inflow) in the top 1%."),
    dict(id="AML-02", name="Repeated transfers to same receiver", family="aml_pattern", feature="max_repeat_same_cp",
         rule="fixed", threshold=2, in_framework=False, artifact=True,
         rationale="Two or more transfers to the same receiver in one week (structuring-like). EXCLUDED from the "
                   "framework: in AMLSim background activity never repeats a receiver within a week, so this rule's "
                   "perfect precision is a simulator artifact, not transferable evidence."),
    dict(id="AML-03", name="Circular flow", family="aml_pattern", feature="cycle_count",
         rule="fixed", threshold=1, in_framework=True,
         rationale="Account lies on >= 1 time-respecting directed cycle (3-6 accounts) within 28 days."),
    dict(id="NET-01", name="Fan-in collector", family="network", feature="fan_in_28",
         rule="quantile", q=config.SIGNAL_QUANTILE, in_framework=True,
         rationale="Distinct senders over 28 days in the top 1% (many-to-one)."),
    dict(id="NET-02", name="Fan-out distributor", family="network", feature="fan_out_28",
         rule="quantile", q=config.SIGNAL_QUANTILE, in_framework=True,
         rationale="Distinct receivers over 28 days in the top 1% (one-to-many)."),
    dict(id="NET-03", name="Flow centrality (PageRank)", family="network", feature="pagerank_28",
         rule="quantile", q=config.SIGNAL_QUANTILE, in_framework=True,
         rationale="Amount-weighted PageRank in the 28-day graph in the top 1% (where money accumulates)."),
]
SIGNAL_BY_ID = {s["id"]: s for s in SIGNALS}
FRAMEWORK_SIGNALS = [s["id"] for s in SIGNALS if s["in_framework"]]
SCORE_FEATURES = list(dict.fromkeys(SIGNAL_BY_ID[i]["feature"] for i in FRAMEWORK_SIGNALS))
FAMILIES = ["transaction", "velocity", "behavioral", "aml_pattern", "network"]


def calibrate_thresholds(aw_dev: pd.DataFrame, tx_dev: pd.DataFrame, q: float | None = None) -> dict[str, float]:
    """Label-free thresholds from the development period only."""
    th = {}
    for s in SIGNALS:
        qq = q if (q is not None and s["rule"] in ("quantile", "quantile_tx")) else s.get("q")
        if s["rule"] == "quantile_tx":
            th[s["id"]] = float(tx_dev["amount"].quantile(qq))
        elif s["rule"] == "quantile":
            vals = aw_dev[s["feature"]]
            if s.get("requires_history"):
                vals = vals[aw_dev["hist_n_tx"] > 0]
            th[s["id"]] = float(vals.dropna().quantile(qq))
        elif s["rule"] == "fixed":
            th[s["id"]] = float(s["threshold"])
        else:  # binary
            th[s["id"]] = 1.0
    return th


def evaluate_signals(aw: pd.DataFrame, th: dict[str, float]) -> pd.DataFrame:
    """Boolean matrix (account-week x signal)."""
    out = {}
    for s in SIGNALS:
        x = aw[s["feature"]]
        hit = x >= th[s["id"]]
        if s["rule"] == "quantile" and th[s["id"]] <= 0:   # guard: a zero threshold would flag everything
            hit = x > th[s["id"]]
        if s.get("requires_history"):
            hit &= aw["hist_n_tx"] > 0
        out[s["id"]] = hit.fillna(False).astype(bool).values
    return pd.DataFrame(out, index=aw.index)


def family_hits(sig: pd.DataFrame, signal_ids: list[str] | None = None) -> pd.DataFrame:
    ids = signal_ids or FRAMEWORK_SIGNALS
    return pd.DataFrame({fam: sig[[i for i in ids if SIGNAL_BY_ID[i]["family"] == fam]].any(axis=1)
                         for fam in FAMILIES if any(SIGNAL_BY_ID[i]["family"] == fam for i in ids)})


class PercentileTransformer:
    """Empirical CDF against a frozen reference sample (development period)."""

    def fit(self, ref: pd.DataFrame, features: list[str]):
        self.features = features
        self.ref = {c: np.sort(ref[c].fillna(0).to_numpy(float)) for c in features}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({c: np.searchsorted(self.ref[c], df[c].fillna(0).to_numpy(float), side="right")
                             / len(self.ref[c]) for c in self.features}, index=df.index)


def fit_nonneg_logit(X: np.ndarray, y: np.ndarray, l2: float = 1e-3) -> tuple[float, np.ndarray]:
    """Logistic regression with non-negative weights (L-BFGS-B), small L2 penalty."""
    n, k = X.shape

    def loss(theta):
        b, w = theta[0], theta[1:]
        z = b + X @ w
        ll = np.logaddexp(0, z) - y * z
        return ll.mean() + l2 * (w @ w)

    def grad(theta):
        b, w = theta[0], theta[1:]
        p = 1 / (1 + np.exp(-(b + X @ w)))
        r = p - y
        return np.concatenate([[r.mean()], X.T @ r / n + 2 * l2 * w])

    bounds = [(None, None)] + [(0, None)] * k
    res = minimize(loss, np.zeros(k + 1), jac=grad, method="L-BFGS-B", bounds=bounds)
    if not res.success:
        raise RuntimeError(res.message)
    return float(res.x[0]), res.x[1:]


class IntegratedRiskModel:
    """Non-negative logistic weighting of standardised framework features.

    * transform: log1p for non-negative count/ratio features (identity for the
      signed amount log-ratio), then standardise with development-period mean/sd
    * fit: accounts of group A, features = each account's maximum over the
      development weeks (an account is labelled, not a week), weights >= 0
    * weekly logit = intercept + sum(w_i * z_i); risk_score (0-100) = percentile of
      the weekly logit among development account-weeks (frozen reference)
    """

    SIGNED = {"amount_log_ratio"}

    def __init__(self, features: list[str] | None = None, l2: float = 1e-3):
        self.features = features or SCORE_FEATURES
        self.l2 = l2

    def _raw(self, aw: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for c in self.features:
            x = aw[c].fillna(0).astype(float)
            out[c] = x if c in self.SIGNED else np.log1p(x.clip(lower=0))
        return pd.DataFrame(out, index=aw.index)

    def standardise(self, aw: pd.DataFrame) -> pd.DataFrame:
        return (self._raw(aw) - self.mu) / self.sd

    def fit(self, aw_dev: pd.DataFrame, train_mask: pd.Series, labels: pd.Series):
        raw = self._raw(aw_dev)
        self.mu, self.sd = raw.mean(), raw.std().replace(0, 1)
        Z = self.standardise(aw_dev[train_mask])
        Z["account_id"] = aw_dev.loc[train_mask, "account_id"].values
        A = Z.groupby("account_id")[self.features].max()
        y = labels.reindex(A.index).to_numpy(float)
        self.intercept, self.weights = fit_nonneg_logit(A.to_numpy(), y, self.l2)
        self.ref_logit = np.sort(self.logit(aw_dev))
        return self

    def contributions(self, aw: pd.DataFrame) -> pd.DataFrame:
        return self.standardise(aw) * self.weights

    def logit(self, aw: pd.DataFrame) -> np.ndarray:
        return self.intercept + self.contributions(aw).sum(axis=1).to_numpy()

    def risk_score(self, aw: pd.DataFrame) -> np.ndarray:
        """0-100: share of development account-weeks with a lower logit."""
        return 100 * np.searchsorted(self.ref_logit, self.logit(aw), side="right") / len(self.ref_logit)

    def to_dict(self) -> dict:
        return {"intercept": self.intercept,
                "weights": dict(zip(self.features, map(float, self.weights))),
                "standardisation_mean": {k: float(v) for k, v in self.mu.items()},
                "standardisation_sd": {k: float(v) for k, v in self.sd.items()}}

    def save(self, path) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)


EVIDENCE_LEVELS = ["Limited", "Moderate", "Strong"]


def evidence_strength(count: pd.Series) -> pd.Series:
    """Confidence is kept separate from risk. `count` is the evidence measure chosen in
    development (see pipeline: signal-family breadth vs alert persistence): 1 -> Limited,
    2 -> Moderate, >= 3 -> Strong."""
    return pd.cut(count, bins=[-1, 1, 2, 10_000], labels=EVIDENCE_LEVELS).astype(str)


QUARANTINE_MIN_ALERTS = 50
QUARANTINE_PRECISION = 0.99


def artifact_quarantine(sig: pd.DataFrame, labels: pd.Series, dev_mask: pd.Series) -> pd.DataFrame:
    """Flag signals whose development precision is (near-)perfect on >= 50 alerts.

    In a simulator, a perfect rule almost always encodes how the data were
    generated rather than behaviour that transfers to a real bank. Flagged
    signals are investigated and kept out of the framework (static
    `in_framework=False`); the pipeline checks the static list matches.
    """
    rows = []
    for s in SIGNALS:
        hit = sig[s["id"]] & dev_mask
        n = int(hit.sum())
        prec = float(labels[hit].mean()) if n else float("nan")
        flag = n >= QUARANTINE_MIN_ALERTS and prec >= QUARANTINE_PRECISION
        rows.append({"signal": s["id"], "dev_alerts": n, "dev_precision": prec, "quarantine_flag": flag,
                     "in_framework": s["in_framework"], "consistent": not (flag and s["in_framework"])})
    return pd.DataFrame(rows)
