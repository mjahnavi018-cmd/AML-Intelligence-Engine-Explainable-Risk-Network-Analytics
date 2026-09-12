"""Data audit: every check produces a row (check, metric, value, status, note).

Status meanings
    PASS  - expectation met
    WARN  - usable, but a limitation/artifact that must be documented
    INFO  - descriptive fact
    FAIL  - blocks the pipeline (the pipeline raises if any FAIL exists)
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

EXPECTED_NODE_COLS = ["nodeid", "isFraud", "init_balance", "fraudStep"]
EXPECTED_TX_COLS = ["sourceNodeId", "targetNodeId", "value", "time"]


def _row(rows, section, check, value, status, note=""):
    rows.append({"section": section, "check": check, "value": value, "status": status, "note": note})


def audit(nodes: pd.DataFrame, tx: pd.DataFrame, metadata: str, dataset: str) -> pd.DataFrame:
    rows: list[dict] = []
    S = "schema"
    _row(rows, S, "nodes columns", ",".join(nodes.columns),
         "PASS" if list(nodes.columns) == EXPECTED_NODE_COLS else "FAIL")
    _row(rows, S, "transactions columns", ",".join(tx.columns),
         "PASS" if list(tx.columns) == EXPECTED_TX_COLS else "FAIL")
    _row(rows, S, "nodes dtypes", ",".join(f"{c}:{t}" for c, t in nodes.dtypes.astype(str).items()), "INFO")
    _row(rows, S, "transactions dtypes", ",".join(f"{c}:{t}" for c, t in tx.dtypes.astype(str).items()), "INFO")

    V = "volume"
    _row(rows, V, "node rows", len(nodes), "INFO")
    _row(rows, V, "transaction rows", len(tx), "INFO")
    m_nodes = re.search(r"nodes:\s*([\d,]+)\s*\((\d+) fraud", metadata)
    m_tx = re.search(r"transactions:\s*([\d,]+)", metadata)
    if m_nodes:
        n_meta = int(m_nodes.group(1).replace(",", ""))
        _row(rows, V, "node rows vs metadata.txt", f"{len(nodes)} vs {n_meta}",
             "PASS" if n_meta == len(nodes) else "FAIL")
        f_meta = int(m_nodes.group(2))
        f_obs = int(nodes["isFraud"].sum())
        _row(rows, V, "labelled accounts vs metadata.txt", f"{f_obs} vs {f_meta}",
             "PASS" if f_meta == f_obs else "WARN",
             "" if f_meta == f_obs else "metadata.txt differs from nodes.csv; nodes.csv is treated as authoritative")
    if m_tx:
        t_meta = int(m_tx.group(1).replace(",", ""))
        _row(rows, V, "transaction rows vs metadata.txt", f"{len(tx)} vs {t_meta}",
             "PASS" if t_meta == len(tx) else "FAIL")

    M = "missingness"
    _row(rows, M, "missing values (nodes)", int(nodes.isna().sum().sum()),
         "PASS" if nodes.isna().sum().sum() == 0 else "FAIL")
    _row(rows, M, "missing values (transactions)", int(tx.isna().sum().sum()),
         "PASS" if tx.isna().sum().sum() == 0 else "FAIL")

    K = "keys_and_integrity"
    _row(rows, K, "duplicate account ids", int(nodes["nodeid"].duplicated().sum()),
         "PASS" if nodes["nodeid"].duplicated().sum() == 0 else "FAIL")
    ids = set(nodes["nodeid"])
    orphan_src = int((~tx["sourceNodeId"].isin(ids)).sum())
    orphan_dst = int((~tx["targetNodeId"].isin(ids)).sum())
    _row(rows, K, "transactions with unknown sender", orphan_src, "PASS" if orphan_src == 0 else "FAIL")
    _row(rows, K, "transactions with unknown receiver", orphan_dst, "PASS" if orphan_dst == 0 else "FAIL")
    active = set(tx["sourceNodeId"]) | set(tx["targetNodeId"])
    _row(rows, K, "unique sending accounts", tx["sourceNodeId"].nunique(), "INFO")
    _row(rows, K, "unique receiving accounts", tx["targetNodeId"].nunique(), "INFO")
    _row(rows, K, "accounts with no transactions", len(ids - active), "WARN" if ids - active else "PASS",
         "kept in account table; excluded from behavioural features")
    self_tx = int((tx["sourceNodeId"] == tx["targetNodeId"]).sum())
    _row(rows, K, "self-transfers (sender == receiver)", self_tx, "WARN" if self_tx else "PASS",
         "kept and flagged; excluded from counterparty and network features")
    pairs = tx.groupby(["sourceNodeId", "targetNodeId"]).size()
    _row(rows, K, "unique directed account pairs", len(pairs), "INFO")

    D = "duplicates"
    dup_mask = tx.duplicated(keep=False)
    n_dup_extra = int(tx.duplicated().sum())
    lab = set(nodes.loc[nodes["isFraud"] == 1, "nodeid"])
    both_lab = tx["sourceNodeId"].isin(lab) & tx["targetNodeId"].isin(lab)
    share = float(both_lab[dup_mask].mean()) if dup_mask.any() else float("nan")
    _row(rows, D, "exact duplicate rows (extra copies)", n_dup_extra, "WARN" if n_dup_extra else "PASS",
         "no transaction id exists, so identical (sender, receiver, amount, step) rows cannot be proven to be "
         "errors; they are kept as repeated transfers and flagged")
    _row(rows, D, "share of duplicated rows between two labelled accounts", round(share, 4), "WARN",
         "if ~1.0 the duplicates are a simulator artifact of the typology generator, not organic behaviour")

    T = "time"
    _row(rows, T, "step min", int(tx["time"].min()), "INFO")
    _row(rows, T, "step max", int(tx["time"].max()), "INFO")
    steps = np.arange(tx["time"].min(), tx["time"].max() + 1)
    missing_steps = sorted(set(steps) - set(tx["time"]))
    _row(rows, T, "steps with zero transactions", len(missing_steps), "PASS" if not missing_steps else "WARN")
    _row(rows, T, "non-integer steps", int((tx["time"] % 1 != 0).sum()), "PASS" if (tx["time"] % 1 == 0).all() else "FAIL")
    per_step = tx.groupby("time").size()
    _row(rows, T, "transactions per step (min / median / max)",
         f"{per_step.min()} / {int(per_step.median())} / {per_step.max()}", "WARN",
         "volume ramps up and down over the simulation (start/end effects); use relative, not absolute, velocity")
    _row(rows, T, "timestamp granularity", "integer simulation step (no clock time, no calendar date)", "WARN",
         "time-of-day and day-of-week analysis is impossible")

    A = "amounts"
    _row(rows, A, "amount min", float(tx["value"].min()), "INFO")
    _row(rows, A, "amount max", float(tx["value"].max()), "INFO")
    _row(rows, A, "zero or negative amounts", int((tx["value"] <= 0).sum()),
         "PASS" if (tx["value"] <= 0).sum() == 0 else "FAIL")
    lt100 = tx["value"] < 100
    _row(rows, A, "amounts below 100 (count)", int(lt100.sum()), "INFO")
    _row(rows, A, "share of <100 amounts between two labelled accounts",
         round(float(both_lab[lt100].mean()), 4) if lt100.any() else float("nan"), "WARN",
         "AMLSim draws background amounts from a range starting at 100; if this share is ~1.0 the <100 band is a "
         "generation artifact and must NOT be used as a detection rule")
    _row(rows, A, "amounts with more than 2 decimals", int(((tx["value"] * 100).round(6) % 1 != 0).sum()),
         "PASS" if ((tx["value"] * 100).round(6) % 1 == 0).all() else "WARN")

    L = "labels"
    n_lab = int(nodes["isFraud"].sum())
    _row(rows, L, "labelled (isFraud=1) accounts", n_lab, "INFO")
    _row(rows, L, "labelled share of accounts", round(n_lab / len(nodes), 4), "WARN",
         "far higher than any realistic prevalence; precision must be re-expressed at realistic prevalence")
    _row(rows, L, "transactions touching >=1 labelled account", int((tx["sourceNodeId"].isin(lab) | tx["targetNodeId"].isin(lab)).sum()), "INFO")
    _row(rows, L, "transactions between two labelled accounts", int(both_lab.sum()), "INFO")
    fs = nodes["fraudStep"]
    _row(rows, L, "unlabelled accounts with fraudStep >= 0", int(((nodes["isFraud"] == 0) & (fs >= 0)).sum()), "INFO")
    _row(rows, L, "labelled accounts with fraudStep = -1", int(((nodes["isFraud"] == 1) & (fs < 0)).sum()), "WARN",
         "fraudStep semantics are undocumented and inconsistent with the label; column excluded from all features "
         "and from evaluation")
    _row(rows, L, "transaction-level label", "not provided", "WARN",
         "only account-level labels exist; evaluation is account-level (account-week alerts inherit the account label)")

    B = "balances"
    _row(rows, B, "init_balance min / max", f"{nodes['init_balance'].min()} / {nodes['init_balance'].max()}", "INFO")
    _row(rows, B, "running balance consistency", "not testable", "WARN",
         "only an initial balance per account is published; no post-transaction balances exist")

    out = pd.DataFrame(rows)
    out.insert(0, "dataset", dataset)
    return out


def assert_no_failures(report: pd.DataFrame) -> None:
    fails = report[report["status"] == "FAIL"]
    if len(fails):
        raise ValueError("Data audit FAILED:\n" + fails.to_string())
