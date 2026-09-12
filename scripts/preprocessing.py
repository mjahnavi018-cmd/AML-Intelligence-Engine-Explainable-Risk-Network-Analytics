"""Preprocessing: raw AMLSim CSVs -> cleaned analytical tables (+ SQLite).

Cleaning rules (all documented in reports/data_audit.md):
* columns renamed to analyst-friendly names; raw values unchanged
* a deterministic transaction id is assigned from the raw row order
* exact duplicate rows are KEPT (no transaction id exists to prove they are
  errors) and flagged with `dup_rank` / `is_repeat_copy`
* self-transfers are KEPT and flagged; they are excluded from counterparty and
  network features downstream
* `fraudStep` is carried as `fraud_step_raw` for transparency only and is never
  used as a feature or in evaluation (semantics unverifiable)
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from . import config


def clean(nodes: pd.DataFrame, tx: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    accounts = nodes.rename(columns={"nodeid": "account_id", "isFraud": "is_suspicious",
                                     "fraudStep": "fraud_step_raw"}).copy()
    accounts["is_suspicious"] = accounts["is_suspicious"].astype(int)

    t = tx.rename(columns={"sourceNodeId": "src", "targetNodeId": "dst", "value": "amount",
                           "time": "step"}).copy()
    t.insert(0, "tx_id", range(1, len(t) + 1))
    t["step"] = t["step"].astype(int)
    t["week"] = (t["step"] - 1) // config.WEEK_LEN + 1
    t["is_self_transfer"] = (t["src"] == t["dst"]).astype(int)
    t["dup_rank"] = t.groupby(["src", "dst", "amount", "step"]).cumcount() + 1
    t["is_repeat_copy"] = (t["dup_rank"] > 1).astype(int)

    # activity summary on the account table (full-period, DESCRIPTIVE ONLY: never
    # used by the point-in-time detection features)
    first = pd.concat([t[["src", "step"]].rename(columns={"src": "a"}),
                       t[["dst", "step"]].rename(columns={"dst": "a"})])
    span = first.groupby("a")["step"].agg(first_step="min", last_step="max")
    accounts = accounts.merge(span, left_on="account_id", right_index=True, how="left")
    return accounts, t


def evaluation_labels(accounts: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    """Transaction-level PROXY label used only for descriptive evaluation.

    A transfer between two labelled accounts is the closest available proxy for a
    typology transaction. It is never a model input.
    """
    lab = set(accounts.loc[accounts["is_suspicious"] == 1, "account_id"])
    out = tx[["tx_id"]].copy()
    out["src_suspicious"] = tx["src"].isin(lab).astype(int)
    out["dst_suspicious"] = tx["dst"].isin(lab).astype(int)
    out["proxy_typology_tx"] = (out["src_suspicious"] & out["dst_suspicious"]).astype(int)
    return out


def write_sqlite(tables: dict[str, pd.DataFrame], db_path: Path = config.DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        for name, df in tables.items():
            df.to_sql(name, con, if_exists="replace", index=False)
        con.executescript(
            """
            CREATE INDEX IF NOT EXISTS ix_tx_src ON transactions(src, step);
            CREATE INDEX IF NOT EXISTS ix_tx_dst ON transactions(dst, step);
            CREATE INDEX IF NOT EXISTS ix_tx_step ON transactions(step);
            CREATE INDEX IF NOT EXISTS ix_acc ON accounts(account_id);
            """
        )
