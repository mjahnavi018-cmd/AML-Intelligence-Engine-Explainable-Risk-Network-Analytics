"""Execute sql/analytical_queries.sql against the SQLite database, export each
result to outputs/tables/sql/, and cross-check key numbers against pandas."""
from __future__ import annotations

import re
import sqlite3
import time

import pandas as pd

from . import config


def parse_queries(path=config.SQL_DIR / "analytical_queries.sql") -> dict[str, str]:
    text = path.read_text()
    parts = re.split(r"^-- name:\s*(\S+)\s*$", text, flags=re.M)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


def run_queries(db_path=config.DB_PATH) -> dict[str, pd.DataFrame]:
    out_dir = config.TABLES_DIR / "sql"
    out_dir.mkdir(parents=True, exist_ok=True)
    res, timing = {}, []
    with sqlite3.connect(db_path) as con:
        for name, q in parse_queries().items():
            t0 = time.time()
            df = pd.read_sql_query(q, con)
            timing.append({"query": name, "rows": len(df), "seconds": round(time.time() - t0, 3)})
            df.to_csv(out_dir / f"{name}.csv", index=False)
            res[name] = df
    pd.DataFrame(timing).to_csv(out_dir / "_query_log.csv", index=False)
    return res


def crosscheck(res: dict[str, pd.DataFrame], ctx: dict) -> pd.DataFrame:
    tx, acc, run, queue = ctx["tx"], ctx["accounts"], ctx["run"], ctx["queue"]
    ov = res["q01_data_overview"].iloc[0]
    rows = [
        ("transactions", ov["transactions"], len(tx)),
        ("total value", ov["total_value"], round(tx["amount"].sum(), 2)),
        ("accounts", ov["accounts"], len(acc)),
        ("labelled accounts", ov["labelled_accounts"], int(acc["is_suspicious"].sum())),
        ("exact duplicate copies", ov["exact_duplicate_copies"], int(tx["is_repeat_copy"].sum())),
        ("self transfers", ov["self_transfers"], int(tx["is_self_transfer"].sum())),
    ]
    q9 = res["q09_alert_volume_by_rule"].set_index("period")
    for period in ("dev", "test"):
        m = run.masks[period]
        rows.append((f"integrated alerts ({period})", q9.at[period, "integrated_alerts"], int(run.alert[m].sum())))
        rows.append((f"VEL-01 alerts ({period})", q9.at[period, "VEL-01"], int(run.sig.loc[m, "VEL-01"].sum())))
    rows.append(("alert queue size", int(res["q10_alert_workload_by_week_and_tier"]["alerts"].sum()), len(queue)))
    q14 = res["q14_alert_label_overlap_by_tier"].set_index("risk_level")
    tv = ctx["tv"].set_index("risk_level")
    for t in q14.index:
        rows.append((f"tier precision {t} (held-out test)", round(q14.at[t, "precision"], 3), round(tv.at[t, "precision"], 3)))
    wk = res["q02_weekly_volume_trend"].set_index("week")["tx"]
    rows.append(("weekly volume series", int(wk.sum()), int(ctx["weekly"]["transactions"].sum())))
    q13 = res["q13_reciprocal_relationships"].iloc[0]
    import networkx as nx
    rows.append(("reciprocity %", round(q13["reciprocity_pct"], 3), round(100 * nx.reciprocity(ctx["G"]), 3)))
    df = pd.DataFrame(rows, columns=["check", "sql_value", "python_value"])
    df["match"] = [abs(float(a) - float(b)) <= 1e-6 * max(1, abs(float(b))) + 1e-3 for a, b in zip(df.sql_value, df.python_value)]
    return df


def run_all(ctx: dict) -> pd.DataFrame:
    res = run_queries()
    qc = crosscheck(res, ctx)
    qc.to_csv(config.TABLES_DIR / "qc_sql_vs_python.csv", index=False)
    if not qc["match"].all():
        raise AssertionError("SQL/Python cross-check failed:\n" + qc[~qc["match"]].to_string())
    print(f"SQL: {len(res)} queries executed; {len(qc)} cross-checks passed", flush=True)
    return qc
