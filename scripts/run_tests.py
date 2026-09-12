"""Validation suite. Run after `python run_pipeline.py`:  python tests/run_tests.py
(Functions are named test_* so `pytest tests/run_tests.py` also works if pytest is installed.)

Several checks recompute headline numbers through an INDEPENDENT path (raw CSVs,
the SQLite alert queue) and compare them with outputs/tables/key_metrics.json.
"""
import json
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config, ingestion  # noqa: E402

T = config.TABLES_DIR
KM = json.loads((T / "key_metrics.json").read_text())


def test_raw_checksums():
    assert all(ingestion.verify_checksums().values())


def test_raw_counts_match_outputs():
    folder = ingestion.extract("combined")
    tx = pd.read_csv(folder / "transactions.csv")
    nodes = pd.read_csv(folder / "nodes.csv")
    assert len(tx) == KM["n_tx"] and len(nodes) == KM["n_accounts"]
    assert int(nodes.isFraud.sum()) == KM["n_labelled"]
    assert abs(tx.value.sum() - KM["total_value"]) < 1e-6 * KM["total_value"]


def test_audit_has_no_failures():
    dq = pd.read_csv(T / "data_quality_report.csv")
    assert not (dq.status == "FAIL").any()


def test_sql_python_crosschecks():
    qc = pd.read_csv(T / "qc_sql_vs_python.csv")
    assert qc["match"].all() and len(qc) >= 15


def test_leakage_checks():
    lk = pd.read_csv(T / "leakage_checks.csv")
    assert (lk.result == "PASS").all()


def test_integrated_precision_independent_recompute():
    """Recompute held-out test alert precision from the SQLite alert queue + labels + groups."""
    with sqlite3.connect(config.DB_PATH) as con:
        q = pd.read_sql_query("""SELECT q.account_id, a.is_suspicious FROM alert_queue q
                                 JOIN accounts a USING (account_id) JOIN account_groups g USING (account_id)
                                 WHERE q.week BETWEEN 13 AND 21 AND g."group" = 1""", con)
    b = KM["baseline"]["B5b Integrated risk score (top 1%)"]
    assert len(q) == b["alerts"]
    assert abs(q.is_suspicious.mean() - b["alert_precision"]) < 1e-9


def test_confusion_matrix_consistent():
    for name, r in KM["baseline"].items():
        assert r["tp"] + r["fn"] == KM["heldout_test_labelled"], name
        assert r["tp"] + r["fp"] + r["fn"] + r["tn"] == KM["heldout_test_accounts"], name


def test_thresholds_label_free_and_frozen():
    th = json.loads((config.MODELS_DIR / "signal_thresholds.json").read_text())["thresholds"]
    assert th == KM["thresholds"]


def test_model_weights_nonnegative():
    w = json.loads((config.MODELS_DIR / "integrated_risk_model.json").read_text())["weights"]
    assert all(v >= 0 for v in w.values())


def test_alert_queue_has_evidence():
    q = pd.read_csv(T / "alert_queue.csv")
    assert q["evidence"].notna().all() and q["risk_level"].isin(["CRITICAL", "HIGH", "MEDIUM", "LOW"]).all()
    assert "is_suspicious" not in q.columns  # analysts must not see labels


def test_reports_fully_rendered():
    files = list((ROOT / "reports").glob("*.md")) + [ROOT / n for n in ("README.md", "DATA_SOURCES.md", "DATASET_EVALUATION.md")]
    for f in files:
        text = f.read_text()
        assert "{{" not in text, f
        assert "nan%" not in text, f


def test_readme_images_exist():
    text = (ROOT / "README.md").read_text()
    for img in re.findall(r'src="([^"]+\.png)"', text):
        assert (ROOT / img).exists(), img


def test_notebooks_executed_without_errors():
    for nb in sorted((ROOT / "notebooks").glob("*.ipynb")):
        cells = [c for c in json.loads(nb.read_text())["cells"] if c["cell_type"] == "code"]
        assert all(c.get("execution_count") for c in cells), nb.name
        assert not any(o.get("output_type") == "error" for c in cells for o in c.get("outputs", [])), nb.name
    assert len(list((ROOT / "notebooks").glob("*.ipynb"))) == 10


def test_required_files_present():
    need = ["README.md", "DATA_SOURCES.md", "DATASET_EVALUATION.md", "PROGRESS_LOG.md", "requirements.txt",
            ".gitignore", "LICENSE", "sql/analytical_queries.sql", "dashboard/app.py", "reports/final_report.md",
            "reports/methodology.md", "reports/findings.md", "reports/data_audit.md", "reports/leakage_audit.md",
            "outputs/tables/data_quality_report.csv", "data/README.md"]
    missing = [n for n in need if not (ROOT / n).exists()]
    assert not missing, missing


def test_dashboard_data_layer():
    sys.path.insert(0, str(ROOT / "dashboard"))
    import data_access as da
    ok, msg = da.ready()
    assert ok, msg
    k = da.overview_kpis()
    assert k["transactions"] == KM["n_tx"]
    acc = int(da.table("alert_queue").sort_values("risk_score").account_id.iloc[-1])
    w = da.account_weeks(acc)
    assert len(w) and len(da.ranked_evidence(w.iloc[-1])) >= 0 and len(da.account_transactions(acc))


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)
