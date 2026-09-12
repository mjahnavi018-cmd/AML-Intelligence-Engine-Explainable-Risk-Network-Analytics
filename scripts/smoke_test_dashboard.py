"""Execute every dashboard page against a stub Streamlit API.

This verifies that all data access, calculations, tables and figures used by the
dashboard run without errors on the real pipeline outputs. It does NOT prove the
real Streamlit app renders; run `streamlit run dashboard/app.py` for that.
Usage: python tests/smoke_test_dashboard.py
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "streamlit_stub"))
sys.path.insert(0, str(ROOT / "dashboard"))
import streamlit as st  # noqa: E402  (the stub)

PAGES = ["1 · Executive risk overview", "2 · Alert queue", "3 · Account investigation", "4 · AML monitoring",
         "5 · Mule accounts & network", "6 · Labelled-activity analytics", "7 · Rule & model performance",
         "8 · Data, method & limitations"]
failures = 0
for p in PAGES:
    st.PAGE["value"] = p
    st.CALLS.clear()
    try:
        runpy.run_path(str(ROOT / "dashboard" / "app.py"), run_name="__main__")
        kinds = {}
        for k, _ in st.CALLS:
            kinds[k] = kinds.get(k, 0) + 1
        errors = [o for k, o in st.CALLS if k == "error"]
        if errors:
            raise RuntimeError(errors)
        print(f"PASS  {p}: " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
    except st._StopException:
        print(f"STOP  {p}: page called st.stop()")
        failures += 1
    except Exception as exc:  # noqa: BLE001
        failures += 1
        print(f"FAIL  {p}: {type(exc).__name__}: {exc}")
sys.exit(1 if failures else 0)
