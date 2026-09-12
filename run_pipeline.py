"""Run the full analytical pipeline: audit -> features -> signals -> scoring ->
alerts -> evaluation -> robustness -> case studies -> figures -> SQL -> reports.

Usage:
    python run_pipeline.py            # everything (about 2-4 minutes)
    python run_pipeline.py --fast     # skip robustness re-runs and report rendering
"""
import argparse
import sys

from src import pipeline


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="skip robustness analyses")
    args = ap.parse_args()
    ctx = pipeline.main(skip_robustness=args.fast)
    from src import reporting, sql_runner, visualization
    visualization.make_all(ctx)
    sql_runner.run_all(ctx)
    if args.fast:
        print("--fast: robustness skipped, so reports were not re-rendered (run without --fast)")
    else:
        reporting.render_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
