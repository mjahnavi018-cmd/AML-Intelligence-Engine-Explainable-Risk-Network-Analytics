"""Central configuration: paths, dataset registry, and analysis design constants.

Every design constant below is referenced in reports/methodology.md. Changing a
constant and re-running `python run_pipeline.py` regenerates every output.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "amlsim_sample"
PROCESSED_DIR = ROOT / "data" / "processed"
EXTRACT_DIR = PROCESSED_DIR / "extracted"          # verbatim extraction of raw archives
DB_PATH = PROCESSED_DIR / "banking_risk.db"
TABLES_DIR = ROOT / "outputs" / "tables"
FIGURES_DIR = ROOT / "outputs" / "figures"
MODELS_DIR = ROOT / "outputs" / "models"
REPORTS_DIR = ROOT / "reports"
SQL_DIR = ROOT / "sql"

# ---------------------------------------------------------------------------
# Dataset registry. All three are official IBM AMLSim sample outputs
# (github.com/IBM/AMLSim, folder sample/, Apache-2.0). They share the same
# 20,000-account background population but differ in the injected AML typologies.
# ---------------------------------------------------------------------------
DATASETS = {
    "combined": {"archive": "20K_fanin200cycle200.tgz", "folder": "20K_fanin200cycle200",
                 "typologies": "200 fan-in + 200 cycle"},
    "fanin": {"archive": "20K_fanin200.tgz", "folder": "20K_fanin200",
              "typologies": "200 fan-in"},
    "cycle": {"archive": "20K_cycle200.tgz", "folder": "20K_cycle200",
              "typologies": "200 cycle"},
}
PRIMARY_DATASET = "combined"
AMLSIM_COMMIT = "7338a4bcb1af9bcfea2201ad7daccfe2a4d569ca"

# ---------------------------------------------------------------------------
# Monitoring design (point-in-time). AMLSim time is an integer simulation
# step; AMLSim's configuration describes steps as days from a base date, so a
# 7-step block is treated as one weekly monitoring cycle.
# ---------------------------------------------------------------------------
WEEK_LEN = 7
FIRST_RUN_WEEK = 5            # weeks 1-4 are warm-up history only (no alerts)
DEV_WEEKS = list(range(5, 13))    # calibration period: steps 29-84
TEST_WEEKS = list(range(13, 22))  # held-out evaluation period: steps 85-147
# steps 148-149 form an incomplete week 22 and are excluded from monitoring runs
NETWORK_WINDOW_DAYS = 28      # look-back window for network / pass-through signals
DORMANCY_WEEKS = 4            # weeks without activity that define dormancy
MIN_HISTORY_TX = 5            # minimum historical transactions for amount z-scores
FORWARD_DAYS = 3              # "rapid forwarding" = outflow within 3 days of an inflow
CYCLE_MAX_LEN = 6             # longest directed cycle searched in each window

SIGNAL_QUANTILE = 0.99        # default empirical quantile for rule thresholds
SENSITIVITY_QUANTILES = [0.90, 0.95, 0.975, 0.99, 0.995]
RANDOM_SEED = 42
ALERT_PERCENTILE = 99.0       # integrated alert: risk score in the top 1% of development account-weeks
ALERT_PERCENTILE_GRID = [97.0, 98.0, 99.0, 99.5, 99.8]
REALISTIC_PREVALENCES = [0.001, 0.005, 0.01, 0.02, 0.05]   # for precision re-projection only
