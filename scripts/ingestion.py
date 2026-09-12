"""Ingestion: verify raw archives and extract them without modification.

Lineage step: SOURCE (IBM AMLSim GitHub) -> RAW (data/raw, checksummed, never
modified) -> EXTRACTED (data/processed/extracted, byte-identical CSVs).
"""
from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path

import pandas as pd

from . import config


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checksums() -> dict[str, bool]:
    """Compare every raw archive with data/raw/amlsim_sample/SHA256SUMS."""
    expected = {}
    for line in (config.RAW_DIR / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        expected[name] = digest
    result = {name: sha256(config.RAW_DIR / name) == digest for name, digest in expected.items()}
    bad = [k for k, ok in result.items() if not ok]
    if bad:
        raise ValueError(f"Checksum mismatch for raw files: {bad}")
    return result


def extract(dataset: str) -> Path:
    """Extract one archive (idempotent). Returns the folder with nodes.csv / transactions.csv."""
    meta = config.DATASETS[dataset]
    target = config.EXTRACT_DIR / meta["folder"]
    if not (target / "transactions.csv").exists():
        config.EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        with tarfile.open(config.RAW_DIR / meta["archive"], "r:gz") as tf:
            members = [m for m in tf.getmembers() if m.isfile() and m.name.endswith((".csv", ".txt"))]
            tf.extractall(config.EXTRACT_DIR, members=members, filter="data")
    return target


def load_raw(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Return (nodes, transactions, metadata_text) exactly as published."""
    folder = extract(dataset)
    nodes = pd.read_csv(folder / "nodes.csv")
    tx = pd.read_csv(folder / "transactions.csv")
    meta = (folder / "metadata.txt").read_text().strip()
    return nodes, tx, meta
