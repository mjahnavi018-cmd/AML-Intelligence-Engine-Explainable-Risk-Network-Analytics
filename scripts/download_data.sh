#!/usr/bin/env bash
# Re-acquire the raw AMLSim sample archives from the official IBM repository and
# verify them against data/raw/amlsim_sample/SHA256SUMS. The archives are already
# committed (Apache-2.0 permits redistribution); use this to prove provenance.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
git clone --depth 1 https://github.com/IBM/AMLSim "$TMP/AMLSim"
git -C "$TMP/AMLSim" log -1 --format='AMLSim commit %H (%cd)'
for f in 20K_fanin200cycle200.tgz 20K_fanin200.tgz 20K_cycle200.tgz; do
  cp "$TMP/AMLSim/sample/$f" "$ROOT/data/raw/amlsim_sample/$f"
done
cp "$TMP/AMLSim/LICENSE" "$ROOT/data/raw/amlsim_sample/AMLSim_LICENSE_Apache-2.0.txt"
( cd "$ROOT/data/raw/amlsim_sample" && sha256sum -c SHA256SUMS )
rm -rf "$TMP"
