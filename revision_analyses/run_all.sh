#!/usr/bin/env bash
# Run every revision analysis. Usage:  bash run_all.sh [--quick]
# Set AURA_DATA (data dir) and AURA_REV_OUT (output dir) first; see README.
set -euo pipefail
cd "$(dirname "$0")"
: "${AURA_REV_OUT:?set AURA_REV_OUT (e.g. the Google Drive Revision/outputs folder)}"
# Per-script logs go to AURA_REV_LOG (default: a local directory). Writing them
# into a cloud-synced folder can fail with "Operation timed out" while the
# sync client holds the file, which would abort the whole run under pipefail.
LOGDIR="${AURA_REV_LOG:-$HOME/aura_revision/logs}"
mkdir -p "$LOGDIR"
for s in 01_recompute_r2 07_centering 06_dispersion 03_neighborhood_scale \
         02_clustering_sim 04_panel_size 05_ipf_downsample; do
  echo "=== $s  $(date)"
  python -W ignore "$s.py" "$@" 2>&1 | tee "$LOGDIR/$s.log"
done
echo "=== all done $(date)"
