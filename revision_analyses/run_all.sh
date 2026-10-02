#!/usr/bin/env bash
# Run every revision analysis. Usage:  bash run_all.sh [--quick]
# Set AURA_DATA (data dir) and AURA_REV_OUT (output dir) first; see README.
set -euo pipefail
cd "$(dirname "$0")"
: "${AURA_REV_OUT:?set AURA_REV_OUT (e.g. the Google Drive Revision/outputs folder)}"
mkdir -p "$AURA_REV_OUT/logs"
for s in 01_recompute_r2 07_centering 06_dispersion 03_neighborhood_scale \
         02_clustering_sim 04_panel_size 05_ipf_downsample; do
  echo "=== $s  $(date)"
  python -W ignore "$s.py" "$@" 2>&1 | tee "$AURA_REV_OUT/logs/$s.log"
done
echo "=== all done $(date)"
