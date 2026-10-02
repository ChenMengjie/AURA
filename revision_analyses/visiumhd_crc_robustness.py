"""
Robustness of the Visium HD CRC findings to the calibration options
(S11: large-scale spatial trend; S12: centering leakage).

Variants per focal type (2,000 permutations):
    default            v0.1 settings
    trend_L            spatial_trend = L µm for L in {1000, 500, 250}
    nonmarker          per-cell centering on genes that are not canonical
                       markers of any non-focal lineage
    nonmarker+trend500 both

Outputs: robustness_runs.csv (one row per focal x variant),
         robustness_genes.csv (per gene: significant / clean under each variant)
Usage: python visiumhd_crc_robustness.py <cells.h5ad> <out_dir> [--quick]
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

import aura
from aura.model import run_model
from common import result_table, spillover_flags, non_marker_genes, jaccard, log
from visiumhd_crc_config import FOCAL, MAX_CELLS, FAMILY

VARIANTS = {
    "default": {},
    "trend_1000": dict(spatial_trend=1000),
    "trend_500": dict(spatial_trend=500),
    "trend_250": dict(spatial_trend=250),
    "nonmarker": "nonmarker",
    "nonmarker+trend500": "nonmarker+trend",
}


def main(h5ad, out, quick=False):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("aura").setLevel(logging.WARNING)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    n_perm = 200 if quick else 2000

    adata = sc.read_h5ad(h5ad)
    frac = adata.obs["lineage"].value_counts(normalize=True)
    adata.obs["lineage"] = adata.obs["lineage"].astype(str).replace(
        {r: "Unassigned" for r in frac.index[frac < 0.005]})
    names = sorted(adata.obs["lineage"].unique())
    types = np.array([names.index(t) for t in adata.obs["lineage"]])
    xy = adata.obs[["x_centroid", "y_centroid"]].values
    lab = adata[adata.obs.lineage != "Unassigned"]
    canonical, _ = aura.learn_canonical(lab, type_column="lineage",
                                        min_fold=2.0, min_detection=0.05)
    own_canonical = {k: set(v) for k, v in canonical.items()}
    fam = lab.copy()
    fam.obs["family"] = fam.obs["lineage"].map(FAMILY).fillna(fam.obs["lineage"])
    fam_markers, _ = aura.learn_canonical(fam, type_column="family",
                                          min_fold=2.0, min_detection=0.05)
    for l_, f_ in FAMILY.items():
        if l_ in canonical:
            canonical[l_] = set(canonical[l_]) | set(fam_markers.get(f_, set()))

    runs, genes_rows = [], []
    for focal in (FOCAL[:2] if quick else FOCAL):
        idx = np.where((adata.obs.cell_type == focal) & adata.obs.focal_ok)[0]
        cap = MAX_CELLS.get(focal)
        if cap and len(idx) > cap:
            idx = np.sort(np.random.default_rng(42).choice(idx, cap, replace=False))
        X = adata.X[idx]
        counts = np.asarray(X.todense() if sp.issparse(X) else X, dtype=np.float64)
        keep = counts.mean(axis=0) >= 0.1
        counts, genes = counts[:, keep], np.asarray(adata.var_names)[keep]
        lin = adata.obs.lineage.iloc[idx].mode().iloc[0]
        ref = non_marker_genes(genes, canonical, lin)

        tabs = {}
        for vname, kw in VARIANTS.items():
            if kw == "nonmarker":
                kw = dict(center_genes=ref)
            elif kw == "nonmarker+trend":
                kw = dict(center_genes=ref, spatial_trend=500)
            res = run_model(counts, xy[idx], xy, types, k=30, n_perm=n_perm,
                            seed=42, **kw)
            df = spillover_flags(result_table(res, genes, names), canonical, lin,
                                 names, P_sd=res["P"].std(axis=0),
                                 own_markers=own_canonical.get(lin, set()))
            tabs[vname] = df.set_index("gene")
        base = tabs["default"]
        base_sig = set(base.index[base.significant])
        base_clean = set(base.index[base.significant & ~base.spillover_suspect])
        for vname, t in tabs.items():
            sig = set(t.index[t.significant])
            clean = set(t.index[t.significant & ~t.spillover_suspect])
            runs.append(dict(focal=focal, variant=vname, n_cells=len(idx),
                             n_genes=len(t), n_sig=len(sig), n_clean=len(clean),
                             jaccard_sig_vs_default=jaccard(sig, base_sig),
                             retained_clean_of_default=len(clean & base_clean)
                             / max(len(base_clean), 1),
                             n_new_clean=len(clean - base_clean)))
            log.info("%s %s: %s", focal, vname, runs[-1])
        allg = sorted(set().union(*[set(t.index[t.significant]) for t in tabs.values()]))
        for g in allg:
            row = dict(focal=focal, gene=g)
            for vname, t in tabs.items():
                row[f"sig_{vname}"] = bool(t.at[g, "significant"])
                row[f"clean_{vname}"] = bool(t.at[g, "significant"]
                                             and not t.at[g, "spillover_suspect"])
            row["driver_default"] = base.at[g, "driver_axis"]
            row["R2_total_default"] = base.at[g, "R2_total"]
            genes_rows.append(row)
        pd.DataFrame(runs).to_csv(out / "robustness_runs.csv", index=False)
        pd.DataFrame(genes_rows).to_csv(out / "robustness_genes.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], "--quick" in sys.argv)
