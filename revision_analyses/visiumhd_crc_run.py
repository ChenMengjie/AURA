"""
Run AURA on the cell-segmented Visium HD CRC section (editor request:
test on 10x Visium data; Reviewer 1, M2).

Same settings as the Xenium analyses: k = 30, 5,000 permutations,
FDR < 0.05, genes with focal mean >= 0.1, data-driven spillover filter.
Composition uses all cells (11 lineages + 'Unassigned'); focal cells are
restricted to cells in RCTD singlet bins.

Usage: python visiumhd_crc_run.py <cells.h5ad> <out_dir> [--quick]
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import aura
from aura.io import TissueData
from aura.model import run_model
from common import result_table, spillover_flags, write_meta, done, log
from visiumhd_crc_config import FOCAL, MAX_CELLS, FAMILY


def main(h5ad, out, quick=False):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("aura").setLevel(logging.WARNING)
    out = Path(out)
    (out / "results").mkdir(parents=True, exist_ok=True)
    n_perm = 300 if quick else 5000
    write_meta(out, h5ad=h5ad, n_perm=n_perm, k=30, focal=FOCAL,
               max_cells=MAX_CELLS)

    import scanpy as sc
    import scipy.sparse as sp
    adata = sc.read_h5ad(h5ad)
    # fold rare lineages (< 0.5% of cells) into 'Unassigned'
    frac = adata.obs["lineage"].value_counts(normalize=True)
    rare = list(frac.index[frac < 0.005])
    log.info("folding rare lineages into Unassigned: %s", rare)
    adata.obs["lineage"] = adata.obs["lineage"].astype(str).replace(
        {r: "Unassigned" for r in rare})
    names = sorted(adata.obs["lineage"].unique())
    types = np.array([names.index(t) for t in adata.obs["lineage"]])
    tissue = TissueData(adata=adata,
                        all_xy=adata.obs[["x_centroid", "y_centroid"]].values,
                        all_types=types, type_names=names,
                        type_to_int={t: i for i, t in enumerate(names)})
    lab = adata.obs[adata.obs.lineage != "Unassigned"]
    canonical, _ = aura.learn_canonical(adata[lab.index], type_column="lineage",
                                        min_fold=2.0, min_detection=0.05)
    # markers shared within a lineage family (tumor vs normal epithelium,
    # fibroblast vs smooth muscle) fail the 2-fold rule for either member;
    # learn them at family level too and credit them to every member
    fam = adata[lab.index].copy()
    fam.obs["family"] = fam.obs["lineage"].map(FAMILY).fillna(fam.obs["lineage"])
    fam_markers, _ = aura.learn_canonical(fam, type_column="family",
                                          min_fold=2.0, min_detection=0.05)
    own_canonical = {k: set(v) for k, v in canonical.items()}
    for lin_, f_ in FAMILY.items():
        if lin_ in canonical:
            canonical[lin_] = set(canonical[lin_]) | set(fam_markers.get(f_, set()))
    # per-lineage mean expression (library-size normalized) for the
    # relative spillover rule
    norm = adata[lab.index].copy()
    sc.pp.normalize_total(norm, target_sum=1e3)
    lineage_means = pd.DataFrame(
        {l: np.asarray(norm[norm.obs.lineage == l].X.mean(axis=0)).ravel()
         for l in names if l != "Unassigned"}, index=norm.var_names)

    rows = []
    for focal in FOCAL:
        path = out / "results" / f"{focal.replace(' ', '_')}.csv"
        idx = np.where((adata.obs.cell_type == focal) & adata.obs.focal_ok)[0]
        cap = MAX_CELLS.get(focal)
        if cap and len(idx) > cap:
            idx = np.sort(np.random.default_rng(42).choice(idx, cap, replace=False))
        X = adata.X[idx]
        counts = np.asarray(X.todense() if sp.issparse(X) else X, dtype=np.float64)
        keep = counts.mean(axis=0) >= 0.1
        counts, genes = counts[:, keep], np.asarray(adata.var_names)[keep]
        lin = adata.obs.lineage.iloc[idx].mode().iloc[0]
        if not done(path):
            log.info("%s: %d cells, %d genes", focal, len(idx), len(genes))
            res = run_model(counts, tissue.all_xy[idx], tissue.all_xy, types,
                            k=30, n_perm=n_perm, seed=42)
            df = result_table(res, genes, names)
            nf = adata[adata.obs_names[idx]].copy()
            sc.pp.normalize_total(nf, target_sum=1e3)
            focal_means = pd.Series(np.asarray(nf.X.mean(axis=0)).ravel(),
                                    index=nf.var_names)
            spillover_flags(df, canonical, lin, names,
                            P_sd=res["P"].std(axis=0),
                            own_markers=own_canonical.get(lin, set()),
                            lineage_means=lineage_means,
                            focal_means=focal_means).to_csv(path, index=False)
        df = pd.read_csv(path)
        sig = df[df.significant]
        rows.append(dict(focal=focal, lineage=lin, n_cells=len(idx),
                         n_genes=len(df), n_sig=len(sig),
                         pct_sig=len(sig) / len(df),
                         n_spillover=int(sig.spillover_suspect.sum()),
                         n_spillover_rel=int(sig.spillover_suspect_rel.sum()),
                         n_clean_both=int((~sig.spillover_suspect & ~sig.spillover_suspect_rel).sum()),
                         top_drivers=str(sig.driver_axis.value_counts().head(3).to_dict()),
                         median_R2_total=sig.R2_total.median(),
                         median_R2_total_legacy=sig.R2_total_legacy.median(),
                         top_genes=", ".join(sig.sort_values("R2_total",
                                             ascending=False).gene.head(8))))
        log.info("%s: %s", focal, rows[-1])
        pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], "--quick" in sys.argv)
