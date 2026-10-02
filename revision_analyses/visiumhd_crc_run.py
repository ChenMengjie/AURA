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
from visiumhd_crc_config import FOCAL, MAX_CELLS


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
    names = sorted(adata.obs["lineage"].unique())
    types = np.array([names.index(t) for t in adata.obs["lineage"]])
    tissue = TissueData(adata=adata,
                        all_xy=adata.obs[["x_centroid", "y_centroid"]].values,
                        all_types=types, type_names=names,
                        type_to_int={t: i for i, t in enumerate(names)})
    lab = adata.obs[adata.obs.lineage != "Unassigned"]
    canonical, _ = aura.learn_canonical(adata[lab.index], type_column="lineage",
                                        min_fold=2.0, min_detection=0.05)

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
            spillover_flags(df, canonical, lin, names).to_csv(path, index=False)
        df = pd.read_csv(path)
        sig = df[df.significant]
        rows.append(dict(focal=focal, lineage=lin, n_cells=len(idx),
                         n_genes=len(df), n_sig=len(sig),
                         pct_sig=len(sig) / len(df),
                         n_spillover=int(sig.spillover_suspect.sum()),
                         median_R2_total=sig.R2_total.median(),
                         median_R2_total_legacy=sig.R2_total_legacy.median(),
                         top_genes=", ".join(sig.sort_values("R2_total",
                                             ascending=False).gene.head(8))))
        log.info("%s: %s", focal, rows[-1])
        pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], "--quick" in sys.argv)
