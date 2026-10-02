"""
06 — Shared vs gene-specific NB dispersion (Reviewer 1, m1).

Per gene, Pearson residuals differ between dispersion models only by a
constant factor, so permutation p-values can change only through cell
centering. This compares shared φ with regularized gene-specific φ
(`dispersion='gene'`) on the same fits: Spearman of -log10 p, Jaccard of
significant sets, R2_total agreement, and how many genes change status
because of the excess-variance filter.

Outputs: dispersion_compare.csv, phi_trend__<dataset>__<focal>.csv
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from aura.model import estimate_dispersion, estimate_gene_dispersion
from common import (parse_args, outdir, write_meta, load_dataset, focal_data,
                    fit, result_table, jaccard)

TARGETS = {"lymphnode": ["Tfh", "Macrophage", "Endothelial"],
           "ipf": ["AT2", "SPP1+ Macrophages", "Basal"],
           "nsclc": ["macrophage", "fibroblast"]}


def main():
    args = parse_args(__doc__, lambda ap: ap.add_argument(
        "--datasets", nargs="+", default=list(TARGETS)), default_n_perm=2000)
    d = outdir("06_dispersion")
    write_meta(d, **vars(args))
    rows = []
    for ds in args.datasets:
        tissue = load_dataset(ds)
        for focal in TARGETS[ds][:1] if args.quick else TARGETS[ds]:
            fd = focal_data(tissue, ds, focal)
            tabs = {}
            for disp in ("shared", "gene"):
                res = fit(tissue, fd, args.n_perm, seed=args.seed, dispersion=disp)
                tabs[disp] = result_table(res, fd["genes"], tissue.type_names)
            a, b = tabs["shared"], tabs["gene"]
            raw_a, raw_b = a.qvalue <= 0.05, b.qvalue <= 0.05
            rows.append(dict(
                dataset=ds, focal=focal, n_genes=len(a),
                n_sig_shared=int(a.significant.sum()),
                n_sig_gene=int(b.significant.sum()),
                jaccard_sig=jaccard(a.gene[a.significant], b.gene[b.significant]),
                jaccard_raw_fdr=jaccard(a.gene[raw_a], b.gene[raw_b]),
                spearman_logp=spearmanr(-np.log10(a.pvalue), -np.log10(b.pvalue))[0],
                max_abs_logp_diff=float(np.max(np.abs(np.log10(a.pvalue)
                                                      - np.log10(b.pvalue)))),
                spearman_R2_total=spearmanr(a.R2_total, b.R2_total)[0],
                n_excess_filtered_shared=int((raw_a & ~a.has_excess).sum()),
                n_excess_filtered_gene=int((raw_b & ~b.has_excess).sum()),
            ))
            mu = fd["counts"].mean(axis=0)
            pd.DataFrame(dict(
                gene=fd["genes"], mean=mu, var=fd["counts"].var(axis=0, ddof=1),
                phi_gene=estimate_gene_dispersion(fd["counts"]),
                phi_shared=estimate_dispersion(fd["counts"]),
            )).to_csv(d / f"phi_trend__{ds}__{focal.replace('/', '_')}.csv",
                      index=False)
    pd.DataFrame(rows).to_csv(d / "dispersion_compare.csv", index=False)


if __name__ == "__main__":
    main()
