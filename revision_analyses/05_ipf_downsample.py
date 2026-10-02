"""
05 — Is IPF-vs-healthy rewiring a sample-size artifact? (Reviewer 1, m2)

Healthy focal populations are ~6x smaller than IPF (225k vs 1.4M cells
overall). For each focal type:
  * IPF downsampled to the healthy donor count, then to the healthy focal
    cell count (reps), and refit
  * % significant and median IPF-vs-healthy β cosine (genes significant in
    either condition, as in Fig. 5c) for full IPF, downsampled IPF
  * noise ceilings: split-half cosine within healthy (donor halves) and
    within IPF at matched size — the cosine expected with no rewiring

Outputs: downsample_runs.csv, downsample_summary.csv
"""

import numpy as np
import pandas as pd

from common import (FOCAL, parse_args, outdir, write_meta, load_dataset,
                    focal_data, fit, result_table, cosine_rows, log)


def subset(fd, keep):
    out = {k: (v[keep] if k in ("counts", "xy", "idx", "sample_ids") else v)
           for k, v in fd.items()}
    return out


def fit_table(tissue, fd, n_perm, seed):
    res = fit(tissue, fd, n_perm, seed=seed)
    return result_table(res, fd["genes"], tissue.type_names).set_index("gene")


def cosine_between(a, b, genes_mask_from="either"):
    """Median cosine of β over shared genes significant in either table."""
    g = a.index.intersection(b.index)
    a, b = a.loc[g], b.loc[g]
    sel = a.significant | b.significant if genes_mask_from == "either" \
        else a.significant
    cols = sorted(set(c for c in a.columns if c.startswith("beta_")) &
                  set(c for c in b.columns if c.startswith("beta_")))
    if not sel.any():
        return np.nan, 0
    cos = cosine_rows(a.loc[sel, cols].values, b.loc[sel, cols].values)
    return float(np.nanmedian(cos)), int(sel.sum())


def main():
    args = parse_args(__doc__, lambda ap: ap.add_argument(
        "--reps", type=int, default=10), default_n_perm=2000)
    reps = 2 if args.quick else args.reps
    d = outdir("05_ipf_downsample")
    write_meta(d, **vars(args))

    ipf = load_dataset("ipf", "Disease")
    hea = load_dataset("ipf", "Healthy")
    rows = []
    for focal in FOCAL["ipf"]:
        fi = focal_data(ipf, "ipf", focal)
        fh = focal_data(hea, "ipf", focal)
        if len(fh["idx"]) < 100:
            log.warning("%s: only %d healthy cells, skipped", focal, len(fh["idx"]))
            continue
        pat_i = ipf.adata.obs["patient"].astype(str).values[fi["idx"]]
        pat_h = hea.adata.obs["patient"].astype(str).values[fh["idx"]]
        n_h, donors_h = len(fh["idx"]), np.unique(pat_h)

        t_h = fit_table(hea, fh, args.n_perm, args.seed)
        t_i = fit_table(ipf, fi, args.n_perm, args.seed)
        base = dict(focal=focal, n_ipf=len(fi["idx"]), n_healthy=n_h,
                    donors_ipf=len(np.unique(pat_i)), donors_healthy=len(donors_h))
        cos, ng = cosine_between(t_i, t_h)
        rows.append(dict(base, analysis="full_ipf_vs_healthy", rep=0,
                         pct_sig_ipf=t_i.significant.mean(),
                         pct_sig_healthy=t_h.significant.mean(),
                         median_cosine=cos, n_genes_cosine=ng))
        log.info("%s full: %s", focal, rows[-1])

        for r in range(reps):
            rng = np.random.default_rng(500 + r)
            # IPF matched to healthy donor count and focal cell count
            dsel = rng.choice(np.unique(pat_i), min(len(donors_h),
                                                    len(np.unique(pat_i))),
                              replace=False)
            pool = np.where(np.isin(pat_i, dsel))[0]
            keep = np.sort(rng.choice(pool, min(n_h, len(pool)), replace=False))
            t_d = fit_table(ipf, subset(fi, keep), args.n_perm, args.seed + r)
            cos, ng = cosine_between(t_d, t_h)
            rows.append(dict(base, analysis="downsampled_ipf_vs_healthy", rep=r,
                             n_cells_used=len(keep),
                             pct_sig_ipf=t_d.significant.mean(),
                             pct_sig_healthy=t_h.significant.mean(),
                             median_cosine=cos, n_genes_cosine=ng))

            # noise ceiling: healthy donor halves
            perm = rng.permutation(donors_h)
            halves = [np.where(np.isin(pat_h, perm[:len(perm) // 2]))[0],
                      np.where(np.isin(pat_h, perm[len(perm) // 2:]))[0]]
            if min(map(len, halves)) >= 50:
                ta, tb = (fit_table(hea, subset(fh, h), args.n_perm, args.seed)
                          for h in halves)
                cos, ng = cosine_between(ta, tb)
                rows.append(dict(base, analysis="ceiling_healthy_split_half",
                                 rep=r, n_cells_used=int(np.mean(list(map(len, halves)))),
                                 median_cosine=cos, n_genes_cosine=ng))

            # noise ceiling: IPF donor halves at the same per-half size
            perm = rng.permutation(np.unique(pat_i))
            m = len(perm) // 2
            halves = []
            for dset in (perm[:m], perm[m:]):
                pool = np.where(np.isin(pat_i, dset))[0]
                size = min(len(pool), max(n_h // 2, 50))
                halves.append(np.sort(rng.choice(pool, size, replace=False)))
            ta, tb = (fit_table(ipf, subset(fi, h), args.n_perm, args.seed)
                      for h in halves)
            cos, ng = cosine_between(ta, tb)
            rows.append(dict(base, analysis="ceiling_ipf_split_half_matched",
                             rep=r, n_cells_used=int(np.mean(list(map(len, halves)))),
                             median_cosine=cos, n_genes_cosine=ng))
            log.info("%s rep %d done", focal, r)
            pd.DataFrame(rows).to_csv(d / "downsample_runs.csv", index=False)

    runs = pd.DataFrame(rows)
    runs.to_csv(d / "downsample_runs.csv", index=False)
    summ = (runs.groupby(["focal", "analysis"])
            [["pct_sig_ipf", "pct_sig_healthy", "median_cosine", "n_genes_cosine"]]
            .agg(["mean", "std"]))
    summ.to_csv(d / "downsample_summary.csv")


if __name__ == "__main__":
    main()
