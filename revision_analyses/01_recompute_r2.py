"""
01 — Re-run every published focal analysis with AURA v0.2 defaults.

Defaults reproduce v0.1 p-values / β / significance exactly, so this run
(a) verifies reproduction against the manuscript counts and
(b) adds the redefined R2_total next to the legacy value for every gene.

Reviewer items: R2.1 (R2_total), R2.3 (SVD percentages), S7.
Outputs: results/<dataset>__<focal>.csv, summary_r2.csv, svd_modes.csv
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import (FOCAL, parse_args, outdir, write_meta, done, load_dataset,
                    focal_data, fit, result_table, canonical_markers,
                    focal_lineage, spillover_flags, log)

# Manuscript counts used to verify reproduction (significant genes)
PUBLISHED = {("lymphnode", "Tfh"): 122, ("ipf", "AT2"): 92}
WATCH = {("lymphnode", "Tfh"): ["TOX2", "IKZF3"],
         ("lymphnode", "Endothelial"): ["CLEC4M", "CR2"],
         ("lymphnode", "Macrophage"): ["MRC1", "CD163", "CD209", "CD14"]}


def cumulative_shares(r2):
    r2 = np.sort(np.clip(r2, 0, None))[::-1]
    tot = r2.sum()
    if tot <= 0 or len(r2) == 0:
        return np.nan, np.nan
    top = lambda f: r2[:max(1, int(round(f * len(r2))))].sum() / tot
    return top(0.01), top(0.10)


def svd_modes(beta):
    """Share of squared β magnitude per mode: uncentered (as published) and
    gene-centered (variance in the usual sense)."""
    out = {}
    for name, B in (("uncentered", beta), ("centered", beta - beta.mean(axis=0))):
        s = np.linalg.svd(B, compute_uv=False)
        out[name] = (s ** 2 / (s ** 2).sum())[:3]
    return out


def main():
    args = parse_args(__doc__, lambda ap: ap.add_argument(
        "--datasets", nargs="+", default=["lymphnode", "ipf", "nsclc"]))
    d = outdir("01_recompute_r2")
    (d / "results").mkdir(exist_ok=True)
    write_meta(d, n_perm=args.n_perm, seed=args.seed)

    rows, svd_rows = [], []
    for ds in args.datasets:
        tissue = load_dataset(ds)
        canonical = canonical_markers(tissue, ds)
        for focal in FOCAL[ds]:
            path = d / "results" / f"{ds}__{focal.replace('/', '_')}.csv"
            fd = focal_data(tissue, ds, focal)
            if len(fd["idx"]) < 50:
                log.warning("%s/%s: %d focal cells, skipped", ds, focal,
                            len(fd["idx"]))
                continue
            if not done(path):
                res = fit(tissue, fd, args.n_perm, seed=args.seed)
                df = result_table(res, fd["genes"], tissue.type_names)
                df = spillover_flags(df, canonical, focal_lineage(tissue, fd),
                                     tissue.type_names)
                df.to_csv(path, index=False)
            df = pd.read_csv(path)

            sig = df[df.significant]
            clean = sig[~sig.spillover_suspect]
            rho = spearmanr(sig.R2_total, sig.R2_total_legacy)[0] \
                if len(sig) > 2 else np.nan
            ratio = np.median(sig.R2_total / sig.R2_total_legacy.replace(0, np.nan))
            t1n, t10n = cumulative_shares(clean.R2_total.values)
            t1l, t10l = cumulative_shares(clean.R2_total_legacy.values)
            row = dict(dataset=ds, focal=focal, n_cells=len(fd["idx"]),
                       n_genes=len(df), n_sig=len(sig),
                       published_sig=PUBLISHED.get((ds, focal)),
                       n_spillover=int(sig.spillover_suspect.sum()),
                       median_R2_total=sig.R2_total.median(),
                       median_R2_total_adj=sig.R2_total_adj.median(),
                       median_R2_total_legacy=sig.R2_total_legacy.median(),
                       median_ratio_new_over_legacy=ratio,
                       spearman_new_vs_legacy=rho,
                       median_var_retained=sig.var_retained.median(),
                       top1_share_new=t1n, top10_share_new=t10n,
                       top1_share_legacy=t1l, top10_share_legacy=t10l)
            for g in WATCH.get((ds, focal), []):
                hit = df[df.gene == g]
                if len(hit):
                    row[f"{g}_R2_total"] = float(hit.R2_total.iloc[0])
                    row[f"{g}_R2_total_legacy"] = float(hit.R2_total_legacy.iloc[0])
            rows.append(row)

            if len(sig) >= 3:
                bcols = [f"beta_{t}" for t in tissue.type_names]
                m = svd_modes(sig[bcols].values)
                svd_rows.append(dict(dataset=ds, focal=focal, n_sig=len(sig),
                                     **{f"{k}_PC{i+1}": v[i] for k, v in m.items()
                                        for i in range(len(v))}))
            log.info("%s/%s: %d sig (published %s), median R2_total %.4f vs "
                     "legacy %.4f", ds, focal, len(sig),
                     PUBLISHED.get((ds, focal)), row["median_R2_total"],
                     row["median_R2_total_legacy"])

    pd.DataFrame(rows).to_csv(d / "summary_r2.csv", index=False)
    pd.DataFrame(svd_rows).to_csv(d / "svd_modes.csv", index=False)


if __name__ == "__main__":
    main()
