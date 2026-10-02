"""
04 — Sensitivity to gene-panel size and composition (Reviewer 1, M4; S8).

The lymph-node Xenium 5K panel is subsampled to targeted-panel sizes and
AURA is re-run under three per-cell centering choices:
    mean       (v0.1 default)
    median     (robust location)
    nonmarker  (center on genes that are not canonical markers of any
                non-focal lineage, learned on the *subsampled* panel —
                what a user with only that panel could do)

Panel strategies:
    random          uniform over genes
    top_expressed   highest tissue-wide mean (deterministic)
    marker_enriched half canonical lineage markers, half random — mimics a
                    targeted panel such as the 343-gene IPF panel

Each panel is compared to the full-panel fit with the same centering (the
panel-size effect) and, in columns prefixed vsmean_, to the full-panel fit
with mean centering (the combined effect), on the genes they share: recall/precision of significant calls, median ratio of
R2_total, median β cosine, Spearman of -log10 p; plus number of canonical
markers and spillover-suspect rate learned on the subsampled panel.

Outputs: full_panel__<focal>__<center>.csv, panel_runs.csv
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import (parse_args, outdir, write_meta, done, load_dataset,
                    focal_data, fit, result_table, canonical_markers,
                    focal_lineage, spillover_flags, non_marker_genes,
                    cosine_rows, log)


def extra(ap):
    ap.add_argument("--focals", nargs="+", default=["Tfh", "Macrophage"])
    ap.add_argument("--sizes", nargs="+", type=int, default=[150, 343, 600])
    ap.add_argument("--reps", type=int, default=10)


def run_fit(tissue, fd, center, canonical, lin, n_perm, seed):
    kw = {}
    if center == "median":
        kw["center"] = "median"
    elif center == "nonmarker":
        ref = non_marker_genes(fd["genes"], canonical, lin)
        if ref.sum() < 20:
            log.warning("only %d non-marker genes; skipping nonmarker", ref.sum())
            return None
        kw["center_genes"] = ref
    res = fit(tissue, fd, n_perm, seed=seed, **kw)
    df = result_table(res, fd["genes"], tissue.type_names)
    return spillover_flags(df, canonical, lin, tissue.type_names)


def compare(full, sub):
    f = full.set_index("gene")
    s = sub.set_index("gene")
    g = f.index.intersection(s.index)
    f, s = f.loc[g], s.loc[g]
    tp = (f.significant & s.significant).sum()
    both = f.significant & s.significant
    bcols = [c for c in f.columns if c.startswith("beta_")]
    return dict(
        n_genes_tested=len(g), n_sig_full=int(f.significant.sum()),
        n_sig_panel=int(s.significant.sum()),
        recall=tp / max(f.significant.sum(), 1),
        precision=tp / max(s.significant.sum(), 1),
        median_R2_total_ratio=float(np.median(s.R2_total[both] / f.R2_total[both]))
        if both.any() else np.nan,
        median_beta_cosine=float(np.nanmedian(cosine_rows(
            f.loc[both, bcols].values, s.loc[both, bcols].values)))
        if both.any() else np.nan,
        spearman_logp=spearmanr(-np.log10(f.pvalue), -np.log10(s.pvalue))[0],
        median_var_retained=float(s.var_retained.median()),
        spillover_rate=float(s.spillover_suspect[s.significant].mean())
        if s.significant.any() else np.nan,
    )


def main():
    args = parse_args(__doc__, extra, default_n_perm=2000)
    reps = 1 if args.quick else args.reps
    sizes = args.sizes[1:2] if args.quick else args.sizes
    d = outdir("04_panel_size")
    write_meta(d, **vars(args))

    tissue = load_dataset("lymphnode")
    all_genes = np.asarray(tissue.adata.var_names)
    canon_full = canonical_markers(tissue, "lymphnode")
    markers = sorted(set().union(*canon_full.values()) & set(all_genes))
    X = tissue.adata.X
    tissue_mean = np.asarray(X.mean(axis=0)).ravel()
    centers = ["mean", "median", "nonmarker"]

    rows = []
    for focal in args.focals:
        fd_full = focal_data(tissue, "lymphnode", focal)
        lin = focal_lineage(tissue, fd_full)
        full = {}
        for c in centers:
            p = d / f"full_panel__{focal}__{c}.csv"
            if not done(p):
                df = run_fit(tissue, fd_full, c, canon_full, lin, args.n_perm,
                             args.seed)
                if df is not None:
                    df.to_csv(p, index=False)
            if p.exists():
                full[c] = pd.read_csv(p)
        for c in centers:
            if c in full and c != "mean":
                rows.append(dict(focal=focal, size=len(all_genes),
                                 strategy="full", rep=0, center=c,
                                 n_markers_panel=len(markers),
                                 **compare(full[c], full[c]),
                                 **{f"vsmean_{k}": v for k, v in
                                    compare(full["mean"], full[c]).items()}))

        for size in sizes:
            for strategy in ("random", "top_expressed", "marker_enriched"):
                for r in range(1 if strategy == "top_expressed" else reps):
                    rng = np.random.default_rng(10_000 + 100 * size + r)
                    if strategy == "random":
                        panel = rng.choice(all_genes, size, replace=False)
                    elif strategy == "top_expressed":
                        panel = all_genes[np.argsort(-tissue_mean)[:size]]
                    else:
                        m = rng.choice(markers, min(size // 2, len(markers)),
                                       replace=False)
                        rest = np.setdiff1d(all_genes, m)
                        panel = np.concatenate(
                            [m, rng.choice(rest, size - len(m), replace=False)])
                    fd = focal_data(tissue, "lymphnode", focal, genes=panel)
                    canon = canonical_markers(tissue, "lymphnode", genes=panel)
                    n_mark = sum(len(v) for v in canon.values())
                    for c in centers:
                        df = run_fit(tissue, fd, c, canon, lin, args.n_perm,
                                     args.seed + r)
                        if df is None or c not in full:
                            continue
                        rows.append(dict(focal=focal, size=size,
                                         strategy=strategy, rep=r, center=c,
                                         n_markers_panel=n_mark,
                                         **compare(full[c], df),
                                         **{f"vsmean_{k}": v for k, v in
                                            compare(full["mean"], df).items()}))
                        log.info("%s size %d %s rep %d %s: %s", focal, size,
                                 strategy, r, c, {k: rows[-1][k] for k in
                                                  ("recall", "precision",
                                                   "median_R2_total_ratio")})
                    pd.DataFrame(rows).to_csv(d / "panel_runs.csv", index=False)
    pd.DataFrame(rows).to_csv(d / "panel_runs.csv", index=False)


if __name__ == "__main__":
    main()
