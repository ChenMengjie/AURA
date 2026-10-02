"""
07 — Per-cell centering choices on the real targeted panels (S8, S9).

For every published IPF and NSCLC focal type (multi-sample mode), refit with
    mean              v0.1 default
    median            robust per-cell location
    nonmarker         center on genes that are not canonical markers of any
                      non-focal lineage (the genes through which spillover
                      enters)
    mean+samples      mean, plus within-sample removal of gene and
                      composition means (`center_samples=True`)

and report significant counts, overlap with the default, spillover-suspect
rates, and, for genes that lose significance under nonmarker centering,
whether their dominant driver was a lineage whose markers dominate the
panel (the leakage signature).

Outputs: results/<dataset>__<focal>__<variant>.csv, centering_summary.csv
"""

import numpy as np
import pandas as pd

from common import (FOCAL, parse_args, outdir, write_meta, done, load_dataset,
                    focal_data, fit, result_table, canonical_markers,
                    focal_lineage, spillover_flags, non_marker_genes, jaccard,
                    log)

VARIANTS = {"mean": {}, "median": {"center": "median"},
            "nonmarker": None, "mean+samples": {"center_samples": True}}


def main():
    args = parse_args(__doc__, lambda ap: ap.add_argument(
        "--datasets", nargs="+", default=["ipf", "nsclc"]), default_n_perm=5000)
    d = outdir("07_centering")
    (d / "results").mkdir(exist_ok=True)
    write_meta(d, **vars(args))
    rows = []
    for ds in args.datasets:
        tissue = load_dataset(ds)
        canonical = canonical_markers(tissue, ds)
        n_mark = {k: len(v) for k, v in canonical.items()}
        for focal in FOCAL[ds][:1] if args.quick else FOCAL[ds]:
            fd = focal_data(tissue, ds, focal)
            lin = focal_lineage(tissue, fd)
            tabs = {}
            for name, kw in VARIANTS.items():
                path = d / "results" / f"{ds}__{focal.replace('/', '_')}__{name}.csv"
                if not done(path):
                    if kw is None:
                        ref = non_marker_genes(fd["genes"], canonical, lin)
                        log.info("%s/%s: centering on %d non-marker genes of %d",
                                 ds, focal, ref.sum(), len(ref))
                        kw = {"center_genes": ref}
                    res = fit(tissue, fd, args.n_perm, seed=args.seed, **kw)
                    df = result_table(res, fd["genes"], tissue.type_names)
                    spillover_flags(df, canonical, lin,
                                    tissue.type_names).to_csv(path, index=False)
                tabs[name] = pd.read_csv(path)
            ref = tabs["mean"]
            ref_sig = set(ref.gene[ref.significant])
            for name, t in tabs.items():
                sig = t[t.significant]
                lost = ref[ref.significant & ~ref.gene.isin(sig.gene)]
                rows.append(dict(
                    dataset=ds, focal=focal, variant=name, n_genes=len(t),
                    n_center_genes=int(non_marker_genes(t.gene, canonical, lin).sum())
                    if name == "nonmarker" else len(t),
                    n_sig=len(sig), jaccard_vs_mean=jaccard(sig.gene, ref_sig),
                    n_lost_vs_mean=len(lost),
                    n_gained_vs_mean=len(set(sig.gene) - ref_sig),
                    spillover_rate=float(sig.spillover_suspect.mean())
                    if len(sig) else np.nan,
                    top_driver_of_lost=lost.driver_axis.mode().iloc[0]
                    if len(lost) else None,
                    markers_of_top_driver=n_mark.get(
                        lost.driver_axis.mode().iloc[0]) if len(lost) else None,
                    median_R2_total=float(sig.R2_total.median()) if len(sig) else np.nan,
                ))
    pd.DataFrame(rows).to_csv(d / "centering_summary.csv", index=False)


if __name__ == "__main__":
    main()
