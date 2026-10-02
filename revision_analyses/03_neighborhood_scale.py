"""
03 — Neighborhood scale: k, physical radius and concentric rings
(Reviewer 1, M3; editor: parameter choice).

For each dataset/focal type:
  * kNN with k in {10, 15, 30, 50, 100}, reporting the physical radius each
    k corresponds to (distance to the k-th neighbor)
  * fixed radii expressed in units of the dataset's median cell spacing s
    (distance to nearest neighbor), so Xenium and CosMx are compared at the
    same scale in cells; µm are also reported (Xenium coordinates are µm)
  * three rings [0, 3s), [3s, 8s), [8s, 20s) — contact / short-range
    paracrine / longer-range — with per-ring p-values

Agreement with the default (k = 30): Jaccard of significant sets, Spearman
of -log10 p over all genes, median cosine of β over shared significant genes.

Outputs: neighborhood_runs.csv, scale_summary.csv, ring_genes.csv,
results/<dataset>__<focal>__<setting>.csv
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from common import (parse_args, outdir, write_meta, done, load_dataset,
                    focal_data, fit, result_table, jaccard, cosine_rows, log)

TARGETS = {"lymphnode": ["Tfh", "Macrophage"],
           "ipf": ["AT2", "SPP1+ Macrophages"],
           "nsclc": ["macrophage"]}
KS = [10, 15, 30, 50, 100]
RADII_S = [2, 3, 5, 10, 20]          # in units of median cell spacing
RINGS_S = [3, 8, 20]


def spacing(tissue, fd):
    """Median nearest-neighbor distance and distance to the k-th neighbor."""
    tree = cKDTree(tissue.all_xy)
    d, _ = tree.query(fd["xy"], k=max(KS) + 1)
    s = float(np.median(d[:, 1]))
    return s, {k: float(np.median(d[:, k])) for k in KS}


def main():
    args = parse_args(__doc__, lambda ap: ap.add_argument(
        "--datasets", nargs="+", default=list(TARGETS)), default_n_perm=2000)
    d = outdir("03_neighborhood_scale")
    (d / "results").mkdir(exist_ok=True)
    write_meta(d, **vars(args))

    runs = []
    for ds in args.datasets:
        tissue = load_dataset(ds)
        for focal in TARGETS[ds]:
            fd = focal_data(tissue, ds, focal)
            s, r_k = spacing(tissue, fd)
            log.info("%s/%s: median spacing %.2f; radius at k: %s", ds, focal,
                     s, {k: round(v, 1) for k, v in r_k.items()})
            settings = [(f"knn_k{k}", dict(k=k), r_k[k]) for k in KS]
            settings += [(f"radius_{m}s", dict(radius=m * s), m * s) for m in RADII_S]
            settings += [("rings_3_8_20s", dict(rings=[m * s for m in RINGS_S]),
                          RINGS_S[-1] * s)]
            if args.quick:
                settings = [x for x in settings if x[0] in
                            ("knn_k10", "knn_k30", "radius_5s", "rings_3_8_20s")]
            for name, kw, radius in settings:
                tag = f"{ds}__{focal.replace('/', '_')}__{name}"
                path = d / "results" / f"{tag}.csv"
                if not done(path):
                    res = fit(tissue, fd, args.n_perm, seed=args.seed,
                              min_neighbors=3, **kw)
                    genes = fd["genes"]
                    df = result_table(res, genes, tissue.type_names)
                    df["n_cells_used"] = int(res["cell_mask"].sum())
                    df["median_neighbors"] = float(np.median(res["n_neighbors"].sum(1)))
                    df.to_csv(path, index=False)
                df = pd.read_csv(path)
                runs.append(dict(dataset=ds, focal=focal, setting=name,
                                 radius_units=radius, radius_in_spacings=radius / s,
                                 median_spacing=s, n_cells=int(df.n_cells_used.iloc[0]),
                                 median_neighbors=df.median_neighbors.iloc[0],
                                 n_genes=len(df), n_sig=int(df.significant.sum()),
                                 path=str(path)))
    runs = pd.DataFrame(runs)
    runs.to_csv(d / "neighborhood_runs.csv", index=False)

    # agreement with the default k = 30
    summ, ring_rows = [], []
    for (ds, focal), grp in runs.groupby(["dataset", "focal"]):
        ref = pd.read_csv(grp[grp.setting == "knn_k30"].path.iloc[0]).set_index("gene")
        bref = [c for c in ref.columns if c.startswith("beta_") and "_ring" not in c]
        for _, row in grp.iterrows():
            df = pd.read_csv(row.path).set_index("gene")
            common = ref.index.intersection(df.index)
            a, b = ref.loc[common], df.loc[common]
            out = row.drop("path").to_dict()
            out["jaccard_vs_k30"] = jaccard(a.index[a.significant], b.index[b.significant])
            out["spearman_logp_vs_k30"] = spearmanr(-np.log10(a.pvalue),
                                                    -np.log10(b.pvalue))[0]
            shared = a.significant & b.significant
            bcols = [c for c in bref if c in b.columns]
            if shared.any() and len(bcols) == len(bref):
                out["median_beta_cosine_vs_k30"] = float(np.nanmedian(cosine_rows(
                    a.loc[shared, bref].values, b.loc[shared, bcols].values)))
            summ.append(out)

            if row.setting.startswith("rings"):
                pr = [c for c in df.columns if c.startswith("pvalue_ring")]
                sig = df[df.significant]
                for g, r in sig.iterrows():
                    ring_rows.append(dict(dataset=ds, focal=focal, gene=g,
                                          **{c: r[c] for c in pr},
                                          strongest_ring=int(np.argmin(r[pr].values))))
    pd.DataFrame(summ).to_csv(d / "scale_summary.csv", index=False)
    pd.DataFrame(ring_rows).to_csv(d / "ring_genes.csv", index=False)


if __name__ == "__main__":
    main()
