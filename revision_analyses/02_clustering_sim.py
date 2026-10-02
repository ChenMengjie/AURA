"""
02 — Effect of clustering errors on calibration and power (Reviewer 1, M1).

Simulations keep the real lymph-node geometry and composition and simulate
counts from the real focal-type means/dispersion (Fig. 2 design).

  A  under-clustering: a fraction f of a second cell type is merged into the
     focal population; it differs in mean expression for 20% of genes
     (unresolved subtype)
  B  over-clustering: analyse (i) random halves/quarters of the focal type,
     (ii) halves split by k-means on composition
  C  neighbor-label noise: composition computed from labels with a fraction
     e of non-focal cells relabelled (random, or only within confusable pairs)
  D  real data: composition granularity coarse / default / fine

Outputs: sim_A.csv, sim_B.csv, sim_C.csv, granularity_D.csv
"""

import numpy as np
import pandas as pd

from aura.model import (compute_composition_vectors, composition_kernel,
                        estimate_dispersion, run_model)
from common import (parse_args, outdir, write_meta, done, load_dataset,
                    focal_data, plant_effects, simulate_counts, sim_metrics,
                    jaccard, log)

COARSE = {"B": "B_all", "GC_B": "B_all", "T_conv": "T_all", "Tfh": "T_all",
          "Treg": "T_all", "Mac": "Myeloid", "FDC": "Stromal_all",
          "Stromal": "Stromal_all", "Endo": "Endo", "Other": "Other"}
CONFUSABLE = [("B", "GC_B"), ("T_conv", "Treg"), ("FDC", "Stromal")]


def extra(ap):
    ap.add_argument("--focal", default="Tfh")
    ap.add_argument("--contaminant", default="Treg",
                    help="cell_type_nebula label merged into the focal set (A)")
    ap.add_argument("--scale", type=float, default=0.75)
    ap.add_argument("--n-sig", type=int, default=100)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--k", type=int, default=30)


def null_params(counts):
    mu = np.maximum(counts.mean(axis=0), 1e-3)
    return mu, estimate_dispersion(counts)


def run(counts, xy, tissue, all_types, args, seed):
    return run_model(counts, xy, tissue.all_xy, all_types, k=args.k,
                     n_perm=args.n_perm, seed=seed)


def main():
    args = parse_args(__doc__, extra, default_n_perm=2000)
    reps = 1 if args.quick else args.reps
    d = outdir("02_clustering_sim")
    write_meta(d, **vars(args))

    tissue = load_dataset("lymphnode")
    fd = focal_data(tissue, "lymphnode", args.focal)
    mu, phi = null_params(fd["counts"])
    G, K = len(mu), len(tissue.type_names)
    P = compute_composition_vectors(fd["xy"], tissue.all_xy, tissue.all_types,
                                    k=args.k, n_types=K)
    _, Pt = composition_kernel(P)
    log.info("focal %s: %d cells, %d genes, phi=%.2f", args.focal,
             len(fd["idx"]), G, phi)

    # ── A: under-clustering ─────────────────────────────────────────
    if not done(d / "sim_A.csv"):
        cont = focal_data(tissue, "lymphnode", args.contaminant,
                          min_gene_mean=0)
        cxy_all = cont["xy"]
        rows = []
        for f in [0, 0.02, 0.05, 0.1, 0.2, 0.3]:
            for r in range(reps):
                rng = np.random.default_rng(1000 * r + int(f * 100))
                # contaminants make up a fraction f of the merged population
                n_c = min(int(round(f / (1 - f) * len(fd["idx"]))), len(cxy_all))
                pick = rng.choice(len(cxy_all), n_c, replace=False)
                xy = np.vstack([fd["xy"], cxy_all[pick]])
                Pa = compute_composition_vectors(xy, tissue.all_xy,
                                                 tissue.all_types, k=args.k,
                                                 n_types=K)
                _, Pta = composition_kernel(Pa)
                B, truth = plant_effects(G, K, args.n_sig, args.scale, rng)
                shift = np.zeros((len(xy), G))
                sub = rng.random(G) < 0.2
                shift[len(fd["xy"]):, sub] = rng.normal(0, 1, sub.sum()) * np.log(2)
                X = simulate_counts(Pta, mu, phi, B, rng, log_shift=shift)
                res = run(X, xy, tissue, tissue.all_types, args, seed=r)
                rows.append(dict(scenario="A_under", frac_merged=f,
                                 n_contaminant=n_c, rep=r,
                                 **sim_metrics(res, truth, B)))
                log.info("A f=%.2f rep %d: %s", f, r, rows[-1])
        pd.DataFrame(rows).to_csv(d / "sim_A.csv", index=False)

    # ── B: over-clustering ──────────────────────────────────────────
    if not done(d / "sim_B.csv"):
        from scipy.cluster.vq import kmeans2
        rows = []
        for r in range(reps):
            rng = np.random.default_rng(2000 + r)
            B, truth = plant_effects(G, K, args.n_sig, args.scale, rng)
            X = simulate_counts(Pt, mu, phi, B, rng)
            splits = {"full": [np.arange(len(X))]}
            for m in (2, 4):
                perm = rng.permutation(len(X))
                splits[f"random_{m}"] = np.array_split(perm, m)
            _, lab = kmeans2(P, 2, seed=r, minit="++")
            splits["composition_kmeans_2"] = [np.where(lab == c)[0] for c in (0, 1)]
            for name, parts in splits.items():
                for j, idx in enumerate(parts):
                    if len(idx) < 100:
                        continue
                    res = run(X[idx], fd["xy"][idx], tissue, tissue.all_types,
                              args, seed=r)
                    rows.append(dict(scenario="B_over", split=name, part=j,
                                     n_cells=len(idx), rep=r,
                                     **sim_metrics(res, truth, B)))
                    log.info("B %s part %d: %s", name, j, rows[-1])
        pd.DataFrame(rows).to_csv(d / "sim_B.csv", index=False)

    # ── C: neighbor-label noise ─────────────────────────────────────
    if not done(d / "sim_C.csv"):
        names = tissue.type_names
        focal_type = np.bincount(tissue.all_types[fd["idx"]]).argmax()
        nonfocal = np.where(tissue.all_types != focal_type)[0]
        pair_of = {}
        for a, b in CONFUSABLE:
            if a in names and b in names:
                ia, ib = names.index(a), names.index(b)
                pair_of[ia], pair_of[ib] = ib, ia
        rows = []
        for r in range(reps):
            rng = np.random.default_rng(3000 + r)
            B, truth = plant_effects(G, K, args.n_sig, args.scale, rng)
            X = simulate_counts(Pt, mu, phi, B, rng)
            for mode in ("random", "confusable"):
                for e in [0, 0.05, 0.1, 0.2, 0.3]:
                    noisy = tissue.all_types.copy()
                    flip = nonfocal[rng.random(len(nonfocal)) < e]
                    if mode == "random":
                        others = np.array([t for t in range(K) if t != focal_type])
                        new = others[rng.integers(0, len(others), len(flip))]
                        same = new == noisy[flip]
                        new[same] = others[(np.searchsorted(others, new[same]) + 1)
                                           % len(others)]
                        noisy[flip] = new
                    else:
                        flip = flip[np.isin(noisy[flip], list(pair_of))]
                        noisy[flip] = [pair_of[t] for t in noisy[flip]]
                    res = run(X, fd["xy"], tissue, noisy, args, seed=r)
                    rows.append(dict(scenario="C_label_noise", mode=mode,
                                     flip_rate=e, n_flipped=len(flip), rep=r,
                                     **sim_metrics(res, truth, B)))
                    log.info("C %s e=%.2f: %s", mode, e, rows[-1])
        pd.DataFrame(rows).to_csv(d / "sim_C.csv", index=False)

    # ── D: real-data granularity ────────────────────────────────────
    if not done(d / "granularity_D.csv"):
        obs = tissue.adata.obs
        levels = {
            "coarse": obs["lineage_aura"].astype(str).map(COARSE).fillna("Other"),
            "default": obs["lineage_aura"].astype(str),
            "fine": obs["cell_type_nebula"].astype(str),
        }
        fits = {}
        for lvl, lab in levels.items():
            names = sorted(lab.unique())
            types = np.array([names.index(x) for x in lab])
            res = run(fd["counts"], fd["xy"], tissue, types, args, seed=0)
            fits[lvl] = (res, names)
        ref, ref_names = fits["default"]
        ref_sig = set(fd["genes"][ref["significant"]])
        rows = []
        for lvl, (res, names) in fits.items():
            sig = set(fd["genes"][res["significant"]])
            row = dict(level=lvl, n_types=len(names), n_sig=len(sig),
                       jaccard_vs_default=jaccard(sig, ref_sig),
                       recall_of_default=len(sig & ref_sig) / max(len(ref_sig), 1))
            if lvl == "fine":
                # dominant fine driver mapped to its lineage == default driver?
                fine2lin = (obs.groupby(obs["cell_type_nebula"].astype(str))
                            ["lineage_aura"].agg(lambda s: s.astype(str).mode()[0]))
                both = res["significant"] & ref["significant"]
                drv_f = np.array(names)[np.abs(res["beta"][both]).argmax(1)]
                drv_d = np.array(ref_names)[np.abs(ref["beta"][both]).argmax(1)]
                row["driver_agreement"] = float(np.mean(
                    [fine2lin.get(a) == b for a, b in zip(drv_f, drv_d)])) \
                    if both.any() else np.nan
            rows.append(row)
        pd.DataFrame(rows).to_csv(d / "granularity_D.csv", index=False)


if __name__ == "__main__":
    main()
