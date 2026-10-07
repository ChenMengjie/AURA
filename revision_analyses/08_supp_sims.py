"""
08 — Two small simulations for supplementary figures S9 and S12.

(a) Ground-truth R2_total on the lymph node Tfh scaffold: counts simulated
    with known composition effects; the exact R2_total and the legacy
    approximation are compared with the true count-variance share of the
    composition term.
(b) Centering leakage: coherent composition effects in many genes leak into
    null genes under per-cell mean centering; reference-gene centering
    prevents it (same design as tests/test_revision.py, more permutations).

Outputs: r2_groundtruth.csv, r2_groundtruth_summary.csv, centering_leakage.csv
"""
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
from aura.model import (compute_composition_vectors, composition_kernel,
                        compute_pearson_residuals, permutation_test,
                        estimate_dispersion, bh_fdr, run_model)
from common import (parse_args, outdir, write_meta, load_dataset, focal_data,
                    plant_effects, simulate_counts, log)


def main():
    args = parse_args(__doc__, default_n_perm=2000)
    d = outdir("08_supp_sims"); write_meta(d, n_perm=args.n_perm, seed=args.seed)
    tissue = load_dataset("lymphnode"); fd = focal_data(tissue, "lymphnode", "Tfh")
    P = compute_composition_vectors(fd["xy"], tissue.all_xy, tissue.all_types, k=30,
                                    n_types=len(tissue.type_names))
    _, Pt = composition_kernel(P)
    counts0 = fd["counts"]; mu = counts0.mean(0); phi = estimate_dispersion(counts0)
    n, G, K = len(Pt), 200, Pt.shape[1]
    rows = []
    for rep in range(3):
        rng = np.random.default_rng(100 + rep)
        gsel = rng.choice(counts0.shape[1], G, replace=False); mu_g = mu[gsel]
        B, truth = plant_effects(G, K, 60, 1.0, rng)
        alpha = rng.normal(0, 0.15, (n, 1))
        eta = Pt @ B.T + alpha
        lam = mu_g[None, :] * np.exp(eta)
        X = rng.negative_binomial(phi, phi / (phi + lam)).astype(float)
        # true composition share: variance of the composition-driven mean,
        # marginal over alpha, relative to total count variance
        m_comp = mu_g[None, :] * np.exp(Pt @ B.T) * np.exp(alpha).mean()
        truth_share = m_comp.var(0) / X.var(0)
        res = run_model(X, fd["xy"], tissue.all_xy, tissue.all_types, k=30,
                        n_perm=args.n_perm, seed=rep)
        for g in range(G):
            rows.append(dict(rep=rep, gene=g, planted=bool(truth[g]),
                             truth_share=truth_share[g], R2_total=res["R2_total"][g],
                             R2_total_legacy=res["R2_total_legacy"][g],
                             R2_total_adj=res["R2_total_adj"][g],
                             significant=bool(res["significant"][g]), pvalue=res["pvalues"][g]))
    df = pd.DataFrame(rows); df.to_csv(d / "r2_groundtruth.csv", index=False)
    pl = df[df.planted & (df.truth_share > 0)]
    summ = dict(n_planted=len(pl),
                median_ratio_exact=float(np.median(pl.R2_total / pl.truth_share)),
                median_ratio_legacy=float(np.median(pl.R2_total_legacy / pl.truth_share)),
                pearson_exact=pearsonr(pl.R2_total, pl.truth_share)[0],
                pearson_legacy=pearsonr(pl.R2_total_legacy, pl.truth_share)[0],
                spearman_exact=spearmanr(pl.R2_total, pl.truth_share)[0],
                spearman_legacy=spearmanr(pl.R2_total_legacy, pl.truth_share)[0],
                median_truth=float(pl.truth_share.median()),
                median_null_exact=float(df[~df.planted].R2_total.median()),
                median_null_adj=float(df[~df.planted].R2_total_adj.median()),
                n_null_sig=int(df[~df.planted].significant.sum()), n_null=int((~df.planted).sum()),
                power=float(pl.significant.mean()))
    pd.DataFrame([summ]).to_csv(d / "r2_groundtruth_summary.csv", index=False)
    log.info("ground truth: %s", summ)

    # (b) centering leakage
    rows = []
    for rep in range(3):
        rng = np.random.default_rng(12 + rep)
        n2, G2, K2, n_sig = 1500, 120, 5, 50
        P2 = rng.dirichlet(np.ones(K2) * 0.5, size=n2); _, Pt2 = composition_kernel(P2)
        beta = np.zeros((G2, K2)); beta[:n_sig] = np.outer(rng.uniform(1, 3, n_sig), rng.normal(0, 1, K2))
        eta = Pt2 @ beta.T; lam = 4 * np.exp(eta) / np.exp(eta).mean(axis=0)
        X = rng.poisson(lam).astype(float); phi2 = estimate_dispersion(X)
        ref = np.zeros(G2, bool); ref[n_sig + 20:] = True; null = np.arange(G2) >= n_sig
        for name, kw in [("mean", {}), ("median", dict(center="median")), ("reference", dict(center_genes=ref))]:
            r = compute_pearson_residuals(X, phi2, **kw)
            _, p = permutation_test(r, Pt2, n_perm=args.n_perm, seed=1)
            q, _ = bh_fdr(p)
            rows.append(dict(rep=rep, centering=name, fpr_null=float((p[null] < .05).mean()),
                             n_null_fdr=int((q[null] <= .05).sum()), n_null=int(null.sum()),
                             tpr=float((q[~null] <= .05).mean())))
            log.info("leakage %s: %s", name, rows[-1])
    pd.DataFrame(rows).to_csv(d / "centering_leakage.csv", index=False)


if __name__ == "__main__":
    main()
