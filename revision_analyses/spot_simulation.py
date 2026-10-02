"""
Supplementary simulation: why AURA is not applied to standard (55 µm) Visium.

A spot-level analogue of AURA tests whether a spot's expression depends on the
composition of surrounding spots after adjusting for its own composition
(Frisch–Waugh–Lovell residualization of residuals and neighbor composition on
own composition, then the omnibus permutation test). On a simulated Visium
grid with mixture expression, the test is calibrated only when own
composition is known exactly; realistic deconvolution error leaves own-mixture
signal in the residuals, which spatially smooth neighbor composition proxies
for, and the false-positive rate approaches 1. Adding own-expression PCs does
not repair it.

Run: python spot_simulation.py  (prints the calibration table)
"""
import logging

import numpy as np
from scipy.spatial import cKDTree

from aura.model import (_estimate_phi, compute_pearson_residuals,
                        permutation_test, _finalize)

log = logging.getLogger(__name__)

def spot_spacing(xy):
    """Median center-to-center distance between nearest spots."""
    d, _ = cKDTree(xy).query(xy, k=2)
    return float(np.median(d[:, 1]))


def neighbor_composition(xy, comp, rings=(1.5,), spacing=None, weights=None):
    """Average composition of surrounding spots in concentric rings.

    Args:
        xy: (S, 2) spot coordinates
        comp: (S, K) per-spot composition (rows sum to 1)
        rings: outer radii of the rings in units of spot spacing; the
            default (1.5,) is the first hexagonal ring (6 neighbors).
            (1.5, 2.5) adds the second ring as a separate block.
        spacing: spot spacing (estimated if None)
        weights: optional (S,) per-spot weights, e.g. estimated cell counts,
            so that neighbors contribute in proportion to their cells

    Returns:
        N: (S, K * len(rings)) neighbor composition (each block sums to 1,
           or 0 if a spot has no neighbors in a ring)
        n_neighbors: (S, len(rings))
    """
    if spacing is None:
        spacing = spot_spacing(xy)
    w = np.ones(len(xy)) if weights is None else np.asarray(weights, float)
    edges = np.concatenate([[0.0], np.asarray(rings, float) * spacing])
    tree = cKDTree(xy)
    pairs = tree.sparse_distance_matrix(tree, edges[-1], output_type="ndarray")
    keep = pairs["v"] >= 1e-10
    i, j, d = pairs["i"][keep], pairs["j"][keep], pairs["v"][keep]
    ring = np.searchsorted(edges, d, side="right") - 1
    ok = ring < len(rings)
    i, j, ring = i[ok], j[ok], ring[ok]

    S, K = comp.shape
    N = np.zeros((S, len(rings), K))
    n_nbrs = np.zeros((S, len(rings)))
    wsum = np.zeros((S, len(rings)))
    np.add.at(N, (i, ring), comp[j] * w[j, None])
    np.add.at(wsum, (i, ring), w[j])
    np.add.at(n_nbrs, (i, ring), 1)
    N = N / np.maximum(wsum, 1e-12)[:, :, None]
    return N.reshape(S, -1), n_nbrs


def _residualize(M, W):
    """M minus its least-squares projection on the columns of W."""
    coef, _, _, _ = np.linalg.lstsq(W, M, rcond=1e-10)
    return M - W @ coef


def expression_pcs(counts, n_pcs=20, target_sum=1e4):
    """Top principal components of each spot's own log-normalized expression.

    Used as additional own-spot covariates: they summarize the spot's
    cellular mixture directly from its expression, absorbing the
    composition signal that deconvolution error leaves behind.
    """
    lib = np.maximum(counts.sum(axis=1, keepdims=True), 1)
    Y = np.log1p(counts / lib * target_sum)
    Y = Y - Y.mean(axis=0)
    sd = Y.std(axis=0)
    Y = Y[:, sd > 0] / sd[sd > 0]
    U, s, _ = np.linalg.svd(Y, full_matrices=False)
    return U[:, :n_pcs] * s[:n_pcs]


def run_model_spot(counts, xy, comp, focal_mask=None, rings=(1.5,),
                   spacing=None, weights=None, adjust_own=True,
                   covariates=None, n_pcs=0,
                   n_perm=5000, alpha=0.05, min_mean=0.5, seed=42,
                   min_neighbors=3, dispersion="shared", center="mean",
                   center_genes=None):
    """Test dependence of spot expression on neighboring-spot composition.

    Args:
        counts: (S, G) spot counts
        xy: (S, 2) spot coordinates
        comp: (S, K) own composition from deconvolution (rows sum to 1)
        focal_mask: optional (S,) boolean; spots to test (e.g. spots where a
            lineage exceeds a purity threshold). Neighbor composition always
            uses all spots.
        rings: ring outer radii in spot spacings (see neighbor_composition)
        spacing: spot spacing (estimated if None)
        weights: optional per-spot cell-count weights for neighbor averaging
        adjust_own: residualize on own composition (default True; False is
            only for diagnostics)
        covariates: optional (S, C) additional own-spot covariates
        n_pcs: number of own-expression principal components to add as
            covariates (see `expression_pcs`); recommended when composition
            comes from deconvolution
        n_perm, alpha, min_mean, seed, dispersion, center, center_genes:
            as in `run_model`
        min_neighbors: spots with fewer neighbors in any ring are excluded

    Returns:
        dict with the keys of `run_model`; `P` / `P_tilde` hold the raw and
        residualized neighbor composition, and `own_comp` the adjustment
        covariates for the tested spots.
    """
    counts = np.asarray(counts, dtype=np.float64)
    comp = np.asarray(comp, dtype=np.float64)
    S, K = comp.shape
    N, n_nbrs = neighbor_composition(xy, comp, rings=rings, spacing=spacing,
                                     weights=weights)

    cell_mask = (n_nbrs >= min_neighbors).all(axis=1)
    if focal_mask is not None:
        cell_mask &= np.asarray(focal_mask, bool)
    log.info("Spot mode: testing %d of %d spots (K=%d, rings=%s)",
             cell_mask.sum(), S, K, list(rings))
    extra = []
    if n_pcs:
        extra.append(expression_pcs(counts, n_pcs=n_pcs))
    if covariates is not None:
        extra.append(np.asarray(covariates, float))
    extra = np.column_stack(extra)[cell_mask] if extra else None
    counts, N, own = counts[cell_mask], N[cell_mask], comp[cell_mask]

    phi = _estimate_phi(counts, dispersion, min_mean)
    residuals = compute_pearson_residuals(counts, phi, center=center,
                                          center_genes=center_genes)

    # own composition has rank K-1 given the intercept; drop one column
    W = np.column_stack([np.ones(len(own)), own[:, :-1]]) if adjust_own \
        else np.ones((len(own), 1))
    if extra is not None:
        W = np.column_stack([W, extra])
    residuals = _residualize(residuals, W)
    N_tilde = _residualize(N, W)

    blocks = None
    if len(rings) > 1:
        blocks = [np.arange(b * K, (b + 1) * K) for b in range(len(rings))]
    out = permutation_test(residuals, N_tilde, n_perm=n_perm, seed=seed,
                           blocks=blocks)
    Q, pvalues = out[0], out[1]
    block_pvalues = out[2] if len(out) == 3 else None

    res = _finalize(counts, phi, residuals, N, N_tilde, None, Q, pvalues,
                    block_pvalues, alpha, cell_mask=cell_mask,
                    n_neighbors=n_nbrs,
                    neighborhood={"type": "spot_rings", "rings": list(rings),
                                  "adjust_own": adjust_own, "n_pcs": n_pcs,
                                  "n_covariates": W.shape[1]})
    res["own_comp"] = own
    return res


def main():
    import logging
    from scipy.stats import chi2
    from aura.model import bh_fdr
    logging.getLogger('aura').setLevel(logging.WARNING)
    rng = np.random.default_rng(0)
    rows, cols = 60, 70
    xy = np.array([(c*100 + (r % 2)*50, r*86.6) for r in range(rows) for c in range(cols)], float)
    S, K, G = len(xy), 8, 300
    cent = rng.uniform(0, xy.max(), (60, 2)); typ = rng.integers(0, K, 60)
    w = np.exp(-((xy[:, None]-cent[None])**2).sum(-1)/(2*250**2))
    logit = np.stack([np.log(w[:, typ == k].sum(1)+1e-3) for k in range(K)], 1)
    comp = np.exp(1.5*logit); comp /= comp.sum(1, keepdims=True)
    sig = np.exp(rng.normal(0, 1.5, (K, G)))
    ncell = rng.poisson(8, S)+1
    N, _ = neighbor_composition(xy, comp, rings=(1.5,)); Nc = N-N.mean(0)
    lam_gc = lambda p: np.median(chi2.isf(np.clip(p, 1e-300, 1), 1))/chi2.ppf(.5, 1)

    def sim(n_sig=0, scale=1.0, het=0.0, smooth=0.0):
        mu = (comp*ncell[:, None])@sig*0.3
        beta = np.zeros((G, K)); truth = np.zeros(G, bool)
        if n_sig:
            idx = rng.choice(G, n_sig, replace=False); truth[idx] = True
            beta[idx] = rng.normal(0, scale, (n_sig, K))
            mu = mu*np.exp(Nc@beta.T)
        if het:   # spot-level capture efficiency
            mu = mu*rng.gamma(1/het, het, (S, 1))
        if smooth:  # unmodeled smooth spatial field per gene (not composition)
            f = np.sin(xy[:, :1]/700*rng.uniform(.5, 2, G) + rng.uniform(0, 6, G))
            mu = mu*np.exp(smooth*f)
        return rng.negative_binomial(5, 5/(5+mu)).astype(float), truth

    def noisy(c, conc):
        return np.vstack([rng.dirichlet(conc*ci+1e-3) for ci in c])

    def report(name, X, truth, own=comp, **kw):
        r = run_model_spot(X, xy, own, n_perm=1000, seed=1, **kw)
        p = r['pvalues']; q, _ = bh_fdr(p); null = ~truth
        tpr = f"{(q[truth] <= .05).mean():.2f}" if truth.any() else "  - "
        print(f"{name:44s} {np.mean(p[null] < .05):6.3f} {lam_gc(p[null]):6.2f} {int((q[null] <= .05).sum()):5d} {tpr}")

    print(f"{'scenario':44s} {'FPR':>6s} {'lamGC':>6s} {'nFDR0':>5s} TPR")
    X, t = sim(); report('null', X, t)
    report('null, no own-composition adjustment', X, t, adjust_own=False)
    X, t = sim(het=0.3); report('null + spot capture variation', X, t)
    for conc in (200, 50, 20):
        X, t = sim(); report(f'null, noisy deconvolution (conc={conc})', X, t, own=noisy(comp, conc))
    X, t = sim(smooth=0.3); report('null + smooth non-composition field', X, t)
    X, t = sim(n_sig=30, scale=0.5); report('30 genes with neighbor effects', X, t)
    X, t = sim(n_sig=30, scale=0.5); report('  ... with noisy deconvolution (conc=50)', X, t, own=noisy(comp, 50))
    print('--- with own-expression PCs as covariates')
    for L in (10, 20, 40):
        X, t = sim(); report(f'null, noisy deconv (conc=50), PCs={L}', X, t, own=noisy(comp, 50), n_pcs=L)
    X, t = sim(); report('null, exact composition, PCs=20', X, t, n_pcs=20)
    X, t = sim(het=0.3); report('null + capture variation, noisy, PCs=20', X, t, own=noisy(comp, 50), n_pcs=20)
    X, t = sim(n_sig=30, scale=1.0); report('30 effect genes (scale 1), exact, PCs=0', X, t)
    X, t = sim(n_sig=30, scale=1.0); report('30 effect genes (scale 1), noisy, PCs=20', X, t, own=noisy(comp, 50), n_pcs=20)
    X, t = sim(smooth=0.3); report('null + smooth field, PCs=20', X, t, own=noisy(comp, 50), n_pcs=20)


if __name__ == "__main__":
    main()
