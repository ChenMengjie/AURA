"""
AURA spot mode: composition dependence for multi-cell spots (e.g. 10x Visium).

At spot resolution a spot's own cell mixture determines most of its
expression, so the cell-level question "does this cell's expression depend
on its neighbors?" becomes "does a spot's expression depend on the
composition of the *surrounding* spots, beyond its own composition?".

Model, per gene g over spots s:

    r_sg   Pearson residual under the NB null (spot-centered, as in run_model)
    W      [1, own composition pi_s]                 (adjustment covariates)
    N      composition of neighboring spots (rings), excluding spot s

Both r and N are residualized on W (Frisch–Waugh–Lovell), and the omnibus
statistic Q_g = ||N~^T r~_g||² is tested by permuting rows of N~, as in the
single-cell model. β_g are the partial effects of neighbor composition given
own composition.
"""

import logging

import numpy as np
from scipy.spatial import cKDTree

from .model import (_estimate_phi, compute_pearson_residuals, permutation_test,
                    _finalize)

log = logging.getLogger(__name__)

__all__ = ["spot_spacing", "neighbor_composition", "expression_pcs",
           "run_model_spot"]


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
