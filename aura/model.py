"""
AURA core algorithm: composition-dependent context variance model.

Tests whether residual gene expression variance within a fixed cell type
is structured by local neighborhood composition.
"""

import logging
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize_scalar

log = logging.getLogger(__name__)

__all__ = [
    "estimate_dispersion",
    "estimate_gene_dispersion",
    "compute_pearson_residuals",
    "compute_composition_vectors",
    "composition_kernel",
    "permutation_test",
    "bh_fdr",
    "effect_size",
    "variance_decomposition",
    "run_model",
    "run_model_sample",
    "run_model_multisample",
]


def estimate_dispersion(counts, min_mean=0.5):
    """Estimate shared NB dispersion from cell-gene count matrix.

    Args:
        counts: (n_cells, n_genes) count matrix for focal cell type
        min_mean: minimum gene mean for inclusion in fit

    Returns:
        phi: shared dispersion parameter
    """
    gene_means = counts.mean(axis=0)
    gene_vars = counts.var(axis=0, ddof=1)
    mask = gene_means >= min_mean
    mu = gene_means[mask]
    sv = gene_vars[mask]

    def objective(log_phi):
        phi = np.exp(log_phi)
        predicted_var = mu + mu ** 2 / phi
        return np.sum((sv - predicted_var) ** 2)

    result = minimize_scalar(objective, bounds=(-2, 10), method="bounded")
    return np.exp(result.x)


def estimate_gene_dispersion(counts, min_mean=0.5, n_bins=20, phi_max=1e4):
    """Estimate regularized gene-specific NB dispersion (mean-dependent trend).

    Per-gene method-of-moments estimates phi_g = mu_g^2 / (var_g - mu_g) are
    smoothed against log mean by taking the median log phi within quantile
    bins of log mean and interpolating (constant extrapolation outside the
    fitted range). This follows the regularization logic of sctransform.

    Unregularized per-gene dispersion is deliberately not offered: a gene
    whose variance is inflated by composition would absorb that variance
    into its own phi_g, driving its excess variance to zero by construction.
    The per-bin median is robust to the minority of composition-responsive
    genes within each bin.

    Args:
        counts: (n_cells, n_genes) count matrix for focal cell type
        min_mean: minimum gene mean for inclusion in the trend fit
        n_bins: number of quantile bins of log mean
        phi_max: cap for genes at or below Poisson variance

    Returns:
        phi: (n_genes,) dispersion per gene
    """
    gene_means = counts.mean(axis=0)
    gene_vars = counts.var(axis=0, ddof=1)
    fit = gene_means >= min_mean
    if fit.sum() < 10:
        log.warning("Only %d genes with mean >= %.2f; falling back to shared phi",
                    fit.sum(), min_mean)
        return np.full(counts.shape[1], estimate_dispersion(counts, min_mean))

    mu = gene_means[fit]
    over = gene_vars[fit] - mu
    phi_raw = np.where(over > 0, mu ** 2 / np.maximum(over, 1e-12), phi_max)
    log_phi = np.log(np.clip(phi_raw, 1e-2, phi_max))
    log_mu = np.log(mu)

    n_bins = int(max(1, min(n_bins, fit.sum() // 10)))
    edges = np.quantile(log_mu, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.searchsorted(edges, log_mu, side="right") - 1, 0, n_bins - 1)
    centers, medians = [], []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            centers.append(np.median(log_mu[m]))
            medians.append(np.median(log_phi[m]))

    log_mu_all = np.log(np.maximum(gene_means, 1e-12))
    return np.exp(np.interp(log_mu_all, centers, medians))


def _estimate_phi(counts, dispersion, min_mean):
    if dispersion == "shared":
        return estimate_dispersion(counts, min_mean=min_mean)
    if dispersion == "gene":
        return estimate_gene_dispersion(counts, min_mean=min_mean)
    raise ValueError(f"Unknown dispersion: {dispersion!r}")


def _phi_summary(phi):
    return float(phi) if np.ndim(phi) == 0 else float(np.median(phi))


def _cell_center(residuals, center="mean", center_genes=None, trim=0.1):
    """Subtract a per-cell location estimate computed across genes."""
    R = residuals if center_genes is None else residuals[:, center_genes]
    if center == "mean":
        alpha_i = R.mean(axis=1, keepdims=True)
    elif center == "median":
        alpha_i = np.median(R, axis=1, keepdims=True)
    elif center == "trimmed":
        from scipy.stats import trim_mean
        alpha_i = trim_mean(R, trim, axis=1)[:, np.newaxis]
    else:
        raise ValueError(f"Unknown center: {center!r}")
    return residuals - alpha_i


def compute_pearson_residuals(counts, phi, center_cells=True,
                              mu_ref=None, sample_ids=None,
                              center="mean", center_genes=None):
    """Compute Pearson residuals under NB null with per-gene mean.

    Uses unconditional per-gene mean (not library-size-conditioned) to
    avoid confounding: conditioning on n_i introduces a dependency between
    genes because n_i includes counts from all genes, including any
    context-dependent ones.

    When center_cells=True, subtracts the per-cell location of the residuals
    across genes (alpha_i) from each gene's residual. This cell-level
    centering removes the shared cell-level shift caused by library size
    variation correlated with composition, isolating gene-specific context
    effects (delta_ig) from global RNA effects.

    Args:
        counts: (n_cells, n_genes) count matrix
        phi: shared dispersion (scalar) or (n_genes,) gene-specific dispersion
        center_cells: if True, apply cell-level centering. Default True.
        mu_ref: optional (n_genes,) per-gene mean to use instead of the
                in-sample mean. Used for control-conditioned residuals
                (e.g. IPF disease cells centered on healthy donor means).
        sample_ids: accepted for API compatibility. Cell-level centering is
                    a per-cell operation, so it gives identical results with
                    or without sample labels; sample-level gene centering is
                    `center_samples` in `run_model_multisample`.
        center: per-cell location estimator: 'mean' (default), 'median', or
                'trimmed' (10% trimmed mean). Robust estimators limit the
                influence of composition-responsive genes in small panels.
        center_genes: optional boolean mask or index array of genes used to
                compute alpha_i (e.g. a composition-insensitive reference
                set). Residuals of all genes are centered.

    Returns:
        residuals: (n_cells, n_genes) Pearson residuals
    """
    if mu_ref is None:
        mu = counts.mean(axis=0, keepdims=True)
    else:
        mu = np.asarray(mu_ref).reshape(1, -1)
    mu_bc = np.broadcast_to(mu, counts.shape)
    var = mu_bc + mu_bc ** 2 / phi
    residuals = (counts - mu_bc) / np.sqrt(np.maximum(var, 1e-10))

    if center_cells:
        residuals = _cell_center(residuals, center=center,
                                 center_genes=center_genes)

    return residuals


def _knn_counts(focal_xy, all_xy, all_types, k, n_types):
    """(n_c, 1, K) neighbor-type counts among the k nearest neighbors."""
    tree = cKDTree(all_xy)
    k_query = min(k + 1, len(all_xy))
    dists, indices = tree.query(focal_xy, k=k_query)
    if k_query == 1:
        dists, indices = dists[:, None], indices[:, None]
    is_self = dists[:, 0] < 1e-10
    k_eff = min(k, k_query - 1)
    nbr_indices = np.where(
        is_self[:, np.newaxis],
        indices[:, 1:k_eff + 1],
        indices[:, :k_eff],
    )
    nbr_types = all_types[nbr_indices]
    C = np.zeros((focal_xy.shape[0], 1, n_types))
    for t in range(n_types):
        C[:, 0, t] = (nbr_types == t).sum(axis=1)
    return C


def _radius_counts(focal_xy, all_xy, all_types, edges, n_types, chunk=20000):
    """(n_c, n_rings, K) neighbor-type counts in annuli [edges[j], edges[j+1])."""
    edges = np.asarray(edges, dtype=float)
    n_rings = len(edges) - 1
    n_c = focal_xy.shape[0]
    tree_all = cKDTree(all_xy)
    flat = np.zeros(n_c * n_rings * n_types)
    for start in range(0, n_c, chunk):
        fx = focal_xy[start:start + chunk]
        pairs = cKDTree(fx).sparse_distance_matrix(
            tree_all, edges[-1], output_type="ndarray")
        d = pairs["v"]
        keep = d >= 1e-10  # the cell itself is not its own neighbor
        rows, cols, d = pairs["i"][keep], pairs["j"][keep], d[keep]
        ring = np.searchsorted(edges, d, side="right") - 1
        ok = ring < n_rings
        idx = ((start + rows[ok]) * n_rings + ring[ok]) * n_types \
            + all_types[cols[ok]]
        flat += np.bincount(idx, minlength=flat.size)
    return flat.reshape(n_c, n_rings, n_types)


def _neighborhood_edges(radius, rings):
    if rings is not None:
        edges = [0.0] + [float(r) for r in rings]
        if np.any(np.diff(edges) <= 0):
            raise ValueError("rings must be strictly increasing positive radii")
        return edges
    if radius is not None:
        return [0.0, float(radius)]
    return None


def compute_composition_vectors(focal_xy, all_xy, all_types, k=30, n_types=None,
                                radius=None, rings=None,
                                return_n_neighbors=False):
    """Compute local neighborhood composition for each focal cell.

    Neighborhood is defined over ALL cells (any type). Three definitions:

    - k nearest neighbors (default; `radius` and `rings` None)
    - all cells within a physical `radius` (same units as coordinates)
    - concentric annuli `rings` = [r1, r2, ...] giving blocks [0, r1),
      [r1, r2), ...; composition is computed within each annulus and the
      blocks are concatenated, so P has K * len(rings) columns.

    Args:
        focal_xy: (n_c, 2) coordinates of focal cells
        all_xy: (M, 2) coordinates of all cells
        all_types: (M,) integer type labels 0..K-1
        k: number of nearest neighbors
        n_types: number of cell types K (inferred if None)
        radius: neighborhood radius (overrides k)
        rings: increasing annulus radii (overrides k and radius)
        return_n_neighbors: also return neighbor counts per block

    Returns:
        P: (n_c, K * n_blocks) composition matrix (each block sums to 1,
           or 0 for a block with no neighbors)
        n_neighbors: (n_c, n_blocks), only if return_n_neighbors
    """
    if n_types is None:
        n_types = int(all_types.max()) + 1
    edges = _neighborhood_edges(radius, rings)
    if edges is None:
        C = _knn_counts(focal_xy, all_xy, all_types, k, n_types)
    else:
        C = _radius_counts(focal_xy, all_xy, all_types, edges, n_types)
    n_nbrs = C.sum(axis=2)
    # Guard against div-by-zero if a focal cell has no neighbors in a block
    P = (C / np.maximum(n_nbrs, 1)[:, :, np.newaxis]).reshape(len(focal_xy), -1)
    if return_n_neighbors:
        return P, n_nbrs
    return P


def _blocks(n_cols, n_types):
    n_blocks = n_cols // n_types
    if n_blocks <= 1:
        return None
    return [np.arange(b * n_types, (b + 1) * n_types) for b in range(n_blocks)]


def composition_kernel(P, build_kernel=False):
    """Compute centered composition factor (and optionally the full kernel).

    The downstream test (`permutation_test`) only needs the **factor**
    P_tilde, not the full n×n kernel — Q_g = ||P_tilde^T r_g||² is the
    low-rank form. The n×n kernel matrix is huge for large focal
    populations (110K cells → 96 GB) and slow to materialize, so we
    skip it by default.

    Args:
        P: (n_c, K) composition matrix
        build_kernel: if True, also materialize K_comp = P_tilde P_tilde^T.
            Off by default — only enable for diagnostic/legacy code that
            needs the full kernel matrix explicitly.

    Returns:
        K_comp: (n_c, n_c) kernel matrix or None if build_kernel=False
        P_tilde: (n_c, K) centered composition matrix
    """
    P_tilde = P - P.mean(axis=0, keepdims=True)
    K_comp = P_tilde @ P_tilde.T if build_kernel else None
    return K_comp, P_tilde


def _permutation_core(residuals, P_tilde, draw, n_perm, batch_size, blocks):
    """Batched float-32 permutation test shared by both permutation schemes.

    `draw()` returns one permutation of focal-cell indices. When `blocks`
    (a list of column-index arrays of P_tilde) is given, a p-value is also
    returned for each block statistic Q_g^(j) = ||P_tilde[:, j]^T r_g||²,
    computed from the same permutations as the omnibus statistic.
    """
    n_c, n_genes = residuals.shape
    K_types = P_tilde.shape[1]

    proj_obs = P_tilde.T @ residuals
    Q_obs = (proj_obs ** 2).sum(axis=0)

    Pt32 = P_tilde.astype(np.float32)
    res32 = residuals.astype(np.float32)
    Q_obs32 = Q_obs.astype(np.float32)
    if blocks is not None:
        Qb_obs32 = np.stack([(proj_obs[b] ** 2).sum(axis=0)
                             for b in blocks]).astype(np.float32)
        count_b = np.zeros((len(blocks), n_genes))

    count_ge = np.zeros(n_genes)
    n_done = 0
    while n_done < n_perm:
        B = min(batch_size, n_perm - n_done)
        perm_idx = np.empty((B, n_c), dtype=np.intp)
        for b in range(B):
            perm_idx[b] = draw()
        P_stack = (Pt32[perm_idx.ravel()]
                   .reshape(B, n_c, K_types)
                   .transpose(0, 2, 1)
                   .reshape(B * K_types, n_c))
        proj = P_stack @ res32
        sq = proj.reshape(B, K_types, n_genes) ** 2
        Q_batch = sq.sum(axis=1)
        count_ge += (Q_batch >= Q_obs32[np.newaxis, :]).sum(axis=0)
        if blocks is not None:
            for j, cols in enumerate(blocks):
                count_b[j] += (sq[:, cols, :].sum(axis=1)
                               >= Qb_obs32[j][np.newaxis, :]).sum(axis=0)
        n_done += B

    pvalues = (1 + count_ge) / (1 + n_perm)
    block_pvalues = None if blocks is None else (1 + count_b) / (1 + n_perm)
    return Q_obs, pvalues, block_pvalues


def permutation_test(residuals, P_tilde, n_perm=5000, seed=42,
                     batch_size=100, blocks=None):
    """Permutation test for composition-structured variance.

    Uses the low-rank decomposition Q_g = ||P_tilde^T r_g||^2 (since
    K_comp = P_tilde @ P_tilde^T has rank ≤ K-1 by the simplex constraint).
    Cost is O(n * K * G) per permutation instead of O(n^2 * G), and the
    permutations are batched into a single GEMM in float32 for further
    speedup.

    Args:
        residuals: (n_c, n_genes)
        P_tilde: (n_c, K) centered composition matrix
        n_perm: number of permutations
        seed: random seed
        batch_size: number of permutations per GEMM batch
        blocks: optional list of column-index arrays (e.g. one per ring);
            if given, per-block p-values are returned as a third value

    Returns:
        Q_obs: (n_genes,) observed test statistics
        pvalues: (n_genes,) permutation p-values
        block_pvalues: (n_blocks, n_genes), only if `blocks` is given

    Note:
        The legacy `K_comp` parameter (full n×n kernel + per-permutation
        Python loop) was removed 2026-04-09 — it produced numerically
        divergent results from the batched float32 path and was unreached
        by every call site in the package.
    """
    rng = np.random.default_rng(seed)
    n_c = residuals.shape[0]
    Q_obs, pvalues, block_pvalues = _permutation_core(
        residuals, P_tilde, lambda: rng.permutation(n_c),
        n_perm, batch_size, blocks)
    if blocks is None:
        return Q_obs, pvalues
    return Q_obs, pvalues, block_pvalues


def bh_fdr(pvalues, alpha=0.05):
    """Benjamini-Hochberg FDR correction.

    Args:
        pvalues: (n_genes,) p-values
        alpha: FDR threshold

    Returns:
        qvalues: (n_genes,) adjusted p-values
        significant: (n_genes,) boolean mask
    """
    pvalues = np.asarray(pvalues)
    n = len(pvalues)
    if n == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=bool)
    sorted_idx = np.argsort(pvalues)
    sorted_p = pvalues[sorted_idx]
    qvalues = np.zeros(n)
    # Largest-rank q-value: sorted_p[-1] * n / n = sorted_p[-1]
    qvalues[sorted_idx[-1]] = sorted_p[-1]
    # Step-down with monotonicity constraint
    for i in range(n - 2, -1, -1):
        qvalues[sorted_idx[i]] = min(
            qvalues[sorted_idx[i + 1]],
            sorted_p[i] * n / (i + 1),
        )
    significant = qvalues <= alpha
    return qvalues, significant


def effect_size(residuals, P_tilde):
    """Compute R^2 and beta coefficients for composition effect.

    Args:
        residuals: (n_c, n_genes) Pearson residuals (typically post-centering)
        P_tilde: (n_c, K) centered composition matrix

    Returns:
        beta: (n_genes, K) composition effect vectors
        R2: (n_genes,) fraction of residual variance explained

    Note:
        P_tilde has rank ≤ K-1 because the rows of P sum to 1; only β
        vectors in the rowspace (zero-sum, after centering) are exactly
        recoverable. lstsq's rcond is pinned to 1e-12 for reproducibility
        across numpy versions.
    """
    beta, _, _, _ = np.linalg.lstsq(P_tilde, residuals, rcond=1e-12)
    fitted = P_tilde @ beta
    ss_model = (fitted ** 2).sum(axis=0)
    ss_total = (residuals ** 2).sum(axis=0)
    R2 = np.where(ss_total > 0, ss_model / ss_total, 0.0)
    return beta.T, R2


def variance_decomposition(counts, phi, residuals, R2, rank, mu_ref=None):
    """Express the composition fit as a share of each gene's count variance.

    Let r_g = (X_g - mu_g) / sqrt(V_g) be the Pearson residual before cell
    centering, r~_g the cell-centered residual, and H the projection onto
    the column space of P_tilde (rank d). With

        R2_g        = ||H r~_g||² / ||r~_g||²            (from effect_size)
        retained_g  = ||r~_g||²   / ||r_g||²

    the composition share of total count variance is

        R2_total_g  = ||H r~_g||² / ||r_g||² = R2_g * retained_g.

    Because r_g is X_g rescaled by the per-gene constant 1/sqrt(V_g), the
    ratio equals the share of sum_i (X_ig - mu_g)² captured by the count-
    scale composition fit sqrt(V_g) H r~_g: V_g cancels, so R2_total does
    not depend on the NB variance model except through cell centering.

    Under the permutation null (rows of P_tilde exchangeable, residual
    columns mean-zero), E[H] = d/(n-1) (I - 11ᵀ/n), so E[R2_g] = d/(n-1)
    exactly. R2_adj removes this chance-level fit:

        R2_adj       = (R2 - d/(n-1)) / (1 - d/(n-1))
        R2_total_adj = R2_adj * retained

    Adjusted values are unbiased under the null and may be negative.

    `R2_total_legacy` reproduces the v0.1 quantity R2 * excess_var /
    total_var for comparison only; it is not a bound in either direction.

    Args:
        counts: (n_c, n_genes) raw counts
        phi: NB dispersion, scalar or (n_genes,)
        residuals: (n_c, n_genes) cell-centered residuals used for R2
        R2: (n_genes,) from `effect_size`
        rank: rank d of P_tilde
        mu_ref: optional reference mean used for the residuals

    Returns:
        dict of (n_genes,) arrays
    """
    n_c = counts.shape[0]
    mu_g = counts.mean(axis=0)
    baseline_var = mu_g + mu_g ** 2 / phi
    total_var = counts.var(axis=0, ddof=1)
    excess_var = np.maximum(total_var - baseline_var, 0.0)
    has_excess = excess_var > 0

    mu0 = mu_g if mu_ref is None else np.asarray(mu_ref)
    V0 = np.maximum(mu0 + mu0 ** 2 / phi, 1e-10)
    ss_raw = ((counts - mu0) ** 2).sum(axis=0) / V0
    ss_cent = (residuals ** 2).sum(axis=0)
    var_retained = np.where(ss_raw > 0, ss_cent / np.maximum(ss_raw, 1e-300), 0.0)

    R2_total = R2 * var_retained
    e0 = rank / max(n_c - 1, 1)
    R2_adj = (R2 - e0) / (1 - e0) if e0 < 1 else np.zeros_like(R2)
    R2_total_adj = R2_adj * var_retained

    R2_total_legacy = np.where(total_var > 0, R2 * excess_var / total_var, 0.0)
    R2_total_legacy[~has_excess] = 0.0

    return dict(
        total_var=total_var, baseline_var=baseline_var,
        excess_var=excess_var, has_excess=has_excess,
        R2_total=R2_total, R2_adj=R2_adj, R2_total_adj=R2_total_adj,
        R2_total_legacy=R2_total_legacy, var_retained=var_retained,
        R2_null=np.full_like(R2, e0),
    )


def _finalize(counts, phi, residuals, P, P_tilde, K_comp, Q, pvalues,
              block_pvalues, alpha, mu_ref=None, sample_ids=None,
              cell_mask=None, n_neighbors=None, neighborhood=None):
    """Stages 4+: FDR, effect sizes, variance decomposition, excess filter."""
    qvalues, significant = bh_fdr(pvalues, alpha=alpha)
    beta, R2 = effect_size(residuals, P_tilde)
    rank = int(np.linalg.matrix_rank(P_tilde)) if P_tilde.size else 0
    vd = variance_decomposition(counts, phi, residuals, R2, rank, mu_ref=mu_ref)

    n_raw_sig = significant.sum()
    n_filtered = (~vd["has_excess"] & significant).sum()
    significant = significant & vd["has_excess"]

    log.info("Dispersion phi = %.2f%s", _phi_summary(phi),
             "" if np.ndim(phi) == 0 else " (median of gene-specific)")
    log.info("Significant genes: %d / %d (FDR < %.2f)",
             significant.sum(), len(significant), alpha)
    if n_filtered > 0:
        log.info("Filtered %d genes with no excess variance (%d before filter)",
                 n_filtered, n_raw_sig)

    return dict(
        Q=Q, pvalues=pvalues, qvalues=qvalues, significant=significant,
        R2=R2, beta=beta, phi=phi, residuals=residuals,
        P=P, P_tilde=P_tilde, K_comp=K_comp, rank_P=rank,
        block_pvalues=block_pvalues, cell_mask=cell_mask,
        n_neighbors=n_neighbors, neighborhood=neighborhood,
        sample_ids=sample_ids,
        **vd,
    )


def _neighborhood_spec(k, radius, rings):
    if rings is not None:
        return {"type": "rings", "rings": list(rings)}
    if radius is not None:
        return {"type": "radius", "radius": radius}
    return {"type": "knn", "k": k}


def _drop_sparse_cells(n_nbrs, min_neighbors):
    cell_mask = (n_nbrs >= min_neighbors).all(axis=1)
    n_drop = (~cell_mask).sum()
    if n_drop:
        log.warning("Excluding %d of %d focal cells with < %d neighbors in "
                    "some neighborhood block", n_drop, len(cell_mask),
                    min_neighbors)
    return cell_mask


def run_model(counts, focal_xy, all_xy, all_types, k=30, n_perm=5000,
              alpha=0.05, min_mean=0.5, seed=42,
              radius=None, rings=None, min_neighbors=1,
              dispersion="shared", center="mean", center_genes=None):
    """Run the full composition-dependent context variance model.

    Args:
        counts: (n_c, n_genes) count matrix for focal cell type
        focal_xy: (n_c, 2) spatial coordinates of focal cells
        all_xy: (M, 2) coordinates of all cells in tissue
        all_types: (M,) integer type labels for all cells
        k: neighborhood size for kNN composition
        n_perm: permutations for test
        alpha: FDR threshold
        min_mean: min gene mean for dispersion estimation
        seed: random seed
        radius: physical neighborhood radius (overrides k)
        rings: increasing annulus radii (overrides k, radius); adds
            per-ring p-values in `block_pvalues`
        min_neighbors: focal cells with fewer neighbors in any block are
            excluded (radius/ring neighborhoods only; see `cell_mask`)
        dispersion: 'shared' (one phi) or 'gene' (regularized trend)
        center: per-cell centering estimator ('mean', 'median', 'trimmed')
        center_genes: optional genes used to compute the per-cell center

    Returns:
        dict with keys: Q, pvalues, qvalues, significant, R2, R2_adj,
            R2_total, R2_total_adj, R2_total_legacy, var_retained, R2_null,
            beta, phi, residuals, P, P_tilde, K_comp, rank_P, block_pvalues,
            total_var, baseline_var, excess_var, has_excess, cell_mask,
            n_neighbors, neighborhood, sample_ids
    """
    n_types = int(all_types.max()) + 1

    P, n_nbrs = compute_composition_vectors(
        focal_xy, all_xy, all_types, k=k, n_types=n_types,
        radius=radius, rings=rings, return_n_neighbors=True)
    cell_mask = _drop_sparse_cells(n_nbrs, min_neighbors)
    if not cell_mask.all():
        counts, P = counts[cell_mask], P[cell_mask]

    phi = _estimate_phi(counts, dispersion, min_mean)
    residuals = compute_pearson_residuals(counts, phi, center=center,
                                          center_genes=center_genes)
    K_comp, P_tilde = composition_kernel(P)

    out = permutation_test(residuals, P_tilde, n_perm=n_perm, seed=seed,
                           blocks=_blocks(P.shape[1], n_types))
    Q, pvalues = out[0], out[1]
    block_pvalues = out[2] if len(out) == 3 else None

    return _finalize(counts, phi, residuals, P, P_tilde, K_comp, Q, pvalues,
                     block_pvalues, alpha, cell_mask=cell_mask,
                     n_neighbors=n_nbrs,
                     neighborhood=_neighborhood_spec(k, radius, rings))


# ============================================================
# Multi-sample model: within-core kNN + within-sample permutation
# ============================================================

def compute_composition_vectors_multisample(focal_xy, all_xy, all_types,
                                            focal_sample_ids, all_sample_ids,
                                            k=30, n_types=None, radius=None,
                                            rings=None,
                                            return_n_neighbors=False):
    """Compute within-core composition for each focal cell.

    For each focal cell, neighbors are found among ALL cells in the same
    sample/core (not across cores). See `compute_composition_vectors` for
    the k / radius / rings neighborhood definitions.

    Args:
        focal_xy: (n_c, 2) coordinates of focal cells
        all_xy: (M, 2) coordinates of all cells
        all_types: (M,) integer type labels 0..K-1
        focal_sample_ids: (n_c,) sample label for each focal cell
        all_sample_ids: (M,) sample label for each cell
        k: number of nearest neighbors
        n_types: number of cell types K (inferred if None)
        radius: neighborhood radius (overrides k)
        rings: increasing annulus radii (overrides k and radius)
        return_n_neighbors: also return neighbor counts per block

    Returns:
        P: (n_c, K * n_blocks) composition matrix
        n_neighbors: (n_c, n_blocks), only if return_n_neighbors
    """
    if n_types is None:
        n_types = int(all_types.max()) + 1
    edges = _neighborhood_edges(radius, rings)
    n_blocks = 1 if edges is None else len(edges) - 1

    P = np.zeros((len(focal_xy), n_types * n_blocks))
    n_nbrs = np.zeros((len(focal_xy), n_blocks))

    for s in np.unique(focal_sample_ids):
        focal_idx = np.where(focal_sample_ids == s)[0]
        if len(focal_idx) == 0:
            continue
        all_mask = all_sample_ids == s
        P_s, n_s = compute_composition_vectors(
            focal_xy[focal_idx], all_xy[all_mask], all_types[all_mask],
            k=k, n_types=n_types, radius=radius, rings=rings,
            return_n_neighbors=True)
        P[focal_idx] = P_s
        n_nbrs[focal_idx] = n_s

    if return_n_neighbors:
        return P, n_nbrs
    return P


def _within_sample_permutation(sample_ids, rng):
    """Generate a permutation that only swaps cells within each sample."""
    perm = np.arange(len(sample_ids))
    for s in np.unique(sample_ids):
        idx = np.where(sample_ids == s)[0]
        perm[idx] = rng.permutation(idx)
    return perm


def permutation_test_within_sample(residuals, P_tilde, sample_ids,
                                   n_perm=5000, seed=42, batch_size=50,
                                   blocks=None):
    """Within-sample permutation test.

    Permutes cell labels only within each sample, preserving sample structure.
    Uses the low-rank P_tilde form for efficiency.

    Args:
        residuals: (n_c, n_genes)
        P_tilde: (n_c, K) centered composition matrix
        sample_ids: (n_c,) integer sample labels
        n_perm: number of permutations
        seed: random seed
        batch_size: permutations per GEMM batch
        blocks: optional list of column-index arrays; if given, per-block
            p-values are returned as a third value

    Returns:
        Q_obs: (n_genes,) observed test statistics
        pvalues: (n_genes,) permutation p-values
        block_pvalues: (n_blocks, n_genes), only if `blocks` is given
    """
    rng = np.random.default_rng(seed)
    Q_obs, pvalues, block_pvalues = _permutation_core(
        residuals, P_tilde, lambda: _within_sample_permutation(sample_ids, rng),
        n_perm, batch_size, blocks)
    if blocks is None:
        return Q_obs, pvalues
    return Q_obs, pvalues, block_pvalues


def run_model_multisample(counts, focal_xy, all_xy, all_types,
                          focal_sample_ids, all_sample_ids,
                          k=30, n_perm=5000, alpha=0.05,
                          min_mean=0.5, seed=42,
                          null_counts=None,
                          dispersion_mode='pooled',
                          dispersion='shared',
                          radius=None, rings=None, min_neighbors=1,
                          center='mean', center_genes=None,
                          center_samples=False):
    """Run AURA with within-core composition and within-sample permutation.

    For multi-sample spatial data (TMA, multi-section). Each cell's composition
    is computed from neighbors within its own core/sample. The permutation
    null preserves sample structure by only swapping cells within each sample.

    Args:
        counts: (n_c, n_genes) count matrix for focal cell type
        focal_xy: (n_c, 2) spatial coordinates of focal cells
        all_xy: (M, 2) coordinates of all cells in tissue
        all_types: (M,) integer type labels for all cells
        focal_sample_ids: (n_c,) sample label for each focal cell
        all_sample_ids: (M,) sample label for each cell
        k: neighborhood size for within-core kNN
        n_perm: permutations for test
        alpha: FDR threshold
        min_mean: min gene mean for dispersion estimation
        seed: random seed
        null_counts: optional (n_ctrl, n_genes) control counts for null params
        dispersion_mode: how to pool the shared NB dispersion φ.
            - 'pooled' (default): one φ from the pooled focal counts.
              Ignores sample structure → underestimates φ when samples
              have batch-level variance, biasing the test conservative.
            - 'median_per_sample': fit φ within each sample separately
              (samples with <200 focal cells are skipped) and report
              the median across samples. More principled for TMA designs
              with strong sample batch effects, but produces slightly
              different sig counts than pooled.
        dispersion: 'shared' (default) or 'gene' (regularized trend, pooled
            across samples; incompatible with median_per_sample)
        radius, rings, min_neighbors: see `run_model`
        center, center_genes: see `compute_pearson_residuals`
        center_samples: if True, subtract per-sample means from each gene's
            residuals and from each composition column, so that both the
            test statistic and β use only within-sample variation (the
            between-sample component is constant under within-sample
            permutation). Off by default to reproduce v0.1 results.

    Returns:
        dict with the same keys as `run_model`
    """
    n_c, n_genes = counts.shape
    n_types = int(all_types.max()) + 1

    # Stage 2 first, so cells without neighbors can be excluded up front
    P, n_nbrs = compute_composition_vectors_multisample(
        focal_xy, all_xy, all_types,
        focal_sample_ids, all_sample_ids,
        k=k, n_types=n_types, radius=radius, rings=rings,
        return_n_neighbors=True,
    )
    cell_mask = _drop_sparse_cells(n_nbrs, min_neighbors)
    if not cell_mask.all():
        counts, P = counts[cell_mask], P[cell_mask]
        focal_sample_ids = focal_sample_ids[cell_mask]
    unique_samples = np.unique(focal_sample_ids)
    S = len(unique_samples)

    # Stage 1: null model and residuals
    ref = null_counts if null_counts is not None else counts
    if dispersion == 'gene':
        if dispersion_mode != 'pooled':
            raise ValueError("dispersion='gene' requires dispersion_mode='pooled'")
        phi = estimate_gene_dispersion(ref, min_mean=min_mean)
    elif dispersion != 'shared':
        raise ValueError(f"Unknown dispersion: {dispersion!r}")
    elif dispersion_mode == 'pooled':
        phi = estimate_dispersion(ref, min_mean=min_mean)
    elif dispersion_mode == 'median_per_sample':
        if null_counts is not None:
            raise ValueError(
                "median_per_sample dispersion is incompatible with "
                "null_counts (control-conditioned residuals)")
        phis = []
        for s in unique_samples:
            s_mask = focal_sample_ids == s
            if s_mask.sum() < 200:  # too few cells for stable fit
                continue
            phis.append(estimate_dispersion(counts[s_mask], min_mean=min_mean))
        if len(phis) == 0:
            log.warning("No sample has ≥200 focal cells; falling back to pooled φ")
            phi = estimate_dispersion(ref, min_mean=min_mean)
        else:
            phi = float(np.median(phis))
            log.info("dispersion_mode=median_per_sample: %d samples used, "
                     "phi range %.2f-%.2f, median %.2f",
                     len(phis), min(phis), max(phis), phi)
    else:
        raise ValueError(f"Unknown dispersion_mode: {dispersion_mode!r}")
    mu_ref = ref.mean(axis=0) if null_counts is not None else None
    residuals = compute_pearson_residuals(
        counts, phi, center_cells=True, mu_ref=mu_ref,
        center=center, center_genes=center_genes,
    )

    if center_samples:
        P = P.copy()
        for s in unique_samples:
            s_mask = focal_sample_ids == s
            residuals[s_mask] -= residuals[s_mask].mean(axis=0, keepdims=True)
            P[s_mask] -= P[s_mask].mean(axis=0, keepdims=True)
    K_comp, P_tilde = composition_kernel(P)

    # Stage 3: within-sample permutation test
    out = permutation_test_within_sample(
        residuals, P_tilde, focal_sample_ids,
        n_perm=n_perm, seed=seed, blocks=_blocks(P.shape[1], n_types),
    )
    Q, pvalues = out[0], out[1]
    block_pvalues = out[2] if len(out) == 3 else None

    log.info("Samples: %d, Cells: %d, Types: %d", S, len(counts), n_types)
    log.info("Cells per sample: min=%d, max=%d, median=%d",
             min((focal_sample_ids == s).sum() for s in unique_samples),
             max((focal_sample_ids == s).sum() for s in unique_samples),
             int(np.median([(focal_sample_ids == s).sum() for s in unique_samples])))

    return _finalize(counts, phi, residuals, P, P_tilde, K_comp, Q, pvalues,
                     block_pvalues, alpha, mu_ref=mu_ref,
                     sample_ids=focal_sample_ids, cell_mask=cell_mask,
                     n_neighbors=n_nbrs,
                     neighborhood=_neighborhood_spec(k, radius, rings))


def run_model_sample(counts, sample_ids, sample_composition, n_perm=5000,
                     alpha=0.05, min_mean=0.5, min_gene_mean=0.1, seed=42,
                     null_counts=None, gene_names=None,
                     dispersion='shared', center='mean', center_genes=None):
    """Run AURA with sample-level composition (for TMA / multi-sample data).

    Instead of computing composition from k-NN spatial neighbors, each cell
    inherits the composition of its sample (tissue core). Tests whether gene
    expression varies with sample-level niche composition.

    Args:
        counts: (n_c, n_genes) count matrix for focal cell type (test set)
        sample_ids: (n_c,) integer sample label for each cell (0..S-1)
        sample_composition: (S, K) composition matrix
        n_perm: permutations for test
        alpha: FDR threshold
        min_mean: min gene mean for **dispersion** estimation
        min_gene_mean: min gene mean for inclusion in the test (drops
            very-low-expression genes that would otherwise contribute
            noise to the permutation null). Default 0.1.
        seed: random seed
        null_counts: (n_ctrl, n_genes) optional control counts for null params
        gene_names: optional (n_genes,) gene name array. If provided, will
            be filtered alongside `counts` and returned in the result dict
            so the caller can map columns back to gene symbols.
        dispersion, center, center_genes: see `run_model`

    Returns:
        dict with same keys as run_model. If `gene_names` was provided,
        the dict also contains a `gene_names` field reflecting the
        post-filter gene set.
    """
    counts = np.asarray(counts)
    if min_gene_mean is not None and min_gene_mean > 0:
        gene_mask = counts.mean(axis=0) >= min_gene_mean
        counts = counts[:, gene_mask]
        if null_counts is not None:
            null_counts = np.asarray(null_counts)[:, gene_mask]
        if gene_names is not None:
            gene_names = np.asarray(gene_names)[gene_mask]
        if center_genes is not None:
            center_genes = np.asarray(center_genes)
            if center_genes.dtype == bool:
                center_genes = center_genes[gene_mask]
            else:
                raise ValueError("center_genes must be a boolean mask when "
                                 "min_gene_mean filtering is applied")
    n_c, n_genes = counts.shape
    S, K = sample_composition.shape

    ref = null_counts if null_counts is not None else counts
    phi = _estimate_phi(ref, dispersion, min_mean)
    mu_ref = ref.mean(axis=0) if null_counts is not None else None
    residuals = compute_pearson_residuals(
        counts, phi, center_cells=True, mu_ref=mu_ref,
        center=center, center_genes=center_genes,
    )

    P = sample_composition[sample_ids]
    K_comp, P_tilde = composition_kernel(P)

    Q, pvalues = permutation_test(residuals, P_tilde, n_perm=n_perm, seed=seed)

    log.info("Samples: %d, Cells: %d, Composition axes: %d", S, n_c, K)
    out = _finalize(counts, phi, residuals, P, P_tilde, K_comp, Q, pvalues,
                    None, alpha, mu_ref=mu_ref, sample_ids=sample_ids,
                    cell_mask=np.ones(n_c, dtype=bool),
                    neighborhood={"type": "sample"})
    if gene_names is not None:
        out['gene_names'] = gene_names
    return out
