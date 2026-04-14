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
    "compute_pearson_residuals",
    "compute_composition_vectors",
    "composition_kernel",
    "permutation_test",
    "bh_fdr",
    "effect_size",
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


def compute_pearson_residuals(counts, phi, center_cells=True,
                              mu_ref=None, sample_ids=None):
    """Compute Pearson residuals under NB null with per-gene mean.

    Uses unconditional per-gene mean (not library-size-conditioned) to
    avoid confounding: conditioning on n_i introduces a dependency between
    genes because n_i includes counts from all genes, including any
    context-dependent ones.

    When center_cells=True, subtracts the per-cell mean residual (alpha_i)
    from each gene's residual. This cell-level centering removes the shared
    cell-level shift caused by library size variation correlated with
    composition, isolating gene-specific context effects (delta_ig) from
    global RNA effects.

    Args:
        counts: (n_cells, n_genes) count matrix
        phi: shared dispersion
        center_cells: if True, apply cell-level centering (subtract per-cell
                      mean residual). Default True.
        mu_ref: optional (n_genes,) per-gene mean to use instead of the
                in-sample mean. Used for control-conditioned residuals
                (e.g. IPF disease cells centered on healthy donor means).
        sample_ids: optional (n_cells,) integer sample labels. If provided,
                    cell-level centering is done **within sample** instead
                    of pooled — needed for multi-sample TMA designs.

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
        if sample_ids is None:
            alpha_i = residuals.mean(axis=1, keepdims=True)
            residuals = residuals - alpha_i
        else:
            for s in np.unique(sample_ids):
                s_mask = sample_ids == s
                alpha_s = residuals[s_mask].mean(axis=1, keepdims=True)
                residuals[s_mask] -= alpha_s

    return residuals


def compute_composition_vectors(focal_xy, all_xy, all_types, k=15, n_types=None):
    """Compute local neighborhood composition for each focal cell.

    Neighborhood is defined over ALL cells (any type). Composition is the
    fraction of each type among k nearest neighbors.

    Args:
        focal_xy: (n_c, 2) coordinates of focal cells
        all_xy: (M, 2) coordinates of all cells
        all_types: (M,) integer type labels 0..K-1
        k: number of nearest neighbors
        n_types: number of cell types K (inferred if None)

    Returns:
        P: (n_c, K) composition matrix (rows sum to 1)
    """
    if n_types is None:
        n_types = int(all_types.max()) + 1
    tree = cKDTree(all_xy)
    dists, indices = tree.query(focal_xy, k=k + 1)
    is_self = dists[:, 0] < 1e-10
    nbr_indices = np.where(
        is_self[:, np.newaxis],
        indices[:, 1:k + 1],
        indices[:, :k],
    )
    nbr_types = all_types[nbr_indices]
    P = np.zeros((focal_xy.shape[0], n_types))
    for t in range(n_types):
        P[:, t] = (nbr_types == t).sum(axis=1)
    # Guard against div-by-zero if a focal cell has no neighbors of any
    # type (only possible with k=0, but defensive).
    P = P / np.maximum(P.sum(axis=1, keepdims=True), 1)
    return P


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


def permutation_test(residuals, P_tilde, n_perm=1000, seed=42,
                     batch_size=100):
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

    Returns:
        Q_obs: (n_genes,) observed test statistics
        pvalues: (n_genes,) permutation p-values

    Note:
        The legacy `K_comp` parameter (full n×n kernel + per-permutation
        Python loop) was removed 2026-04-09 — it produced numerically
        divergent results from the batched float32 path and was unreached
        by every call site in the package.
    """
    rng = np.random.default_rng(seed)
    n_c, n_genes = residuals.shape
    K_types = P_tilde.shape[1]

    proj_obs = P_tilde.T @ residuals
    Q_obs = (proj_obs ** 2).sum(axis=0)

    Pt32 = P_tilde.astype(np.float32)
    res32 = residuals.astype(np.float32)
    Q_obs32 = Q_obs.astype(np.float32)

    count_ge = np.zeros(n_genes)
    n_done = 0
    while n_done < n_perm:
        B = min(batch_size, n_perm - n_done)
        perm_idx = np.empty((B, n_c), dtype=np.intp)
        for b in range(B):
            perm_idx[b] = rng.permutation(n_c)
        P_stack = (Pt32[perm_idx.ravel()]
                   .reshape(B, n_c, K_types)
                   .transpose(0, 2, 1)
                   .reshape(B * K_types, n_c))
        proj = P_stack @ res32
        Q_batch = (proj.reshape(B, K_types, n_genes) ** 2).sum(axis=1)
        count_ge += (Q_batch >= Q_obs32[np.newaxis, :]).sum(axis=0)
        n_done += B

    pvalues = (1 + count_ge) / (1 + n_perm)
    return Q_obs, pvalues


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


def _variance_decomposition(counts, phi, R2, significant):
    """Compute variance decomposition arrays.

    Args:
        counts: (n_c, n_genes) raw counts
        phi: NB shared dispersion
        R2: (n_genes,) R² from `effect_size` — fraction of *centered*
            Pearson residual variance attributable to composition
        significant: (n_genes,) boolean (unused; kept for API stability)

    Returns:
        total_var: (n_genes,) sample variance of counts
        baseline_var: (n_genes,) NB null variance under the per-gene mean
        excess_var: (n_genes,) max(total_var - baseline_var, 0) — upper
            bound on the variance NOT explained by NB sampling
        has_excess: (n_genes,) excess_var > 0
        R2_total: (n_genes,) ≈ composition share of total count variance
        comp_var: (n_genes,) ≈ composition variance in count units

    Note:
        `comp_var = R2 * excess_var` is an **approximation**: R² is
        computed on cell-centered Pearson residuals whose total variance
        is closer to (excess_var - alpha_contribution), so multiplying
        by `excess_var` slightly over-estimates the composition share for
        genes where the per-cell α effect is large. Subtracting the α
        contribution exactly is unstable because for many genes
        alpha_var > excess_var, which would push comp_var negative.

        For relative ranking (cumulative-R² figures, top-gene tables),
        the approximation is well-behaved and stable. Treat the absolute
        R²_total values as upper bounds on the true composition share.
    """
    mu_g = counts.mean(axis=0)
    baseline_var = mu_g + mu_g ** 2 / phi
    total_var = counts.var(axis=0, ddof=1)
    excess_var = np.maximum(total_var - baseline_var, 0.0)
    has_excess = excess_var > 0

    comp_var = R2 * excess_var
    R2_total = np.where(total_var > 0, comp_var / total_var, 0.0)
    R2_total[~has_excess] = 0.0

    return total_var, baseline_var, excess_var, has_excess, R2_total, comp_var


def run_model(counts, focal_xy, all_xy, all_types, k=15, n_perm=1000,
              alpha=0.05, min_mean=0.5, seed=42):
    """Run the full composition-dependent context variance model.

    Args:
        counts: (n_c, n_genes) count matrix for focal cell type
        focal_xy: (n_c, 2) spatial coordinates of focal cells
        all_xy: (M, 2) coordinates of all cells in tissue
        all_types: (M,) integer type labels for all cells
        k: neighborhood size for composition
        n_perm: permutations for test
        alpha: FDR threshold
        min_mean: min gene mean for dispersion estimation
        seed: random seed

    Returns:
        dict with keys: Q, pvalues, qvalues, significant, R2, R2_total,
                        beta, phi, residuals, P, P_tilde, K_comp,
                        total_var, baseline_var, excess_var, has_excess
    """
    n_types = int(all_types.max()) + 1

    phi = estimate_dispersion(counts, min_mean=min_mean)
    residuals = compute_pearson_residuals(counts, phi)

    P = compute_composition_vectors(focal_xy, all_xy, all_types, k=k,
                                    n_types=n_types)
    K_comp, P_tilde = composition_kernel(P)

    Q, pvalues = permutation_test(residuals, P_tilde, n_perm=n_perm, seed=seed)
    qvalues, significant = bh_fdr(pvalues, alpha=alpha)

    beta, R2 = effect_size(residuals, P_tilde)

    total_var, baseline_var, excess_var, has_excess, R2_total, _ = \
        _variance_decomposition(counts, phi, R2, significant)

    n_raw_sig = significant.sum()
    n_filtered = (~has_excess & significant).sum()
    significant = significant & has_excess

    log.info("Dispersion phi = %.2f", phi)
    log.info("Significant genes: %d / %d (FDR < %.2f)",
             significant.sum(), len(significant), alpha)
    if n_filtered > 0:
        log.info("Filtered %d genes with no excess variance (%d before filter)",
                 n_filtered, n_raw_sig)

    return dict(
        Q=Q, pvalues=pvalues, qvalues=qvalues, significant=significant,
        R2=R2, R2_total=R2_total, beta=beta, phi=phi, residuals=residuals,
        P=P, P_tilde=P_tilde, K_comp=K_comp,
        total_var=total_var, baseline_var=baseline_var,
        excess_var=excess_var, has_excess=has_excess,
        sample_ids=None,
    )


# ============================================================
# Multi-sample model: within-core kNN + within-sample permutation
# ============================================================

def compute_composition_vectors_multisample(focal_xy, all_xy, all_types,
                                            focal_sample_ids, all_sample_ids,
                                            k=15, n_types=None):
    """Compute within-core kNN composition for each focal cell.

    For each focal cell, finds k nearest neighbors among ALL cells in the
    same sample/core (not across cores). Composition is the fraction of each
    type among those neighbors.

    Args:
        focal_xy: (n_c, 2) coordinates of focal cells
        all_xy: (M, 2) coordinates of all cells
        all_types: (M,) integer type labels 0..K-1
        focal_sample_ids: (n_c,) sample label for each focal cell
        all_sample_ids: (M,) sample label for each cell
        k: number of nearest neighbors
        n_types: number of cell types K (inferred if None)

    Returns:
        P: (n_c, K) composition matrix
    """
    if n_types is None:
        n_types = int(all_types.max()) + 1

    unique_samples = np.unique(focal_sample_ids)
    P = np.zeros((len(focal_xy), n_types))

    for s in unique_samples:
        focal_mask = focal_sample_ids == s
        focal_idx = np.where(focal_mask)[0]
        if len(focal_idx) == 0:
            continue

        all_mask = all_sample_ids == s
        all_xy_s = all_xy[all_mask]
        all_types_s = all_types[all_mask]

        tree = cKDTree(all_xy_s)
        focal_xy_s = focal_xy[focal_idx]
        k_query = min(k + 1, len(all_xy_s))
        dists, indices = tree.query(focal_xy_s, k=k_query)

        is_self = dists[:, 0] < 1e-10
        k_eff = min(k, k_query - 1)
        nbr_indices = np.where(
            is_self[:, np.newaxis],
            indices[:, 1:k_eff + 1],
            indices[:, :k_eff],
        )
        nbr_types = all_types_s[nbr_indices]

        P_s = np.zeros((len(focal_idx), n_types))
        for t in range(n_types):
            P_s[:, t] = (nbr_types == t).sum(axis=1)
        P_s = P_s / np.maximum(P_s.sum(axis=1, keepdims=True), 1)
        P[focal_idx] = P_s

    return P


def _within_sample_permutation(sample_ids, rng):
    """Generate a permutation that only swaps cells within each sample."""
    perm = np.arange(len(sample_ids))
    for s in np.unique(sample_ids):
        idx = np.where(sample_ids == s)[0]
        perm[idx] = rng.permutation(idx)
    return perm


def permutation_test_within_sample(residuals, P_tilde, sample_ids,
                                   n_perm=1000, seed=42, batch_size=50):
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

    Returns:
        Q_obs: (n_genes,) observed test statistics
        pvalues: (n_genes,) permutation p-values
    """
    rng = np.random.default_rng(seed)
    n_c, n_genes = residuals.shape
    K_types = P_tilde.shape[1]

    proj_obs = P_tilde.T @ residuals
    Q_obs = (proj_obs ** 2).sum(axis=0)

    Pt32 = P_tilde.astype(np.float32)
    res32 = residuals.astype(np.float32)
    Q_obs32 = Q_obs.astype(np.float32)

    count_ge = np.zeros(n_genes)
    n_done = 0
    while n_done < n_perm:
        B = min(batch_size, n_perm - n_done)
        perm_idx = np.empty((B, n_c), dtype=np.intp)
        for b in range(B):
            perm_idx[b] = _within_sample_permutation(sample_ids, rng)
        P_stack = (Pt32[perm_idx.ravel()]
                   .reshape(B, n_c, K_types)
                   .transpose(0, 2, 1)
                   .reshape(B * K_types, n_c))
        proj = P_stack @ res32
        Q_batch = (proj.reshape(B, K_types, n_genes) ** 2).sum(axis=1)
        count_ge += (Q_batch >= Q_obs32[np.newaxis, :]).sum(axis=0)
        n_done += B

    pvalues = (1 + count_ge) / (1 + n_perm)
    return Q_obs, pvalues


def run_model_multisample(counts, focal_xy, all_xy, all_types,
                          focal_sample_ids, all_sample_ids,
                          k=15, n_perm=1000, alpha=0.05,
                          min_mean=0.5, seed=42,
                          null_counts=None,
                          dispersion_mode='pooled'):
    """Run AURA with within-core kNN composition and within-sample permutation.

    For multi-sample spatial data (TMA, multi-section). Each cell's composition
    is computed from k nearest neighbors within its own core/sample. The
    permutation null preserves sample structure by only swapping cells within
    each sample.

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
        dispersion_mode: how to estimate the NB shared dispersion φ.
            - 'pooled' (default): one φ from the pooled focal counts.
              Ignores sample structure → underestimates φ when samples
              have batch-level variance, biasing the test conservative.
            - 'median_per_sample': fit φ within each sample separately
              (samples with <200 focal cells are skipped) and report
              the median across samples. More principled for TMA designs
              with strong sample batch effects, but produces slightly
              different sig counts than pooled.

    Returns:
        dict with keys: Q, pvalues, qvalues, significant, R2, R2_total,
                        beta, phi, residuals, P, P_tilde, K_comp,
                        total_var, baseline_var, excess_var, has_excess,
                        sample_ids
    """
    n_c, n_genes = counts.shape
    n_types = int(all_types.max()) + 1
    unique_samples = np.unique(focal_sample_ids)
    S = len(unique_samples)

    # Stage 1: null model and residuals
    ref = null_counts if null_counts is not None else counts
    if dispersion_mode == 'pooled':
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
        counts, phi, center_cells=True,
        mu_ref=mu_ref, sample_ids=focal_sample_ids,
    )

    # Stage 2: within-core kNN composition
    P = compute_composition_vectors_multisample(
        focal_xy, all_xy, all_types,
        focal_sample_ids, all_sample_ids,
        k=k, n_types=n_types,
    )
    K_comp, P_tilde = composition_kernel(P)

    # Stage 3: within-sample permutation test
    Q, pvalues = permutation_test_within_sample(
        residuals, P_tilde, focal_sample_ids,
        n_perm=n_perm, seed=seed,
    )
    qvalues, significant = bh_fdr(pvalues, alpha=alpha)

    # Stage 4: effect size
    beta, R2 = effect_size(residuals, P_tilde)

    total_var, baseline_var, excess_var, has_excess, R2_total, _ = \
        _variance_decomposition(counts, phi, R2, significant)

    n_raw_sig = significant.sum()
    n_filtered = (~has_excess & significant).sum()
    significant = significant & has_excess

    log.info("Dispersion phi = %.2f", phi)
    log.info("Samples: %d, Cells: %d, Types: %d", S, n_c, n_types)
    log.info("Cells per sample: min=%d, max=%d, median=%d",
             min((focal_sample_ids == s).sum() for s in unique_samples),
             max((focal_sample_ids == s).sum() for s in unique_samples),
             int(np.median([(focal_sample_ids == s).sum() for s in unique_samples])))
    log.info("Significant genes: %d / %d (FDR < %.2f)",
             significant.sum(), n_genes, alpha)
    if n_filtered > 0:
        log.info("Filtered %d genes with no excess variance (%d before filter)",
                 n_filtered, n_raw_sig)

    return dict(
        Q=Q, pvalues=pvalues, qvalues=qvalues, significant=significant,
        R2=R2, R2_total=R2_total, beta=beta, phi=phi, residuals=residuals,
        P=P, P_tilde=P_tilde, K_comp=K_comp,
        total_var=total_var, baseline_var=baseline_var,
        excess_var=excess_var, has_excess=has_excess,
        sample_ids=focal_sample_ids,
    )


def run_model_sample(counts, sample_ids, sample_composition, n_perm=1000,
                     alpha=0.05, min_mean=0.5, min_gene_mean=0.1, seed=42,
                     null_counts=None, gene_names=None):
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
    n_c, n_genes = counts.shape
    S, K = sample_composition.shape

    ref = null_counts if null_counts is not None else counts
    phi = estimate_dispersion(ref, min_mean=min_mean)
    mu_ref = ref.mean(axis=0) if null_counts is not None else None
    residuals = compute_pearson_residuals(
        counts, phi, center_cells=True, mu_ref=mu_ref,
    )

    P = sample_composition[sample_ids]
    K_comp, P_tilde = composition_kernel(P)

    Q, pvalues = permutation_test(residuals, P_tilde, n_perm=n_perm, seed=seed)
    qvalues, significant = bh_fdr(pvalues, alpha=alpha)

    beta, R2 = effect_size(residuals, P_tilde)

    total_var, baseline_var, excess_var, has_excess, R2_total, _ = \
        _variance_decomposition(counts, phi, R2, significant)

    n_raw_sig = significant.sum()
    n_filtered = (~has_excess & significant).sum()
    significant = significant & has_excess

    log.info("Dispersion phi = %.2f", phi)
    log.info("Samples: %d, Cells: %d, Composition axes: %d", S, n_c, K)
    log.info("Significant genes: %d / %d (FDR < %.2f)",
             significant.sum(), len(significant), alpha)
    if n_filtered > 0:
        log.info("Filtered %d genes with no excess variance (%d before filter)",
                 n_filtered, n_raw_sig)

    out = dict(
        Q=Q, pvalues=pvalues, qvalues=qvalues, significant=significant,
        R2=R2, R2_total=R2_total, beta=beta, phi=phi, residuals=residuals,
        P=P, P_tilde=P_tilde, K_comp=K_comp,
        total_var=total_var, baseline_var=baseline_var,
        excess_var=excess_var, has_excess=has_excess,
        sample_ids=sample_ids,
    )
    if gene_names is not None:
        out['gene_names'] = gene_names
    return out
