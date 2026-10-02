"""
Unit tests for aura.model.

Focused on the statistical core: dispersion, residuals, kernel, permutation
test, BH-FDR, effect size. Uses synthetic data — no h5ad dependencies.
"""

import numpy as np
import pytest

from aura.model import (
    estimate_dispersion,
    compute_pearson_residuals,
    compute_composition_vectors,
    composition_kernel,
    permutation_test,
    bh_fdr,
    effect_size,
    run_model,
)


# ─────────────────────────────────────────────────────────────────────
# bh_fdr
# ─────────────────────────────────────────────────────────────────────

def test_bh_fdr_empty():
    """Empty p-value array must not crash (regression: GC_B label refined
    upstream → empty focal cell set → empty p-value array → IndexError)."""
    q, sig = bh_fdr(np.array([]))
    assert len(q) == 0
    assert len(sig) == 0
    assert q.dtype == float
    assert sig.dtype == bool


def test_bh_fdr_single_element():
    q, sig = bh_fdr(np.array([0.01]))
    assert q[0] == pytest.approx(0.01)
    assert sig[0] == True

    q, sig = bh_fdr(np.array([0.5]))
    assert q[0] == pytest.approx(0.5)
    assert sig[0] == False


def test_bh_fdr_monotone():
    """q-values must be monotone non-decreasing in the original index order
    when sorted by p-value (BH guarantee)."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=200)
    q, _ = bh_fdr(p)
    # When re-sorted by p, q should be monotone non-decreasing.
    order = np.argsort(p)
    q_sorted = q[order]
    assert np.all(np.diff(q_sorted) >= -1e-12)


def test_bh_fdr_matches_statsmodels():
    """Cross-check against statsmodels.multitest."""
    from statsmodels.stats.multitest import multipletests
    rng = np.random.default_rng(42)
    p = rng.uniform(0, 1, size=500)
    p[:50] = rng.uniform(0, 0.001, size=50)  # 50 strong signals
    q, sig = bh_fdr(p, alpha=0.05)
    _, q_sm, _, _ = multipletests(p, alpha=0.05, method='fdr_bh')
    assert np.allclose(q, q_sm, atol=1e-12)


# ─────────────────────────────────────────────────────────────────────
# estimate_dispersion + compute_pearson_residuals
# ─────────────────────────────────────────────────────────────────────

def test_estimate_dispersion_recovers_known_phi():
    """Simulate NB(mu=5, phi=2) for 100 genes × 1000 cells; estimator
    should recover phi within ~30% on a single sample."""
    rng = np.random.default_rng(0)
    n_cells, n_genes = 1000, 100
    phi_true = 2.0
    mu = np.full(n_genes, 5.0)
    counts = np.zeros((n_cells, n_genes))
    for g in range(n_genes):
        # numpy NB: n=phi (size), p = phi/(phi+mu)
        p = phi_true / (phi_true + mu[g])
        counts[:, g] = rng.negative_binomial(phi_true, p, size=n_cells)
    phi_hat = estimate_dispersion(counts, min_mean=0.5)
    assert 1.0 < phi_hat < 4.0  # generous bounds


def test_pearson_residuals_zero_mean_after_centering():
    """Per-cell mean residual should be exactly zero after center_cells=True."""
    rng = np.random.default_rng(0)
    counts = rng.poisson(5, size=(100, 50)).astype(float)
    r = compute_pearson_residuals(counts, phi=10.0, center_cells=True)
    cell_means = r.mean(axis=1)
    assert np.allclose(cell_means, 0, atol=1e-12)


def test_pearson_residuals_variance_finite_with_zero_mean_gene():
    """A gene with mean 0 (var=0) shouldn't produce NaN/inf — the var
    floor handles this."""
    counts = np.zeros((10, 5))
    counts[:, 0] = 1  # one nonzero column
    r = compute_pearson_residuals(counts, phi=1.0)
    assert np.isfinite(r).all()


# ─────────────────────────────────────────────────────────────────────
# compute_composition_vectors
# ─────────────────────────────────────────────────────────────────────

def test_composition_vectors_sum_to_one():
    rng = np.random.default_rng(0)
    n = 100
    xy = rng.uniform(0, 100, size=(n, 2))
    types = rng.integers(0, 4, size=n)
    P = compute_composition_vectors(xy, xy, types, k=10, n_types=4)
    assert P.shape == (n, 4)
    assert np.allclose(P.sum(axis=1), 1.0)


def test_composition_vectors_skip_self():
    """When focal_xy == all_xy, the cell should not be its own neighbor."""
    n = 50
    xy = np.column_stack([np.arange(n).astype(float), np.zeros(n)])
    types = np.zeros(n, dtype=int)
    types[0] = 1  # cell 0 is type 1, all others type 0
    P = compute_composition_vectors(xy, xy, types, k=2, n_types=2)
    # Cell 0: nearest neighbors are 1 and 2 (both type 0). It should NOT
    # see itself, so P[0, 1] should be 0.
    assert P[0, 1] == 0
    assert P[0, 0] == 1.0


# ─────────────────────────────────────────────────────────────────────
# composition_kernel
# ─────────────────────────────────────────────────────────────────────

def test_kernel_centered():
    rng = np.random.default_rng(0)
    P = rng.dirichlet(np.ones(5), size=100)
    # Default: skip kernel build (returns None for K_comp)
    K_comp, P_tilde = composition_kernel(P)
    assert P_tilde.shape == P.shape
    assert K_comp is None
    # P_tilde column means should be ~zero (centered)
    assert np.allclose(P_tilde.mean(axis=0), 0, atol=1e-12)


def test_kernel_centered_with_kernel_build():
    rng = np.random.default_rng(0)
    P = rng.dirichlet(np.ones(5), size=100)
    K_comp, P_tilde = composition_kernel(P, build_kernel=True)
    assert P_tilde.shape == P.shape
    assert K_comp.shape == (100, 100)
    assert np.allclose(K_comp, K_comp.T)
    # K_comp = P_tilde @ P_tilde.T (low-rank factor identity)
    assert np.allclose(K_comp, P_tilde @ P_tilde.T)


# ─────────────────────────────────────────────────────────────────────
# permutation_test (calibration)
# ─────────────────────────────────────────────────────────────────────

def test_permutation_test_calibrated_under_null():
    """Random residuals + random composition → p-values should be uniform.

    With n_genes=400 the expected count below p=0.05 is 20 ± ~4 (Poisson),
    so we use a wide tolerance band.
    """
    rng = np.random.default_rng(0)
    n_c, n_genes, K = 200, 400, 4
    residuals = rng.standard_normal(size=(n_c, n_genes))
    P = rng.dirichlet(np.ones(K), size=n_c)
    _, P_tilde = composition_kernel(P)
    _, pvals = permutation_test(residuals, P_tilde, n_perm=2000, seed=42)
    frac = (pvals < 0.05).mean()
    assert 0.02 < frac < 0.10
    # Mean of uniform should be ~0.5
    assert 0.4 < pvals.mean() < 0.6


def test_permutation_test_detects_planted_signal():
    """A gene whose expression is monotone in P[:,0] should get a small p."""
    rng = np.random.default_rng(0)
    n_c, n_genes, K = 200, 400, 4
    P = rng.dirichlet(np.ones(K), size=n_c)
    residuals = rng.standard_normal(size=(n_c, n_genes))
    # Plant a strong signal in gene 0: residual proportional to P[:,0]
    residuals[:, 0] = 5 * (P[:, 0] - P[:, 0].mean()) + 0.1 * residuals[:, 0]
    _, P_tilde = composition_kernel(P)
    _, pvals = permutation_test(residuals, P_tilde, n_perm=2000, seed=42)
    assert pvals[0] < 0.01  # planted signal is detected
    # Other genes should be ~uniform
    other_p = pvals[1:]
    assert (other_p < 0.05).mean() < 0.10


# ─────────────────────────────────────────────────────────────────────
# effect_size
# ─────────────────────────────────────────────────────────────────────

def test_effect_size_recovers_planted_beta():
    """If we plant a zero-sum β (in the rowspace of centered P), the OLS
    estimate should recover it.

    Note: P_tilde has rank K-1 because of the simplex constraint, so only
    zero-sum β vectors are in the rowspace and can be recovered exactly.
    A non-zero-sum β would project onto the rowspace and the recovered
    coefficients would be biased.
    """
    rng = np.random.default_rng(0)
    n_c, n_genes, K = 500, 10, 4
    P = rng.dirichlet(np.ones(K), size=n_c)
    _, P_tilde = composition_kernel(P)
    # Zero-sum β so it's in the rowspace of P_tilde
    true_beta_one = np.array([2.0, -1.0, -0.5, -0.5])
    assert np.isclose(true_beta_one.sum(), 0)
    true_beta = np.tile(true_beta_one, (n_genes, 1))  # (n_genes, K)
    # Construct residuals: P_tilde @ true_beta.T → (n_c, n_genes)
    signal = P_tilde @ true_beta.T
    residuals = signal + 0.01 * rng.standard_normal(size=(n_c, n_genes))
    beta_hat, R2 = effect_size(residuals, P_tilde)
    assert beta_hat.shape == (n_genes, K)
    # Recovery should be tight because zero-sum β is exactly representable
    assert np.allclose(beta_hat, true_beta, atol=0.05)
    # R² should be high (signal dominates 100:1 over noise)
    assert (R2 > 0.99).all()


# ─────────────────────────────────────────────────────────────────────
# run_model end-to-end smoke test
# ─────────────────────────────────────────────────────────────────────

EXPECTED_RUN_KEYS = {
    'Q', 'pvalues', 'qvalues', 'significant', 'R2', 'R2_total', 'beta',
    'phi', 'residuals', 'P', 'P_tilde', 'K_comp',
    'total_var', 'baseline_var', 'excess_var', 'has_excess', 'sample_ids',
    'R2_adj', 'R2_total_adj', 'R2_total_legacy', 'var_retained', 'R2_null',
    'rank_P', 'block_pvalues', 'cell_mask', 'n_neighbors', 'neighborhood',
}


def test_run_model_smoke():
    """End-to-end smoke test for single-tissue run_model."""
    rng = np.random.default_rng(0)
    n_focal = 200
    n_other = 800
    n_genes = 30

    focal_xy = rng.uniform(0, 100, size=(n_focal, 2))
    other_xy = rng.uniform(0, 100, size=(n_other, 2))
    all_xy = np.vstack([focal_xy, other_xy])
    all_types = np.concatenate([
        np.zeros(n_focal, dtype=int),       # focal cells = type 0
        rng.integers(1, 4, size=n_other),   # others = types 1, 2, 3
    ])
    counts = rng.poisson(5, size=(n_focal, n_genes)).astype(float)

    result = run_model(
        counts=counts, focal_xy=focal_xy, all_xy=all_xy, all_types=all_types,
        k=15, n_perm=200, alpha=0.05, seed=42,
    )
    assert EXPECTED_RUN_KEYS <= set(result.keys())
    assert result['pvalues'].shape == (n_genes,)
    assert result['beta'].shape == (n_genes, 4)
    # Single-tissue: sample_ids should be None
    assert result['sample_ids'] is None
    # Under the null (random Poisson) most genes shouldn't be significant
    assert result['significant'].sum() < n_genes * 0.30


def test_run_model_multisample_dispersion_mode_per_sample():
    """median_per_sample dispersion fits per-sample φ and pools by median.
    Verify it runs and produces a similar result to pooled (synthetic data
    where there's no batch effect → both should agree)."""
    from aura.model import run_model_multisample
    rng = np.random.default_rng(0)
    n_focal_per_sample = 250
    n_samples = 4
    n_focal = n_focal_per_sample * n_samples
    n_genes = 30
    n_other = 1000

    focal_xy = rng.uniform(0, 100, size=(n_focal, 2))
    other_xy = rng.uniform(0, 100, size=(n_other, 2))
    all_xy = np.vstack([focal_xy, other_xy])
    all_types = np.concatenate([
        np.zeros(n_focal, dtype=int),
        rng.integers(1, 4, size=n_other),
    ])
    focal_sample = np.repeat(np.arange(n_samples), n_focal_per_sample)
    all_sample = np.concatenate([focal_sample, rng.integers(0, n_samples, size=n_other)])
    counts = rng.poisson(5, size=(n_focal, n_genes)).astype(float)

    r_pooled = run_model_multisample(
        counts=counts, focal_xy=focal_xy, all_xy=all_xy, all_types=all_types,
        focal_sample_ids=focal_sample, all_sample_ids=all_sample,
        k=10, n_perm=50, seed=42, dispersion_mode='pooled')
    r_persample = run_model_multisample(
        counts=counts, focal_xy=focal_xy, all_xy=all_xy, all_types=all_types,
        focal_sample_ids=focal_sample, all_sample_ids=all_sample,
        k=10, n_perm=50, seed=42, dispersion_mode='median_per_sample')

    # Both should give the same dict shape and finite phi
    assert set(r_pooled) == set(r_persample)
    assert np.isfinite(r_pooled['phi'])
    assert np.isfinite(r_persample['phi'])
    # On Poisson-only synthetic data with no batch effect, the two phi
    # estimates should be in the same order of magnitude (within 3x).
    ratio = r_persample['phi'] / r_pooled['phi']
    assert 0.3 < ratio < 3.0


def test_run_model_sample_filters_low_mean_genes():
    """run_model_sample should drop genes with mean < min_gene_mean
    before fitting (otherwise low-expression noise contaminates the test)."""
    from aura.model import run_model_sample
    rng = np.random.default_rng(0)
    n_c, n_genes_total, K = 100, 30, 4
    sample_ids = np.array([0] * 50 + [1] * 50)
    sample_comp = rng.dirichlet(np.ones(K), size=2)
    counts = rng.poisson(5, size=(n_c, n_genes_total)).astype(float)
    # Make the last 10 genes essentially unexpressed (mean ≈ 0.01)
    counts[:, -10:] = (rng.random((n_c, 10)) < 0.01).astype(float)
    gene_names = np.array([f'g{i}' for i in range(n_genes_total)])

    r = run_model_sample(
        counts=counts, sample_ids=sample_ids, sample_composition=sample_comp,
        n_perm=50, min_gene_mean=0.1, seed=42, gene_names=gene_names,
    )
    # 10 genes should be filtered out → 20 remain
    assert r['pvalues'].shape == (20,)
    assert 'gene_names' in r
    assert len(r['gene_names']) == 20
    assert 'g29' not in set(r['gene_names'])  # unexpressed → dropped
    assert 'g0' in set(r['gene_names'])       # well-expressed → kept


def test_run_model_dict_keys_unified():
    """All three run_model entry points must return the same dict keys.

    Regression: previously run_model returned K_comp but not sample_ids;
    run_model_multisample returned sample_ids but not K_comp; the IPF
    pipeline had to recompute one or the other depending on which entry
    point it called.
    """
    from aura.model import run_model, run_model_sample, run_model_multisample
    rng = np.random.default_rng(0)
    n_focal, n_other, n_genes, K = 100, 300, 20, 4
    focal_xy = rng.uniform(0, 100, size=(n_focal, 2))
    all_xy = np.vstack([focal_xy, rng.uniform(0, 100, size=(n_other, 2))])
    all_types = np.concatenate([
        np.zeros(n_focal, dtype=int),
        rng.integers(1, K, size=n_other),
    ])
    counts = rng.poisson(5, size=(n_focal, n_genes)).astype(float)

    # 1. single-tissue
    r1 = run_model(counts=counts, focal_xy=focal_xy, all_xy=all_xy,
                   all_types=all_types, k=10, n_perm=50, seed=42)

    # 2. multisample (assign half the cells to each of 2 samples)
    sample_ids_focal = np.array([0] * 50 + [1] * 50)
    sample_ids_all = np.concatenate([
        sample_ids_focal,
        rng.integers(0, 2, size=n_other),
    ])
    r2 = run_model_multisample(
        counts=counts, focal_xy=focal_xy, all_xy=all_xy, all_types=all_types,
        focal_sample_ids=sample_ids_focal, all_sample_ids=sample_ids_all,
        k=10, n_perm=50, seed=42,
    )

    # 3. sample-level: each cell inherits its sample's composition vector
    sample_comp = rng.dirichlet(np.ones(K), size=2)
    r3 = run_model_sample(
        counts=counts, sample_ids=sample_ids_focal,
        sample_composition=sample_comp, n_perm=50, seed=42,
    )

    keys1, keys2, keys3 = set(r1), set(r2), set(r3)
    assert keys1 == keys2 == keys3 == EXPECTED_RUN_KEYS, \
        f"key mismatch: r1={keys1 - keys2 - keys3}, r2={keys2 - keys1 - keys3}, r3={keys3 - keys1 - keys2}"
    # sample_ids semantics: None for single-tissue, populated for multi/sample
    assert r1['sample_ids'] is None
    assert r2['sample_ids'] is not None
    assert r3['sample_ids'] is not None
