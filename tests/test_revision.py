"""
Tests for v0.2 additions: variance decomposition, physical-scale
neighborhoods, gene-specific dispersion and reference-gene centering.
"""

import numpy as np
import pytest

from aura.model import (
    compute_composition_vectors,
    compute_pearson_residuals,
    composition_kernel,
    effect_size,
    estimate_dispersion,
    estimate_gene_dispersion,
    permutation_test,
    run_model,
    run_model_multisample,
)


def _tissue(rng, n_focal=300, n_other=1200, n_types=4, size=300.0):
    focal_xy = rng.uniform(0, size, size=(n_focal, 2))
    all_xy = np.vstack([focal_xy, rng.uniform(0, size, size=(n_other, 2))])
    all_types = np.concatenate([
        np.zeros(n_focal, dtype=int),
        rng.integers(1, n_types, size=n_other),
    ])
    return focal_xy, all_xy, all_types


# ─────────────────────────────────────────────────────────────────────
# variance decomposition
# ─────────────────────────────────────────────────────────────────────

def test_r2_total_identity_and_legacy():
    rng = np.random.default_rng(0)
    focal_xy, all_xy, all_types = _tissue(rng)
    counts = rng.poisson(5, size=(300, 40)).astype(float)
    r = run_model(counts, focal_xy, all_xy, all_types, k=10, n_perm=50)
    assert np.allclose(r['R2_total'], r['R2'] * r['var_retained'])
    legacy = np.where(r['total_var'] > 0,
                      r['R2'] * r['excess_var'] / r['total_var'], 0.0)
    legacy[~r['has_excess']] = 0.0
    assert np.allclose(r['R2_total_legacy'], legacy)


def test_r2_total_without_centering_is_count_scale_r2():
    """Without cell centering, R2_total equals the OLS R² of raw counts on
    composition, independent of phi."""
    rng = np.random.default_rng(1)
    n, G = 400, 20
    P = rng.dirichlet(np.ones(5), size=n)
    _, P_tilde = composition_kernel(P)
    counts = rng.poisson(3 + 10 * P[:, :1], size=(n, G)).astype(float)
    from aura.model import variance_decomposition
    for phi in (0.5, 5.0, 50.0):
        res = compute_pearson_residuals(counts, phi, center_cells=False)
        _, R2 = effect_size(res, P_tilde)
        vd = variance_decomposition(counts, phi, res, R2, rank=4)
        Xc = counts - counts.mean(axis=0)
        _, R2_counts = effect_size(Xc, P_tilde)
        assert np.allclose(vd['R2_total'], R2_counts)


def test_null_r2_matches_d_over_n_minus_1():
    rng = np.random.default_rng(2)
    n, G, K = 200, 3000, 6
    P = rng.dirichlet(np.ones(K), size=n)
    _, P_tilde = composition_kernel(P)
    res = rng.standard_normal((n, G))
    res -= res.mean(axis=0)
    _, R2 = effect_size(res, P_tilde)
    d = np.linalg.matrix_rank(P_tilde)
    assert d == K - 1
    assert R2.mean() == pytest.approx(d / (n - 1), rel=0.05)


def test_pvalues_invariant_to_phi_without_centering():
    """Per-gene rescaling leaves Q's permutation distribution unchanged."""
    rng = np.random.default_rng(3)
    n, G = 200, 30
    P = rng.dirichlet(np.ones(4), size=n)
    _, P_tilde = composition_kernel(P)
    counts = rng.negative_binomial(2, 0.3, size=(n, G)).astype(float)
    p_list = []
    for phi in (0.5, 2.0, np.linspace(0.5, 5, G)):
        res = compute_pearson_residuals(counts, phi, center_cells=False)
        _, p = permutation_test(res, P_tilde, n_perm=400, seed=7)
        p_list.append(p)
    for p in p_list[1:]:
        assert np.abs(p - p_list[0]).max() <= 2 / 401


# ─────────────────────────────────────────────────────────────────────
# radius / ring neighborhoods
# ─────────────────────────────────────────────────────────────────────

def test_radius_composition_matches_bruteforce():
    rng = np.random.default_rng(4)
    focal_xy, all_xy, all_types = _tissue(rng, n_focal=50, n_other=300)
    P, n_nbrs = compute_composition_vectors(
        focal_xy, all_xy, all_types, n_types=4, radius=40,
        return_n_neighbors=True)
    D = np.linalg.norm(focal_xy[:, None] - all_xy[None], axis=2)
    for i in range(len(focal_xy)):
        nb = (D[i] < 40) & (D[i] >= 1e-10)
        assert n_nbrs[i, 0] == nb.sum()
        if nb.sum():
            expect = np.bincount(all_types[nb], minlength=4) / nb.sum()
            assert np.allclose(P[i], expect)


def test_rings_blocks_and_pvalues():
    rng = np.random.default_rng(5)
    focal_xy, all_xy, all_types = _tissue(rng)
    counts = rng.poisson(5, size=(300, 25)).astype(float)
    r = run_model(counts, focal_xy, all_xy, all_types,
                  rings=[20, 60], n_perm=50, min_neighbors=1)
    n_kept = r['cell_mask'].sum()
    assert r['P'].shape == (n_kept, 8)
    assert r['beta'].shape == (25, 8)
    assert r['block_pvalues'].shape == (2, 25)
    blocks = r['P'].reshape(n_kept, 2, 4).sum(axis=2)
    assert np.allclose(blocks, 1.0)
    assert r['neighborhood'] == {'type': 'rings', 'rings': [20, 60]}


def test_isolated_cells_are_excluded():
    rng = np.random.default_rng(6)
    focal_xy, all_xy, all_types = _tissue(rng, n_focal=100, n_other=400)
    focal_xy[0] = all_xy[0] = [1e5, 1e5]  # far from everything
    counts = rng.poisson(5, size=(100, 10)).astype(float)
    r = run_model(counts, focal_xy, all_xy, all_types, radius=30, n_perm=20)
    assert not r['cell_mask'][0]
    assert r['residuals'].shape[0] == r['cell_mask'].sum()


def test_multisample_radius_stays_within_sample():
    rng = np.random.default_rng(7)
    focal_xy, all_xy, all_types = _tissue(rng, n_focal=200, n_other=800)
    focal_s = rng.integers(0, 2, 200)
    all_s = np.concatenate([focal_s, rng.integers(0, 2, 800)])
    counts = rng.poisson(5, size=(200, 15)).astype(float)
    r = run_model_multisample(counts, focal_xy, all_xy, all_types,
                              focal_s, all_s, radius=50, n_perm=20)
    assert r['P'].shape[1] == 4
    assert r['neighborhood']['type'] == 'radius'


# ─────────────────────────────────────────────────────────────────────
# dispersion and centering options
# ─────────────────────────────────────────────────────────────────────

def test_gene_dispersion_tracks_mean_trend():
    rng = np.random.default_rng(8)
    n, G = 2000, 300
    mu = np.exp(rng.uniform(-0.5, 3, G))
    phi_true = 0.5 + 0.5 * mu            # dispersion increasing with mean
    counts = rng.negative_binomial(phi_true, phi_true / (phi_true + mu),
                                   size=(n, G)).astype(float)
    phi_g = estimate_gene_dispersion(counts, min_mean=0.5)
    assert phi_g.shape == (G,)
    ok = mu > 1
    ratio = np.median(phi_g[ok] / phi_true[ok])
    assert 0.7 < ratio < 1.4
    assert np.corrcoef(np.log(phi_g[ok]), np.log(phi_true[ok]))[0, 1] > 0.9


def test_run_model_gene_dispersion_runs():
    rng = np.random.default_rng(9)
    focal_xy, all_xy, all_types = _tissue(rng)
    counts = rng.negative_binomial(3, 0.4, size=(300, 40)).astype(float)
    r = run_model(counts, focal_xy, all_xy, all_types, k=10, n_perm=30,
                  dispersion='gene')
    assert np.ndim(r['phi']) == 1 and len(r['phi']) == 40
    assert np.isfinite(r['R2_total']).all()


@pytest.mark.parametrize('center', ['median', 'trimmed'])
def test_robust_centering_options(center):
    rng = np.random.default_rng(10)
    counts = rng.poisson(5, size=(50, 30)).astype(float)
    r = compute_pearson_residuals(counts, phi=10.0, center=center)
    raw = compute_pearson_residuals(counts, phi=10.0, center_cells=False)
    shift = raw - r
    assert np.allclose(shift, shift[:, :1])  # one constant per cell


def test_center_genes_reference_set():
    rng = np.random.default_rng(11)
    counts = rng.poisson(5, size=(50, 30)).astype(float)
    ref = np.zeros(30, dtype=bool)
    ref[10:] = True
    r = compute_pearson_residuals(counts, phi=10.0, center_genes=ref)
    assert np.allclose(r[:, ref].mean(axis=1), 0, atol=1e-12)


def test_reference_centering_prevents_leakage():
    """Coherent composition effects in many genes leak into null genes
    through mean centering; centering on null reference genes prevents it."""
    rng = np.random.default_rng(12)
    n, G, K, n_sig = 1500, 120, 5, 50
    P = rng.dirichlet(np.ones(K) * 0.5, size=n)
    _, P_tilde = composition_kernel(P)
    beta = np.zeros((G, K))
    beta[:n_sig] = np.outer(rng.uniform(1, 3, n_sig), rng.normal(0, 1, K))
    eta = P_tilde @ beta.T
    lam = 4 * np.exp(eta) / np.exp(eta).mean(axis=0)
    counts = rng.poisson(lam).astype(float)
    phi = estimate_dispersion(counts)
    ref = np.zeros(G, dtype=bool)
    ref[n_sig + 20:] = True
    null = slice(n_sig, G)

    res_mean = compute_pearson_residuals(counts, phi)
    _, p_mean = permutation_test(res_mean, P_tilde, n_perm=200, seed=1)
    res_ref = compute_pearson_residuals(counts, phi, center_genes=ref)
    _, p_ref = permutation_test(res_ref, P_tilde, n_perm=200, seed=1)

    assert (p_mean[null] < 0.05).mean() > 0.5
    assert (p_ref[null] < 0.05).mean() < 0.15


# ─────────────────────────────────────────────────────────────────────
# I/O with ring neighborhoods
# ─────────────────────────────────────────────────────────────────────

def test_save_results_ring_columns(tmp_path):
    from aura.io import save_results, load_results
    rng = np.random.default_rng(13)
    focal_xy, all_xy, all_types = _tissue(rng)
    counts = rng.poisson(5, size=(300, 12)).astype(float)
    r = run_model(counts, focal_xy, all_xy, all_types, rings=[25, 70],
                  n_perm=20)
    path = tmp_path / 'res.csv'
    save_results(path, np.array([f'g{i}' for i in range(12)]), r,
                 ['A', 'B', 'C', 'D'])
    df = load_results(path)
    for col in ('R2_total_adj', 'R2_total_legacy', 'pvalue_ring0',
                'pvalue_ring1', 'beta_A_ring0', 'beta_D_ring1'):
        assert col in df.columns
