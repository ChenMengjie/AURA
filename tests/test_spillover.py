"""
Unit tests for aura.spillover.

All tests use synthetic data: a tiny tissue with two cell types and a few
genes whose expression we control directly. No external h5ad required.
"""

import numpy as np
import pandas as pd
import pytest

import aura
from aura.spillover import (
    learn_canonical,
    spillover_filter,
    spillover_filter_multi,
    spillover_distance_test,
    DEFAULT_CANONICAL,
    CANONICAL_IPF,
    CANONICAL_LYMPHNODE,
)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

class FakeAdata:
    """Minimal duck-typed AnnData replacement with .X and .obs."""
    def __init__(self, X, obs, var_names):
        self.X = X
        self.obs = pd.DataFrame(obs)
        # var_names is treated as a 1-D array everywhere in spillover.py.
        self.var_names = np.asarray(list(var_names))


def make_synthetic_adata(seed=0):
    """Tissue with 200 cells: 100 type_A, 100 type_B, 6 genes.

      gene1: A-marker  (mean 5 in A, 0 in B) → fold ≈ ∞
      gene2: B-marker  (mean 0 in A, 5 in B) → fold ≈ ∞
      gene3: shared    (mean 2 in both)      → fold = 1
      gene4: weak A    (mean 1 in A, 0.4 in B) → fold = 2.5
      gene5: undetected (mean 0 in both)
      gene6: noisy     (mean 0.6 in A, 0.5 in B) → fold = 1.2
    """
    rng = np.random.default_rng(seed)
    n_per = 100
    n_genes = 6
    X = np.zeros((2 * n_per, n_genes), dtype=np.float32)
    # Type A
    X[:n_per, 0] = rng.poisson(5, size=n_per)
    X[:n_per, 1] = rng.poisson(0.05, size=n_per)
    X[:n_per, 2] = rng.poisson(2, size=n_per)
    X[:n_per, 3] = rng.poisson(1.0, size=n_per)
    X[:n_per, 5] = rng.poisson(0.6, size=n_per)
    # Type B
    X[n_per:, 0] = rng.poisson(0.05, size=n_per)
    X[n_per:, 1] = rng.poisson(5, size=n_per)
    X[n_per:, 2] = rng.poisson(2, size=n_per)
    X[n_per:, 3] = rng.poisson(0.4, size=n_per)
    X[n_per:, 5] = rng.poisson(0.5, size=n_per)
    obs = {
        'cell_type': ['type_A'] * n_per + ['type_B'] * n_per,
    }
    return FakeAdata(X, obs, [f'gene{i}' for i in range(1, n_genes + 1)])


# ─────────────────────────────────────────────────────────────────────
# learn_canonical
# ─────────────────────────────────────────────────────────────────────

def test_learn_canonical_returns_tuple():
    adata = make_synthetic_adata(seed=42)
    canonical, stats_df = learn_canonical(adata, type_column='cell_type',
                                          min_fold=2.0, min_detection=0.05)
    assert isinstance(canonical, dict)
    assert isinstance(stats_df, pd.DataFrame)
    assert set(canonical.keys()) == {'type_A', 'type_B'}


def test_learn_canonical_finds_strong_markers():
    adata = make_synthetic_adata(seed=42)
    canonical, _ = learn_canonical(adata, type_column='cell_type',
                                   min_fold=2.0, min_detection=0.05)
    assert 'gene1' in canonical['type_A']        # strong A-marker
    assert 'gene2' in canonical['type_B']        # strong B-marker
    assert 'gene3' not in canonical['type_A']    # shared gene
    assert 'gene3' not in canonical['type_B']
    assert 'gene6' not in canonical['type_A']    # noisy, fold ≈ 1.2 < 2
    assert 'gene6' not in canonical['type_B']


def test_learn_canonical_fold_threshold_is_respected():
    adata = make_synthetic_adata(seed=42)
    # gene4 has fold ≈ 2.5; lower threshold should pick it up.
    canonical_low, _ = learn_canonical(adata, type_column='cell_type',
                                       min_fold=1.5, min_detection=0.05)
    canonical_high, _ = learn_canonical(adata, type_column='cell_type',
                                        min_fold=3.0, min_detection=0.05)
    assert 'gene4' in canonical_low['type_A']
    assert 'gene4' not in canonical_high['type_A']


def test_learn_canonical_stats_df_columns():
    adata = make_synthetic_adata(seed=42)
    _, stats_df = learn_canonical(adata, type_column='cell_type')
    expected_cols = {'gene', 'top_type', 'top_mean', 'second_type',
                     'second_mean', 'fold_enrichment', 'top_detection',
                     'is_marker'}
    assert expected_cols <= set(stats_df.columns)


# ─────────────────────────────────────────────────────────────────────
# spillover_filter
# ─────────────────────────────────────────────────────────────────────

class FakeResult:
    """Minimal AuraResult duck-type for spillover_filter."""
    def __init__(self, gene_names, type_names, beta, significant,
                 R2_total, qvalues):
        self.gene_names = np.asarray(gene_names)
        self.type_names = list(type_names)
        self.beta = np.asarray(beta, dtype=float)
        self.significant = np.asarray(significant, dtype=bool)
        self.R2_total = np.asarray(R2_total, dtype=float)
        self.qvalues = np.asarray(qvalues, dtype=float)


def make_fake_result():
    """3-gene synthetic AuraResult for spillover_filter testing.

    Composition groups: A, B, C.
    Genes:
      g_real      sig, driver +A, NOT a canonical marker → clean
      g_spillover sig, driver +B, IS a canonical B marker → suspect
      g_owned     sig, driver +A, IS a canonical A marker → clean (focal owns A)
    """
    return FakeResult(
        gene_names=['g_real', 'g_spillover', 'g_owned'],
        type_names=['A', 'B', 'C'],
        beta=[[2.0, 0.1, 0.0],   # g_real, driver A
              [0.1, 3.0, 0.2],   # g_spillover, driver B
              [4.0, 0.0, 0.5]],  # g_owned, driver A
        significant=[True, True, True],
        R2_total=[0.05, 0.10, 0.20],
        qvalues=[0.01, 0.001, 0.0001],
    )


def test_spillover_filter_flags_spillover():
    result = make_fake_result()
    canonical = {
        'A': {'g_owned'},
        'B': {'g_spillover'},
        'C': set(),
    }
    df = spillover_filter(result=result, focal_lineage='A',
                          canonical=canonical)
    df = df.set_index('gene')
    assert df.loc['g_real', 'spillover_suspect'] == False
    assert df.loc['g_spillover', 'spillover_suspect'] == True
    assert df.loc['g_owned', 'spillover_suspect'] == False  # focal owns A
    assert df.loc['g_spillover', 'driver_axis'] == 'B'


def test_spillover_filter_returns_only_significant():
    result = make_fake_result()
    result.significant = np.array([True, False, True])
    df = spillover_filter(result=result, focal_lineage='A',
                          canonical={'A': set(), 'B': set(), 'C': set()})
    assert set(df['gene']) == {'g_real', 'g_owned'}


def test_spillover_filter_reason_string():
    result = make_fake_result()
    canonical = {'A': set(), 'B': {'g_spillover'}, 'C': set()}
    df = spillover_filter(result=result, focal_lineage='A',
                          canonical=canonical)
    suspect = df[df['spillover_suspect']].iloc[0]
    assert 'g_spillover' in suspect['reason']
    assert 'B' in suspect['reason']


def test_spillover_filter_focal_lineage_list():
    """focal_lineage can be a list of multiple groups (e.g. macrophage subtypes)."""
    result = make_fake_result()
    canonical = {'A': {'g_real'}, 'B': {'g_spillover'}, 'C': set()}
    df = spillover_filter(result=result, focal_lineage=['A', 'C'],
                          canonical=canonical)
    df = df.set_index('gene')
    # g_real is now an own-marker since A is in focal_lineage → not flagged
    assert df.loc['g_real', 'spillover_suspect'] == False


# ─────────────────────────────────────────────────────────────────────
# spillover_filter_multi
# ─────────────────────────────────────────────────────────────────────

def test_spillover_filter_multi_concatenates(tmp_path):
    """Two synthetic 'focal' CSVs → concatenated table with focal_type col."""
    # Build a CSV that spillover_filter(csv_path=) can read.
    def write_csv(path, focal_owns, suspect_gene):
        df = pd.DataFrame({
            'gene': ['g_clean', suspect_gene],
            'significant': [True, True],
            'R2_total': [0.05, 0.10],
            'qvalue': [0.01, 0.001],
            'beta_A': [2.0, 0.1],
            'beta_B': [0.1, 3.0],
        })
        df.to_csv(path, index=False)

    csv1 = tmp_path / 'focal1.csv'
    csv2 = tmp_path / 'focal2.csv'
    write_csv(csv1, focal_owns='A', suspect_gene='g_in_B_1')
    write_csv(csv2, focal_owns='A', suspect_gene='g_in_B_2')

    canonical = {'A': set(), 'B': {'g_in_B_1', 'g_in_B_2'}}
    out = spillover_filter_multi(
        focals=[
            ('focal_alpha', str(csv1), 'A'),
            ('focal_beta',  str(csv2), 'A'),
        ],
        canonical=canonical,
    )
    assert set(out['focal_type']) == {'focal_alpha', 'focal_beta'}
    assert len(out) == 4
    suspects = out[out['spillover_suspect']]
    assert set(suspects['gene']) == {'g_in_B_1', 'g_in_B_2'}


def test_spillover_filter_multi_empty():
    """Empty focals list returns an empty DataFrame with the right columns."""
    out = spillover_filter_multi(focals=[], canonical={})
    assert len(out) == 0
    expected = {'gene', 'significant', 'R2_total', 'driver_axis',
                'beta_driver', 'qvalue', 'spillover_suspect', 'reason',
                'focal_type'}
    assert expected <= set(out.columns)


# ─────────────────────────────────────────────────────────────────────
# spillover_distance_test
# ─────────────────────────────────────────────────────────────────────

def make_distance_decay_data(n_focal=2000, decay='sharp', seed=0):
    """Build synthetic spatial data where the test gene shows distance decay.

    All cells in one sample. Driver-type cells live in a tight cluster at
    (0, 0). Focal cells are scattered radially with enough density near
    the origin to populate the small distance bins.

      decay='sharp'  → test gene strongly expressed in focal cells within
                       30 µm of the driver cluster, near-zero further away
      decay='none'   → test gene expressed uniformly across distances
    """
    rng = np.random.default_rng(seed)
    n_driver = 50
    driver_xy = rng.normal(loc=0, scale=3, size=(n_driver, 2))

    # Sample focal cells radially: r ~ uniform(0, 300), theta ~ uniform(0, 2π).
    # This guarantees uniform density per radial bin (vs uniform 2D which
    # severely under-samples small radii).
    r = rng.uniform(0, 300, size=n_focal)
    theta = rng.uniform(0, 2 * np.pi, size=n_focal)
    focal_xy = np.column_stack([r * np.cos(theta), r * np.sin(theta)])

    all_xy = np.vstack([focal_xy, driver_xy])
    all_types = np.array([0] * n_focal + [1] * n_driver)  # 0=focal, 1=driver
    all_sample = np.zeros(len(all_xy), dtype=int)
    focal_sample = np.zeros(n_focal, dtype=int)

    counts = np.zeros((n_focal, 2), dtype=np.float32)
    if decay == 'sharp':
        d = np.linalg.norm(focal_xy, axis=1)
        # Strong expression within 30µm, near-zero beyond
        prob = np.where(d < 30, 1.0, 0.02)
        counts[:, 0] = rng.poisson(prob * 8)
    else:
        counts[:, 0] = rng.poisson(2.0, size=n_focal)
    counts[:, 1] = rng.poisson(2.0, size=n_focal)  # control: uniform

    return dict(
        focal_xy=focal_xy, focal_counts=counts,
        focal_gene_names=np.array(['g_test', 'g_control']),
        focal_sample_ids=focal_sample,
        all_xy=all_xy, all_types=all_types, all_sample_ids=all_sample,
    )


def test_distance_test_flags_spillover():
    data = make_distance_decay_data(decay='sharp', seed=0)
    out = spillover_distance_test(
        gene_name='g_test', driver_type_idx=1,
        type_names=['focal', 'driver'],
        control_gene='g_control', verbose=False,
        **data,
    )
    assert out['verdict'] == 'spillover'
    assert out['fold_decay'] > 3
    assert out['control_fold'] < 1.5


def test_distance_test_clean_for_uniform_gene():
    data = make_distance_decay_data(decay='none', seed=0)
    out = spillover_distance_test(
        gene_name='g_test', driver_type_idx=1,
        type_names=['focal', 'driver'],
        control_gene='g_control', verbose=False,
        **data,
    )
    assert out['verdict'] == 'clean'


def test_distance_test_returns_required_keys():
    data = make_distance_decay_data(decay='sharp', seed=0)
    out = spillover_distance_test(
        gene_name='g_test', driver_type_idx=1, control_gene='g_control',
        verbose=False, **data,
    )
    expected_keys = {'gene', 'verdict', 'fold_decay', 'control_fold',
                     'control_gene', 'driver_name', 'n_valid', 'median_dist',
                     'bins_df'}
    assert expected_keys <= set(out.keys())
    assert isinstance(out['bins_df'], pd.DataFrame)


def test_distance_test_unknown_gene_raises():
    data = make_distance_decay_data()
    with pytest.raises(ValueError, match='not found'):
        spillover_distance_test(
            gene_name='g_unknown', driver_type_idx=1, **data,
        )


# ─────────────────────────────────────────────────────────────────────
# Presets are loadable + non-empty
# ─────────────────────────────────────────────────────────────────────

def test_presets_are_dicts_of_sets():
    for preset in [DEFAULT_CANONICAL, CANONICAL_IPF, CANONICAL_LYMPHNODE]:
        assert isinstance(preset, dict)
        assert len(preset) > 0
        for k, v in preset.items():
            assert isinstance(k, str)
            assert isinstance(v, set)
            assert len(v) > 0


def test_canonical_ipf_has_lung_groups():
    expected = {'Alveolar_epi', 'Airway_epi', 'Fibroblast', 'Macrophage',
                'Endothelial', 'T_cell', 'B_Plasma', 'SMC_Peri', 'Granulocyte'}
    assert expected == set(CANONICAL_IPF.keys())


def test_canonical_lymphnode_has_ln_groups():
    expected = {'B', 'GC_B', 'T_conv', 'Tfh', 'Treg', 'Mac', 'FDC', 'Endo',
                'Stromal'}
    assert expected == set(CANONICAL_LYMPHNODE.keys())


# ─────────────────────────────────────────────────────────────────────
# driver_axes (contribution rule)
# ─────────────────────────────────────────────────────────────────────

def test_driver_axes_ignores_rare_axis():
    """A rare neighbor type has a huge, noisy beta but almost no variation in
    local fraction; the contribution rule must not pick it."""
    from aura.spillover import driver_axes
    beta = np.array([[0.8, 0.0, 40.0]])      # axis 2: rare type
    sd = np.array([0.20, 0.10, 0.001])
    assert driver_axes(beta)[0][0] == 2                 # default: abs_beta
    idx, contrib = driver_axes(beta, sd, "contribution")
    assert idx[0] == 0
    assert contrib[0] == pytest.approx(0.16)


def test_driver_axes_prefers_positive_contribution():
    """With compositions summing to one, a source lineage can appear as a
    large negative coefficient on the dominant lineage; spillover drivers
    must be positive associations."""
    from aura.spillover import driver_axes
    beta = np.array([[-3.0, 1.5, 0.2],       # 'tumor' negative, 'fibro' positive
                     [-1.0, -2.0, -0.5]])    # no positive contribution
    sd = np.array([0.3, 0.2, 0.2])
    idx, contrib = driver_axes(beta, sd, "contribution")
    assert idx[0] == 1
    assert idx[1] == 1                       # no positive: largest |contribution| (-0.4)
    assert contrib[1] < 0


def test_driver_axes_requires_sd():
    from aura.spillover import driver_axes
    with pytest.raises(ValueError):
        driver_axes(np.ones((2, 3)), method="contribution")


def test_spillover_filter_uses_composition_sd(tmp_path):
    """End to end: a marker of a common lineage with modest beta is flagged,
    although a rare lineage has the largest |beta|; the same holds when
    reading a saved CSV (sdP_* columns)."""
    from aura.io import save_results
    from aura.spillover import spillover_filter
    rng = np.random.default_rng(0)
    n = 500
    P = np.column_stack([rng.uniform(0, 0.6, n), np.zeros(n), np.zeros(n)])
    P[:5, 2] = 0.05                          # rare lineage C
    P[:, 1] = 1 - P[:, 0] - P[:, 2]
    beta = np.array([[2.0, 0.0, 60.0],       # GENE1: driver A by contribution
                     [0.1, 0.0, 0.1]])
    res = dict(significant=np.array([True, True]),
               has_excess=np.array([True, True]),
               R2_total=np.array([0.1, 0.01]), R2=np.array([0.1, 0.01]),
               total_var=np.ones(2), baseline_var=np.ones(2),
               excess_var=np.ones(2), Q=np.ones(2),
               pvalues=np.array([1e-4, 1e-3]), qvalues=np.array([1e-3, 1e-2]),
               beta=beta, P=P)
    canonical = {'A': {'GENE1'}, 'B': set(), 'C': set()}
    path = tmp_path / 'r.csv'
    save_results(path, np.array(['GENE1', 'GENE2']), res, ['A', 'B', 'C'])

    out = spillover_filter(csv_path=path, canonical=canonical,
                           focal_lineage='B',
                           driver='contribution').set_index('gene')
    assert out.loc['GENE1', 'driver_axis'] == 'A'
    assert out.loc['GENE1', 'driver_axis_abs_beta'] == 'C'
    assert bool(out.loc['GENE1', 'spillover_suspect'])

    # default = manuscript rule (argmax |beta|)
    default = spillover_filter(csv_path=path, canonical=canonical,
                               focal_lineage='B').set_index('gene')
    assert default.loc['GENE1', 'driver_axis'] == 'C'
    assert not bool(default.loc['GENE1', 'spillover_suspect'])
