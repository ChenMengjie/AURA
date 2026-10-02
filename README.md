# AURA

AURA (Analysis of microenvironmental Regulatory Activity) tests whether gene expression in a focal cell type varies with local neighborhood composition beyond what is expected from baseline abundance and per-cell transcriptional output. It detects composition-dependent expression programs — genes whose expression is structured by *which cell types surround each cell* — using a kernel-based permutation test with built-in confound correction and spillover detection.

## Key features

- **Composition kernel test** — omnibus quadratic form across all composition axes, avoiding per-axis multiple testing
- **Batched float-32 GEMM** — low-rank permutation test processes B permutations in a single `(BK x n) . (n x G)` matrix multiply, 60–166x faster than naive looping
- **Confound-robust residuals** — NB null model with unconditional mean + cell-level centering removes library-size/composition confounding
- **Multi-sample mode** — within-sample kNN and within-sample permutation for TMA/multi-section designs
- **Data-driven spillover filter** — `learn_canonical` derives canonical markers from the data; `spillover_filter` flags transcript leakage artifacts
- **Distance-decay validation** — `spillover_distance_test` confirms suspects with spatial decay profiles
- **Effect decomposition** — per-gene beta vectors decomposed by SVD into principal composition modes
- **Variance decomposition** — partitions gene variance into baseline NB, library-size (alpha), composition (beta), and unexplained components

## Tutorials

| Dataset | Notebook | Description |
|---------|----------|-------------|
| **Human Lymph Node** (Xenium 5K, 702K cells, 4624 genes) | [`tutorial_lymphnode.ipynb`](tutorials/tutorial_lymphnode.ipynb) | Single-sample mode, Tfh + Macrophage focal types, SVD interpretation, gene zoom, cross-focal comparison |
| **IPF Lung** (Xenium custom, 1.6M cells, 343 genes) | [`tutorial_ipf.ipynb`](tutorials/tutorial_ipf.ipynb) | Multi-sample mode (45 TMA cores), beta heatmap, data-driven spillover detection, distance-decay validation |
| **NSCLC** (CosMx 1K, 766K cells, 960 genes) | [`tutorial_nsclc.ipynb`](tutorials/tutorial_nsclc.ipynb) | Multi-sample mode, tumor microenvironment, cross-focal driver comparison, spillover filter at intermediate suspect rate |

## Installation

```bash
git clone https://github.com/ChenMengjie/AURA.git
cd aura
pip install -e .
```

**Requirements:** Python >= 3.9, numpy, scipy, pandas, scanpy, matplotlib.

## Quick start

### Single-sample analysis

```python
import aura

# 1. Load tissue data
tissue = aura.load_adata(
    'tissue.h5ad',
    xy_columns=('x_centroid', 'y_centroid'),
    type_column='lineage',       # composition axes (e.g., 10 lineage groups)
)

# 2. Run AURA on a focal cell type
result = aura.run_aura(
    tissue,
    focal_label='Tfh',           # focal cell type to analyze
    label_column='cell_type',    # column to select focal cells
    k=30, n_perm=5000, alpha=0.05,
)

print(f'Significant: {result.n_significant}/{result.n_genes}')

# 3. Diagnostics
aura.diagnostic_panel(result, title='Tfh')

# 4. SVD result panel
aura.result_panel(result=result, title='Tfh')

# 5. Single-gene deep dive
aura.gene_zoom_panel('TOX2', result, all_xy=tissue.all_xy)

# 6. Save
result.save('results/aura_Tfh.csv')
```

### Multi-sample analysis (TMA / multi-section)

```python
import aura

tissue = aura.load_adata('ipf.h5ad', type_column='comp_group')

result = aura.run_aura_multisample(
    tissue,
    focal_label='AT2',
    sample_column='sample',       # within-sample kNN + permutation
    label_column='cell_type',
    k=30, n_perm=5000,
)
```

### Spillover detection

```python
import aura

# Learn canonical markers from the data
canonical, stats_df = aura.learn_canonical(
    adata, type_column='comp_group',
    min_fold=2.0, min_detection=0.05,
)

# Filter a single focal type
clean_df = aura.spillover_filter(
    result=result,
    focal_lineage='Alveolar_epi',
    canonical=canonical,
)

# Filter multiple focal types at once
combined = aura.spillover_filter_multi(
    [
        ('AT2',      result_at2,  'Alveolar_epi'),
        ('SPP1_Mac', result_spp1, 'Macrophage'),
        ('Basal',    result_bas,  'Airway_epi'),
    ],
    canonical=canonical,
)

# Validate a suspect gene with distance-decay test
verdict = aura.spillover_distance_test(
    gene_name='SFTPC',
    focal_xy=result.focal_xy,
    focal_counts=result.counts,
    focal_gene_names=result.gene_names,
    focal_sample_ids=focal_sample_ids,
    all_xy=tissue.all_xy,
    all_types=tissue.all_types,
    all_sample_ids=all_sample_ids,
    driver_type_idx=tissue.type_to_int['Alveolar_epi'],
    type_names=tissue.type_names,
    verbose=True,
)
```

## Method

### Stage 1: Null model and residuals

For a focal cell type with count matrix X (n cells x G genes), AURA fits a negative binomial null with shared dispersion phi and computes Pearson residuals:

```
r_ig = (X_ig - mu_g) / sqrt(mu_g + mu_g^2 / phi)
```

Cell-level centering removes the shared per-cell transcriptional shift: `r_tilde_ig = r_ig - mean_g(r_ig)`. This removes global cell-level effects correlated with library size and composition while limiting contamination from composition-responsive genes.

### Stage 2: Composition kernel

For each focal cell, neighborhood composition is computed as the fraction of each cell type among its k nearest neighbors. The centered composition matrix P_tilde is used directly as a low-rank factor — no n x n kernel is materialized.

### Stage 3: Permutation test

The omnibus test statistic measures total squared projection of centered residuals onto the composition subspace:

```
Q_g = ||P_tilde^T r_tilde_g||^2
```

Significance is assessed by permuting cell labels relative to the fixed composition structure. The low-rank form reduces cost from O(n^2 G) to O(nKG) per permutation and processes B permutations simultaneously via a single batched GEMM.

### Stage 4: Effect decomposition

For significant genes, cell-type-resolved beta coefficients are computed by OLS regression:

```
beta_g = (P_tilde^T P_tilde)^+ P_tilde^T r_tilde_g
```

The beta matrix is decomposed by SVD to identify principal composition modes — shared patterns of context-dependent regulation across significant genes.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | 30 | Number of spatial nearest neighbors for composition |
| `n_perm` | 5000 | Number of permutations for the test |
| `alpha` | 0.05 | FDR threshold (Benjamini-Hochberg) |
| `min_gene_mean` | 0.1 | Minimum gene mean for inclusion |
| `seed` | 42 | Random seed for reproducibility |
| `radius` | None | Physical neighborhood radius (coordinate units); overrides `k` |
| `rings` | None | Increasing annulus radii, e.g. `[30, 100]`; adds per-ring p-values |
| `min_neighbors` | 1 | Focal cells with fewer neighbors (radius/rings) are excluded |
| `dispersion` | `'shared'` | `'shared'` NB dispersion, or `'gene'` (regularized mean trend) |
| `center` | `'mean'` | Per-cell centering: `'mean'`, `'median'`, or `'trimmed'` |
| `center_genes` | None | Reference genes (names or mask) used for per-cell centering |
| `center_samples` | False | Multi-sample only: remove per-sample means (within-sample effects) |
| `min_fold` | 2.0 | Fold-enrichment threshold for canonical markers |
| `min_detection` | 0.05 | Detection rate threshold for canonical markers |

## Modules

| Module | Description |
|--------|-------------|
| `aura.model` | Core statistical engine: NB null, residuals, composition kernel, permutation test, effect decomposition |
| `aura.io` | Data loading (`load_adata`), focal cell extraction, CSV save/load |
| `aura.workflow` | High-level API: `run_aura`, `run_aura_multisample`, `AuraResult` container |
| `aura.spillover` | Spillover detection: `learn_canonical`, `spillover_filter`, `spillover_distance_test` |
| `aura.plot` | Visualization: `diagnostic_panel`, `variance_panel`, `result_panel`, `gene_zoom_panel` |

## Input data requirements

| Input | Format | Notes |
|-------|--------|-------|
| Expression matrix | AnnData `.X` | Raw counts (not normalized or log-transformed) |
| Spatial coordinates | `adata.obs` columns | Default: `x_centroid`, `y_centroid` |
| Composition labels | `adata.obs` column | Lineage-level groups (10-15 types recommended) |
| Focal cell labels | `adata.obs` column | Fine cell type labels for focal selection |
| Sample labels | `adata.obs` column | Required for multi-sample mode only |

The composition column (`type_column`) defines the axes of the composition kernel — typically 10-15 lineage-level groups. The focal cell column (`label_column`) can be a finer annotation used to select the specific cell population to analyze.

## Citation

If you use AURA in your research, please cite:

> [Citation to be added upon publication]

## License

AURA is released under the MIT License (see `LICENSE`).
