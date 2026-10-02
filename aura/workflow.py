"""
AURA workflow: single focal type pipeline.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .model import run_model, run_model_multisample
from .io import TissueData, FocalData, load_adata, extract_focal, save_results

log = logging.getLogger(__name__)


@dataclass
class AuraResult:
    """Container for AURA analysis results."""
    # Input references
    focal_data: FocalData
    tissue_data: TissueData
    type_names: list

    # Model outputs (from run_model dict)
    Q: np.ndarray
    pvalues: np.ndarray
    qvalues: np.ndarray
    significant: np.ndarray
    R2: np.ndarray
    R2_total: np.ndarray
    beta: np.ndarray
    phi: float
    residuals: np.ndarray
    P: np.ndarray
    P_tilde: np.ndarray
    total_var: np.ndarray
    baseline_var: np.ndarray
    excess_var: np.ndarray
    has_excess: np.ndarray
    sample_ids: np.ndarray = None  # populated for multi-sample runs only
    R2_adj: np.ndarray = None
    R2_total_adj: np.ndarray = None
    R2_total_legacy: np.ndarray = None
    var_retained: np.ndarray = None
    block_pvalues: np.ndarray = None  # (n_rings, n_genes) for ring neighborhoods
    neighborhood: dict = None

    # Convenience properties
    @property
    def gene_names(self):
        return self.focal_data.gene_names

    @property
    def n_significant(self):
        return self.significant.sum()

    @property
    def n_genes(self):
        return len(self.focal_data.gene_names)

    @property
    def counts(self):
        return self.focal_data.counts

    @property
    def focal_xy(self):
        return self.focal_data.focal_xy

    def to_dict(self):
        """Return the raw model dict (backward compat)."""
        return dict(
            Q=self.Q, pvalues=self.pvalues, qvalues=self.qvalues,
            significant=self.significant, R2=self.R2, R2_total=self.R2_total,
            beta=self.beta, phi=self.phi, residuals=self.residuals,
            P=self.P, P_tilde=self.P_tilde,
            total_var=self.total_var, baseline_var=self.baseline_var,
            excess_var=self.excess_var, has_excess=self.has_excess,
            R2_adj=self.R2_adj, R2_total_adj=self.R2_total_adj,
            R2_total_legacy=self.R2_total_legacy,
            var_retained=self.var_retained, block_pvalues=self.block_pvalues,
        )

    def save(self, path):
        """Save results to CSV."""
        save_results(path, self.gene_names, self.to_dict(), self.type_names)


_RESULT_KEYS = [
    "Q", "pvalues", "qvalues", "significant",
    "R2", "R2_total", "beta", "phi", "residuals",
    "P", "P_tilde", "total_var", "baseline_var",
    "excess_var", "has_excess", "sample_ids",
    "R2_adj", "R2_total_adj", "R2_total_legacy", "var_retained",
    "block_pvalues", "neighborhood",
]


def _build_result(result, focal, tissue):
    """Wrap a run_model dict, dropping focal cells the model excluded."""
    cell_mask = result.get("cell_mask")
    if cell_mask is not None and not np.all(cell_mask):
        focal = FocalData(counts=focal.counts[cell_mask],
                          focal_xy=focal.focal_xy[cell_mask],
                          gene_names=focal.gene_names)
    return AuraResult(
        focal_data=focal,
        tissue_data=tissue,
        type_names=tissue.type_names,
        **{key: result.get(key) for key in _RESULT_KEYS},
    )


def _resolve_center_genes(model_kwargs, focal):
    """Allow center_genes to be given as gene names (after gene filtering)."""
    cg = model_kwargs.get("center_genes")
    if cg is not None and len(cg) and isinstance(next(iter(cg)), str):
        names = set(cg)
        mask = np.array([g in names for g in focal.gene_names])
        log.info("Centering on %d of %d reference genes present in panel",
                 mask.sum(), len(names))
        model_kwargs = dict(model_kwargs, center_genes=mask)
    return model_kwargs


def run_aura(tissue, focal_label, label_column="cell_type_nebula",
             k=30, n_perm=5000, alpha=0.05,
             min_gene_mean=0.1, max_cells=None, seed=42,
             out_path=None, **model_kwargs):
    """Run AURA on a single focal type from loaded tissue data.

    Args:
        tissue: TissueData from load_adata, or path to h5ad file
        focal_label: cell type label(s) to select as focal
        label_column: .obs column to match focal_label against
        k: neighborhood size for kNN composition
        n_perm: number of permutations
        alpha: FDR threshold
        min_gene_mean: minimum gene mean for inclusion
        max_cells: subsample focal cells if exceeding this
        seed: random seed
        out_path: optional path to save results CSV
        **model_kwargs: passed to `run_model` (radius, rings, min_neighbors,
            dispersion, center, center_genes)

    Returns:
        AuraResult
    """
    if isinstance(tissue, str):
        tissue = load_adata(tissue)

    focal = extract_focal(tissue, focal_label, label_column=label_column,
                          min_gene_mean=min_gene_mean, max_cells=max_cells,
                          seed=seed)
    model_kwargs = _resolve_center_genes(model_kwargs, focal)

    result = run_model(
        counts=focal.counts,
        focal_xy=focal.focal_xy,
        all_xy=tissue.all_xy,
        all_types=tissue.all_types,
        k=k, n_perm=n_perm, alpha=alpha, seed=seed,
        **model_kwargs,
    )

    aura_result = _build_result(result, focal, tissue)

    if out_path:
        aura_result.save(out_path)

    return aura_result


def run_aura_multisample(tissue, focal_label, sample_column,
                         label_column="cell_type_nebula",
                         k=30, n_perm=5000, alpha=0.05,
                         min_gene_mean=0.1, max_cells=None, seed=42,
                         null_counts=None, out_path=None,
                         dispersion_mode='pooled', **model_kwargs):
    """Run AURA with within-core kNN and within-sample permutation.

    For multi-sample spatial data (TMA, multi-section). Each cell's composition
    is computed from k nearest neighbors within its own sample/core.

    Args:
        tissue: TissueData from load_adata, or path to h5ad file
        focal_label: cell type label(s) to select as focal
        sample_column: .obs column containing sample/core identifiers
        label_column: .obs column to match focal_label against
        k: neighborhood size for within-core kNN
        n_perm: number of permutations
        alpha: FDR threshold
        min_gene_mean: minimum gene mean for inclusion
        max_cells: subsample focal cells if exceeding this
        seed: random seed
        null_counts: optional (n_ctrl, n_genes) control counts for null params
        out_path: optional path to save results CSV
        dispersion_mode: see `run_model_multisample`
        **model_kwargs: passed to `run_model_multisample` (dispersion,
            radius, rings, min_neighbors, center, center_genes,
            center_samples)

    Returns:
        AuraResult
    """
    if isinstance(tissue, str):
        tissue = load_adata(tissue)

    focal = extract_focal(tissue, focal_label, label_column=label_column,
                          min_gene_mean=min_gene_mean, max_cells=max_cells,
                          seed=seed)
    model_kwargs = _resolve_center_genes(model_kwargs, focal)

    # Build sample ID arrays
    adata = tissue.adata
    if isinstance(focal_label, str):
        focal_label = [focal_label]
    focal_mask = adata.obs[label_column].isin(focal_label).values

    # Re-derive the focal mask accounting for subsampling
    # Use the focal_xy coordinates to match back
    sample_names = sorted(adata.obs[sample_column].unique())
    sample_to_int = {s: i for i, s in enumerate(sample_names)}
    all_sample_ids = np.array([sample_to_int[s] for s in adata.obs[sample_column]])

    # For the subsampled focal cells, match by coordinate to get sample IDs
    # Since extract_focal may have subsampled, we need the mask it used
    # Simpler: re-extract with the same params to get consistent indexing
    import scipy.sparse as sp
    mask = adata.obs[label_column].isin(focal_label).values
    if max_cells and mask.sum() > max_cells:
        rng = np.random.default_rng(seed)
        indices = np.where(mask)[0]
        chosen = rng.choice(indices, max_cells, replace=False)
        mask = np.zeros(len(mask), dtype=bool)
        mask[chosen] = True

    focal_sample_ids = all_sample_ids[mask]

    result = run_model_multisample(
        counts=focal.counts,
        focal_xy=focal.focal_xy,
        all_xy=tissue.all_xy,
        all_types=tissue.all_types,
        focal_sample_ids=focal_sample_ids,
        all_sample_ids=all_sample_ids,
        k=k, n_perm=n_perm, alpha=alpha, seed=seed,
        null_counts=null_counts,
        dispersion_mode=dispersion_mode,
        **model_kwargs,
    )

    aura_result = _build_result(result, focal, tissue)

    if out_path:
        aura_result.save(out_path)

    return aura_result
