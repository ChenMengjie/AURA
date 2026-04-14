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
        )

    def save(self, path):
        """Save results to CSV."""
        save_results(path, self.gene_names, self.to_dict(), self.type_names)


def run_aura(tissue, focal_label, label_column="cell_type_nebula",
             k=15, n_perm=1000, alpha=0.05,
             min_gene_mean=0.1, max_cells=None, seed=42,
             out_path=None):
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

    Returns:
        AuraResult
    """
    if isinstance(tissue, str):
        tissue = load_adata(tissue)

    focal = extract_focal(tissue, focal_label, label_column=label_column,
                          min_gene_mean=min_gene_mean, max_cells=max_cells,
                          seed=seed)

    result = run_model(
        counts=focal.counts,
        focal_xy=focal.focal_xy,
        all_xy=tissue.all_xy,
        all_types=tissue.all_types,
        k=k, n_perm=n_perm, alpha=alpha, seed=seed,
    )

    aura_result = AuraResult(
        focal_data=focal,
        tissue_data=tissue,
        type_names=tissue.type_names,
        **{key: result[key] for key in [
            "Q", "pvalues", "qvalues", "significant",
            "R2", "R2_total", "beta", "phi", "residuals",
            "P", "P_tilde", "total_var", "baseline_var",
            "excess_var", "has_excess",
        ]},
        sample_ids=result.get("sample_ids"),
    )

    if out_path:
        aura_result.save(out_path)

    return aura_result


def run_aura_multisample(tissue, focal_label, sample_column,
                         label_column="cell_type_nebula",
                         k=15, n_perm=1000, alpha=0.05,
                         min_gene_mean=0.1, max_cells=None, seed=42,
                         null_counts=None, out_path=None,
                         dispersion_mode='pooled'):
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

    Returns:
        AuraResult
    """
    if isinstance(tissue, str):
        tissue = load_adata(tissue)

    focal = extract_focal(tissue, focal_label, label_column=label_column,
                          min_gene_mean=min_gene_mean, max_cells=max_cells,
                          seed=seed)

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
    )

    aura_result = AuraResult(
        focal_data=focal,
        tissue_data=tissue,
        type_names=tissue.type_names,
        **{key: result[key] for key in [
            "Q", "pvalues", "qvalues", "significant",
            "R2", "R2_total", "beta", "phi", "residuals",
            "P", "P_tilde", "total_var", "baseline_var",
            "excess_var", "has_excess",
        ]},
        sample_ids=result.get("sample_ids"),
    )

    if out_path:
        aura_result.save(out_path)

    return aura_result
