"""
AURA I/O: load spatial transcriptomics data, extract focal cells, save/load results.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class TissueData:
    """Tissue-wide data extracted from an h5ad file."""
    adata: object  # AnnData
    all_xy: np.ndarray          # (M, 2)
    all_types: np.ndarray       # (M,) int
    type_names: list             # sorted unique type labels
    type_to_int: dict            # str -> int mapping


@dataclass
class FocalData:
    """Focal cell subset ready for AURA."""
    counts: np.ndarray          # (n_c, n_genes) dense
    focal_xy: np.ndarray        # (n_c, 2)
    gene_names: np.ndarray      # (n_genes,)


def load_adata(path, xy_columns=("x_centroid", "y_centroid"),
               type_column="lineage_aura"):
    """Load h5ad and extract tissue-wide coordinates and cell types.

    Args:
        path: path to .h5ad file
        xy_columns: tuple of column names for spatial coordinates in .obs
        type_column: column name for cell type labels in .obs

    Returns:
        TissueData
    """
    import scanpy as sc
    adata = sc.read_h5ad(path)

    all_xy = adata.obs[list(xy_columns)].values.astype(np.float64)

    type_names = sorted(adata.obs[type_column].unique())
    type_to_int = {t: i for i, t in enumerate(type_names)}
    all_types = np.array([type_to_int[t] for t in adata.obs[type_column]])

    log.info("Loaded %d cells, %d genes, %d types from %s",
             adata.n_obs, adata.n_vars, len(type_names), path)

    return TissueData(
        adata=adata,
        all_xy=all_xy,
        all_types=all_types,
        type_names=type_names,
        type_to_int=type_to_int,
    )


def extract_focal(tissue: TissueData, focal_label,
                  label_column="cell_type_nebula",
                  min_gene_mean=0.1, max_cells=None, seed=42):
    """Extract focal cell counts and coordinates.

    Args:
        tissue: TissueData from load_adata
        focal_label: string or list of strings — cell type labels to select
        label_column: column in .obs to match focal_label against
        min_gene_mean: filter genes below this mean in focal cells
        max_cells: subsample if more focal cells than this
        seed: random seed for subsampling

    Returns:
        FocalData
    """
    adata = tissue.adata
    if isinstance(focal_label, str):
        focal_label = [focal_label]

    mask = adata.obs[label_column].isin(focal_label).values
    n_total = mask.sum()
    log.info("Focal cells (%s): %d", ", ".join(focal_label), n_total)

    if max_cells and n_total > max_cells:
        rng = np.random.default_rng(seed)
        indices = np.where(mask)[0]
        chosen = rng.choice(indices, max_cells, replace=False)
        mask = np.zeros(len(mask), dtype=bool)
        mask[chosen] = True
        log.info("Subsampled to %d", max_cells)

    focal_xy = tissue.all_xy[mask]

    import scipy.sparse as sp
    X = adata.X[mask]
    counts = np.asarray(X.todense()) if sp.issparse(X) else np.asarray(X)

    gene_means = counts.mean(axis=0)
    gene_mask = gene_means >= min_gene_mean
    counts = counts[:, gene_mask]
    gene_names = np.array(adata.var_names)[gene_mask]

    log.info("Genes (mean >= %.2f): %d", min_gene_mean, len(gene_names))

    return FocalData(counts=counts, focal_xy=focal_xy, gene_names=gene_names)


def save_results(path, gene_names, result, type_names):
    """Save AURA results to CSV in canonical schema.

    Args:
        path: output CSV path
        gene_names: (n_genes,) array
        result: dict from run_model / run_model_sample
        type_names: list of composition axis names
    """
    df = pd.DataFrame({
        "gene": gene_names,
        "significant": result["significant"],
        "has_excess": result["has_excess"],
        "R2_total": result["R2_total"],
        "R2_resid": result["R2"],
        "total_var": result["total_var"],
        "baseline_var": result["baseline_var"],
        "excess_var": result["excess_var"],
        "Q": result["Q"],
        "pvalue": result["pvalues"],
        "qvalue": result["qvalues"],
    })
    for key in ("R2_adj", "R2_total_adj", "R2_total_legacy", "var_retained"):
        if result.get(key) is not None:
            df[key] = result[key]
    block_p = result.get("block_pvalues")
    if block_p is not None:
        for j in range(block_p.shape[0]):
            df[f"pvalue_ring{j}"] = block_p[j]

    beta = result["beta"]
    n_blocks = beta.shape[1] // len(type_names)
    for j in range(n_blocks):
        suffix = "" if n_blocks == 1 else f"_ring{j}"
        for i, name in enumerate(type_names):
            df[f"beta_{name}{suffix}"] = beta[:, j * len(type_names) + i]

    df = df.sort_values("pvalue")
    df.to_csv(path, index=False)
    log.info("Saved results: %s (%d genes, %d significant)",
             path, len(df), df["significant"].sum())


def load_results(path):
    """Load AURA results CSV.

    Returns:
        DataFrame with canonical columns
    """
    return pd.read_csv(path)
