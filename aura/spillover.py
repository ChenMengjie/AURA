"""
AURA spillover detection.

Flags significant genes whose AURA driver axis matches a lineage that
canonically expresses the gene — a hallmark of transcript misassignment
from neighboring cells rather than genuine composition-dependent regulation.

Provides:
  1. learn_canonical: derive canonical markers from the data
  2. query_gene: check which lineage a gene belongs to
  3. spillover_filter: flag suspect genes from AURA results
  4. spillover_distance_test: validate individual genes with distance-decay analysis
  5. DEFAULT_CANONICAL: default canonical marker sets (editable)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

__all__ = [
    "driver_axes",
    "learn_canonical",
    "query_gene",
    "spillover_filter",
    "spillover_filter_multi",
    "spillover_distance_test",
    "spillover_check",
    "DEFAULT_CANONICAL",
    "CANONICAL_IPF",
    "CANONICAL_LYMPHNODE",
]

# ─────────────────────────────────────────────────────────────────────
# Canonical marker presets
# ─────────────────────────────────────────────────────────────────────
# These presets map composition-group labels (the 'beta_<group>' axis
# names AURA produces) to sets of canonical marker gene symbols.
# Users should edit or supply their own dict for their tissue/panel.

# Generic mixed preset — kept for backward compat. Includes both
# composition-group keys (Alveolar_epi, Macrophage, …) and lymph-node
# fine-type keys (GC_B, FDC, Tfh). Prefer the tissue-specific presets
# below.
DEFAULT_CANONICAL = {
    "Alveolar_epi": {"SFTPC", "SFTPD", "NAPSA", "PGC", "LAMP3", "AGER",
                     "SFTPA1", "SFTPA2", "SFTPB"},
    "Airway_epi":   {"SCGB1A1", "SCGB3A2", "MUC5B", "MUC5AC", "FOXJ1"},
    "Fibroblast":   {"COL1A1", "COL3A1", "COL1A2", "DCN", "LUM",
                     "POSTN", "FN1"},
    "Macrophage":   {"CD68", "LYZ", "CCL18", "MS4A7", "MARCO", "AIF1"},
    "Endothelial":  {"PECAM1", "VWF", "PLVAP", "ACKR1", "CDH5"},
    "T_cell":       {"CD3D", "CD3E", "CD4", "CD8A", "CD8B", "PTPRC"},
    "B_Plasma":     {"JCHAIN", "IGHG1", "CD79A", "MS4A1", "CD19"},
    "SMC_Peri":     {"ACTA2", "TAGLN", "MYH11", "RGS5"},
    "Granulocyte":  {"S100A8", "S100A9", "FCER1G"},
    "GC_B":         {"AICDA", "BCL6", "CR2"},
    "FDC":          {"CXCL13", "CR2", "FDCSP"},
    "Tfh":          {"CXCL13", "PDCD1", "ICOS", "TOX2"},
}

# IPF lung preset — matches the 9 composition groups used by the
# IPF Xenium pipeline (Vannan et al. 2025; ipf_xenium.h5ad).
# Tested against `code/build_clean_findings.py` to reproduce the
# 493-row / 428-clean spillover-filtered findings table that the IPF
# panel 1 / panel 2 figures depend on.
CANONICAL_IPF = {
    "Alveolar_epi": {"SFTPC", "SFTPD", "NAPSA", "PGC", "LAMP3", "AGER"},
    "Airway_epi":   {"SCGB1A1", "SCGB3A2", "MUC5B", "MUC5AC", "FOXJ1"},
    "Fibroblast":   {"COL1A1", "COL3A1", "COL1A2", "DCN", "LUM",
                     "POSTN", "FN1", "ACTA2"},
    "Macrophage":   {"CD68", "LYZ", "CCL18", "MS4A7", "MARCO"},
    "Endothelial":  {"PECAM1", "VWF", "PLVAP", "ACKR1"},
    "T_cell":       {"CD3D", "CD3E", "CD4", "CD8A", "PTPRC"},
    "B_Plasma":     {"JCHAIN", "IGHG1", "CD79A", "MS4A1"},
    "SMC_Peri":     {"ACTA2", "TAGLN", "MYH11"},
    "Granulocyte":  {"S100A8", "S100A9", "FCER1G"},
}

# Lymph node preset — composition groups used by the lymph node
# AURA pipeline (10x Xenium 5K, NEBULA-processed).
CANONICAL_LYMPHNODE = {
    "B":        {"MS4A1", "CD79A", "CD19", "BANK1"},
    "GC_B":     {"AICDA", "BCL6", "CR2", "STMN1"},
    "T_conv":   {"CD3D", "CD3E", "CD4", "PTPRC"},
    "Tfh":      {"PDCD1", "ICOS", "TOX2", "CXCL13"},
    "Treg":     {"FOXP3", "IL2RA", "CTLA4"},
    "Mac":      {"CD68", "LYZ", "CD163", "C1QA"},
    "FDC":      {"CXCL13", "FDCSP", "CR2"},
    "Endo":     {"PECAM1", "VWF", "PLVAP", "CDH5"},
    "Stromal":  {"COL1A1", "COL3A1", "DCN", "LUM"},
}



def driver_axes(beta, axis_sd=None, method="abs_beta"):
    """Dominant neighbor axis for each gene.

    Args:
        beta: (n_genes, K) composition coefficients
        axis_sd: (K,) standard deviation of each composition axis across the
            focal cells; required for method='contribution'
        method:
            'abs_beta' (default): argmax |beta_k|, the rule used in the
                AURA manuscript.
            'contribution': the axis with the largest positive
                contribution beta_k * sd(P_k), i.e. the neighbor type whose
                observed range of local fraction raises expression most;
                genes with no positive contribution take the largest
                |beta_k * sd(P_k)|.
                Recommended when some neighbor types are rare (their
                fractions barely vary, so their coefficients are large and
                noisy and dominate |beta|) or when one type dominates the
                tissue (with compositions summing to one, |beta| can pick a
                negative coefficient on that type).

    Returns:
        idx: (n_genes,) driver axis index
        contribution: (n_genes,) beta * sd at the driver (beta for abs_beta)
    """
    beta = np.atleast_2d(np.asarray(beta, dtype=float))
    rows = np.arange(beta.shape[0])
    if method == "abs_beta":
        idx = np.abs(beta).argmax(axis=1)
        return idx, beta[rows, idx]
    if method != "contribution":
        raise ValueError(f"Unknown driver method: {method!r}")
    if axis_sd is None:
        raise ValueError("method='contribution' requires axis_sd")
    C = beta * np.asarray(axis_sd, dtype=float)[None, :]
    idx = np.where(C.max(axis=1) > 0, C.argmax(axis=1), np.abs(C).argmax(axis=1))
    return idx, C[rows, idx]

def learn_canonical(adata, type_column, gene_names=None,
                    min_fold=2.0, min_detection=0.05):
    """Derive canonical marker sets from the data.

    For each gene, computes mean expression and detection rate per cell type.
    A gene is a canonical marker of a type if:
      1. That type has the highest mean expression for the gene
      2. The fold-enrichment over the second-highest type exceeds min_fold
      3. The detection rate in that type exceeds min_detection

    Args:
        adata: AnnData object with .X and .obs[type_column]
        type_column: column in .obs defining composition groups
        gene_names: subset of genes to evaluate (default: all)
        min_fold: minimum fold-enrichment over 2nd-highest type
        min_detection: minimum detection rate in top type

    Returns:
        (canonical, stats_df) tuple where:
          canonical: dict {type_name -> set(gene_names)}
          stats_df: per-gene table with top_type, top_mean, second_type,
                    second_mean, fold_enrichment, top_detection, is_marker
    """
    import scipy.sparse as sp

    types = adata.obs[type_column].values
    unique_types = sorted(set(types))

    # Normalize var_names to a 1-D string array (handles AnnData Index,
    # plain list, or numpy array uniformly).
    all_genes = np.asarray(list(adata.var_names))
    if gene_names is not None:
        gene_mask = np.isin(all_genes, gene_names)
        all_genes = all_genes[gene_mask]
        X = adata.X[:, gene_mask]
    else:
        X = adata.X

    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    n_types = len(unique_types)
    n_genes = len(all_genes)

    # Per-type mean and detection rate
    means = np.zeros((n_types, n_genes))
    det_rates = np.zeros((n_types, n_genes))
    for i, t in enumerate(unique_types):
        mask = types == t
        X_t = X[mask]
        means[i] = X_t.mean(axis=0)
        det_rates[i] = (X_t > 0).mean(axis=0)

    # For each gene: top type + fold enrichment
    canonical = {t: set() for t in unique_types}
    stats_rows = []
    for g in range(n_genes):
        gene = all_genes[g]
        m = means[:, g]
        d = det_rates[:, g]
        order = np.argsort(m)[::-1]
        top_type = unique_types[order[0]]
        top_mean = m[order[0]]
        second_mean = m[order[1]] if n_types > 1 else 0
        fold = top_mean / max(second_mean, 1e-6)
        top_det = d[order[0]]
        is_marker = (fold >= min_fold and top_det >= min_detection)
        stats_rows.append({
            'gene': gene,
            'top_type': top_type,
            'top_mean': top_mean,
            'second_type': unique_types[order[1]] if n_types > 1 else '',
            'second_mean': second_mean,
            'fold_enrichment': fold,
            'top_detection': top_det,
            'is_marker': is_marker,
        })
        if is_marker:
            canonical[top_type].add(gene)

    stats_df = pd.DataFrame(stats_rows).sort_values(
        'fold_enrichment', ascending=False)

    n_markers = sum(len(v) for v in canonical.values())
    n_types_with = sum(1 for v in canonical.values() if len(v) > 0)
    log.info("Learned %d canonical markers across %d/%d types "
             "(min_fold=%.1f, min_det=%.0f%%)",
             n_markers, n_types_with, n_types, min_fold, min_detection * 100)

    return canonical, stats_df


def query_gene(gene_name, canonical=None, adata=None, type_column=None):
    """Check which lineage(s) a gene is a canonical marker of.

    Args:
        gene_name: gene to query (case-sensitive)
        canonical: dict from learn_canonical or DEFAULT_CANONICAL
        adata: optional — if provided with type_column, computes per-type stats
        type_column: .obs column for cell types (required if adata provided)

    Returns:
        dict with keys: marker_of (list of types), stats (per-type means if adata given)
    """
    import scipy.sparse as sp

    result = {'gene': gene_name, 'marker_of': [], 'stats': None}

    if canonical is None:
        canonical = DEFAULT_CANONICAL

    for group, markers in canonical.items():
        if isinstance(markers, set) and gene_name in markers:
            result['marker_of'].append(group)

    # If adata provided, compute per-type stats
    if adata is not None and type_column is not None:
        gene_idx = np.where(np.array(adata.var_names) == gene_name)[0]
        if len(gene_idx) > 0:
            gene_idx = gene_idx[0]
            types = adata.obs[type_column].values
            unique_types = sorted(set(types))
            X_col = adata.X[:, gene_idx]
            if sp.issparse(X_col):
                X_col = np.asarray(X_col.todense()).ravel()
            else:
                X_col = np.asarray(X_col).ravel()

            rows = []
            for t in unique_types:
                mask = types == t
                expr = X_col[mask]
                rows.append({
                    'type': t,
                    'mean': expr.mean(),
                    'detection': (expr > 0).mean(),
                    'n_cells': mask.sum(),
                })
            result['stats'] = pd.DataFrame(rows).sort_values('mean', ascending=False)

    # Print summary
    if result['marker_of']:
        print(f"{gene_name}: canonical marker of {', '.join(result['marker_of'])}")
    else:
        print(f"{gene_name}: not a canonical marker of any group")

    if result['stats'] is not None:
        print(f"\n  {'Type':20s} {'Mean':>8s} {'Det%':>6s} {'n_cells':>8s}")
        for _, r in result['stats'].head(10).iterrows():
            print(f"  {r['type']:20s} {r['mean']:8.3f} {r['detection']:5.1%} {r['n_cells']:>8,}")

    return result


def spillover_filter(result=None, csv_path=None, gene_names=None,
                     focal_lineage=None, canonical=None,
                     type_names=None, driver="abs_beta", axis_sd=None):
    """Flag significant genes as spillover-suspect or clean.

    A gene is spillover-suspect if:
      1. Its driver axis (see `driver_axes`) has the gene in its canonical
         marker set
      2. The gene is NOT a canonical marker of the focal cell's own lineage

    Args:
        result: AuraResult (preferred). If provided, extracts beta, significant,
                gene_names, and type_names automatically.
        csv_path: alternative — path to AURA results CSV.
        gene_names: required if csv_path is used and gene column differs.
        focal_lineage: string or list of strings — which composition group(s)
                       the focal cell type belongs to. Genes that are canonical
                       markers of the focal lineage are never flagged.
                       Example: "Macrophage" for SPP1+ Macrophages.
        canonical: dict mapping composition group names to sets of canonical
                   marker gene names. Defaults to DEFAULT_CANONICAL.
        type_names: list of composition axis names. Required if csv_path is used.
        driver: 'abs_beta' (default; manuscript rule) or 'contribution';
                see `driver_axes`.
        axis_sd: (K,) SD of each composition axis over the focal cells,
                 needed for driver='contribution'. Taken from result.P or
                 from the CSV's sdP_* columns when not given; if unavailable,
                 falls back to 'abs_beta' with a warning.

    Returns:
        DataFrame with columns:
            gene, significant, R2_total, driver_axis, beta_driver,
            contribution_driver, driver_axis_abs_beta, qvalue,
            spillover_suspect, reason
        Only includes significant genes. Sorted by R2_total descending.
    """
    if canonical is None:
        canonical = DEFAULT_CANONICAL

    # Parse input
    if result is not None:
        beta = result.beta
        significant = result.significant
        genes = result.gene_names
        tnames = result.type_names
        r2_total = result.R2_total
        qvalues = result.qvalues
        if axis_sd is None and getattr(result, "P", None) is not None:
            axis_sd = np.asarray(result.P).std(axis=0)
    elif csv_path is not None:
        df = pd.read_csv(csv_path)
        beta_cols = [c for c in df.columns if c.startswith('beta_')]
        tnames = type_names or [c.replace('beta_', '') for c in beta_cols]
        beta = df[beta_cols].values.astype(float)
        significant = df['significant'].values
        genes = df['gene'].values if gene_names is None else gene_names
        r2_total = df['R2_total'].values
        qvalues = df['qvalue'].values
        sd_cols = [f"sdP_{t}" for t in tnames]
        if axis_sd is None and all(c in df.columns for c in sd_cols):
            axis_sd = df[sd_cols].iloc[0].values.astype(float)
    else:
        raise ValueError("Provide either result or csv_path")

    tnames = list(tnames)
    if beta.shape[1] != len(tnames):
        n_blocks = beta.shape[1] // len(tnames)
        tnames = [f"{t}_ring{j}" for j in range(n_blocks) for t in tnames]
    if driver == "contribution" and axis_sd is None:
        log.warning("spillover_filter: composition SDs unavailable; using the "
                    "v0.1 abs_beta driver rule")
        driver = "abs_beta"
    drv_idx, drv_contrib = driver_axes(beta, axis_sd, method=driver)
    abs_idx, _ = driver_axes(beta, method="abs_beta")

    # Focal cell's own markers (never flagged)
    own = set()
    if focal_lineage is not None:
        if isinstance(focal_lineage, str):
            focal_lineage = [focal_lineage]
        for lin in focal_lineage:
            own |= canonical.get(lin, set())

    # Process significant genes
    sig_idx = np.where(significant)[0]
    rows = []
    for i in sig_idx:
        gene = genes[i]
        driver_idx = drv_idx[i]
        driver_name = tnames[driver_idx]
        lineage = driver_name.split("_ring")[0]
        beta_val = beta[i, driver_idx]

        suspect = False
        reason = ""
        if lineage in canonical and gene in canonical[lineage]:
            if gene not in own:
                suspect = True
                reason = f"{gene} is canonical marker of {lineage}"

        rows.append({
            'gene': gene,
            'significant': True,
            'R2_total': r2_total[i],
            'driver_axis': driver_name,
            'beta_driver': beta_val,
            'contribution_driver': drv_contrib[i],
            'driver_axis_abs_beta': tnames[abs_idx[i]],
            'qvalue': qvalues[i],
            'spillover_suspect': suspect,
            'reason': reason,
        })

    out = pd.DataFrame(rows).sort_values('R2_total', ascending=False)
    n_suspect = out['spillover_suspect'].sum()
    n_clean = len(out) - n_suspect

    log.info("Spillover filter: %d significant, %d suspect (%.0f%%), %d clean",
             len(out), n_suspect, n_suspect / max(len(out), 1) * 100, n_clean)

    return out


def spillover_filter_multi(focals, canonical=None, type_names=None,
                           driver="abs_beta"):
    """Run `spillover_filter` across multiple focal types and concatenate.

    Convenience wrapper for the common pattern in disease/multi-tissue
    pipelines (e.g. the IPF six-focal scan): run the filter on each focal
    type with its own `focal_lineage`, then merge into a single findings
    table with a `focal_type` column.

    Args:
        focals: list of (focal_name, source, focal_lineage) tuples where
            `focal_name` is a string label for the column,
            `source` is either an AuraResult or a path to an AURA results CSV,
            `focal_lineage` is a string or list of strings — the composition
            group(s) the focal cell type "owns" (its own canonical markers
            are never flagged as spillover).
        canonical: dict mapping composition group names to sets of canonical
            marker gene names. Defaults to DEFAULT_CANONICAL — supply
            CANONICAL_IPF, CANONICAL_LYMPHNODE, or your own.
        type_names: optional list of composition axis names. Required only
            if any source is a CSV path that doesn't follow the
            `beta_<group>` column convention.
        driver: driver rule passed to `spillover_filter`.

    Returns:
        Single DataFrame with all significant rows from every focal type
        concatenated, plus a `focal_type` column. Columns:
        gene, significant, R2_total, driver_axis, beta_driver, qvalue,
        spillover_suspect, reason, focal_type
    """
    if canonical is None:
        canonical = DEFAULT_CANONICAL

    parts = []
    for entry in focals:
        if len(entry) != 3:
            raise ValueError(
                f"focals entry must be (focal_name, source, focal_lineage); "
                f"got {entry!r}")
        focal_name, source, focal_lineage = entry
        if isinstance(source, str):
            df_focal = spillover_filter(
                csv_path=source, focal_lineage=focal_lineage,
                canonical=canonical, type_names=type_names, driver=driver)
        else:
            df_focal = spillover_filter(
                result=source, focal_lineage=focal_lineage, canonical=canonical,
                driver=driver)
        df_focal = df_focal.copy()
        df_focal['focal_type'] = focal_name
        parts.append(df_focal)

    if not parts:
        return pd.DataFrame(columns=['gene', 'significant', 'R2_total',
                                     'driver_axis', 'beta_driver',
                                     'contribution_driver',
                                     'driver_axis_abs_beta', 'qvalue',
                                     'spillover_suspect', 'reason',
                                     'focal_type'])

    out = pd.concat(parts, ignore_index=True)
    n_total = len(out)
    n_suspect = int(out['spillover_suspect'].sum())
    log.info("spillover_filter_multi: %d focal types, %d total significant, "
             "%d clean, %d spillover-suspect",
             len(focals), n_total, n_total - n_suspect, n_suspect)
    return out


def spillover_distance_test(gene_name, focal_xy, focal_counts, focal_gene_names,
                            focal_sample_ids, all_xy, all_types, all_sample_ids,
                            driver_type_idx, type_names=None,
                            distance_bins=None, control_gene=None,
                            verbose=False):
    """Test a specific gene for distance-decay spillover signature.

    For each focal cell, computes the distance to the nearest cell of the
    driver type (within the same sample). Then bins focal cells by distance
    and computes detection rate per bin. A sharp decay with distance is the
    hallmark of spillover.

    Args:
        gene_name: gene to test
        focal_xy: (n_c, 2) focal cell coordinates
        focal_counts: (n_c, n_genes) count matrix
        focal_gene_names: (n_genes,) gene name array
        focal_sample_ids: (n_c,) sample labels for focal cells
        all_xy: (M, 2) all cell coordinates
        all_types: (M,) integer type labels
        all_sample_ids: (M,) sample labels for all cells
        driver_type_idx: integer type label of the driver cell type, OR
                         list of integers for a composition group
        type_names: optional names for reporting
        distance_bins: bin edges in µm. Default:
                       [0, 10, 20, 30, 50, 100, 200, 500, 1000]
        control_gene: gene name to use as the "stable" reference. If None,
                      defaults to the highest-detection gene in `focal_counts`
                      (a heuristic — usually a housekeeping gene).
        verbose: if True, print a human-readable summary of the result.

    Returns:
        dict with keys:
            gene, verdict ('spillover'/'possible'/'clean'/'inconclusive'),
            fold_decay, control_fold, control_gene, driver_name, n_valid,
            median_dist, bins_df
    """
    from scipy.spatial import cKDTree

    if distance_bins is None:
        distance_bins = [0, 10, 20, 30, 50, 100, 200, 500, 1000]

    if isinstance(driver_type_idx, int):
        driver_type_idx = [driver_type_idx]
    driver_mask_all = np.isin(all_types, driver_type_idx)

    gidx = np.where(focal_gene_names == gene_name)[0]
    if len(gidx) == 0:
        raise ValueError(f"Gene {gene_name} not found in focal_gene_names")
    gidx = gidx[0]
    expr = focal_counts[:, gidx]

    # Resolve control gene
    if control_gene is None:
        det_rates_all = (focal_counts > 0).mean(axis=0)
        control_idx = int(np.argmax(det_rates_all))
        control_gene = str(focal_gene_names[control_idx])
    else:
        c_idx = np.where(focal_gene_names == control_gene)[0]
        if len(c_idx) == 0:
            raise ValueError(
                f"control_gene {control_gene} not in focal_gene_names")
        control_idx = int(c_idx[0])
    control_expr = focal_counts[:, control_idx]

    # Per-cell distance to nearest driver-type cell, within sample
    unique_samples = np.unique(focal_sample_ids)
    dist_to_driver = np.full(len(focal_xy), np.inf)
    for s in unique_samples:
        f_idx = np.where(focal_sample_ids == s)[0]
        if len(f_idx) == 0:
            continue
        d_mask = driver_mask_all & (all_sample_ids == s)
        driver_xy = all_xy[d_mask]
        if len(driver_xy) == 0:
            continue
        tree = cKDTree(driver_xy)
        d, _ = tree.query(focal_xy[f_idx], k=1)
        dist_to_driver[f_idx] = d

    valid = dist_to_driver < np.inf

    # Bin
    rows = []
    for i in range(len(distance_bins) - 1):
        lo, hi = distance_bins[i], distance_bins[i + 1]
        mask = valid & (dist_to_driver >= lo) & (dist_to_driver < hi)
        n = int(mask.sum())
        if n < 20:
            continue
        rows.append({
            'dist_lo': lo, 'dist_hi': hi, 'n_cells': n,
            'detection_rate': float((expr[mask] > 0).mean()),
            'mean_expr': float(expr[mask].mean()),
            'control_detection': float((control_expr[mask] > 0).mean()),
        })
    bins_df = pd.DataFrame(rows)

    driver_name = (f"type {driver_type_idx}" if type_names is None
                   else ", ".join(type_names[i] for i in driver_type_idx
                                  if i < len(type_names)))
    out = {
        'gene': gene_name,
        'verdict': 'inconclusive',
        'fold_decay': float('nan'),
        'control_fold': float('nan'),
        'control_gene': control_gene,
        'driver_name': driver_name,
        'n_valid': int(valid.sum()),
        'median_dist': float(np.median(dist_to_driver[valid]))
                       if valid.any() else float('nan'),
        'bins_df': bins_df,
    }

    if len(bins_df) >= 2:
        near_det = bins_df.iloc[0]['detection_rate']
        far_det = bins_df.iloc[-1]['detection_rate']
        near_ctrl = bins_df.iloc[0]['control_detection']
        far_ctrl = bins_df.iloc[-1]['control_detection']
        fold_decay = near_det / max(far_det, 0.001)
        ctrl_fold = near_ctrl / max(far_ctrl, 0.001)
        out['fold_decay'] = float(fold_decay)
        out['control_fold'] = float(ctrl_fold)
        if fold_decay > 3 and ctrl_fold < 1.5:
            out['verdict'] = 'spillover'
        elif fold_decay > 2:
            out['verdict'] = 'possible'
        else:
            out['verdict'] = 'clean'

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Spillover Distance Test: {gene_name}")
        print(f"{'='*60}")
        print(f"  Driver type: {driver_name}")
        print(f"  Cells with driver neighbors: {out['n_valid']:,}/{len(valid):,}")
        if not np.isnan(out['median_dist']):
            print(f"  Median distance to nearest driver: "
                  f"{out['median_dist']:.0f} µm")
        print()
        if len(bins_df) >= 2:
            print(f"  {'Distance':>12s} {'n_cells':>8s} {'Det%':>6s} "
                  f"{'Mean':>6s} {'Ctrl_det%':>10s}")
            for _, r in bins_df.iterrows():
                print(f"  {r['dist_lo']:>5.0f}-{r['dist_hi']:<5.0f}µm "
                      f"{int(r['n_cells']):>8,} {r['detection_rate']:5.1%} "
                      f"{r['mean_expr']:5.2f} {r['control_detection']:9.1%}")
            print()
            print(f"  {gene_name} fold-decay (near/far): {out['fold_decay']:.1f}x")
            print(f"  Control ({control_gene}) fold-change: "
                  f"{out['control_fold']:.1f}x")
            print()
            verdict_msg = {
                'spillover': f"** LIKELY SPILLOVER: decays {out['fold_decay']:.0f}x"
                             f" while control is stable.",
                'possible':  f"⚠ POSSIBLE SPILLOVER: {out['fold_decay']:.1f}x decay.",
                'clean':     "✓ No strong distance-decay pattern.",
            }
            print(f"  {verdict_msg.get(out['verdict'], out['verdict'])}")
        else:
            print(f"  Not enough populated distance bins to compute verdict.")

    return out


def spillover_check(result, focal_lineage, canonical=None,
                    distance_test=False, tissue=None, sample_column=None,
                    type_to_int=None, distance_bins=None,
                    control_gene=None):
    """High-level spillover audit for one focal type.

    Always runs `spillover_filter` (the cheap canonical-marker check). If
    `distance_test=True`, additionally runs `spillover_distance_test` on
    every gene flagged as suspect and merges the verdict back in.

    Args:
        result: AuraResult for the focal type.
        focal_lineage: composition group(s) the focal cell type owns.
        canonical: marker dict (defaults to DEFAULT_CANONICAL).
        distance_test: if True, run distance-decay validation on every
            spillover-suspect gene.
        tissue: TissueData (required if distance_test=True). Provides
            all_xy, all_types, and the type-name → int mapping.
        sample_column: .obs column with sample/core IDs (required if
            distance_test=True). For single-tissue runs (e.g. lymph node)
            pass any column with a constant value.
        type_to_int: optional override for the lineage→int mapping; by
            default uses tissue.type_to_int.
        distance_bins, control_gene: forwarded to spillover_distance_test.

    Returns:
        DataFrame from spillover_filter, with extra columns when
        distance_test=True: distance_verdict, distance_fold_decay,
        distance_control_fold.
    """
    if canonical is None:
        canonical = DEFAULT_CANONICAL

    df = spillover_filter(result=result, focal_lineage=focal_lineage,
                          canonical=canonical)
    if not distance_test:
        return df

    if tissue is None or sample_column is None:
        raise ValueError(
            "distance_test=True requires tissue and sample_column")
    if type_to_int is None:
        type_to_int = tissue.type_to_int

    # Per-cell sample IDs for the focal cells (need to align with result.P)
    adata = tissue.adata
    sample_to_int = {s: i for i, s in enumerate(sorted(adata.obs[sample_column].unique()))}
    all_sample_ids = np.array([sample_to_int[s] for s in adata.obs[sample_column]])
    # focal_sample_ids: best-effort by spatial match
    focal_xy = result.focal_xy
    focal_counts = result.counts
    focal_gene_names = np.asarray(result.gene_names)
    # Match focal_xy back to all_xy by exact coordinate equality
    from scipy.spatial import cKDTree
    tree = cKDTree(tissue.all_xy)
    _, idx = tree.query(focal_xy, k=1)
    focal_sample_ids = all_sample_ids[idx]

    suspects = df[df['spillover_suspect']]
    verdicts = {}
    for _, row in suspects.iterrows():
        gene = row['gene']
        driver = row['driver_axis']
        if driver not in type_to_int:
            continue
        driver_idx = type_to_int[driver]
        try:
            r = spillover_distance_test(
                gene_name=gene,
                focal_xy=focal_xy,
                focal_counts=focal_counts,
                focal_gene_names=focal_gene_names,
                focal_sample_ids=focal_sample_ids,
                all_xy=tissue.all_xy,
                all_types=tissue.all_types,
                all_sample_ids=all_sample_ids,
                driver_type_idx=driver_idx,
                type_names=tissue.type_names,
                distance_bins=distance_bins,
                control_gene=control_gene,
                verbose=False,
            )
            verdicts[gene] = r
        except Exception as e:
            log.warning("distance test failed for %s: %s", gene, e)

    df = df.copy()
    df['distance_verdict'] = df['gene'].map(
        lambda g: verdicts[g]['verdict'] if g in verdicts else '')
    df['distance_fold_decay'] = df['gene'].map(
        lambda g: verdicts[g]['fold_decay'] if g in verdicts else float('nan'))
    df['distance_control_fold'] = df['gene'].map(
        lambda g: verdicts[g]['control_fold'] if g in verdicts else float('nan'))
    return df
