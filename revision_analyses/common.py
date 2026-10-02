"""
Shared configuration and loaders for the Genome Biology revision analyses.

Set AURA_DATA to the directory holding the three datasets (default below
matches the tutorials) and AURA_REV_OUT for outputs. Every script accepts
--quick for a fast smoke test (few permutations / replicates).
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import aura
from aura.io import TissueData, FocalData
from aura.model import run_model, run_model_multisample

DATA = Path(os.environ.get("AURA_DATA",
                           "/Users/mchen12/SpatialHeterogeneity/data"))
OUT = Path(os.environ.get("AURA_REV_OUT", "revision_outputs"))

PATHS = {
    "lymphnode": DATA / "lymphnode_nebula_counts.h5ad",
    "ipf": DATA / "IPF" / "ipf_xenium.h5ad",
    "nsclc": DATA / "CosMx_NSCLC" / "cosmx_nsclc_prepped.h5ad",
}

# Focal populations analysed in the manuscript.
# LN labels other than Tfh / Macrophage: check against
# adata.obs['cell_type_nebula'].unique() and edit if they differ.
FOCAL = {
    "lymphnode": ["Tfh", "Treg", "T_naive", "T_CD8", "GC_LZ", "GC_DZ",
                  "FDC", "Macrophage", "Endothelial"],
    "ipf": ["AT2", "SPP1+ Macrophages", "Basal", "Capillary",
            "Interstitial Macrophages", "Alveolar FBs"],
    "nsclc": ["macrophage", "fibroblast", "T CD4 memory", "endothelial"],
}
LABEL_COLUMN = {"lymphnode": "cell_type_nebula", "ipf": "final_CT",
                "nsclc": "cell_type"}
SAMPLE_COLUMN = {"lymphnode": None, "ipf": "sample", "nsclc": "sample"}

# IPF fine cell type -> 9 composition groups (from tutorial_ipf.ipynb)
IPF_LINEAGE_MAP = {
    'Alveolar FBs': 'Fibroblast', 'Activated Fibrotic FBs': 'Fibroblast',
    'Inflammatory FBs': 'Fibroblast', 'Adventitial FBs': 'Fibroblast',
    'Proliferating FBs': 'Fibroblast', 'Myofibroblasts': 'Fibroblast',
    'Subpleural FBs': 'Fibroblast',
    'Interstitial Macrophages': 'Macrophage', 'SPP1+ Macrophages': 'Macrophage',
    'Alveolar Macrophages': 'Macrophage', 'Macrophages - IFN-activated': 'Macrophage',
    'Monocytes/MDMs': 'Macrophage', 'Proliferating Myeloid': 'Macrophage',
    'cDCs': 'Macrophage', 'Migratory DCs': 'Macrophage', 'pDCs': 'Macrophage',
    'Langerhans cells': 'Macrophage',
    'Capillary': 'Endothelial', 'Venous': 'Endothelial', 'Arteriole': 'Endothelial',
    'Lymphatic': 'Endothelial',
    'CD4+ T-cells': 'T_cell', 'CD8+ T-cells': 'T_cell', 'Tregs': 'T_cell',
    'NK': 'T_cell', 'NK/NKT': 'T_cell',
    'Proliferating T-cells': 'T_cell', 'Proliferating NK/NKT': 'T_cell',
    'AT2': 'Alveolar_epi', 'AT1': 'Alveolar_epi', 'Transitional AT2': 'Alveolar_epi',
    'Proliferating AT2': 'Alveolar_epi',
    'Basal': 'Airway_epi', 'Multiciliated': 'Airway_epi', 'Goblet': 'Airway_epi',
    'Secretory': 'Airway_epi', 'KRT5-/KRT17+': 'Airway_epi', 'RASC': 'Airway_epi',
    'PNEC': 'Airway_epi', 'Proliferating Airway': 'Airway_epi',
    'SMCs/Pericytes': 'SMC_Peri', 'Mesothelial': 'SMC_Peri',
    'B cells': 'B_Plasma', 'Plasma': 'B_Plasma', 'Proliferating B cells': 'B_Plasma',
    'Neutrophils': 'Granulocyte', 'Mast': 'Granulocyte', 'Basophils': 'Granulocyte',
}

log = logging.getLogger("revision")


# ─────────────────────────────────────────────────────────────────────
# CLI / bookkeeping
# ─────────────────────────────────────────────────────────────────────

def parse_args(description, extra=None, default_n_perm=5000):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: few permutations and replicates")
    ap.add_argument("--n-perm", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    if extra:
        extra(ap)
    args = ap.parse_args()
    if args.n_perm is None:
        args.n_perm = 200 if args.quick else default_n_perm
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s: %(message)s")
    logging.getLogger("aura").setLevel(logging.WARNING)
    return args


def outdir(name):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_meta(d, **meta):
    meta.update(aura_version=aura.__version__,
                time=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(d / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)


def done(path):
    """Resume support: skip outputs that already exist."""
    if Path(path).exists():
        log.info("exists, skipping: %s", path)
        return True
    return False


# ─────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────

def _tissue_from_adata(adata, type_column):
    type_names = sorted(adata.obs[type_column].astype(str).unique())
    type_to_int = {t: i for i, t in enumerate(type_names)}
    all_types = np.array([type_to_int[t] for t in adata.obs[type_column].astype(str)])
    all_xy = adata.obs[["x_centroid", "y_centroid"]].values.astype(np.float64)
    return TissueData(adata=adata, all_xy=all_xy, all_types=all_types,
                      type_names=type_names, type_to_int=type_to_int)


def load_dataset(name, condition="Disease"):
    """Return TissueData exactly as in the tutorials.

    For IPF, `condition` is 'Disease', 'Healthy' or 'All'. Healthy is
    everything that is not disease_status == 'Disease'; the manuscript
    defined healthy donors by patient ID prefix (VUHD/THD), which is
    checked and reported.
    """
    import scanpy as sc
    if name == "lymphnode":
        return aura.load_adata(str(PATHS[name]), type_column="lineage_aura")

    adata = sc.read_h5ad(PATHS[name])
    if name == "ipf":
        status = adata.obs["disease_status"].astype(str)
        if "patient" in adata.obs:
            pref = adata.obs["patient"].astype(str).str.match(r"^(VUHD|THD)")
            agree = ((status != "Disease") == pref).mean()
            log.info("IPF healthy definition agreement (status vs VUHD/THD "
                     "prefix): %.4f", agree)
        if condition == "Disease":
            adata = adata[status == "Disease"].copy()
        elif condition == "Healthy":
            adata = adata[status != "Disease"].copy()
        adata.obs["comp_group"] = (adata.obs["final_CT"].astype(str)
                                   .map(IPF_LINEAGE_MAP).fillna("Other"))
        return _tissue_from_adata(adata, "comp_group")
    if name == "nsclc":
        return _tissue_from_adata(adata, "lineage")
    raise ValueError(name)


def focal_data(tissue, dataset, focal_label, min_gene_mean=0.1,
               genes=None, max_cells=None, seed=42):
    """Focal counts/coords (+ integer sample ids for multi-sample data).

    `genes` restricts the panel before the min_gene_mean filter (used to
    emulate smaller panels).
    """
    import scipy.sparse as sp
    adata = tissue.adata
    mask = adata.obs[LABEL_COLUMN[dataset]].astype(str).isin(
        [focal_label] if isinstance(focal_label, str) else focal_label).values
    idx = np.where(mask)[0]
    if max_cells and len(idx) > max_cells:
        idx = np.sort(np.random.default_rng(seed).choice(idx, max_cells,
                                                         replace=False))
    gene_names = np.asarray(adata.var_names)
    gcols = np.arange(len(gene_names)) if genes is None else \
        np.where(np.isin(gene_names, list(genes)))[0]
    X = adata.X[idx][:, gcols]
    counts = np.asarray(X.todense()) if sp.issparse(X) else np.asarray(X)
    counts = counts.astype(np.float64)
    keep = counts.mean(axis=0) >= min_gene_mean
    out = dict(counts=counts[:, keep], xy=tissue.all_xy[idx],
               genes=gene_names[gcols][keep], idx=idx)
    scol = SAMPLE_COLUMN[dataset]
    if scol:
        names = sorted(adata.obs[scol].astype(str).unique())
        s2i = {s: i for i, s in enumerate(names)}
        all_s = np.array([s2i[s] for s in adata.obs[scol].astype(str)])
        out["all_sample_ids"] = all_s
        out["sample_ids"] = all_s[idx]
    return out


def fit(tissue, fd, n_perm, seed=42, all_types=None, **kw):
    """Run the single- or multi-sample model on a focal_data dict."""
    all_types = tissue.all_types if all_types is None else all_types
    if "sample_ids" in fd:
        return run_model_multisample(
            fd["counts"], fd["xy"], tissue.all_xy, all_types,
            fd["sample_ids"], fd["all_sample_ids"], n_perm=n_perm, seed=seed,
            **kw)
    return run_model(fd["counts"], fd["xy"], tissue.all_xy, all_types,
                     n_perm=n_perm, seed=seed, **kw)


def result_table(res, genes, type_names):
    """Per-gene table from a run_model dict."""
    kept = genes
    df = pd.DataFrame({
        "gene": kept, "pvalue": res["pvalues"], "qvalue": res["qvalues"],
        "significant": res["significant"], "has_excess": res["has_excess"],
        "Q": res["Q"], "R2": res["R2"], "R2_adj": res["R2_adj"],
        "R2_total": res["R2_total"], "R2_total_adj": res["R2_total_adj"],
        "R2_total_legacy": res["R2_total_legacy"],
        "var_retained": res["var_retained"],
    })
    beta = res["beta"]
    nb = beta.shape[1] // len(type_names)
    for j in range(nb):
        sfx = "" if nb == 1 else f"_ring{j}"
        for i, t in enumerate(type_names):
            df[f"beta_{t}{sfx}"] = beta[:, j * len(type_names) + i]
    if res.get("block_pvalues") is not None:
        for j, p in enumerate(res["block_pvalues"]):
            df[f"pvalue_ring{j}"] = p
    return df


def lambda_gc(p):
    from scipy.stats import chi2
    p = np.clip(np.asarray(p), 1e-300, 1)
    return float(np.median(chi2.isf(p, 1)) / chi2.ppf(0.5, 1))


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(len(a | b), 1)


def cosine_rows(A, B):
    num = (A * B).sum(axis=1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
    return np.where(den > 0, num / np.maximum(den, 1e-300), np.nan)


# ─────────────────────────────────────────────────────────────────────
# Spillover helpers
# ─────────────────────────────────────────────────────────────────────

TYPE_COLUMN = {"lymphnode": "lineage_aura", "ipf": "comp_group",
               "nsclc": "lineage"}


def canonical_markers(tissue, dataset, genes=None):
    """Data-driven canonical markers (fold >= 2, detection >= 5%)."""
    adata = tissue.adata if genes is None else \
        tissue.adata[:, np.isin(tissue.adata.var_names, list(genes))]
    canonical, _ = aura.learn_canonical(adata, type_column=TYPE_COLUMN[dataset],
                                        min_fold=2.0, min_detection=0.05)
    return canonical


def focal_lineage(tissue, fd):
    types = tissue.all_types[fd["idx"]]
    return tissue.type_names[np.bincount(types).argmax()]


def spillover_flags(df, canonical, focal_lin, type_names, P_sd=None,
                    own_markers=None, lineage_means=None, focal_means=None,
                    rel_fold=2.0):
    """Add driver and spillover columns (significant genes only).

    driver_axis_raw / spillover_suspect_raw: driver = argmax |beta_k|
        (the v0.1 rule used in the manuscript).
    driver_axis / spillover_suspect: with `P_sd` (per-axis SD of the focal
        cells' composition), driver = argmax beta_k * sd(P_k) over positive
        contributions, i.e. the neighbor lineage whose observed range of
        local fraction raises expression most. Raw |beta| favours rare axes,
        whose fractions barely vary and whose coefficients are therefore
        large and noisy, and can pick a negative coefficient on the dominant
        lineage. Without `P_sd` both definitions coincide.
    own_markers: genes exempt as the focal lineage's own markers (defaults
        to canonical[focal_lin]).
    spillover_suspect_rel (if lineage_means / focal_means are given): the
        driver lineage expresses the gene >= rel_fold times more than the
        focal cells themselves. Unlike the canonical-marker rule this also
        catches genes shared by several neighbor lineages (e.g. VIM).
    """
    bcols = [f"beta_{t}" for t in type_names]
    own = canonical.get(focal_lin, set()) if own_markers is None else own_markers
    names = np.array(type_names)

    def flag(driver):
        return np.array([
            bool(sig) and (g in canonical.get(d, set())) and (g not in own)
            for g, d, sig in zip(df["gene"], driver, df["significant"])
        ])

    from aura.spillover import driver_axes
    df = df.copy()
    raw = names[driver_axes(df[bcols].values, method="abs_beta")[0]]
    std = names[driver_axes(df[bcols].values, P_sd, "contribution")[0]] \
        if P_sd is not None else raw
    df["driver_axis_raw"], df["spillover_suspect_raw"] = raw, flag(raw)
    df["driver_axis"], df["spillover_suspect"] = std, flag(std)
    if lineage_means is not None and focal_means is not None:
        lm = lineage_means.reindex(df["gene"])
        drv = np.array([lm.at[g, d] if d in lm.columns else np.nan
                        for g, d in zip(df["gene"], std)], dtype=float)
        fm = focal_means.reindex(df["gene"]).values
        df["driver_over_focal"] = drv / np.maximum(fm, 1e-6)
        df["spillover_suspect_rel"] = df["significant"].values & \
            (df["driver_over_focal"].values >= rel_fold) & \
            ~df["gene"].isin(own).values
    return df


def non_marker_genes(genes, canonical, focal_lin):
    """Reference set for centering: genes that are not canonical markers of
    any non-focal lineage (the genes through which spillover enters)."""
    other = set().union(*[v for k, v in canonical.items() if k != focal_lin]) \
        if canonical else set()
    return np.array([g not in other for g in genes])


# ─────────────────────────────────────────────────────────────────────
# Simulation on a real spatial scaffold (same design as Fig. 2)
# ─────────────────────────────────────────────────────────────────────

def plant_effects(G, K, n_sig, scale, rng, n_axes=3):
    """Composition effects on n_sig genes, each over n_axes random axes with
    exponential magnitudes and random signs (as in the Fig. 2 power sims)."""
    B = np.zeros((G, K))
    sig = rng.choice(G, n_sig, replace=False)
    for g in sig:
        ax = rng.choice(K, min(n_axes, K), replace=False)
        B[g, ax] = rng.exponential(1.0, len(ax)) * rng.choice([-1, 1], len(ax))
    truth = np.zeros(G, dtype=bool)
    truth[sig] = True
    return B * scale, truth


def simulate_counts(P_tilde, mu, phi, B, rng, alpha_sd=0.15, log_shift=None):
    """NB counts with per-cell output effect alpha_i ~ N(0, alpha_sd) and
    composition effect P_tilde @ B.T; optional per-cell log-mean shifts
    (n, G) e.g. for unresolved subtypes."""
    n = P_tilde.shape[0]
    eta = P_tilde @ B.T + rng.normal(0, alpha_sd, (n, 1))
    if log_shift is not None:
        eta = eta + log_shift
    lam = mu[None, :] * np.exp(eta)
    return rng.negative_binomial(phi, phi / (phi + lam)).astype(np.float64)


def sim_metrics(res, truth, B_true=None, alpha=0.05):
    """Calibration and power of one simulated fit."""
    from aura.model import bh_fdr
    p = res["pvalues"]
    q, _ = bh_fdr(p, alpha=alpha)
    called = q <= alpha
    called_ex = res["significant"]           # after excess-variance filter
    null = ~truth
    out = dict(
        fpr_nominal=float(np.mean(p[null] < 0.05)),
        lambda_gc_null=lambda_gc(p[null]),
        n_null_fdr=int(called[null].sum()),
        n_null_fdr_excess=int(called_ex[null].sum()),
        tpr=float(called[truth].mean()) if truth.any() else np.nan,
        tpr_nominal=float(np.mean(p[truth] < 0.05)) if truth.any() else np.nan,
        tpr_excess=float(called_ex[truth].mean()) if truth.any() else np.nan,
        emp_fdr=float(called[null].sum() / max(called.sum(), 1)),
    )
    if B_true is not None and truth.any():
        hit = truth & called
        if hit.any():
            Bt = B_true[hit] - B_true[hit].mean(axis=1, keepdims=True)
            Be = res["beta"][hit] - res["beta"][hit].mean(axis=1, keepdims=True)
            out["beta_cosine_median"] = float(np.nanmedian(cosine_rows(Be, Bt)))
    return out
