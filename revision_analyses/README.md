# AURA revision analyses (Genome Biology, 2026)

Handoff notes, decisions and the list of remaining tasks are in
`Ongoing/Spatial_variance_component/Revision/RESUME_HERE.md` on Google Drive.

Scripts that produce the new analyses requested in review. Each script is
resumable (finished outputs are skipped) and accepts `--quick` for a smoke
test and `--n-perm N` to override permutations.

| Script | Reviewer item | What it answers |
|---|---|---|
| `01_recompute_r2.py` | R2.1, R2.3, S7 | Reproduces every published fit with v0.2 defaults (bitwise-identical p/β), adds redefined `R2_total` next to the legacy value, SVD percentages (uncentered as published vs centered) |
| `02_clustering_sim.py` | R1.M1 | Under-/over-clustering and neighbor-label noise: FPR, λ_GC, power, β accuracy; real-data granularity (coarse/default/fine) |
| `03_neighborhood_scale.py` | R1.M3 | k sweep, physical radii in cell spacings, rings (contact / short / long range) |
| `04_panel_size.py` | R1.M4, S8 | LN 5K panel subsampled to 150/343/600 genes (random, top-expressed, marker-enriched) × three centering choices |
| `05_ipf_downsample.py` | R1.m2 | IPF downsampled to healthy donor and cell counts; split-half noise ceilings for the IPF–healthy cosine |
| `06_dispersion.py` | R1.m1 | Shared vs gene-specific dispersion |
| `07_centering.py` | S8, S9 | IPF and NSCLC with mean / median / non-marker centering and `center_samples` |

## Setup on the analysis machine

```bash
# 1. get the code (either from GitHub, once the branch is pushed ...)
git clone -b revision-gb https://github.com/ChenMengjie/AURA.git
# ... or from the git bundle synced through Google Drive
git clone -b revision-gb "<Drive>/Ongoing/Spatial_variance_component/Revision/aura_revision-gb.bundle" AURA

# 2. environment
conda create -y -n aura -c conda-forge python=3.11 numpy scipy pandas scanpy matplotlib pytest statsmodels
conda activate aura
cd AURA && pip install -e . && pytest -q          # expect 50 passed

# 3. paths
export AURA_DATA=/Users/mchen12/SpatialHeterogeneity/data
export AURA_REV_OUT="<Drive>/Ongoing/Spatial_variance_component/Revision/outputs"

# 4. smoke test, then the full run
cd revision_analyses
bash run_all.sh --quick        # minutes; checks labels/paths
rm -rf "$AURA_REV_OUT"/0*      # discard quick outputs
bash run_all.sh                # full run
```

Writing outputs into the Google Drive folder makes them available for
assembling figures and the response letter without copying.

## Check before the full run

1. **Lymph-node focal labels** (`common.FOCAL['lymphnode']`): only `Tfh` and
   `Macrophage` are confirmed from the tutorial. Compare with
   `adata.obs['cell_type_nebula'].unique()` and edit.
2. **NSCLC focal label** `'T CD4 memory'` is a guess; check
   `adata.obs['cell_type'].unique()`.
3. **IPF composition groups**: the tutorial maps unmapped fine types to
   `'Other'` and keeps them (10 groups), whereas the manuscript reports 9.
   `01_recompute_r2.py` logs significant counts against the published ones
   (Tfh 122, AT2 92); if AT2 does not reproduce, drop `'Other'` in
   `common.load_dataset` and rerun.
4. **IPF healthy definition**: the log reports agreement between
   `disease_status != 'Disease'` and the VUHD/THD patient-ID prefix used in
   the manuscript. It should be 1.0.

## Not covered here (need the original figure code)

* Fig. 3g–h label "Thf" → "Tfh"; Fig. S8c title "ρ=ρ=0.17" → "ρ=0.17".
* R2.2: false-positive rate at moderate confounding from the Fig. 2e
  simulation (the text currently quotes only the extreme ρ = 0.61 setting).

## Expected cost (rough, single machine)

01 and 07 refit every published focal type at 5,000 permutations (similar
to the original runs). 02, 04 and 05 are the largest (≈150–300 fits each at
2,000 permutations); they write partial results as they go and resume if
interrupted.
