# Changelog

## 0.2.0 (2026-10-08)

### Changed
- `R2_total` is now the share of each gene's count variance captured by the
  composition fit, `||H r~_g||² / ||r_g||² = R2 × var_retained` (see
  `aura.model.variance_decomposition` for the derivation). The v0.1 quantity
  `R2 × excess_var / total_var` is still reported as `R2_total_legacy`.
- Package defaults now match the documentation and manuscript: `k=30`,
  `n_perm=5000` (previously `k=15`, `n_perm=1000` in the functions).
- Packaging: `pip install` from the repository root now works (setuptools
  previously found both `aura` and `tutorials` as top-level packages).

### Added
- `R2_adj`, `R2_total_adj`: R² corrected for the chance-level fit d/(n−1)
  under the permutation null; `R2_null`, `var_retained`, `rank_P`.
- Physical-scale neighborhoods: `radius=` and concentric `rings=` (per-ring
  p-values in `block_pvalues`, ring-specific β columns).
- `dispersion='gene'`: regularized gene-specific NB dispersion.
- `center=` ('mean', 'median', 'trimmed') and `center_genes=` for per-cell
  centering on a reference gene set.
- `center_samples=` in `run_model_multisample`.
- `driver='contribution'` option for `spillover_filter`,
  `spillover_filter_multi`, `variance_summary` and `gene_zoom_panel`
  (`aura.spillover.driver_axes`): driver = the neighbor type with the
  largest positive contribution β_k·sd(P_k). Recommended when some neighbor
  types are rare or one type dominates. The default remains argmax|β_k|
  (manuscript rule); output reports both drivers. `save_results` stores
  per-axis composition SDs (`sdP_*`) so the option also works on CSVs.
- `spatial_trend=` (length scale): residualize residuals and composition on
  a tensor-product cubic B-spline basis (`spatial_trend_basis`) before
  testing, within each sample in multi-sample mode, so tissue-scale
  gradients unrelated to local composition are not called. Effective only
  for gradients well above the knot spacing; warns when the basis is large
  relative to the number of focal cells.
- `LICENSE` file (MIT).

### Unchanged
- With default arguments, p-values, Q, β, R², significance calls and
  spillover calls (driver = argmax|β|) are
  bitwise identical to v0.1.0.
