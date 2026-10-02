# Changelog

## 0.2.0 (unreleased)

### Changed
- `R2_total` is now the share of each gene's count variance captured by the
  composition fit, `||H r~_g||² / ||r_g||² = R2 × var_retained` (see
  `aura.model.variance_decomposition` for the derivation). The v0.1 quantity
  `R2 × excess_var / total_var` is still reported as `R2_total_legacy`.
- Package defaults now match the documentation and manuscript: `k=30`,
  `n_perm=5000` (previously `k=15`, `n_perm=1000` in the functions).
- Packaging: `pip install` from the repository root now works (setuptools
  previously found both `aura` and `tutorials` as top-level packages).

### Changed (spillover filter)
- Driver axis is now the neighbor type with the largest positive
  contribution β_k·sd(P_k) (`driver_axes(..., method='contribution')`).
  The v0.1 rule argmax|β_k| is biased toward rare neighbor types, whose
  local fractions barely vary and whose coefficients are large and noisy,
  and can select a negative coefficient on the dominant type. The v0.1 rule
  remains available as `driver='abs_beta'` and is reported in
  `driver_axis_abs_beta`. The same rule is used by `variance_summary` and
  `gene_zoom_panel`. `save_results` stores per-axis composition SDs
  (`sdP_*`) so the rule also applies to saved CSVs.

### Added
- `R2_adj`, `R2_total_adj`: R² corrected for the chance-level fit d/(n−1)
  under the permutation null; `R2_null`, `var_retained`, `rank_P`.
- Physical-scale neighborhoods: `radius=` and concentric `rings=` (per-ring
  p-values in `block_pvalues`, ring-specific β columns).
- `dispersion='gene'`: regularized gene-specific NB dispersion.
- `center=` ('mean', 'median', 'trimmed') and `center_genes=` for per-cell
  centering on a reference gene set.
- `center_samples=` in `run_model_multisample`.
- `LICENSE` file (MIT).

### Unchanged
- With default arguments, p-values, Q, β, R² and significance calls are
  bitwise identical to v0.1.0.
