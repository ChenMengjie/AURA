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
