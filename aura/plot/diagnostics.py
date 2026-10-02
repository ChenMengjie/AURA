"""
AURA diagnostic and variance decomposition plots.

1. diagnostic_panel: 8-panel figure checking model fit and null calibration
2. variance_panel: 6-panel 4-way variance decomposition
3. variance_summary: text-based summary
4. subtype_check: UMAP of focal cells colored by composition and transcriptomic PCs
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import kstest, spearmanr

from ._style import VARIANCE_COLORS as _COLORS



def _phi_label(phi):
    if np.ndim(phi) == 0:
        return f"{float(phi):.2f}"
    return f"gene, median {float(np.median(phi)):.2f}"

def diagnostic_panel(result, title="AURA Diagnostic", out_path=None):
    """Standard 8-panel diagnostic figure.

    Args:
        result: AuraResult or dict with model outputs + counts/gene_names
        title: figure title
        out_path: save path (optional)

    Returns:
        matplotlib Figure
    """
    # Unpack — accept AuraResult or raw dict
    if hasattr(result, 'counts'):
        counts = result.counts
        phi = result.phi
        residuals = result.residuals
        pvalues = result.pvalues
        qvalues = result.qvalues
        significant = result.significant
        R2_resid = result.R2
        R2_total = result.R2_total
        has_excess = result.has_excess
        total_var = result.total_var
        baseline_var = result.baseline_var
    else:
        raise ValueError("Pass an AuraResult object")

    # Raw residuals (before cell-level centering) for cell effect plot
    from ..model import compute_pearson_residuals
    residuals_raw = compute_pearson_residuals(counts, phi, center_cells=False)

    mu = counts.mean(axis=0)
    s2 = counts.var(axis=0, ddof=1)
    lib_size = counts.sum(axis=1)
    alpha_i = residuals_raw.mean(axis=1)
    n_cells, n_genes = counts.shape

    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.35)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # (a) Mean-variance fit
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(mu, s2, s=3, alpha=0.3, c='#4878CF')
    order = np.argsort(mu)
    order = order[mu[order] > 0]
    mu_s = mu[order]
    phi_s = phi if np.ndim(phi) == 0 else np.asarray(phi)[order]
    ax.plot(mu_s, mu_s + mu_s ** 2 / phi_s, 'r-', lw=2,
            label=f'NB(phi={_phi_label(phi)})')
    ax.plot(mu_s, mu_s, 'k--', lw=1, alpha=0.5, label='Poisson')
    ax.set_xlabel('Gene mean'); ax.set_ylabel('Gene variance')
    ax.set_title('(a) Mean-variance fit', fontsize=10)
    ax.legend(fontsize=7); ax.set_xscale('log'); ax.set_yscale('log')

    # (b) Per-gene residual variance
    ax = fig.add_subplot(gs[0, 1])
    residuals_centered = residuals  # already cell-level centered
    gvar = residuals_centered.var(axis=0)
    ax.hist(gvar, bins=50, color='#4878CF', alpha=0.7, edgecolor='k', linewidth=0.3)
    ax.axvline(1.0, color='red', ls='--', lw=2, label='Expected=1')
    ax.set_xlabel('Residual variance per gene'); ax.set_ylabel('Count')
    ax.set_title(f'(b) Residual variance (med={np.median(gvar):.2f})', fontsize=10)
    ax.legend(fontsize=7)

    # (c) Cell effect
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(alpha_i, bins=50, color='#4878CF', alpha=0.7, edgecolor='k', linewidth=0.3)
    ax.axvline(0, color='red', ls='--', lw=1)
    ax.set_xlabel('alpha_i (cell effect)'); ax.set_ylabel('Count')
    ax.set_title(f'(c) Cell effect (std={alpha_i.std():.3f})', fontsize=10)

    # (d) alpha_i vs library size
    ax = fig.add_subplot(gs[0, 3])
    ax.scatter(lib_size, alpha_i, s=2, alpha=0.2, c='#4878CF')
    rho_lib, _ = spearmanr(lib_size, alpha_i)
    ax.set_xlabel('Library size'); ax.set_ylabel('alpha_i')
    ax.set_title(f'(d) alpha_i vs lib size (rho={rho_lib:.2f})', fontsize=10)

    # (e) QQ all genes
    ax = fig.add_subplot(gs[1, 0])
    sorted_p = np.sort(pvalues)
    expected = np.linspace(0, 1, len(sorted_p) + 2)[1:-1]
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    sig_order = significant[np.argsort(pvalues)]
    colors_qq = np.where(sig_order, '#D65F5F', '#4878CF')
    ax.scatter(expected, sorted_p, s=5, alpha=0.5, c=colors_qq)
    ax.set_xlabel('Expected (uniform)'); ax.set_ylabel('Observed')
    n_sig = significant.sum()
    ax.set_title(f'(e) QQ all genes ({n_sig}/{n_genes} sig)', fontsize=10)
    ax.set_aspect('equal'); ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)

    # (f) QQ scrambled composition (calibration control)
    # Permute composition labels to break the real association, then re-test.
    # If the test is well-calibrated, scrambled p-values should be uniform.
    ax = fig.add_subplot(gs[1, 1])
    from ..model import composition_kernel, permutation_test
    rng_scr = np.random.default_rng(99)
    P_scr = result.P[rng_scr.permutation(len(result.P))]
    _, P_tilde_scr = composition_kernel(P_scr)
    _, pv_scr = permutation_test(residuals, P_tilde_scr, n_perm=1000, seed=42)
    sp_scr = np.sort(pv_scr)
    ep_scr = np.linspace(0, 1, len(sp_scr) + 2)[1:-1]
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.scatter(ep_scr, sp_scr, s=5, alpha=0.5, c='#70AD47')
    _, ks_scr = kstest(pv_scr, 'uniform')
    from scipy.stats import chi2
    lam_scr = np.median(chi2.ppf(1 - pv_scr, 1)) / 0.4549
    ax.set_xlabel('Expected'); ax.set_ylabel('Observed')
    ax.set_title(f'(f) QQ scrambled composition\n'
                 f'KS={ks_scr:.2f}, λ_GC={lam_scr:.2f}', fontsize=10)
    ax.set_aspect('equal'); ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)

    # (g) p-value histogram with pi0 estimate
    ax = fig.add_subplot(gs[1, 2])
    ax.hist(pvalues, bins=20, color='#4878CF', alpha=0.7, edgecolor='k', linewidth=0.3)
    ax.axhline(n_genes / 20, color='red', ls='--', lw=1, label='Uniform')
    ax.set_xlabel('p-value'); ax.set_ylabel('Count')
    ax.set_title('(g) p-value histogram', fontsize=10)
    ax.legend(fontsize=7)

    # (h) R2_resid vs R2_total
    ax = fig.add_subplot(gs[1, 3])
    ax.scatter(R2_resid[~significant], R2_total[~significant],
               s=5, alpha=0.3, c='#CCCCCC', label='NS')
    ax.scatter(R2_resid[significant], R2_total[significant],
               s=10, alpha=0.5, c='#D65F5F', label='Significant')
    ax.set_xlabel('R2_resid'); ax.set_ylabel('R2_total')
    ax.set_title('(h) R2_resid vs R2_total', fontsize=10)
    ax.legend(fontsize=7)

    # ================================================================
    # Row 3: π₀ estimation and effect-size stratified QQ
    # ================================================================
    from ..model import effect_size as _effect_size
    beta, _ = _effect_size(residuals, result.P_tilde)
    beta_norm = np.sqrt((beta ** 2).sum(axis=1))

    # Storey π₀ estimation
    lambdas_pi0 = np.arange(0.05, 0.95, 0.01)
    pi0_ests = np.array([(pvalues > l).sum() / (n_genes * (1 - l))
                         for l in lambdas_pi0])
    pi0_ests = np.minimum(pi0_ests, 1.0)
    # Use bootstrap-smoothed estimate: median above λ=0.5
    pi0 = np.median(pi0_ests[lambdas_pi0 >= 0.5])
    pi0 = min(pi0, 1.0)

    # (i) Storey π₀(λ) curve
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(lambdas_pi0, pi0_ests, '-', color='#4878CF', lw=1.5)
    ax.axhline(pi0, color='#D65F5F', ls='--', lw=1.5,
               label=f'π̂₀ = {pi0:.2f}')
    ax.set_xlabel('λ'); ax.set_ylabel('π̂₀(λ)')
    ax.set_title('(i) Storey π₀ estimate', fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, max(1.05, pi0_ests.max() * 1.1))
    ax.legend(fontsize=8)

    # (j) p-value histogram with π₀ shading
    ax = fig.add_subplot(gs[2, 1])
    counts_hist, edges, patches = ax.hist(
        pvalues, bins=20, color='#4878CF', alpha=0.7,
        edgecolor='k', linewidth=0.3)
    null_height = n_genes / 20 * pi0
    ax.axhline(null_height, color='#70AD47', ls='--', lw=1.5)
    # Shade the null portion of each bar
    for patch in patches:
        x0 = patch.get_x()
        w = patch.get_width()
        h = patch.get_height()
        null_h = min(null_height, h)
        ax.bar(x0, null_h, width=w, color='#70AD47', alpha=0.25,
               edgecolor='none')
    ax.set_xlabel('p-value'); ax.set_ylabel('Count')
    ax.set_title(f'(j) p-values with π₀ shading\n'
                 f'π̂₀={pi0:.2f} → ~{pi0*n_genes:.0f} null, '
                 f'~{(1-pi0)*n_genes:.0f} non-null', fontsize=10)

    # (k) QQ by effect-size tier: top 10%, middle, bottom 10% |β|
    ax = fig.add_subplot(gs[2, 2])
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)

    tiers = [
        ('Top 10% |β|', beta_norm >= np.percentile(beta_norm, 90), '#D65F5F'),
        ('Middle 80%', (beta_norm > np.percentile(beta_norm, 10)) &
                       (beta_norm < np.percentile(beta_norm, 90)), '#BBBBBB'),
        ('Bottom 10% |β|', beta_norm <= np.percentile(beta_norm, 10), '#4878CF'),
    ]
    for label, mask, color in tiers:
        p_tier = pvalues[mask]
        sp = np.sort(p_tier)
        ep = np.linspace(0, 1, len(sp) + 2)[1:-1]
        from scipy.stats import chi2 as _chi2
        lam_t = np.median(_chi2.ppf(1 - p_tier, 1)) / 0.4549 if len(p_tier) > 0 else 1.0
        ax.scatter(ep, sp, s=4, alpha=0.5, c=color,
                   label=f'{label} (λ={lam_t:.1f})')
    ax.set_xlabel('Expected'); ax.set_ylabel('Observed')
    ax.set_title('(k) QQ by effect-size tier', fontsize=10)
    ax.set_aspect('equal'); ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, loc='lower right')

    # (l) Cumulative contribution: genes ranked by effect size
    ax = fig.add_subplot(gs[2, 3])
    # Rank by |β| and R²_total, show cumulative share of total composition variance
    # Total composition variance = sum of R2_total * total_var across genes
    comp_var_g = R2_total * result.total_var
    total_comp = comp_var_g.sum()

    # By |β|
    order_beta = np.argsort(beta_norm)[::-1]
    cum_beta = np.cumsum(comp_var_g[order_beta]) / total_comp if total_comp > 0 else np.zeros(n_genes)

    # By R²_total
    order_r2 = np.argsort(R2_total)[::-1]
    cum_r2 = np.cumsum(comp_var_g[order_r2]) / total_comp if total_comp > 0 else np.zeros(n_genes)

    x_frac = np.arange(1, n_genes + 1) / n_genes
    ax.plot(x_frac, cum_beta, '-', color='#4878CF', lw=2, label='Ranked by |β|')
    ax.plot(x_frac, cum_r2, '-', color='#D65F5F', lw=2, label='Ranked by R²_total')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, lw=1)

    # Mark key thresholds
    for pct, ls in [(0.05, ':'), (0.10, '--')]:
        idx_beta = int(pct * n_genes)
        idx_r2 = int(pct * n_genes)
        ax.plot(pct, cum_beta[idx_beta - 1], 'o', color='#4878CF', markersize=5)
        ax.plot(pct, cum_r2[idx_r2 - 1], 'o', color='#D65F5F', markersize=5)
        ax.annotate(f'{cum_r2[idx_r2-1]:.0%}', (pct, cum_r2[idx_r2 - 1]),
                    textcoords='offset points', xytext=(5, -10), fontsize=7,
                    color='#D65F5F')

    # Mark n_sig boundary
    sig_frac = n_sig / n_genes
    ax.axvline(sig_frac, color='gray', ls=':', lw=1, alpha=0.5)
    ax.text(sig_frac + 0.01, 0.05, f'{n_sig} sig\n({sig_frac:.0%})',
            fontsize=7, color='gray')

    ax.set_xlabel('Fraction of genes (ranked)'); ax.set_ylabel('Cumulative composition variance')
    ax.set_title('(l) Cumulative contribution', fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc='lower right')

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
    return fig


def variance_panel(result, title="AURA Variance Decomposition", out_path=None):
    """Standard 6-panel 4-way variance decomposition figure.

    Args:
        result: AuraResult
        title: figure title
        out_path: save path (optional)

    Returns:
        matplotlib Figure
    """
    counts = result.counts
    phi = result.phi
    R2_total = result.R2_total
    R2_resid = result.R2
    total_var = result.total_var
    significant = result.significant
    gene_names = result.gene_names

    n_genes = len(gene_names)
    n_sig = significant.sum()

    # 4-way decomposition
    mu = counts.mean(axis=0)
    s2 = counts.var(axis=0, ddof=1)
    mu_bc = np.broadcast_to(mu[np.newaxis, :], counts.shape)
    var_nb = mu_bc + mu_bc ** 2 / phi
    r_raw = (counts - mu_bc) / np.sqrt(np.maximum(var_nb, 1e-10))
    alpha_i = r_raw.mean(axis=1)
    var_alpha = np.var(alpha_i)

    baseline_nb = mu + mu ** 2 / phi
    alpha_var = mu ** 2 * var_alpha * (1 + 1 / phi)
    comp_var = R2_total * total_var
    unexplained = np.maximum(s2 - baseline_nb - alpha_var - comp_var, 0.0)

    frac_nb = np.where(s2 > 0, np.minimum(baseline_nb / s2, 1.0), 1.0)
    frac_alpha = np.where(s2 > 0, np.minimum(alpha_var / s2, 1.0 - frac_nb), 0.0)
    frac_comp = np.where(s2 > 0, np.minimum(comp_var / s2,
                         1.0 - frac_nb - frac_alpha), 0.0)
    frac_unexp = np.maximum(1.0 - frac_nb - frac_alpha - frac_comp, 0.0)

    fig = plt.figure(figsize=(22, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # (a) All genes
    ax = fig.add_subplot(gs[0, :])
    order = np.argsort(R2_total)[::-1]
    x = np.arange(n_genes)
    b1 = frac_nb[order]; b2 = frac_alpha[order]
    b3 = frac_unexp[order]; b4 = frac_comp[order]
    ax.bar(x, b1, color=_COLORS['nb'], width=1.0, label='Baseline (NB)')
    ax.bar(x, b2, bottom=b1, color=_COLORS['alpha'], width=1.0,
           label='Library size (α)')
    ax.bar(x, b3, bottom=b1+b2, color=_COLORS['unexp'], width=1.0,
           label='Unexplained')
    ax.bar(x, b4, bottom=b1+b2+b3, color=_COLORS['comp'], width=1.0,
           label='Composition (β)')
    ax.axvline(n_sig - 0.5, color='k', ls='--', lw=1, alpha=0.5)
    ax.text(n_sig/2, 1.02, f'{n_sig} sig', ha='center', fontsize=8)
    ax.set_xlim(-0.5, n_genes-0.5); ax.set_ylim(0, 1.08)
    ax.set_xlabel('Genes (sorted by R2_total)')
    ax.set_ylabel('Fraction of total variance')
    ax.set_title('(a) 4-way decomposition — all genes', fontsize=11)
    ax.legend(fontsize=8, loc='center right')

    # (b) Significant genes
    ax = fig.add_subplot(gs[1, 0])
    sig_idx = np.where(significant)[0]
    sig_order = sig_idx[np.argsort(R2_total[sig_idx])[::-1]]
    x2 = np.arange(len(sig_order))
    ax.bar(x2, frac_nb[sig_order], color=_COLORS['nb'], width=1.0)
    ax.bar(x2, frac_alpha[sig_order], bottom=frac_nb[sig_order],
           color=_COLORS['alpha'], width=1.0)
    ax.bar(x2, frac_unexp[sig_order],
           bottom=frac_nb[sig_order]+frac_alpha[sig_order],
           color=_COLORS['unexp'], width=1.0)
    ax.bar(x2, frac_comp[sig_order],
           bottom=frac_nb[sig_order]+frac_alpha[sig_order]+frac_unexp[sig_order],
           color=_COLORS['comp'], width=1.0)
    for j, idx in enumerate(sig_order[:8]):
        top_val = frac_nb[idx]+frac_alpha[idx]+frac_unexp[idx]+frac_comp[idx]
        ax.text(j, top_val+0.01, gene_names[idx], fontsize=5,
                rotation=90, va='bottom', ha='center')
    ax.set_xlim(-0.5, len(sig_order)-0.5); ax.set_ylim(0, 1.15)
    ax.set_xlabel('Significant genes'); ax.set_ylabel('Fraction')
    ax.set_title(f'(b) Significant genes ({len(sig_order)})', fontsize=11)

    # (c) Pie
    ax = fig.add_subplot(gs[1, 1])
    if n_sig > 0:
        med = [np.median(frac_nb[significant]),
               np.median(frac_alpha[significant]),
               np.median(frac_unexp[significant]),
               np.median(frac_comp[significant])]
        total_med = sum(med)
        sizes = [m/total_med for m in med]
        labels = [f'NB baseline\n{med[0]*100:.0f}%',
                  f'Library size\n{med[1]*100:.1f}%',
                  f'Unexplained\n{med[2]*100:.0f}%',
                  f'Composition\n{med[3]*100:.2f}%']
        ax.pie(sizes, labels=labels,
               colors=[_COLORS['nb'], _COLORS['alpha'],
                       _COLORS['unexp'], _COLORS['comp']],
               startangle=90, textprops={'fontsize': 8})
    ax.set_title('(c) Median (sig genes)', fontsize=11)

    # (d) Alpha fraction vs gene mean
    ax = fig.add_subplot(gs[1, 2])
    ax.scatter(mu[~significant], frac_alpha[~significant],
               s=5, alpha=0.2, c='#CCCCCC')
    ax.scatter(mu[significant], frac_alpha[significant],
               s=10, alpha=0.5, c='#17BECF')
    ax.set_xlabel('Gene mean'); ax.set_ylabel('Library size (α) fraction')
    ax.set_xscale('log')
    ax.set_title('(d) Library size fraction vs mean', fontsize=11)

    # (e) Top 20 genes
    ax = fig.add_subplot(gs[2, :2])
    top20 = sig_order[:20] if n_sig >= 20 else sig_order
    y_pos = np.arange(len(top20))
    ax.barh(y_pos, frac_nb[top20], color=_COLORS['nb'])
    ax.barh(y_pos, frac_alpha[top20], left=frac_nb[top20],
            color=_COLORS['alpha'])
    ax.barh(y_pos, frac_unexp[top20],
            left=frac_nb[top20]+frac_alpha[top20], color=_COLORS['unexp'])
    ax.barh(y_pos, frac_comp[top20],
            left=frac_nb[top20]+frac_alpha[top20]+frac_unexp[top20],
            color=_COLORS['comp'])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f'{gene_names[i]} (R2t={R2_total[i]:.1%})' for i in top20],
        fontsize=7)
    ax.set_xlabel('Fraction of total variance')
    ax.set_title('(e) Top genes', fontsize=11)
    ax.set_xlim(0, 1); ax.invert_yaxis()
    ax.legend(['Baseline (NB)', 'Library size (α)',
               'Unexplained', 'Composition (β)'],
              fontsize=7, loc='lower right')

    # (f) Text summary
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    lines = [
        f"Genes tested:    {n_genes}",
        f"Significant:     {n_sig} ({n_sig/n_genes*100:.0f}%)",
        f"phi (NB size):   {_phi_label(phi)}",
        f"std(alpha_i):    {np.sqrt(var_alpha):.3f}",
        "",
        "Median decomposition (sig genes):",
    ]
    if n_sig > 0:
        lines += [
            f"  Baseline (NB):    {np.median(frac_nb[significant])*100:.0f}%",
            f"  Library size (α): {np.median(frac_alpha[significant])*100:.1f}%",
            f"  Composition (β):  {np.median(frac_comp[significant])*100:.2f}%",
            f"  Unexplained:      {np.median(frac_unexp[significant])*100:.1f}%",
            "",
            f"Top gene: {gene_names[sig_order[0]]}",
            f"  R2_total = {R2_total[sig_order[0]]:.1%}",
        ]
    ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes,
            fontsize=9, va='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.3))
    ax.set_title('(f) Summary', fontsize=11)

    plt.savefig(out_path, dpi=150, bbox_inches='tight') if out_path else None
    return fig


def variance_summary(result=None, csv_path=None, focal_name=""):
    """Print text-based variance decomposition summary.

    Args:
        result: AuraResult (preferred)
        csv_path: alternative — path to results CSV
        focal_name: label for the focal type

    Returns:
        summary string
    """
    import pandas as pd

    if csv_path is not None:
        df = pd.read_csv(csv_path)
    elif result is not None:
        df = pd.DataFrame({
            "gene": result.gene_names,
            "significant": result.significant,
            "R2_resid": result.R2,
            "R2_total": result.R2_total,
            "total_var": result.total_var,
            "baseline_var": result.baseline_var,
            "excess_var": result.excess_var,
            "has_excess": result.has_excess,
            "qvalue": result.qvalues,
        })
        for i, name in enumerate(result.type_names):
            df[f"beta_{name}"] = result.beta[:, i]
        if result.P is not None and result.beta.shape[1] == len(result.type_names):
            for name, sd in zip(result.type_names, np.asarray(result.P).std(axis=0)):
                df[f"sdP_{name}"] = sd
    else:
        raise ValueError("Provide either result or csv_path")

    gene_names = df['gene'].values
    sig = df['significant'].values
    r2_resid = df['R2_resid'].values
    r2_total = df['R2_total'].values
    total_var = df['total_var'].values
    baseline_var = df['baseline_var'].values
    excess_var = df['excess_var'].values
    has_excess = df['has_excess'].values
    n_genes = len(df)
    n_sig = sig.sum()

    frac_bl = np.where(total_var > 0, np.minimum(baseline_var / total_var, 1.0), 1.0)
    comp_var = r2_resid * excess_var
    frac_comp = np.where(total_var > 0, comp_var / total_var, 0.0)
    frac_excess = np.where(total_var > 0, excess_var / total_var, 0.0)

    beta_cols = [c for c in df.columns if c.startswith('beta_')]
    axis_names = [c.replace('beta_', '') for c in beta_cols]

    lines = []
    lines.append(f"{'='*70}")
    lines.append(f"  AURA Variance Decomposition — {focal_name}")
    lines.append(f"{'='*70}")
    lines.append("")
    lines.append(f"Genes tested:  {n_genes}")
    lines.append(f"Significant:   {n_sig} ({n_sig/n_genes*100:.0f}%)")
    n_filtered = ((~has_excess) & (df['qvalue'].values <= 0.05)).sum()
    if n_filtered > 0:
        lines.append(f"Filtered (no excess var): {n_filtered}")
    lines.append("")

    lines.append("--- Variance decomposition (median across genes) ---")
    lines.append("")
    lines.append(f"{'':30s} {'All genes':>12s} {'Significant':>12s}")
    lines.append(f"{'':30s} {'─'*12:>12s} {'─'*12:>12s}")
    lines.append(f"{'Baseline (NB sampling)':30s} "
                 f"{np.median(frac_bl)*100:11.0f}% "
                 f"{np.median(frac_bl[sig])*100:11.0f}%")
    lines.append(f"{'Excess variance':30s} "
                 f"{np.median(frac_excess)*100:11.0f}% "
                 f"{np.median(frac_excess[sig])*100:11.0f}%")
    lines.append(f"{'  ├─ Composition-explained':30s} "
                 f"{np.median(frac_comp)*100:11.2f}% "
                 f"{np.median(frac_comp[sig])*100:11.2f}%")
    lines.append(f"{'  └─ Unexplained':30s} "
                 f"{np.median(frac_excess - frac_comp)*100:11.1f}% "
                 f"{np.median(frac_excess[sig] - frac_comp[sig])*100:11.1f}%")
    lines.append("")

    lines.append("--- Top 10 composition-responsive genes ---")
    lines.append("")
    lines.append(f"{'Gene':15s} {'R2_total':>8s} {'Baseline':>9s} "
                 f"{'Excess':>7s} {'Comp':>6s} {'Driver axis'}")
    lines.append(f"{'─'*15:15s} {'─'*8:>8s} {'─'*9:>9s} "
                 f"{'─'*7:>7s} {'─'*6:>6s} {'─'*20}")
    sig_idx = np.where(sig)[0]
    from ..spillover import driver_axes
    sd_cols = [f"sdP_{a}" for a in axis_names]
    if all(c in df.columns for c in sd_cols):
        drivers, _ = driver_axes(df[beta_cols].values,
                                 df[sd_cols].iloc[0].values, "contribution")
    else:
        drivers, _ = driver_axes(df[beta_cols].values, method="abs_beta")
    if len(sig_idx) > 0:
        top = sig_idx[np.argsort(r2_total[sig_idx])[::-1][:10]]
        for i in top:
            betas = [df[c].iloc[i] for c in beta_cols]
            driver = drivers[i]
            sign = '+' if betas[driver] > 0 else '-'
            lines.append(
                f"{gene_names[i]:15s} {r2_total[i]:7.1%} "
                f"{frac_bl[i]:8.0%} {frac_excess[i]:6.0%} "
                f"{frac_comp[i]:5.1%} {sign}{axis_names[driver]}")
    lines.append("")

    lines.append("--- Dominant composition axes ---")
    lines.append("")
    if len(sig_idx) > 0:
        axis_counts = {a: 0 for a in axis_names}
        for i in sig_idx:
            axis_counts[axis_names[drivers[i]]] += 1
        for a, cnt in sorted(axis_counts.items(), key=lambda x: -x[1]):
            if cnt > 0:
                lines.append(f"  {a:15s}: {cnt:3d} genes "
                             f"({cnt/n_sig*100:.0f}% of significant)")

    text = '\n'.join(lines)
    print(text)
    return text


def subtype_check(result, n_pcs=15, title="Focal Cell Subtype Check", out_path=None):
    """UMAP of focal cells to check for unresolved internal subtypes.

    6-panel figure:
      Row 1: UMAP colored by composition PC1, PC2, top gene expression
      Row 2: UMAP colored by dominant neighbor fraction, library size, transcriptomic PC1

    Also prints Spearman correlations between transcriptomic PCs and composition
    to quantify whether AURA signal is confounded with internal structure.

    Args:
        result: AuraResult
        n_pcs: number of PCs for UMAP input
        title: figure title
        out_path: save path (optional)

    Returns:
        matplotlib Figure
    """
    from sklearn.decomposition import PCA
    try:
        import umap
    except ImportError:
        raise ImportError("umap-learn is required for subtype_check: pip install umap-learn")

    counts = result.counts.astype(np.float32)
    lib_size = counts.sum(axis=1, keepdims=True)
    norm = counts / np.maximum(lib_size, 1) * np.median(lib_size)
    X = np.log1p(norm)

    pca = PCA(n_components=min(n_pcs, X.shape[1], X.shape[0]))
    pcs = pca.fit_transform(X)

    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42)
    emb = reducer.fit_transform(pcs[:, :min(n_pcs, pcs.shape[1])])

    # AURA composition PCs
    sig_mask = result.significant
    if sig_mask.sum() >= 3:
        beta_sig = result.beta[sig_mask]
        _, S_svd, Vt = np.linalg.svd(beta_sig, full_matrices=False)
        V = Vt.T
        comp_pc1 = result.P_tilde @ V[:, 0]
        comp_pc2 = result.P_tilde @ V[:, 1] if V.shape[1] > 1 else np.zeros(len(comp_pc1))
    else:
        comp_pc1 = result.P_tilde[:, 0]
        comp_pc2 = result.P_tilde[:, 1] if result.P_tilde.shape[1] > 1 else np.zeros(len(comp_pc1))

    # Top gene by R2_total
    sig_idx = np.where(sig_mask)[0]
    if len(sig_idx) > 0:
        top_gene_idx = sig_idx[np.argmax(result.R2_total[sig_idx])]
        top_gene_name = result.gene_names[top_gene_idx]
        top_gene_expr = counts[:, top_gene_idx]
    else:
        top_gene_name = result.gene_names[0]
        top_gene_expr = counts[:, 0]

    # Dominant neighbor axis (highest variance in P)
    comp_var = result.P.var(axis=0)
    dom_idx = np.argmax(comp_var)
    dom_name = result.type_names[dom_idx]
    dom_frac = result.P[:, dom_idx]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    panels = [
        (axes[0, 0], comp_pc1, 'Composition PC1 score', 'RdBu_r', True),
        (axes[0, 1], comp_pc2, 'Composition PC2 score', 'RdBu_r', True),
        (axes[0, 2], top_gene_expr, f'{top_gene_name} expression', 'Purples', False),
        (axes[1, 0], dom_frac, f'{dom_name} neighbor fraction', 'YlOrRd', False),
        (axes[1, 1], counts.sum(axis=1).astype(float), 'Library size', 'viridis', False),
        (axes[1, 2], pcs[:, 0], 'Transcriptomic PC1', 'RdBu_r', True),
    ]

    for ax, vals, label, cmap, diverging in panels:
        if diverging:
            vmax = np.percentile(np.abs(vals), 99)
            kwargs = dict(vmin=-vmax, vmax=vmax)
        else:
            kwargs = dict(vmin=np.percentile(vals, 1))
        sc_plot = ax.scatter(emb[:, 0], emb[:, 1], c=vals, s=1, alpha=0.5,
                             cmap=cmap, rasterized=True, **kwargs)
        plt.colorbar(sc_plot, ax=ax, shrink=0.7)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2')

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')

    # Print correlation summary
    rho_pc1_comp, _ = spearmanr(pcs[:, 0], comp_pc1)
    rho_pc1_dom, _ = spearmanr(pcs[:, 0], dom_frac)
    rho_lib_comp, _ = spearmanr(counts.sum(axis=1), comp_pc1)
    rho_lib_dom, _ = spearmanr(counts.sum(axis=1), dom_frac)

    print(f"\n{'='*60}")
    print(f"  Subtype Check Summary")
    print(f"{'='*60}")
    print(f"  Transcriptomic PC1 variance explained: {pca.explained_variance_ratio_[0]:.1%}")
    print(f"  Spearman correlations:")
    print(f"    Transcriptomic PC1 vs Composition PC1:  {rho_pc1_comp:+.3f}")
    print(f"    Transcriptomic PC1 vs {dom_name} fraction: {rho_pc1_dom:+.3f}")
    print(f"    Library size vs Composition PC1:         {rho_lib_comp:+.3f}")
    print(f"    Library size vs {dom_name} fraction:      {rho_lib_dom:+.3f}")
    print()
    if abs(rho_pc1_comp) > 0.3 or abs(rho_pc1_dom) > 0.3:
        print("  ⚠ Transcriptomic PC1 is correlated with composition.")
        print("    Consider whether the focal type contains spatially")
        print("    segregated subtypes. If so, split and re-run.")
    else:
        print("  ✓ No strong transcriptomic–composition correlation.")
        print("    AURA signal is unlikely to reflect unresolved subtypes.")

    return fig
