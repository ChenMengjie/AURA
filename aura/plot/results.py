"""
AURA result visualization: SVD decomposition + single-gene zoom.

1. result_panel: 6-panel SVD result figure
2. gene_zoom_panel: single-gene spatial + composition deep dive
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from ._style import POS as _POS, NEG as _NEG, PC_COLORS


def result_panel(result=None, csv_path=None, title="AURA Results",
                 out_path=None, n_top=20):
    """Generate standard 6-panel SVD result figure.

    Args:
        result: AuraResult (preferred)
        csv_path: alternative — path to results CSV
        title: figure title
        out_path: save path
        n_top: number of top genes in bar chart

    Returns:
        matplotlib Figure
    """
    import pandas as pd

    if csv_path is not None:
        df = pd.read_csv(csv_path)
    elif result is not None:
        df = pd.DataFrame({
            "gene": result.gene_names,
            "significant": result.significant,
            "R2_total": result.R2_total,
        })
        beta_cols_list = []
        for i, name in enumerate(result.type_names):
            col = f"beta_{name}"
            df[col] = result.beta[:, i]
            beta_cols_list.append(col)
    else:
        raise ValueError("Provide either result or csv_path")

    beta_cols = [c for c in df.columns if c.startswith('beta_')]
    axis_names = [c.replace('beta_', '') for c in beta_cols]
    K = len(axis_names)

    sig = df[df['significant']].copy()
    n_sig = len(sig)
    n_total = len(df)

    if n_sig < 3:
        print(f"Only {n_sig} significant genes — skipping result panel.")
        return None

    beta_mat = sig[beta_cols].values.astype(float)
    gene_names = sig['gene'].values
    r2t = sig['R2_total'].values

    # SVD
    U, S, Vt = np.linalg.svd(beta_mat, full_matrices=False)
    V = Vt.T
    var_exp = S ** 2 / (S ** 2).sum()
    scores = U * S[np.newaxis, :]
    n_pc = min(3, len(S))

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f'{title}\n({n_sig} significant / {n_total} tested, '
                 f'top {n_pc} PCs = {var_exp[:n_pc].sum()*100:.0f}%)',
                 fontsize=13, fontweight='bold')

    # Row 1: PC mode bar charts
    for pc in range(3):
        ax = axes[0, pc]
        if pc < n_pc:
            mode = V[:, pc]
            bar_colors = [_POS if v > 0 else _NEG for v in mode]
            ax.barh(range(K), mode, color=bar_colors, alpha=0.75)
            ax.set_yticks(range(K))
            ax.set_yticklabels(axis_names if pc == 0 else [], fontsize=9)
            ax.set_xlabel('Loading', fontsize=9)
            ax.set_title(f'PC{pc+1} ({var_exp[pc]*100:.0f}%)', fontsize=11)
            ax.axvline(0, color='gray', lw=0.5)
            ax.invert_yaxis()

            top_pos = np.argsort(scores[:, pc])[::-1][:3]
            top_neg = np.argsort(scores[:, pc])[:3]
            pos_str = ', '.join(gene_names[top_pos])
            neg_str = ', '.join(gene_names[top_neg])
            ax.text(0.02, 0.02, f'+ {pos_str}\n- {neg_str}',
                    transform=ax.transAxes, fontsize=6, va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='wheat', alpha=0.5))
        else:
            ax.set_visible(False)

    # Row 2: PC scatters + top genes
    label_genes = set()
    for pc_i in range(n_pc):
        for idx in np.argsort(np.abs(scores[:, pc_i]))[::-1][:5]:
            label_genes.add(idx)

    is_labeled = np.array([i in label_genes for i in range(n_sig)])

    # PC1 vs PC2
    ax = axes[1, 0]
    ax.scatter(scores[~is_labeled, 0], scores[~is_labeled, 1],
               s=r2t[~is_labeled] * 3000 + 3,
               alpha=0.25, c='#BBBBBB', edgecolors='none')
    ax.scatter(scores[is_labeled, 0], scores[is_labeled, 1],
               s=r2t[is_labeled] * 3000 + 8,
               alpha=0.7, c='#333333', edgecolors='k', linewidths=0.3)
    labeled_12 = set()
    for pc_i in [0, 1]:
        for idx in np.argsort(np.abs(scores[:, pc_i]))[::-1][:5]:
            if idx not in labeled_12:
                ax.annotate(gene_names[idx],
                            (scores[idx, 0], scores[idx, 1]),
                            fontsize=7, color='#8B2020',
                            textcoords='offset points', xytext=(0, 5),
                            ha='center')
                labeled_12.add(idx)
    ax.set_xlabel(f'PC1 ({var_exp[0]*100:.0f}%)')
    ax.set_ylabel(f'PC2 ({var_exp[1]*100:.0f}%)')
    ax.set_title('PC1 vs PC2', fontsize=11)
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.axvline(0, color='gray', lw=0.5, ls='--')

    # PC1 vs PC3
    ax = axes[1, 1]
    if n_pc >= 3:
        ax.scatter(scores[~is_labeled, 0], scores[~is_labeled, 2],
                   s=r2t[~is_labeled] * 3000 + 3,
                   alpha=0.25, c='#BBBBBB', edgecolors='none')
        ax.scatter(scores[is_labeled, 0], scores[is_labeled, 2],
                   s=r2t[is_labeled] * 3000 + 8,
                   alpha=0.7, c='#333333', edgecolors='k', linewidths=0.3)
        labeled_13 = set()
        for pc_i in [0, 2]:
            for idx in np.argsort(np.abs(scores[:, pc_i]))[::-1][:5]:
                if idx not in labeled_13:
                    ax.annotate(gene_names[idx],
                                (scores[idx, 0], scores[idx, 2]),
                                fontsize=7, color='#8B2020',
                                textcoords='offset points', xytext=(0, 5),
                                ha='center')
                    labeled_13.add(idx)
        ax.set_xlabel(f'PC1 ({var_exp[0]*100:.0f}%)')
        ax.set_ylabel(f'PC3 ({var_exp[2]*100:.0f}%)')
        ax.set_title('PC1 vs PC3', fontsize=11)
        ax.axhline(0, color='gray', lw=0.5, ls='--')
        ax.axvline(0, color='gray', lw=0.5, ls='--')
    else:
        ax.set_visible(False)

    # Top genes bar — stacked by PC contribution
    ax = axes[1, 2]
    n_show = min(n_top, n_sig)
    top_idx = np.argsort(r2t)[::-1][:n_show]
    top_genes = gene_names[top_idx]
    top_r2 = r2t[top_idx]

    score_sq = scores ** 2
    beta_var_total = score_sq.sum(axis=1, keepdims=True)
    beta_var_total = np.maximum(beta_var_total, 1e-10)
    pc_frac = score_sq / beta_var_total

    pc_rest_color = '#D0D0D0'
    left = np.zeros(n_show)
    for pc in range(min(n_pc, 3)):
        widths = top_r2 * pc_frac[top_idx, pc]
        ax.barh(range(n_show), widths, left=left,
                color=PC_COLORS[pc], alpha=0.8, edgecolor='white', linewidth=0.3,
                label=f'PC{pc+1} ({var_exp[pc]*100:.0f}%)')
        left += widths
    rest_frac = 1.0 - pc_frac[top_idx, :min(n_pc, 3)].sum(axis=1)
    rest_widths = top_r2 * np.maximum(rest_frac, 0)
    if rest_widths.sum() > 0:
        ax.barh(range(n_show), rest_widths, left=left,
                color=pc_rest_color, alpha=0.6, edgecolor='white', linewidth=0.3,
                label='Other PCs')

    ax.set_yticks(range(n_show))
    ax.set_yticklabels(top_genes, fontsize=7)
    for j in range(n_show):
        ax.text(top_r2[j] + 0.001, j, f'{top_r2[j]:.1%}',
                va='center', fontsize=6)
    ax.set_xlabel('R²_total')
    ax.set_title('Top genes (by PC contribution)', fontsize=11)
    ax.invert_yaxis()
    ax.legend(fontsize=7, loc='lower right')

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
    return fig


def gene_zoom_panel(gene_name, result, all_xy=None, bg_masks=None,
                    n_drivers=3, n_bins=10,
                    dot_size=12, title=None, out_path=None,
                    driver="abs_beta"):
    """Single-gene deep dive: spatial expression, beta profile, per-driver scatters.

    Layout:
      Row 0: spatial expression map (cols 0-1) | effect by neighbor type (col 2)
      Rows 1..n_drivers: driver composition map (cols 0-1) | scatter + detection rate (col 2)

    Args:
        gene_name: gene to zoom on
        result: AuraResult
        all_xy: optional (n_all, 2) all cell coordinates for background
        bg_masks: list of boolean masks into all_xy for background cell types
        n_drivers: number of top driver axes to show
        n_bins: number of bins for scatter trend lines
        dot_size: size of focal cells in spatial maps (uniform across all rows)
        title: figure title override
        out_path: save path
        driver: driver-axis ranking, 'abs_beta' (default) or 'contribution'
            (see `aura.spillover.driver_axes`)

    Returns:
        matplotlib Figure
    """
    from scipy.stats import spearmanr

    focal_xy = result.focal_xy
    counts = result.counts
    gene_names = result.gene_names
    P = result.P
    axis_names = result.type_names

    gidx = np.where(gene_names == gene_name)[0][0]
    expr = counts[:, gidx]
    expr_pos = expr > 0
    detection = expr_pos.mean()
    mean_expr = expr.mean()

    # Beta and R2 for this gene
    gene_row_idx = np.where(gene_names == gene_name)[0][0]
    betas = result.beta[gene_row_idx]
    beta_names = axis_names
    r2_total = result.R2_total[gene_row_idx]

    # Top driver axes by |beta| (default) or, with driver='contribution', by
    # |beta_k| * sd(P_k): the expression change across the observed range of
    # each axis (raw |beta| favours rare neighbor types)
    if (driver == "contribution" and getattr(result, 'P', None) is not None
            and len(betas) == result.P.shape[1]):
        driver_order = np.argsort(np.abs(betas) * result.P.std(axis=0))[::-1][:n_drivers]
    else:
        driver_order = np.argsort(np.abs(betas))[::-1][:n_drivers]

    n_rows = 1 + n_drivers
    fig = plt.figure(figsize=(18, 5 * n_rows))
    gs = GridSpec(n_rows, 3, figure=fig, hspace=0.35, wspace=0.30, top=0.90)
    if title is None:
        title = (f"{gene_name} in focal cells — R²_total = {r2_total:.1%}, "
                 f"detection = {detection:.0%}, mean = {mean_expr:.2f}")
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.97)

    # Shared spatial axis limits
    xlim = (focal_xy[:, 0].min() - 50, focal_xy[:, 0].max() + 50)
    ylim = (focal_xy[:, 1].min() - 50, focal_xy[:, 1].max() + 50)

    # Row 0, Col 0-1: Spatial expression map
    ax = fig.add_subplot(gs[0, :2])
    if all_xy is not None and bg_masks is not None:
        for mask in bg_masks:
            ax.scatter(all_xy[mask, 0], all_xy[mask, 1], c='#E8E8E8',
                       s=0.2, alpha=0.03, rasterized=True)
    ax.scatter(focal_xy[~expr_pos, 0], focal_xy[~expr_pos, 1],
               c='#DDDDDD', s=dot_size, alpha=0.3, rasterized=True)
    if expr_pos.any():
        sc_plot = ax.scatter(focal_xy[expr_pos, 0], focal_xy[expr_pos, 1],
                             c=expr[expr_pos], cmap='Purples', s=dot_size,
                             alpha=0.8, vmin=0, rasterized=True,
                             edgecolors='k', linewidths=0.2)
        plt.colorbar(sc_plot, ax=ax, shrink=0.6, label='count')
    ax.set_title(f'{gene_name} expression\n'
                 f'{expr_pos.sum()}/{len(expr)} cells express', fontsize=9)
    ax.set_aspect('equal')
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_ylabel('y (µm)')

    # Row 0, Col 2: Effect by neighbor type
    ax = fig.add_subplot(gs[0, 2])
    abs_sort = np.argsort(np.abs(betas))[::-1]
    colors_abs = [_POS if betas[i] > 0 else _NEG for i in abs_sort]
    ax.barh(range(len(betas)), betas[abs_sort], color=colors_abs, alpha=0.75)
    ax.set_yticks(range(len(betas)))
    ax.set_yticklabels([beta_names[i] for i in abs_sort], fontsize=9)
    ax.set_xlabel('Beta (effect size)')
    ax.set_title('Effect by neighbor type', fontsize=9)
    ax.axvline(0, color='gray', lw=0.5)
    ax.invert_yaxis()

    # Rows 1+: Per-driver spatial + scatter
    for row_i, d_idx in enumerate(driver_order):
        d_name = beta_names[d_idx]
        d_beta = betas[d_idx]
        comp_vals = P[:, d_idx]

        # Col 0-1: spatial composition map
        ax = fig.add_subplot(gs[1 + row_i, :2])
        sc2 = ax.scatter(focal_xy[:, 0], focal_xy[:, 1],
                         c=comp_vals, cmap='YlOrRd', s=dot_size,
                         alpha=0.5, rasterized=True)
        if expr_pos.any():
            ax.scatter(focal_xy[expr_pos, 0], focal_xy[expr_pos, 1],
                       c='none', edgecolors='#555555', s=dot_size,
                       linewidths=0.8, alpha=0.5)
        plt.colorbar(sc2, ax=ax, shrink=0.6, label=f'{d_name} fraction')
        ax.set_title(f'{d_name} neighborhood\n'
                     f'(circles = {gene_name}+, beta = {d_beta:.1f})',
                     fontsize=9)
        ax.set_aspect('equal')
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_ylabel('y (µm)')
        if row_i == n_drivers - 1:
            ax.set_xlabel('x (µm)')

        # Col 2: scatter + detection rate
        ax = fig.add_subplot(gs[1 + row_i, 2])
        jitter = np.random.default_rng(42).normal(0, 0.15, size=len(expr))
        ax.scatter(comp_vals, expr + jitter, s=3, alpha=0.15, c='#4878CF')

        lo_edge, hi_edge = comp_vals.min(), comp_vals.max()
        bins = np.linspace(lo_edge, hi_edge, n_bins + 1)
        bin_idx = np.digitize(comp_vals, bins)
        bin_x, bin_y, bin_frac = [], [], []
        for b in range(1, len(bins)):
            bm = bin_idx == b
            if bm.sum() >= 10:
                bin_x.append(comp_vals[bm].mean())
                bin_y.append(expr[bm].mean())
                bin_frac.append((expr[bm] > 0).mean())

        ax.plot(bin_x, bin_y, 'o--', color='#333333', markersize=5, lw=1.5,
                alpha=0.8, label='mean expression')

        ax2 = ax.twinx()
        ax2.plot(bin_x, bin_frac, '-s', color='#C44E52', markersize=4, lw=2,
                 alpha=0.8, label='detection rate')
        ax2.set_ylabel('Fraction expressing', fontsize=8)
        ax2.tick_params(axis='y')
        ax2.set_ylim(0, max(bin_frac) * 1.3 if bin_frac else 1)

        rho, pval = spearmanr(comp_vals, expr)
        ax.set_xlabel(f'{d_name} fraction')
        ax.set_ylabel(f'{gene_name} count')
        ax.set_title(f'{gene_name} vs {d_name}\n'
                     f'Spearman rho = {rho:.3f}', fontsize=9)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7,
                  loc='upper left')

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
    return fig
