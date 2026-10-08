"""
Alternative Fig. 6e: core-vs-peripheral regulatory enrichment on a log scale.

The linear panel caps Tfh at 60 because its odds ratio is infinite (no
peripheral Tfh gene is regulatory). Here odds ratios are drawn on a log2
axis as bars from OR = 1; rows with a zero cell (Tfh) use the
Haldane-Anscombe correction (+0.5 to every cell) and are marked with a
dagger. Optional exact (conditional) 95% confidence intervals, which agree
with the Fisher p-values (Woolf intervals can disagree at these small counts).

The 2x2 counts are recovered from the published fractions in
`core_peripheral_newR2.csv` (core = top 10% of significant genes by
R2_total, peripheral = bottom 50%) and verified against its Fisher odds
ratios and p-values.

Usage: python fig6e_log_odds.py <core_peripheral_newR2.csv> <out_dir>
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from scipy.stats.contingency import odds_ratio

COLORS = {"Lymph node": (0.761, 0.384, 0.478), "IPF": (0.557, 0.42, 0.686),
          "NSCLC": (0.482, 0.541, 0.749)}
# Row order and lineage groups of the published panel
ROWS = [("T cell", [("Tfh", "Lymph node"), ("T CD4 mem", "NSCLC"),
                    ("T_naive", "Lymph node"), ("T_CD8", "Lymph node"),
                    ("Treg", "Lymph node")]),
        ("B / GC", [("FDC", "Lymph node"), ("GC_LZ", "Lymph node"),
                    ("GC_DZ", "Lymph node")]),
        ("Macrophage", [("SPP1+Mac", "IPF"), ("Macrophage", "Lymph node"),
                        ("Int.Mac", "IPF"), ("Macrophage", "NSCLC")]),
        ("Fibroblast", [("Alv.FBs", "IPF"), ("Fibroblast", "NSCLC")]),
        ("Endothelial", [("Endothelial", "Lymph node"), ("Capillary", "IPF"),
                         ("Endothelial", "NSCLC")]),
        ("Epithelial", [("Basal", "IPF"), ("AT2", "IPF")])]


def recover_counts(r):
    """Integer 2x2 table reproducing the published fractions, OR and p."""
    n = int(r.n_sig)
    for nc in range(max(1, int(0.08 * n)), int(0.12 * n) + 2):
        a = r.core_reg_frac * nc
        if abs(a - round(a)) > 1e-6:
            continue
        for npp in range(int(0.45 * n), int(0.55 * n) + 2):
            c = r.periph_reg_frac * npp
            if abs(c - round(c)) > 1e-6:
                continue
            a_, c_ = int(round(a)), int(round(c))
            orr, p = fisher_exact([[a_, nc - a_], [c_, npp - c_]])
            same_or = (np.isinf(orr) and np.isinf(r.fisher_or)) or \
                (np.isfinite(orr) and abs(orr - r.fisher_or) < 1e-6)
            if same_or and abs(p - r.fisher_p) <= 1e-9 * max(1, r.fisher_p) + 1e-12:
                return a_, nc - a_, c_, npp - c_
    raise ValueError(f"could not recover counts for {r.label} ({r.dataset})")


def log_or(a, b, c, d):
    """log OR and Woolf SE; Haldane-Anscombe correction if any cell is 0."""
    corrected = min(a, b, c, d) == 0
    if corrected:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    lor = np.log((a * d) / (b * c))
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return lor, se, corrected


def draw(tab, out, with_ci, with_text=True):
    fig, ax = plt.subplots(figsize=(427.68 / 72, 324 / 72))
    y, yticks, ylabels, groups = 0, [], [], []
    lo_lim, hi_lim = np.log2(1 / 16), np.log2(1024)
    for gname, members in ROWS:
        y0 = y
        for label, ds in members:
            r = tab[(tab.label == label) & (tab.dataset == ds)].iloc[0]
            l2 = r.log_or / np.log(2)
            ax.barh(y, l2, left=0, height=0.65, color=COLORS[ds], alpha=0.8,
                    edgecolor="none")
            if with_ci:
                lo = np.log2(max(r.ci_low, 1e-12))
                hi = np.log2(r.ci_high) if np.isfinite(r.ci_high) else np.inf
                ax.plot([max(lo, lo_lim), min(hi, hi_lim)], [y, y], color="0.25",
                        lw=0.8, solid_capstyle="butt")
                for v, lim, mk in ((lo, lo_lim, "<"), (hi, hi_lim, ">")):
                    if (mk == "<" and v < lim) or (mk == ">" and v > lim):
                        ax.plot(lim, y, marker=mk, color="0.25", ms=3)
            if with_text:
                tag = ("*" if r.fisher_p < 0.05 else "") + ("†" if r.corrected else "")
                if tag:
                    if with_ci:   # at the bar end, just above the CI line
                        ax.text(max(l2, 0) + 0.12, y - 0.08, tag, va="bottom",
                                ha="left", fontsize=8)
                    else:
                        ax.text(max(l2, 0) + 0.25, y, tag, va="center",
                                ha="left", fontsize=8)
            yticks.append(y)
            ylabels.append(label)
            y += 1
        groups.append((gname, y0, y - 1))
        y += 0.0
        ax.axhline(y - 0.5, color="0.6", ls="--", lw=0.8)
    ax.axvline(0, color="black", ls="--", lw=0.8)
    ax.set_ylim(y - 0.5, -0.8)
    ax.set_xlim(lo_lim, hi_lim)
    ticks = np.arange(-4, 11, 2)
    ax.set_xticks(ticks)
    ax.set_yticks(yticks)
    if with_text:
        ax.set_xticklabels([f"{2.0 ** t:g}" if t >= 0 else f"1/{int(2 ** -t)}"
                            for t in ticks], fontsize=7)
        ax.set_yticklabels(ylabels, fontsize=6.5)
        ax.set_xlabel("Odds ratio (core vs peripheral), log scale", fontsize=8)
        ax.set_title("Regulatory enrichment by lineage", fontsize=9)
        for gname, a, b in groups:
            ax.text(1.01, (a + b) / 2, gname, transform=ax.get_yaxis_transform(),
                    va="center", ha="left", fontsize=6.5, fontweight="bold",
                    color="0.3")
        note = "* Fisher p < 0.05   † zero cell: Haldane–Anscombe correction"
        if with_ci:
            note += "   lines: exact 95% CI"
        fig.text(0.02, 0.01, note, fontsize=5.5, color="0.3")
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    ax.tick_params(axis="y", length=2)
    fig.subplots_adjust(left=0.15, right=0.86, top=0.92, bottom=0.14)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)


def main(csv, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(csv)
    rows = []
    for _, r in d.iterrows():
        a, b, c, dd = recover_counts(r)
        lor, se, corr = log_or(a, b, c, dd)
        ci = odds_ratio([[a, b], [c, dd]], kind="conditional").confidence_interval(0.95)
        rows.append(dict(label=r.label, dataset=r.dataset, core_reg=a,
                         core_other=b, periph_reg=c, periph_other=dd,
                         fisher_or=r.fisher_or, fisher_p=r.fisher_p,
                         log_or=lor, se=se, corrected=corr,
                         or_plotted=np.exp(lor),
                         ci_low=ci.low, ci_high=ci.high))
    tab = pd.DataFrame(rows)
    tab.to_csv(out_dir / "fig6e_log_odds_table.csv", index=False)
    for with_ci, name in ((False, "A_logOR"), (True, "B_logOR_CI")):
        draw(tab, out_dir / f"Fig6_panel_e_{name}_withtext", with_ci, True)
        draw(tab, out_dir / f"Fig6_panel_e_{name}_notext", with_ci, False)
    print(tab[["label", "dataset", "or_plotted", "ci_low", "ci_high",
               "fisher_p", "corrected"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
