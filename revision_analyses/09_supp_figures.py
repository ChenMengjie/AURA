"""
09 — Draw Supplementary Figures S9–S16 from the revision outputs.

Reads the CSVs written by scripts 01–08 (and the spot-simulation table) under
AURA_REV_OUT and writes figS9_...pdf/png into AURA_REV_FIG (default
<AURA_REV_OUT>/../manuscript/Figures/supp). Panel contents follow
manuscript/Supplementary_Figures_S9-S16.md.
"""
import os, re, glob
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUT = Path(os.environ.get("AURA_REV_OUT", "."))
FIG = Path(os.environ.get("AURA_REV_FIG", OUT.parent / "manuscript" / "Figures" / "supp"))
FIG.mkdir(parents=True, exist_ok=True)
DS = {"lymphnode": ("Lymph node", "#C2627A"), "ipf": ("IPF", "#8E6BAF"), "nsclc": ("NSCLC", "#7B8ABF")}
GRAY, RED, BLUE, ORANGE = "#888888", "#8B2E2E", "#4A6FA5", "#D98C3F"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


def letter(ax, s, x=-0.18, y=1.08):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig); print("saved", name)


def ms(x, y, **kw):
    return dict(marker="o", ms=4, lw=1.2, **kw)


# ───────────────────────── S9: exact vs legacy R2 ─────────────────────────
def fig_s9():
    d = pd.read_csv(OUT / "08_supp_sims/r2_groundtruth.csv")
    pl = d[d.planted & (d.truth_share > 0)]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, col, ttl, lab in [(axes[0], "R2_total", "Exact $R^2_{total}$ vs truth", "a"),
                              (axes[1], "R2_total_legacy", "Original approximation vs truth", "b")]:
        sig, ns = pl[pl.significant], pl[~pl.significant]
        ax.scatter(ns.truth_share, ns[col].clip(1e-5), s=10, facecolors="none", edgecolors=GRAY, lw=0.6, label="planted, not detected")
        ax.scatter(sig.truth_share, sig[col].clip(1e-5), s=14, color=RED, label="planted, FDR < 0.05")
        lo, hi = 1e-4, 1.0
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(1e-5, hi)
        ax.set_xlabel("True composition share of count variance"); ax.set_ylabel(col.replace("_", " "))
        r = (sig[col] / sig.truth_share)
        ax.set_title(ttl); ax.text(0.03, 0.95, f"significant genes: median ratio {r.median():.2f}\n(IQR {r.quantile(.25):.2f}–{r.quantile(.75):.2f})",
                                   transform=ax.transAxes, va="top", fontsize=7)
        letter(ax, lab)
    axes[0].legend(loc="lower right", frameon=False)
    # (c) real data
    ax = axes[2]
    for ds, (name, c) in DS.items():
        for f in glob.glob(str(OUT / f"01_recompute_r2/results/{ds}__*.csv")):
            t = pd.read_csv(f); s = t[t.significant]
            ax.scatter(s.R2_total_legacy.clip(1e-5), s.R2_total.clip(1e-5), s=4, color=c, alpha=0.5, lw=0)
    ax.plot([1e-5, 1], [1e-5, 1], "k--", lw=0.8); ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e-5, 1); ax.set_ylim(1e-5, 1)
    ax.set_xlabel("Original approximation"); ax.set_ylabel("Exact $R^2_{total}$"); ax.set_title("Significant genes, 19 focal types")
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="", color=c, label=n) for n, c in DS.values()], frameon=False, loc="lower right")
    letter(ax, "c")
    # (d) cumulative curves
    ax = axes[3]
    q = np.linspace(0, 1, 101)
    for ds, (name, c) in DS.items():
        curves = {"R2_total": [], "R2_total_legacy": []}
        for f in glob.glob(str(OUT / f"01_recompute_r2/results/{ds}__*.csv")):
            t = pd.read_csv(f); s = t[t.significant & ~t.spillover_suspect]
            for col in curves:
                v = np.sort(s[col].clip(0).values)[::-1]
                if v.sum() <= 0: continue
                cs = np.concatenate([[0], np.cumsum(v) / v.sum()])
                curves[col].append(np.interp(q, np.linspace(0, 1, len(cs)), cs))
        for col, ls in [("R2_total", "-"), ("R2_total_legacy", "--")]:
            ax.plot(q * 100, np.mean(curves[col], 0) * 100, ls=ls, color=c, lw=1.5)
    ax.axvline(10, color=GRAY, ls=":", lw=0.8)
    ax.set_xlabel("Gene rank percentile (%)"); ax.set_ylabel("Cumulative $R^2_{total}$ (%)"); ax.set_title("Mean cumulative curves per dataset")
    ax.legend(handles=[Line2D([0], [0], color="k", ls="-", label="exact"), Line2D([0], [0], color="k", ls="--", label="approximation")]
              + [Patch(color=c, label=n) for n, c in DS.values()], frameon=False, loc="lower right")
    letter(ax, "d")
    fig.tight_layout(); save(fig, "figS9_r2_exact_vs_legacy")


# ───────────────────────── S10: clustering errors ─────────────────────────
def fig_s10():
    A = pd.read_csv(OUT / "02_clustering_sim/sim_A.csv"); B = pd.read_csv(OUT / "02_clustering_sim/sim_B.csv")
    C = pd.read_csv(OUT / "02_clustering_sim/sim_C.csv"); D = pd.read_csv(OUT / "02_clustering_sim/granularity_D.csv")
    fig = plt.figure(figsize=(14, 7.2)); gs = fig.add_gridspec(2, 4, hspace=0.55, wspace=0.45)
    # (a) under-clustering: three stacked metrics
    ga = gs[0, 0].subgridspec(3, 1, hspace=0.15)
    g = A.groupby("frac_merged")
    for i, (m, lab, ref) in enumerate([("fpr_nominal", "FPR (p < 0.05)", 0.05), ("lambda_gc_null", "$\\lambda_{GC}$", 1.0), ("n_null_fdr", "null genes at FDR", None)]):
        ax = fig.add_subplot(ga[i]); mu, sd = g[m].mean(), g[m].std()
        ax.errorbar(mu.index * 100, mu, sd, color=RED, **ms(0, 0)); ax.set_ylabel(lab)
        if ref is not None: ax.axhline(ref, color=GRAY, ls=":", lw=0.8)
        if i < 2: ax.set_xticklabels([])
        if i == 0: ax.set_title("Under-clustering: Treg merged into Tfh"); letter(ax, "a", x=-0.35, y=1.25)
    ax.set_xlabel("Merged cells (% of focal population)")
    # (b) over-clustering
    ax = fig.add_subplot(gs[0, 1]); order = ["full", "random_2", "random_4", "composition_kmeans_2"]
    gb = B.groupby("split"); x = np.arange(len(order))
    ax.bar(x - 0.2, [gb.fpr_nominal.mean()[o] for o in order], 0.4, color=GRAY, label="FPR (p < 0.05)")
    ax.bar(x + 0.2, [gb.tpr.mean()[o] for o in order], 0.4, color=RED, label="sensitivity (FDR)")
    ax.axhline(0.05, color=GRAY, ls=":", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(["full\n(2,773)", "halves\n(1,387)", "quarters\n(693)", "composition\nhalves"], fontsize=7)
    ax.set_title("Over-clustering: focal population split"); ax.legend(frameon=False); letter(ax, "b")
    # (c) label noise
    gc_ = gs[0, 2:].subgridspec(1, 3, wspace=0.45)
    for i, (m, lab, ref) in enumerate([("fpr_nominal", "FPR (p < 0.05)", 0.05), ("lambda_gc_null", "$\\lambda_{GC}$", 1.0), ("tpr", "sensitivity (FDR)", None)]):
        ax = fig.add_subplot(gc_[i])
        for mode, c, mk in [("random", GRAY, "o"), ("confusable", RED, "s")]:
            s = C[C["mode"] == mode].groupby("flip_rate")[m]
            ax.errorbar(s.mean().index * 100, s.mean(), s.std(), color=c, marker=mk, ms=4, lw=1.2, label=mode)
        if ref is not None: ax.axhline(ref, color=GRAY, ls=":", lw=0.8)
        ax.set_xlabel("Neighbor labels flipped (%)"); ax.set_ylabel(lab)
        if i == 0: ax.set_title("Neighbor-label noise"); ax.legend(frameon=False); letter(ax, "c")
    # (d) beta cosine
    ax = fig.add_subplot(gs[1, :2])
    xs, ys, es, labs = [], [], [], []
    for f, s in A.groupby("frac_merged").beta_cosine_median: xs.append(len(xs)); ys.append(s.mean()); es.append(s.std()); labs.append(f"merge {int(f*100)}%")
    for o in order: s = B[B.split == o].beta_cosine_median; xs.append(len(xs)); ys.append(s.mean()); es.append(s.std()); labs.append({"full": "full", "random_2": "halves", "random_4": "quarters", "composition_kmeans_2": "comp. halves"}[o])
    for mode in ["random", "confusable"]:
        for f, s in C[C["mode"] == mode].groupby("flip_rate").beta_cosine_median:
            if f == 0 and mode == "confusable": continue
            xs.append(len(xs)); ys.append(s.mean()); es.append(s.std()); labs.append(f"{mode[:4]} {int(f*100)}%")
    cols = [RED] * 6 + [GRAY] * 4 + [BLUE] * 9
    ax.bar(xs, ys, yerr=es, color=cols, width=0.7, error_kw=dict(lw=0.8))
    ax.set_xticks(xs); ax.set_xticklabels(labs, rotation=60, ha="right", fontsize=6.5); ax.set_ylabel("median cos(β̂, β)")
    ax.set_title("β accuracy over detected planted genes"); letter(ax, "d", x=-0.08)
    ax.legend(handles=[Patch(color=RED, label="under-clustering"), Patch(color=GRAY, label="over-clustering"), Patch(color=BLUE, label="label noise")], frameon=False, ncol=3, loc="upper right")
    # (e) granularity
    ax = fig.add_subplot(gs[1, 2]); x = np.arange(3)
    ax.bar(x, D.n_sig, color=[GRAY, RED, GRAY], width=0.6)
    for i, (n, j, r) in enumerate(zip(D.n_sig, D.jaccard_vs_default, D.recall_of_default)):
        ax.text(i, n + 2, f"J = {j:.2f}\nrecall {r:.2f}" if i != 1 else "default", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([f"{l}\n({k} types)" for l, k in zip(D.level, D.n_types)]); ax.set_ylabel("significant Tfh genes")
    ax.set_ylim(0, D.n_sig.max() * 1.3); ax.set_title("Composition granularity (real data)"); letter(ax, "e")
    save(fig, "figS10_clustering_errors")


# ───────────────────────── S11: neighborhood scale ─────────────────────────
def fig_s11():
    S = pd.read_csv(OUT / "03_neighborhood_scale/scale_summary.csv"); R = pd.read_csv(OUT / "03_neighborhood_scale/ring_genes.csv")
    S["label"] = S.dataset.map(lambda d: DS[d][0]) + " " + S.focal
    S["color"] = S.dataset.map(lambda d: DS[d][1])
    styles = {"Tfh": "-", "Macrophage": "--", "AT2": "-", "SPP1+ Macrophages": "--", "macrophage": "-"}
    fig = plt.figure(figsize=(14, 7)); gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.4)
    knn = S[S.setting.str.startswith("knn")].copy(); knn["k"] = knn.setting.str.extract(r"k(\d+)").astype(int)
    rad = S[S.setting.str.startswith("radius")].copy()
    ax = fig.add_subplot(gs[0, 0])
    for (ds, foc), g in knn.groupby(["dataset", "focal"]):
        g = g.sort_values("k"); ax.plot(g.k, g.radius_in_spacings, ls=styles[foc], color=DS[ds][1], marker="o", ms=3, label=f"{DS[ds][0]} {foc}")
    ax.set_xscale("log"); ax.set_xticks([10, 15, 30, 50, 100]); ax.set_xticklabels([10, 15, 30, 50, 100]); ax.set_xlabel("k"); ax.set_ylabel("median distance to k-th neighbor\n(cell spacings)")
    ax.axvline(30, color=GRAY, ls=":", lw=0.8); ax.set_title("Physical scale of k"); ax.legend(frameon=False, fontsize=6, loc="upper left"); letter(ax, "a")
    ax.text(0.98, 0.04, "k = 30: 19–21 µm lymph node\n43–46 µm IPF; 236 units NSCLC", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5)
    for row, (sub, xcol, xlab, lab, ttl) in enumerate([(knn, "k", "k", "b", "k-NN vs k = 30"), (rad, "radius_in_spacings", "radius (cell spacings)", "c", "fixed radius vs k = 30")]):
        gsub = gs[row, 1:].subgridspec(1, 3, wspace=0.4) if row == 0 else gs[1, :].subgridspec(1, 4, wspace=0.4)
        for i, (m, ylab) in enumerate([("jaccard_vs_k30", "Jaccard (significant genes)"), ("spearman_logp_vs_k30", "Spearman $-\\log_{10}p$"), ("median_beta_cosine_vs_k30", "median β cosine")]):
            ax = fig.add_subplot(gsub[i])
            for (ds, foc), g in sub.groupby(["dataset", "focal"]):
                g = g.sort_values(xcol); ax.plot(g[xcol], g[m], ls=styles[foc], color=DS[ds][1], marker="o", ms=3)
            ax.set_xscale("log"); ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_ylim(0, 1.05)
            if row == 0: ax.set_xticks([10, 15, 30, 50, 100]); ax.set_xticklabels([10, 15, 30, 50, 100]); ax.axvline(30, color=GRAY, ls=":", lw=0.8)
            else: ax.set_xticks([2, 3, 5, 10, 20]); ax.set_xticklabels([2, 3, 5, 10, 20])
            if i == 0: ax.set_title(ttl); letter(ax, lab)
        if row == 1:
            ax = fig.add_subplot(gsub[3]); ct = R.groupby(["dataset", "focal"]).strongest_ring.value_counts().unstack(fill_value=0)
            short = {"Tfh": "Tfh", "Macrophage": "Mac", "AT2": "AT2", "SPP1+ Macrophages": "SPP1+ Mac", "macrophage": "Mac"}
            labs = [f"{ {'lymphnode':'LN','ipf':'IPF','nsclc':'NSCLC'}[d] }\n{short[f]}" for d, f in ct.index]; bottom = np.zeros(len(ct))
            for ring, c, l in [(0, RED, "0–3 spacings (contact)"), (1, ORANGE, "3–8"), (2, BLUE, "8–20")]:
                v = ct.get(ring, 0); ax.bar(range(len(ct)), v, bottom=bottom, color=c, label=l); bottom += v
            ax.set_xticks(range(len(ct))); ax.set_xticklabels(labs, fontsize=6.5); ax.set_ylabel("ring-significant genes"); ax.set_title("Ring with smallest p-value")
            ax.legend(frameon=False, fontsize=6); letter(ax, "d")
    save(fig, "figS11_neighborhood_scale")


# ───────────────────────── S12: panel size & centering ─────────────────────────
def fig_s12():
    P = pd.read_csv(OUT / "04_panel_size/panel_runs.csv"); P = P[P["size"] < 4624]
    L = pd.read_csv(OUT / "08_supp_sims/centering_leakage.csv"); Cn = pd.read_csv(OUT / "07_centering/centering_summary.csv")
    fig = plt.figure(figsize=(14, 7)); gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.4)
    strat = {"random": (GRAY, "o", "random"), "top_expressed": (BLUE, "s", "top expressed"), "marker_enriched": (ORANGE, "^", "marker-enriched")}
    for i, (m, ylab, lab) in enumerate([("recall", "recall of full-panel genes", "a"), ("precision", "precision vs full panel", "b")]):
        ax = fig.add_subplot(gs[0, i]); sub = P[P.center == "mean"]
        for foc, ls in [("Tfh", "-"), ("Macrophage", "--")]:
            for st, (c, mk, nm) in strat.items():
                g = sub[(sub.focal == foc) & (sub.strategy == st)].groupby("size")[m]
                ax.errorbar(g.mean().index, g.mean(), g.std(), color=c, marker=mk, ls=ls, ms=4, lw=1.1)
        ax.set_xscale("log"); ax.set_xticks([150, 343, 600]); ax.set_xticklabels([150, 343, 600]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_xlabel("panel size (genes)"); ax.set_ylabel(ylab); ax.set_ylim(0, 1.05)
        if i == 0:
            ax.set_title("Lymph node panel subsampled (mean centering)")
            ax.legend(handles=[Line2D([0], [0], color=c, marker=mk, ls="", label=nm) for c, mk, nm in strat.values()] + [Line2D([0], [0], color="k", ls="-", label="Tfh"), Line2D([0], [0], color="k", ls="--", label="Macrophage")], frameon=False, fontsize=6, ncol=2)
        letter(ax, lab)
    ax = fig.add_subplot(gs[0, 2]); sub = P[P.focal == "Macrophage"]
    for cen, ls in [("mean", "-"), ("median", "--"), ("nonmarker", ":")]:
        for st, (c, mk, nm) in strat.items():
            g = sub[(sub.center == cen) & (sub.strategy == st)].groupby("size").median_R2_total_ratio
            ax.errorbar(g.mean().index, g.mean(), g.std(), color=c, marker=mk, ls=ls, ms=4, lw=1.1)
    ax.axhline(1, color=GRAY, ls=":", lw=0.8); ax.set_xscale("log"); ax.set_xticks([150, 343, 600]); ax.set_xticklabels([150, 343, 600]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("panel size (genes)"); ax.set_ylabel("panel / full-panel $R^2_{total}$ (median)"); ax.set_title("R² bias by centering (macrophages)")
    ax.legend(handles=[Line2D([0], [0], color="k", ls=ls, label=l) for ls, l in [("-", "mean"), ("--", "median"), (":", "non-marker reference")]], frameon=False, fontsize=6); letter(ax, "c")
    ax = fig.add_subplot(gs[1, 0]); g = L.groupby("centering"); order = ["mean", "median", "reference"]
    ax.bar(range(3), [g.fpr_null.mean()[o] for o in order], yerr=[g.fpr_null.std()[o] for o in order], color=[RED, ORANGE, BLUE], width=0.6)
    for i, o in enumerate(order): ax.text(i, g.fpr_null.mean()[o] + 0.03, f"{g.n_null_fdr.mean()[o]:.0f}/70 at FDR", ha="center", fontsize=7)
    ax.axhline(0.05, color=GRAY, ls=":", lw=0.8); ax.set_xticks(range(3)); ax.set_xticklabels(["per-cell mean", "per-cell median", "reference genes"]); ax.set_ylabel("FPR among null genes"); ax.set_ylim(0, 1.2)
    ax.set_title("Centering leakage (simulation)"); letter(ax, "d")
    ax = fig.add_subplot(gs[1, 1:]); Cn = Cn[Cn.variant != "mean"].copy()
    Cn["lab"] = Cn.dataset.map(lambda d: DS[d][0]) + " " + Cn.focal; piv = Cn.pivot(index="variant", columns="lab", values="jaccard_vs_mean")
    vorder = ["median", "nonmarker", "mean+samples", "trend500", "nonmarker+trend500"]; piv = piv.loc[vorder]
    im = ax.imshow(piv.values, cmap="viridis", vmin=0.4, vmax=1, aspect="auto")
    ax.set_yticks(range(len(vorder))); ax.set_yticklabels(["median", "non-marker reference", "within-sample", "trend 500 µm", "reference + trend"])
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=45, ha="right", fontsize=6.5)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]): ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=6, color="w" if piv.values[i, j] < 0.75 else "k")
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="Jaccard vs default centering"); ax.set_title("Real data: significant genes vs default (mean) centering"); letter(ax, "e", x=-0.1)
    save(fig, "figS12_panel_size_centering")


# ───────────────────────── S13: dispersion ─────────────────────────
def fig_s13():
    D = pd.read_csv(OUT / "06_dispersion/dispersion_compare.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw=dict(wspace=0.4))
    ax = axes[0]
    for f, (ds, foc) in [("phi_trend__lymphnode__Tfh.csv", ("lymphnode", "Tfh")), ("phi_trend__ipf__AT2.csv", ("ipf", "AT2")), ("phi_trend__nsclc__macrophage.csv", ("nsclc", "macrophage"))]:
        t = pd.read_csv(OUT / "06_dispersion" / f).sort_values("mean"); c = DS[ds][1]
        ax.scatter(t["mean"], t.phi_gene, s=4, color=c, alpha=0.6, lw=0, label=f"{DS[ds][0]} {foc}"); ax.axhline(t.phi_shared.iloc[0], color=c, ls="--", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("gene mean"); ax.set_ylabel("dispersion φ (gene-specific; dashed = shared)"); ax.set_title("Dispersion estimates"); ax.legend(frameon=False, fontsize=6); letter(ax, "a")
    labs = [f"{DS[d][0]}\n{f}" for d, f in zip(D.dataset, D.focal)]; x = np.arange(len(D))
    ax = axes[1]; ax.bar(x - 0.2, D.jaccard_raw_fdr, 0.4, color=GRAY, label="Jaccard, FDR sets (pre-filter)"); ax.bar(x + 0.2, D.spearman_logp, 0.4, color=RED, label="Spearman $-\\log_{10}p$")
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=6, rotation=45, ha="right"); ax.set_ylim(0, 1.05); ax.set_title("p-values: shared vs gene-specific φ"); ax.legend(frameon=False, fontsize=6, loc="lower right"); letter(ax, "b")
    ax = axes[2]; ax.bar(x - 0.2, D.n_sig_shared, 0.4, color=GRAY, label="shared φ"); ax.bar(x + 0.2, D.n_sig_gene, 0.4, color=RED, label="gene-specific φ")
    for i, j in enumerate(D.jaccard_sig): ax.text(i, max(D.n_sig_shared[i], D.n_sig_gene[i]) + 5, f"J={j:.2f}", ha="center", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=6, rotation=45, ha="right"); ax.set_ylabel("significant genes after excess filter"); ax.set_title("Final gene sets"); ax.legend(frameon=False, fontsize=6); letter(ax, "c")
    save(fig, "figS13_dispersion")


# ───────────────────────── S14: IPF downsampling ─────────────────────────
def fig_s14():
    R = pd.read_csv(OUT / "05_ipf_downsample/downsample_runs.csv")
    order = ["AT2", "SPP1+ Macrophages", "Basal", "Capillary", "Interstitial Macrophages", "Alveolar FBs"]
    short = ["AT2", "SPP1+ Mac", "Basal", "Capillary", "Int. Mac", "Alv. FBs"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw=dict(wspace=0.35)); x = np.arange(6)
    full = R[R.analysis == "full_ipf_vs_healthy"].set_index("focal"); down = R[R.analysis == "downsampled_ipf_vs_healthy"].groupby("focal")
    hs = R[R.analysis == "ceiling_healthy_split_half"].groupby("focal").median_cosine; isp = R[R.analysis == "ceiling_ipf_split_half_matched"].groupby("focal").median_cosine
    ax = axes[0]
    ax.bar(x - 0.27, [full.pct_sig_ipf[o] * 100 for o in order], 0.27, color=ORANGE, label="IPF, all cells")
    ax.bar(x, [down.pct_sig_ipf.mean()[o] * 100 for o in order], 0.27, yerr=[down.pct_sig_ipf.std()[o] * 100 for o in order], color="#F2C9A0", label="IPF, downsampled to healthy size")
    ax.bar(x + 0.27, [full.pct_sig_healthy[o] * 100 for o in order], 0.27, color=BLUE, label="healthy")
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=7, rotation=30, ha="right"); ax.set_ylabel("significant genes (%)"); ax.set_title("Fraction significant"); ax.legend(frameon=False, fontsize=6); letter(ax, "a")
    ax = axes[1]
    ax.bar(x - 0.27, [full.median_cosine[o] for o in order], 0.27, color=RED, label="IPF vs healthy")
    ax.bar(x, [hs.mean()[o] for o in order], 0.27, yerr=[hs.std()[o] for o in order], color=BLUE, label="healthy split-half ceiling")
    ax.bar(x + 0.27, [isp.mean()[o] for o in order], 0.27, yerr=[isp.std()[o] for o in order], color=ORANGE, label="IPF split-half ceiling (matched n)")
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=7, rotation=30, ha="right"); ax.set_ylabel("median β cosine"); ax.set_title("Cosine vs noise ceilings"); ax.legend(frameon=False, fontsize=6); letter(ax, "b")
    ax = axes[2]
    ax.errorbar(x, [down.median_cosine.mean()[o] for o in order], [down.median_cosine.std()[o] for o in order], fmt="o", color=RED, label="downsampled IPF vs healthy")
    ax.scatter(x, [full.median_cosine[o] for o in order], marker="_", s=200, color="k", label="full IPF vs healthy")
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=7, rotation=30, ha="right"); ax.set_ylabel("median β cosine"); ax.set_ylim(0, 0.8); ax.set_title("IPF–healthy cosine after downsampling"); ax.legend(frameon=False, fontsize=6); letter(ax, "c")
    save(fig, "figS14_ipf_downsampling")


# ───────────────────────── S15: Visium HD ─────────────────────────
def fig_s15():
    S = pd.read_csv(OUT / "visiumhd_crc/summary.csv"); Rb = pd.read_csv(OUT / "visiumhd_crc_robustness/robustness_runs.csv")
    order = ["Macrophage", "CAF", "Endothelial", "Plasma", "Goblet", "Tumor III"]; S = S.set_index("focal").loc[order]
    fig = plt.figure(figsize=(15, 7.2)); gs = fig.add_gridspec(2, 3, hspace=0.6, wspace=0.55, width_ratios=[1, 1, 1.3]); x = np.arange(6)
    ax = fig.add_subplot(gs[0, 0]); clean = S.n_sig - S.n_spillover_contribution
    ax.bar(x, clean, color=RED, label="retained"); ax.bar(x, S.n_spillover_contribution, bottom=clean, color=ORANGE, label="spillover-suspect")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([f"{o}\n({n:,} cells)" for o, n in zip(order, S.n_cells)], fontsize=6.5, rotation=30, ha="right"); ax.set_ylabel("significant genes"); ax.set_title("Significant genes per focal type"); ax.legend(frameon=False); letter(ax, "a")
    ax = fig.add_subplot(gs[0, 1]); ax.bar(x, S.n_spillover_contribution / S.n_sig, color=ORANGE, width=0.6)
    ax.set_xticks(x); ax.set_xticklabels(order, fontsize=7, rotation=30, ha="right"); ax.set_ylabel("fraction of hits flagged as spillover"); ax.set_ylim(0, 1); ax.set_title("Spillover burden"); letter(ax, "b")
    M = pd.read_csv(OUT / "visiumhd_crc/results/Macrophage.csv"); Mc = M[M.significant & ~M.spillover_suspect_contribution].copy()
    bcols = [c for c in M.columns if c.startswith("beta_")]; Mc = Mc.sort_values("beta_Tumor", ascending=False)
    ax = fig.add_subplot(gs[0, 2]); v = Mc[bcols].values; lim = np.abs(v).max()
    im = ax.imshow(v, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto"); ax.set_yticks(range(len(Mc))); ax.set_yticklabels(Mc.gene, fontsize=6.5)
    ax.set_xticks(range(len(bcols))); ax.set_xticklabels([c[5:] for c in bcols], rotation=60, ha="right", fontsize=6.5); ax.set_title("Macrophage: spillover-clean genes, β")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="β"); letter(ax, "c")
    ax = fig.add_subplot(gs[1, 0]); rows = []
    for o in order:
        t = pd.read_csv(OUT / f"visiumhd_crc/results/{o.replace(' ', '_')}.csv"); t = t[t.significant & ~t.spillover_suspect_contribution]
        rows.append(t.driver_axis_contribution.value_counts(normalize=True).rename(o))
    drv = pd.concat(rows, axis=1).fillna(0).T; drv = drv[drv.sum().sort_values(ascending=False).index]
    bottom = np.zeros(6); cmap = plt.get_cmap("tab10")
    for j, c in enumerate(drv.columns): ax.bar(x, drv[c], bottom=bottom, color=cmap(j % 10), label=c); bottom += drv[c].values
    ax.set_xticks(x); ax.set_xticklabels(order, fontsize=7, rotation=30, ha="right"); ax.set_ylabel("fraction of retained genes"); ax.set_title("Dominant driver of retained genes"); ax.legend(frameon=False, fontsize=5.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.32)); letter(ax, "d")
    ax = fig.add_subplot(gs[1, 1:]); ax.set_position([0.5, 0.11, 0.42, 0.3]); var = ["nonmarker", "trend_1000", "trend_500", "trend_250", "nonmarker+trend500"]
    piv = Rb[Rb.variant.isin(var)].pivot(index="variant", columns="focal", values="retained_clean_of_default").loc[var, order]
    im = ax.imshow(piv.values, cmap="viridis", vmin=0, vmax=1, aspect="auto"); ax.set_yticks(range(len(var))); ax.set_yticklabels(["non-marker reference centering", "trend 1000 µm", "trend 500 µm", "trend 250 µm", "reference + trend 500 µm"])
    nclean = Rb[Rb.variant == "default"].set_index("focal").n_clean.loc[order]
    ax.set_xticks(range(6)); ax.set_xticklabels([f"{o}\n({n} clean hits)" for o, n in zip(order, nclean)], fontsize=6.5)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]): ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="w" if piv.values[i, j] < 0.6 else "k")
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="fraction of default clean hits retained"); ax.set_title("Robustness of retained hits"); letter(ax, "e", x=-0.12)
    save(fig, "figS15_visiumhd_crc")


# ───────────────────────── S16: spot simulation ─────────────────────────
def fig_s16():
    txt = open(OUT / "08_supp_sims/spot_simulation_table.txt").read().splitlines()
    rows = []
    for l in txt:
        m = re.match(r"^(.*?)\s+([01]\.\d{3})\s+(\d+\.\d{2})\s+(\d+)\s+(\S+)\s*$", l)
        if m: rows.append(dict(scenario=m.group(1).strip(), fpr=float(m.group(2)), lam=float(m.group(3)), nfdr=int(m.group(4)), tpr=m.group(5)))
    T = pd.DataFrame(rows)
    a = T[~T.scenario.str.contains("PCs")]; b = T[T.scenario.str.contains("PCs")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw=dict(width_ratios=[1.1, 1], wspace=0.9))
    for ax, sub, lab, ttl in [(axes[0], a, "a", "Spot-level test: own composition exact or deconvolved"), (axes[1], b, "b", "Own-expression principal components as covariates")]:
        y = np.arange(len(sub))[::-1]; cols = [RED if f > 0.1 else GRAY for f in sub.fpr]
        ax.barh(y, sub.fpr, color=cols, height=0.65); ax.axvline(0.05, color="k", ls=":", lw=0.8)
        for yi, (f, lam, n, t) in zip(y, zip(sub.fpr, sub.lam, sub.nfdr, sub.tpr)):
            ax.text(min(f, 1.0) + 0.02, yi, f"λ={lam:.2f}, {n}/300 at FDR" + (f", TPR {t}" if t != "-" else ""), va="center", fontsize=6.5)
        ax.set_yticks(y); ax.set_yticklabels(sub.scenario.str.strip(), fontsize=7); ax.set_xlim(0, 1.6); ax.set_xlabel("false positive rate among null genes (p < 0.05)"); ax.set_title(ttl); letter(ax, lab, x=-0.9 if lab == "a" else -0.75)
    save(fig, "figS16_visium_spot_simulation")


if __name__ == "__main__":
    for f in [fig_s9, fig_s10, fig_s11, fig_s12, fig_s13, fig_s14, fig_s15, fig_s16]:
        try:
            f()
        except Exception as e:
            import traceback; traceback.print_exc(); print("FAILED", f.__name__)
