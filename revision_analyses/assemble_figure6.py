"""
Assemble revised Figure 6 on top of the submitted Illustrator figure.

  a  new cumulative-R2 curves (vector, from the regenerated notext crop) placed
     into the original axes boxes; text boxes rewritten with the new values
  b, c  unchanged (taken from the submitted figure)
  d  redrawn: core vs peripheral regulatory fractions, new order by odds ratio
  e  redrawn: odds ratios on a log scale (version A; Haldane-Anscombe for the
     zero cell; * Fisher p < 0.05)

Everything is drawn in Arial with TrueType embedding so the result stays
editable in Illustrator. Geometry (axes boxes, label positions) is taken from
the submitted Figure6.pdf.

Usage: python assemble_figure6.py <Figures dir> <out.pdf>
"""

import sys
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymupdf

sys.path.insert(0, str(Path(__file__).parent))
from fig6e_log_odds import recover_counts, log_or  # noqa: E402

matplotlib.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42,
                            "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                            "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
                            "axes.linewidth": 0.5, "xtick.major.width": 0.5,
                            "ytick.major.width": 0.5, "xtick.major.size": 2,
                            "ytick.major.size": 2})
INK = (0.14, 0.12, 0.13)
COL = {"Lymph node": (0.761, 0.384, 0.478), "IPF": (0.556, 0.421, 0.686),
       "NSCLC": (0.485, 0.544, 0.75)}
NAME = {"Tfh": "Tfh", "T_naive": "T naive", "T_CD8": "CD8 T", "Treg": "Treg",
        "FDC": "FDC", "GC_LZ": "GC LZ", "GC_DZ": "GC DZ",
        "Macrophage": "Macrophage", "Endothelial": "Endothelial",
        "SPP1+Mac": "SPP1$^{+}$ Mac", "Int.Mac": "Int. Mac", "Alv.FBs": "Alv. FBs",
        "Basal": "Basal", "Capillary": "Capillary", "AT2": "AT2",
        "T CD4 mem": "CD4 mem T", "Fibroblast": "Fibroblast"}
GROUPS = [("T cell", ["Tfh", "T CD4 mem", "T_naive", "T_CD8", "Treg"]),
          ("GC/B cell", ["FDC", "GC_LZ", "GC_DZ"]),
          ("Macrophage", ["SPP1+Mac", "Macrophage", "Int.Mac"]),
          ("Fibroblast", ["Alv.FBs", "Fibroblast"]),
          ("Endothelial", ["Endothelial", "Capillary"]),
          ("Epithelial", ["Basal", "AT2"])]

# --- geometry of the submitted figure (PDF points, origin top-left) --------
PAGE = (535.7727, 295.6844)
A_BOXES = [(36.37, 20.5, 167.7, 82.7), (36.37, 107.44, 167.7, 169.65),
           (36.37, 196.38, 167.7, 258.59)]          # spine boxes, y 0..105
CROP_BOXES = [(90.72, 25.92 + k * 207.36, 419.04, 181.44 + k * 207.36)
              for k in range(3)]                     # same data range in crop
A_TEXT = [((94.51, 68.34), "51.5"), ((102.04, 154.85), "47.0"),
          ((102.04, 242.23), "58.6")]
A_VALUES = ["33.6 ± 8.8 %", "36.7 ± 5.1 %", "42.0 ± 6.2 %"]
D_AX = (396.61, 20.5, 527.93, 124.18)
E_AX = (399.61, 161.91, 530.93, 265.59)
D_REGION = (367.6, 8.0, PAGE[0], 151.5)            # keeps the "d)" letter
E_REGION = (367.6, 152.5, PAGE[0], 285.0)          # keeps the "e)" letter


def fig_axes(fig, box):
    x0, y0, x1, y1 = box
    W, H = PAGE
    return fig.add_axes([x0 / W, 1 - y1 / H, (x1 - x0) / W, (y1 - y0) / H])


def load_table(csv):
    d = pd.read_csv(csv)
    rows = []
    for _, r in d.iterrows():
        a, b, c, dd = recover_counts(r)
        lor, _, corr = log_or(a, b, c, dd)
        rows.append(dict(label=r.label, dataset=r.dataset,
                         core=100 * r.core_reg_frac, periph=100 * r.periph_reg_frac,
                         odds=np.exp(lor), corrected=corr, p=r.fisher_p,
                         fisher_or=r.fisher_or))
    return pd.DataFrame(rows)


def draw_d(fig, t):
    ax = fig_axes(fig, D_AX)
    order = []
    for ds in ["Lymph node", "IPF", "NSCLC"]:
        sub = t[t.dataset == ds].sort_values("fisher_or", ascending=False)
        order += list(sub.itertuples())
    w = 0.35
    for i, r in enumerate(order):
        ax.bar(i - w / 2, r.core, w, color=COL[r.dataset], edgecolor="white", lw=0.3)
        ax.bar(i + w / 2, r.periph, w, color=COL[r.dataset], alpha=0.4,
               edgecolor="white", lw=0.3)
    n_ln, n_ipf = (t.dataset == "Lymph node").sum(), (t.dataset == "IPF").sum()
    for xv in (n_ln - 0.5, n_ln + n_ipf - 0.5):
        ax.axvline(xv, color=(0.71, 0.72, 0.73), lw=0.5, ls=(0, (2.96, 1.28)))
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(0, 88)
    ax.set_yticks(range(0, 81, 10))
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([NAME[r.label] for r in order], rotation=45, ha="right",
                       rotation_mode="anchor", fontsize=5, color=INK)
    ax.tick_params(axis="y", labelsize=6, colors=INK)
    ax.tick_params(axis="x", length=1.5, pad=1)
    ax.set_ylabel("Percentage of regulatory genes", fontsize=6, color=INK, labelpad=2)
    ax.set_title("Regulatory gene enrichment", fontsize=6, color=INK, pad=3)
    for s in ax.spines.values():
        s.set_color(INK)
    # dataset labels in clear space above the lower bars of each block
    for x, y, txt in ((4.6, 40, "Lymph node\n4,624 genes"),
                      (12.55, 61, "IPF\n343 genes"),
                      (16.45, 52, "NSCLC\n960 genes")):
        ax.text(x, y, txt, fontsize=5, color=INK, ha="center", va="center",
                linespacing=1.1)
    # legend: gene tier
    from matplotlib.patches import Patch
    leg = ax.legend(handles=[Patch(color=(0.503, 0.51, 0.52), label="Core"),
                             Patch(color=(0.503, 0.51, 0.52), alpha=0.4,
                                   label="Peripheral")],
                    title="Gene tier", fontsize=5, title_fontsize=5,
                    loc="upper left", bbox_to_anchor=(0.07, 0.985), frameon=True,
                    handlelength=1.2, handleheight=0.6, borderpad=0.3,
                    labelspacing=0.2, handletextpad=0.4)
    leg.get_frame().set_linewidth(0.5)
    leg.get_frame().set_edgecolor((0.82, 0.83, 0.83))
    leg._legend_box.align = "left"


def draw_e(fig, t):
    ax = fig_axes(fig, E_AX)
    y, rows, seps, labels = 0, [], [], []
    for gname, members in GROUPS:
        y0 = y
        sub = t[t.label.isin(members) | ((t.label == "Macrophage") & (gname == "Macrophage"))
                | ((t.label == "Endothelial") & (gname == "Endothelial"))]
        sub = sub.sort_values("odds", ascending=False)
        for r in sub.itertuples():
            rows.append((y, r))
            y += 1
        labels.append((gname, (y0 + y - 1) / 2))
        seps.append(y - 0.5)
    lo, hi = 1 / 8, 512
    for yy, r in rows:
        ax.barh(yy, np.log2(r.odds), left=0, height=0.7, color=COL[r.dataset],
                edgecolor="white", lw=0.3)
        tag = ("*" if r.p < 0.05 else "") + ("†" if r.corrected else "")
        if tag:
            ax.text(max(np.log2(r.odds), 0) + 0.15, yy, tag, fontsize=5.5,
                    va="center", ha="left", color=INK)
    for s_ in seps[:-1]:
        ax.axhline(s_, color=(0.71, 0.72, 0.73), lw=0.5, ls=(0, (2.96, 1.28)))
    ax.axvline(0, color=INK, lw=0.5, ls=(0, (1.85, 0.8)))
    for gname, yc in labels:
        ax.text(np.log2(96), yc, gname, fontsize=5.9, color=INK, ha="center",
                va="center")
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(np.log2(lo), np.log2(hi))
    ticks = [-2, 0, 2, 4, 6, 8]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["1/4", "1", "4", "16", "64", "256"], fontsize=6, color=INK)
    ax.set_yticks([yy for yy, _ in rows])
    ax.set_yticklabels([NAME[r.label] for _, r in rows], fontsize=5, color=INK)
    for lab, (_, r) in zip(ax.get_yticklabels(), rows):
        lab.set_color(INK)
    ax.tick_params(axis="y", length=1.5, pad=1)
    ax.tick_params(axis="x", pad=1)
    ax.set_xlabel("Odds ratio (core vs. peripheral), log scale", fontsize=6,
                  color=INK, labelpad=1.5)
    ax.set_title("Regulatory enrichment by lineage", fontsize=6, color=INK, pad=3)
    for s in ax.spines.values():
        s.set_color(INK)


def draw_a_text(fig):
    W, H = PAGE
    for ((x, ybase), _), val in zip(A_TEXT, A_VALUES):
        # line 1 "Top 10% genes R²" (6 pt), line 2 value centred beneath it
        fig.text(x / W, 1 - ybase / H, "Top 10% genes R²", fontsize=6,
                 color=INK, ha="left", va="baseline")
        width = 50.3                                   # line-1 width in points
        fig.text((x + width / 2) / W, 1 - (ybase + 7.2) / H, val, fontsize=6,
                 color=INK, ha="center", va="baseline")


def main(fig_dir, out_pdf):
    fig_dir = Path(fig_dir)
    src = pymupdf.open(fig_dir / "Figure6.pdf")
    crop = pymupdf.open(fig_dir / "panels/crops/Fig6_panel_a_cumulative_curves_notext.pdf")
    # the crop's tick labels lie outside the placed clip but would still be
    # embedded (DejaVu); remove all text so only the curves come across
    cp = crop[0]
    cp.add_redact_annot(cp.rect)
    cp.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE,
                        graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
                        text=pymupdf.PDF_REDACT_TEXT_REMOVE)
    cp.clean_contents(sanitize=True)
    res = crop.xref_get_key(cp.xref, "Resources")
    if res[0] == "xref":                      # drop the now-unused font dict
        crop.xref_set_key(int(res[1].split()[0]), "Font", "null")
    else:
        crop.xref_set_key(cp.xref, "Resources/Font", "null")
    crop = pymupdf.open("pdf", crop.tobytes(garbage=3))
    table = load_table(fig_dir / "panels/core_peripheral_newR2.csv")

    # remove (not just cover) the content being replaced: panel a plot
    # interiors (curves, reference lines, text boxes; spines stay) and the
    # whole of panels d and e (panel letters stay)
    sp = src[0]
    a_inner = [pymupdf.Rect(b) + (0.6, 0.6, -0.6, -0.6) for b in A_BOXES]
    for r in a_inner + [pymupdf.Rect(D_REGION), pymupdf.Rect(E_REGION)]:
        sp.add_redact_annot(r, fill=False)
    sp.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_REMOVE,
                        graphics=pymupdf.PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED,
                        text=pymupdf.PDF_REDACT_TEXT_REMOVE)
    src = pymupdf.open("pdf", src.tobytes(garbage=3))

    out = pymupdf.open()
    page = out.new_page(width=PAGE[0], height=PAGE[1])
    page.show_pdf_page(page.rect, src, 0)

    # panel a: new curves into the original axes boxes
    for box, cbox, inner in zip(A_BOXES, CROP_BOXES, a_inner):
        sx = (cbox[2] - cbox[0]) / (box[2] - box[0])
        sy = (cbox[3] - cbox[1]) / (box[3] - box[1])
        cclip = pymupdf.Rect(cbox[0] + 0.6 * sx, cbox[1] + 0.6 * sy,
                             cbox[2] - 0.6 * sx, cbox[3] - 0.6 * sy)
        page.show_pdf_page(inner, crop, 0, clip=cclip, keep_proportion=False)

    # vector overlay drawn in page coordinates
    fig = plt.figure(figsize=(PAGE[0] / 72, PAGE[1] / 72))
    fig.patch.set_alpha(0)
    draw_a_text(fig)
    draw_d(fig, table)
    draw_e(fig, table)
    buf = BytesIO()
    fig.savefig(buf, format="pdf", transparent=True)
    plt.close(fig)
    overlay = pymupdf.open("pdf", buf.getvalue())
    page.show_pdf_page(page.rect, overlay, 0)

    out.save(out_pdf, garbage=3, deflate=True)
    print("wrote", out_pdf)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
