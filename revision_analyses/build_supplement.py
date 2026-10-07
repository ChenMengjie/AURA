"""
Rebuild the supplementary-figures PDF for the revision.

Starts from the submitted supplement (figure page, legend page, ... for S1–S8),
replaces the figure pages whose panels depend on R2_total (S3, S4, S5, S8),
patches the S8 legend's rho values, and appends S9–S16 (figure PDFs from
09_supp_figures.py, legend text from manuscript/Supplementary_Figures_S9-S16.md).
"""
import os, re, pymupdf
TNR = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
TNRB = "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"
D = os.environ["AURA_PROJECT_DIR"]  # project folder containing Manuscript/ and Revision/
REV = f"{D}/Revision/manuscript"; RES = os.path.expanduser("~/SpatialHeterogeneity/results")
BASE = f"{D}/Manuscript/Submit_to_GB/AURA_Supplementary_Figures.pdf"
REPLACE = {4: f"{RES}/lymph_node/conditional_heterogeneity.pdf",      # S3 (page 5)
           6: f"{RES}/lymph_node/figS_macrophage_panel.pdf",          # S4 (page 7)
           8: f"{RES}/IPF/figS5_ipf_heatmaps.pdf",                    # S5 (page 9)
           14: f"{RES}/figS8_core_peripheral.pdf"}                    # S8 (page 15)
FILES = {9: "figS9_r2_exact_vs_legacy", 10: "figS10_clustering_errors", 11: "figS11_neighborhood_scale",
         12: "figS12_panel_size_centering", 13: "figS13_dispersion", 14: "figS14_ipf_downsampling",
         15: "figS15_visiumhd_crc", 16: "figS16_visium_spot_simulation"}

supp = pymupdf.open(BASE)
for pageno, path in REPLACE.items():
    new = pymupdf.open(path); supp.delete_page(pageno); supp.insert_pdf(new, from_page=0, to_page=0, start_at=pageno)
# S8 legend: rho values
pg = supp[15]
for b in pg.get_text("dict")["blocks"]:
    for l in b.get("lines", []):
        for sp in l["spans"]:
            if "0.17, 0.33, 0.79" in sp["text"]:
                pg.add_redact_annot(pymupdf.Rect(sp["bbox"]), fill=(1, 1, 1)); org, size = sp["origin"], sp["size"]
pg.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE)
pg.insert_text(org, " = 0.28, 0.32, 0.82).", fontsize=size, fontname="tiro")
# S9–S16
md = open(f"{REV}/Supplementary_Figures_S9-S16.md").read()
blocks = re.findall(r"\*\*Supplementary Figure (\d+)\.\*\*\n(.*?)(?=\n\*\*Supplementary Figure|\n---)", md, flags=re.S)
assert len(blocks) == 8, len(blocks)
for num, body in blocks:
    num = int(num); fig = pymupdf.open(f"{REV}/Figures/supp/{FILES[num]}.pdf")
    supp.insert_pdf(fig, from_page=0, to_page=0)
    pg = supp.new_page(width=612, height=792)
    text = " ".join(l.strip() for l in body.strip().splitlines())
    pg.insert_text((72, 80), f"Supplementary Figure {num}.", fontsize=10, fontname="TNRB", fontfile=TNRB)
    rc = pg.insert_textbox(pymupdf.Rect(72, 90, 540, 760), text, fontsize=10, fontname="TNR", fontfile=TNR, lineheight=1.3)
    assert rc >= 0, f"legend {num} overflowed by {rc}"
out = f"{REV}/AURA_Supplementary_Figures_rev.pdf"
supp.save(out + ".tmp", garbage=3, deflate=True); supp.close(); os.replace(out + ".tmp", out)
chk = pymupdf.open(out); print("pages", len(chk))
for i in range(1, len(chk), 2): print(" ", chk[i].get_text().strip().splitlines()[0])
