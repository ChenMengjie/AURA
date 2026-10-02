"""
Build a cell-level AnnData for the Visium HD CRC (P2) section.

Inputs (see visiumhd_crc_config.py for sources):
  segmented_outputs/filtered_feature_cell_matrix.h5
  segmented_outputs/cell_segmentations.geojson
  segmented_outputs/spatial/scalefactors_json.json
  P2CRC_Metadata.parquet  (RCTD labels of 8-µm bins)

Each cell centroid (full-resolution pixels) is mapped to the study's bin
coordinates (low-res pixels: X = x * s, Y = -y * s) and takes the label of
the nearest bin. Cells in singlet or doublet_certain bins take
DeconvolutionLabel1; others are 'Unassigned' (kept as neighbors, never
focal). `focal_ok` marks cells in singlet bins.

Usage: python visiumhd_crc_prepare.py <segmented_outputs dir> <metadata parquet> <out.h5ad>
"""

import json
import sys

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial import cKDTree

from visiumhd_crc_config import LINEAGE_MAP


def main(seg_dir, meta_path, out_path):
    adata = sc.read_10x_h5(f"{seg_dir}/filtered_feature_cell_matrix.h5")
    adata.var_names_make_unique()
    sf = json.load(open(f"{seg_dir}/spatial/scalefactors_json.json"))
    feats = json.load(open(f"{seg_dir}/cell_segmentations.geojson"))["features"]
    ids = np.array([f["properties"]["cell_id"] for f in feats])
    cen = np.array([np.asarray(f["geometry"]["coordinates"][0])[:-1].mean(axis=0)
                    for f in feats])
    pos = pd.DataFrame(cen, columns=["px", "py"],
                       index=[f"cellid_{i:09d}-1" for i in ids])
    missing = ~adata.obs_names.isin(pos.index)
    print(f"cells without polygon: {missing.sum()}")
    adata = adata[~missing].copy()
    pos = pos.loc[adata.obs_names]

    um = sf["microns_per_pixel"]
    adata.obs["x_centroid"] = pos.px.values * um
    adata.obs["y_centroid"] = pos.py.values * um

    meta = pd.read_parquet(meta_path)
    s = sf["tissue_lowres_scalef"]
    q = np.column_stack([pos.px.values * s, -pos.py.values * s])
    d, j = cKDTree(meta[["X", "Y"]].values).query(q)
    b = meta.iloc[j]
    half_bin = 0.5 * 8 / um * s
    print(f"median distance to bin centre {np.median(d):.3f} "
          f"(half bin {half_bin:.3f}); beyond half-diagonal: "
          f"{np.mean(d > half_bin * np.sqrt(2)):.4f}")

    cls = b.DeconvolutionClass.fillna("none").values
    lab1 = b.DeconvolutionLabel1.fillna("Unassigned").values
    usable = np.isin(cls, ["singlet", "doublet_certain"]) & (d <= half_bin * np.sqrt(2))
    fine = np.where(usable, lab1, "Unassigned")
    adata.obs["cell_type"] = fine
    adata.obs["lineage"] = [LINEAGE_MAP.get(t, "Unassigned") if t != "Unassigned"
                            else "Unassigned" for t in fine]
    adata.obs["deconv_class"] = cls
    adata.obs["focal_ok"] = (cls == "singlet") & usable
    adata.obs["dist_to_bin"] = d
    print(adata.obs.lineage.value_counts().to_string())
    print("focal-eligible cells:", int(adata.obs.focal_ok.sum()))
    adata.write_h5ad(out_path, compression="gzip")


if __name__ == "__main__":
    main(*sys.argv[1:4])
