"""
10 — Refit the six IPF healthy-only focal types with v0.2 defaults and write the
exact R2_total into the project's results/IPF/healthy tables (legacy kept as
R2_total_legacy). p-values and significance must match the existing tables.
"""
import os, shutil, numpy as np, pandas as pd
from common import FOCAL, parse_args, load_dataset, focal_data, fit, result_table, log

RES = os.path.expanduser("~/SpatialHeterogeneity/results/IPF/healthy")
BK = os.path.expanduser("~/SpatialHeterogeneity/results/archive/legacy_r2_2026-10-07")
NAME = {"AT2": "AT2", "SPP1+ Macrophages": "SPP1_Mac", "Basal": "Basal", "Capillary": "Capillary",
        "Interstitial Macrophages": "Interstitial_Mac", "Alveolar FBs": "Alveolar_FBs"}

def main():
    args = parse_args(__doc__)
    tissue = load_dataset("ipf", condition="Healthy")
    for focal in FOCAL["ipf"]:
        fd = focal_data(tissue, "ipf", focal)
        res = fit(tissue, fd, args.n_perm, seed=args.seed)
        new = result_table(res, fd["genes"], tissue.type_names)
        path = f"{RES}/aura_ms_{NAME[focal]}.csv"; old = pd.read_csv(path)
        m = old[["gene", "pvalue", "significant", "R2_total"]].merge(new[["gene", "pvalue", "significant", "R2_total", "R2_total_legacy"]], on="gene", suffixes=("_o", "_n"))
        assert len(m) == len(old) == len(new), (focal, len(old), len(new))
        dp = np.abs(m.pvalue_o - m.pvalue_n).max(); agree = (m.significant_o == m.significant_n).mean()
        dl = np.abs(m.R2_total_o - m.R2_total_legacy).max()
        log.info("%s: n=%d max|dp|=%.2e sig agree=%.4f legacy match=%.1e", focal, len(m), dp, agree, dl)
        assert agree == 1.0 and dl < 1e-5, focal
        shutil.copy2(path, f"{BK}/results__IPF__healthy__aura_ms_{NAME[focal]}.csv")
        if "R2_total_legacy" not in old.columns:
            old.insert(old.columns.get_loc("R2_total") + 1, "R2_total_legacy", old["R2_total"].values)
        old["R2_total"] = old["gene"].map(dict(zip(new.gene, new.R2_total))).values
        old.to_csv(path, index=False)

if __name__ == "__main__":
    main()
