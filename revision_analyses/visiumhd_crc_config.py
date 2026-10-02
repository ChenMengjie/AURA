"""
Visium HD colorectal cancer (patient P2; Oliveira et al., Nat Genet 2025).

Cell-level data: Space Ranger 4.0.1 nucleus-based cell segmentation
(220,703 cells, whole-transcriptome probe set), from
https://cf.10xgenomics.com/samples/spatial-exp/4.0.1/Visium_HD_Human_Colon_Cancer/
Cell types: the study's RCTD deconvolution of 8-µm bins
(MetaData/P2CRC_Metadata.parquet, github.com/10XGenomics/HumanColonCancer_VisiumHD),
transferred to each segmented cell from the bin under its centroid; only
bins called 'singlet' are used.
"""

# RCTD fine labels (DeconvolutionLabel1) -> composition lineages
LINEAGE_MAP = {
    'Tumor I': 'Tumor', 'Tumor II': 'Tumor', 'Tumor III': 'Tumor',
    'Tumor IV': 'Tumor', 'Tumor V': 'Tumor',
    'Goblet': 'Epithelial', 'Enterocyte': 'Epithelial', 'Epithelial': 'Epithelial',
    'Tuft': 'Epithelial', 'Neuroendocrine': 'Epithelial',
    'CAF': 'Fibroblast', 'Fibroblast': 'Fibroblast', 'Myofibroblast': 'Fibroblast',
    'Proliferating Fibroblast': 'Fibroblast', 'Vascular Fibroblast': 'Fibroblast',
    'Smooth Muscle': 'SMC_Peri', 'vSM': 'SMC_Peri', 'SM Stress Response': 'SMC_Peri',
    'Pericytes': 'SMC_Peri', 'Unknown III (SM)': 'SMC_Peri',
    'Endothelial': 'Endothelial', 'Lymphatic Endothelial': 'Endothelial',
    'Macrophage': 'Myeloid', 'Proliferating Macrophages': 'Myeloid',
    'cDC I': 'Myeloid', 'mRegDC': 'Myeloid', 'pDC': 'Myeloid', 'Mast': 'Myeloid',
    'Neutrophil': 'Granulocyte',
    'CD4 T cell': 'T_NK', 'CD8 T cell': 'T_NK', 'NK': 'T_NK',
    'Proliferating Immune II': 'T_NK',
    'Plasma': 'B_Plasma', 'Mature B': 'B_Plasma', 'Memory B': 'B_Plasma',
    'Enteric Glial': 'Neural', 'Adipocyte': 'Other',
}

# Focal populations (fine labels), chosen to parallel the Xenium analyses
FOCAL = ['Macrophage', 'CAF', 'Endothelial', 'Plasma', 'Goblet', 'Tumor III']
MAX_CELLS = {'Tumor III': 20000}
