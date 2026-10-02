"""
AURA: composition-dependent context variance model for spatial transcriptomics.

Detects genes whose expression in a focal cell type is structured by
local neighborhood composition.
"""

__version__ = "0.2.0"

from .model import run_model, run_model_sample, run_model_multisample
from .io import load_adata, extract_focal, save_results, load_results
from .workflow import run_aura, run_aura_multisample, AuraResult
from .spillover import (spillover_filter, spillover_filter_multi,
                        spillover_distance_test, spillover_check,
                        learn_canonical, query_gene,
                        DEFAULT_CANONICAL, CANONICAL_IPF, CANONICAL_LYMPHNODE)
from .plot import (
    diagnostic_panel,
    variance_panel,
    variance_summary,
    subtype_check,
    result_panel,
    gene_zoom_panel,
)
