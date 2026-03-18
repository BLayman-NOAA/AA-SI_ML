# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries

"""aa_si_ml - Machine learning tools for NOAA Fisheries AA-SI sonar data analysis.

Provides ML pipeline functions for clustering, normalization, and analysis
of acoustic backscatter data.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("aa-si-ml")
except PackageNotFoundError:
    # Package is not installed (e.g., running from source without install)
    __version__ = "0.0.0.dev"

from .ml import (
    reshape_data_for_ml,
    normalize_data,
    extract_valid_samples_for_sklearn,
    store_ml_results_flattened,
    extract_ml_data_gridded,
    apply_dbscan_clustering,
    apply_kmeans_clustering,
    data_preprocessing_pipeline,
    full_dbscan_iteration,
)

__all__ = [
    "__version__",
    "reshape_data_for_ml",
    "normalize_data",
    "extract_valid_samples_for_sklearn",
    "store_ml_results_flattened",
    "extract_ml_data_gridded",
    "apply_dbscan_clustering",
    "apply_kmeans_clustering",
    "data_preprocessing_pipeline",
    "full_dbscan_iteration",
]
