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
    add_cluster_label_mask,
    add_largest_cluster_mask,
    add_valid_data_mask,
    apply_dbscan_clustering,
    apply_kmeans_clustering,
    apply_min_cluster_size_filter,
    assign_noise_by_soft_membership,
    compute_mvbs,
    create_ml_index_coordinate,
    data_preprocessing_pipeline,
    extract_cluster_statistics,
    extract_data_and_run_hdbscan,
    extract_ml_data_flattened,
    extract_ml_data_gridded,
    extract_valid_samples_for_sklearn,
    full_dbscan_iteration,
    get_grid_coordinates,
    normalize_data,
    plot_cluster_statistics,
    plot_dbscan_cluster_hierarchy,
    print_basic_cluster_stats,
    print_cluster_statistics,
    remove_noise,
    reshape_and_normalize_data,
    reshape_data_for_ml,
    retrieve_background_cluster,
    store_ml_data_flattened,
    store_ml_results_flattened,
    visualize_normalized_data_histogram,
)

__all__ = [
    "__version__",
    "add_cluster_label_mask",
    "add_largest_cluster_mask",
    "add_valid_data_mask",
    "apply_dbscan_clustering",
    "apply_kmeans_clustering",
    "apply_min_cluster_size_filter",
    "assign_noise_by_soft_membership",
    "compute_mvbs",
    "create_ml_index_coordinate",
    "data_preprocessing_pipeline",
    "extract_cluster_statistics",
    "extract_data_and_run_hdbscan",
    "extract_ml_data_flattened",
    "extract_ml_data_gridded",
    "extract_valid_samples_for_sklearn",
    "full_dbscan_iteration",
    "get_grid_coordinates",
    "normalize_data",
    "plot_cluster_statistics",
    "plot_dbscan_cluster_hierarchy",
    "print_basic_cluster_stats",
    "print_cluster_statistics",
    "remove_noise",
    "reshape_and_normalize_data",
    "reshape_data_for_ml",
    "retrieve_background_cluster",
    "store_ml_data_flattened",
    "store_ml_results_flattened",
    "visualize_normalized_data_histogram",
]
