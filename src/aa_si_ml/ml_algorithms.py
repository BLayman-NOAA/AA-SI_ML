# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries

import logging
import time

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import hdbscan

from .plotting_and_logging import print_basic_cluster_stats

logger = logging.getLogger(__name__)


def _subsample_data(X, sample_indices, sample_size, algorithm_name):
    """Return a random subsample of X and sample_indices, or the originals.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        sample_indices (np.ndarray): Grid-index values for each row of X.
        sample_size (int or None): Target subsample size. When None or
            larger than len(X), the originals are returned unchanged.
        algorithm_name (str): Name used in log messages.

    Returns:
        tuple: (X_out, indices_out) arrays of the (possibly subsampled) data.
    """
    if sample_size is not None and sample_size < len(X):
        logger.info(
            "Using random sample of %s points for %s (from %s total)",
            f"{sample_size:,}", algorithm_name, f"{len(X):,}"
        )
        rng = np.random.default_rng(42)
        mask = rng.choice(len(X), size=sample_size, replace=False)
        return X[mask], sample_indices[mask]
    logger.info("Using all %s valid data points for %s", f"{len(X):,}", algorithm_name)
    return X, sample_indices


def apply_dbscan_clustering(
        X_normalized,
        sample_indices,
        eps_values=[0.3, 0.5, 0.7, 1.0],
        min_samples_values=[5, 10, 20],
        sample_size=None,
        calculate_silhouette=True,
        silhouette_sample_size=10000,
        metric="euclidean",
        algorithm="auto",
        use_hdbscan=False,
        min_cluster_size=5,
        cluster_selection_method='eom',
        soft_membership_threshold=None
        ):
    """Run DBSCAN or HDBSCAN clustering over a parameter grid.

    Iterates over the provided parameter values, fits a model for each
    combination, and returns results including labels and scores.

    Args:
        X_normalized (np.ndarray): Feature matrix of shape
            (n_samples, n_features).
        sample_indices (np.ndarray): Grid-index values corresponding to
            each row of X_normalized.
        eps_values (list[float]): Epsilon values for DBSCAN neighbourhood
            radius (ignored when use_hdbscan is True).
            Defaults to [0.3, 0.5, 0.7, 1.0].
        min_samples_values (list[int]): Core-point neighbourhood sizes
            to try. Defaults to [5, 10, 20].
        sample_size (int or None): Sub-sample size for large datasets.
            Defaults to None (all data).
        calculate_silhouette (bool): Compute silhouette scores.
            Defaults to True.
        silhouette_sample_size (int): Sample size for faster silhouette
            calculation. Defaults to 10000.
        metric (str): Distance metric. Defaults to 'euclidean'.
        algorithm (str): DBSCAN algorithm variant. Defaults to 'auto'.
        use_hdbscan (bool): Use HDBSCAN instead of DBSCAN.
            Defaults to False.
        min_cluster_size (int): Minimum cluster size (HDBSCAN parameter /
            DBSCAN post-filter). Defaults to 5.
        cluster_selection_method (str): HDBSCAN cluster selection method.
            Defaults to 'eom'.
        soft_membership_threshold (float or None): If set, reassign
            HDBSCAN noise points whose soft-membership probability
            exceeds this threshold. Defaults to None.

    Returns:
        dict: Keyed by parameter string, each value is a dict with keys
        'method', 'model', 'labels', 'silhouette_score', 'sample_indices',
        'n_clusters', 'n_noise', etc.
    """
    algorithm_name = "HDBSCAN" if use_hdbscan else "DBSCAN"
    X_sample, used_sample_indices = _subsample_data(
        X_normalized, sample_indices, sample_size, algorithm_name
    )

    results = {}

    if use_hdbscan:
        _run_hdbscan_grid(
            results, X_sample, used_sample_indices, X_normalized,
            min_samples_values, min_cluster_size, cluster_selection_method,
            metric, calculate_silhouette, silhouette_sample_size,
            soft_membership_threshold
        )
    else:
        _run_dbscan_grid(
            results, X_sample, used_sample_indices,
            eps_values, min_samples_values, min_cluster_size,
            metric, algorithm, calculate_silhouette, silhouette_sample_size
        )

    return results


def _run_hdbscan_grid(
        results, X_sample, used_sample_indices, X_normalized,
        min_samples_values, min_cluster_size, cluster_selection_method,
        metric, calculate_silhouette, silhouette_sample_size,
        soft_membership_threshold
        ):
    """Fit HDBSCAN for each min_samples value and populate results.

    Args:
        results (dict): Mutable dict to populate with per-run results.
        X_sample (np.ndarray): Feature matrix (possibly subsampled).
        used_sample_indices (np.ndarray): Grid indices for X_sample rows.
        X_normalized (np.ndarray): Full feature matrix (used for covariance).
        min_samples_values (list[int]): min_samples values to iterate over.
        min_cluster_size (int): HDBSCAN min_cluster_size.
        cluster_selection_method (str): HDBSCAN cluster selection method.
        metric (str): Distance metric.
        calculate_silhouette (bool): Whether to compute silhouette scores.
        silhouette_sample_size (int): Max samples for silhouette calculation.
        soft_membership_threshold (float or None): Threshold for noise
            reassignment via soft membership probabilities.
    """
    if calculate_silhouette and len(X_sample) > silhouette_sample_size:
        logger.info(
            "Silhouette scores calculated on sample of %s points for efficiency",
            f"{silhouette_sample_size:,}"
        )

    metric_kwargs = (
        {'V': np.cov(X_normalized, rowvar=False)} if metric == "mahalanobis" else {}
    )

    for min_samples in min_samples_values:
        param_key = f"hdbscan_mincluster_{min_cluster_size}_min_{min_samples}"
        logger.info(
            "Testing HDBSCAN with min_cluster_size=%d, min_samples=%d",
            min_cluster_size, min_samples
        )

        start_time = time.time()
        model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_method=cluster_selection_method,
            prediction_data=True,
            **metric_kwargs
        )
        cluster_labels = model.fit_predict(X_sample)
        logger.info("HDBSCAN fitting took %.2f seconds", time.time() - start_time)

        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        sil_score = _calculate_silhouette(
            X_sample, cluster_labels, n_clusters,
            calculate_silhouette, silhouette_sample_size
        )

        if soft_membership_threshold is not None:
            logger.info("Before noise reassignment:")
            print_basic_cluster_stats(cluster_labels, n_clusters, n_noise, sil_score, calculate_silhouette)
            cluster_labels = assign_noise_by_soft_membership(
                model, threshold=soft_membership_threshold
            )
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            sil_score = _calculate_silhouette(
                X_sample, cluster_labels, n_clusters,
                calculate_silhouette, silhouette_sample_size
            )

        print_basic_cluster_stats(cluster_labels, n_clusters, n_noise, sil_score, calculate_silhouette)

        results[param_key] = {
            'method': 'hdbscan',
            'model': model,
            'labels': cluster_labels,
            'silhouette_score': sil_score,
            'sample_indices': used_sample_indices,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'eps': None,
            'min_samples': min_samples,
            'min_cluster_size': min_cluster_size,
            'metric': metric,
        }


def _run_dbscan_grid(
        results, X_sample, used_sample_indices,
        eps_values, min_samples_values, min_cluster_size,
        metric, algorithm, calculate_silhouette, silhouette_sample_size
        ):
    """Fit DBSCAN for each (eps, min_samples) combination and populate results.

    Args:
        results (dict): Mutable dict to populate with per-run results.
        X_sample (np.ndarray): Feature matrix (possibly subsampled).
        used_sample_indices (np.ndarray): Grid indices for X_sample rows.
        eps_values (list[float]): Epsilon values to iterate over.
        min_samples_values (list[int]): min_samples values to iterate over.
        min_cluster_size (int): Minimum cluster size for post-filtering.
        metric (str): Distance metric.
        algorithm (str): DBSCAN algorithm variant.
        calculate_silhouette (bool): Whether to compute silhouette scores.
        silhouette_sample_size (int): Max samples for silhouette calculation.
    """
    if calculate_silhouette and len(X_sample) > silhouette_sample_size:
        logger.info(
            "Silhouette scores calculated on sample of %s points for efficiency",
            f"{silhouette_sample_size:,}"
        )

    for eps in eps_values:
        for min_samples in min_samples_values:
            param_key = f"eps_{eps}_min_{min_samples}"
            logger.info("Testing DBSCAN with eps=%s, min_samples=%d", eps, min_samples)

            start_time = time.time()
            model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, algorithm=algorithm)
            cluster_labels = model.fit_predict(X_sample)
            cluster_labels = apply_min_cluster_size_filter(cluster_labels, min_cluster_size)
            logger.info("DBSCAN fitting took %.2f seconds", time.time() - start_time)

            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            sil_score = _calculate_silhouette(
                X_sample, cluster_labels, n_clusters,
                calculate_silhouette, silhouette_sample_size
            )

            print_basic_cluster_stats(cluster_labels, n_clusters, n_noise, sil_score, calculate_silhouette)

            results[param_key] = {
                'method': 'dbscan',
                'model': model,
                'labels': cluster_labels,
                'silhouette_score': sil_score,
                'sample_indices': used_sample_indices,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'eps': eps,
                'min_samples': min_samples,
                'min_cluster_size': min_cluster_size,
                'algorithm': algorithm,
                'metric': metric,
            }


def assign_noise_by_soft_membership(clusterer, threshold=0.1):
    """Assign noise points to clusters based on soft membership probabilities.

    Args:
        clusterer (hdbscan.HDBSCAN): Fitted HDBSCAN clusterer with
            prediction_data=True.
        threshold (float): Minimum probability required to assign a noise
            point to a cluster. Defaults to 0.1.

    Returns:
        np.ndarray: Cluster labels with noise points reassigned if their
        max probability exceeds threshold.
    """
    labels = clusterer.labels_.copy()
    soft_clusters = hdbscan.all_points_membership_vectors(clusterer)

    noise_mask = labels == -1
    if np.any(noise_mask):
        noise_soft_clusters = soft_clusters[noise_mask]
        best_clusters = np.argmax(noise_soft_clusters, axis=1)
        max_probs = noise_soft_clusters[np.arange(len(best_clusters)), best_clusters]

        reassign_mask = max_probs > threshold
        noise_indices = np.where(noise_mask)[0]
        labels[noise_indices[reassign_mask]] = best_clusters[reassign_mask]

        logger.info(
            "Reassigned %d of %d noise points via soft membership",
            int(np.sum(reassign_mask)), int(np.sum(noise_mask))
        )

    return labels


def _calculate_silhouette(X_sample, cluster_labels, n_clusters, calculate_silhouette, silhouette_sample_size):
    """Compute silhouette score, optionally on a random subsample.

    Args:
        X_sample (np.ndarray): Feature matrix.
        cluster_labels (np.ndarray): Cluster label per sample.
        n_clusters (int): Number of clusters found.
        calculate_silhouette (bool): Whether to compute the score.
        silhouette_sample_size (int): Maximum samples for calculation.

    Returns:
        float or None: Silhouette score, or None if not computed.
    """
    if not (calculate_silhouette and n_clusters > 1 and n_clusters < len(X_sample)):
        return None

    start_time = time.time()
    try:
        if len(X_sample) > silhouette_sample_size:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X_sample), size=silhouette_sample_size, replace=False)
            X_sub = X_sample[idx]
            labels_sub = cluster_labels[idx]
            if len(set(labels_sub)) <= 1:
                return None
            sil_score = silhouette_score(X_sub, labels_sub)
        else:
            sil_score = silhouette_score(X_sample, cluster_labels)

        logger.debug("Silhouette calculation took %.2f seconds", time.time() - start_time)
        return sil_score
    except Exception as e:
        logger.warning("Silhouette calculation failed: %s", e)
        return None


def apply_kmeans_clustering(X_normalized, sample_indices, k_values=[3, 5, 7], sample_size=None,
                            calculate_silhouette=True, silhouette_sample_size=10000):
    """Run K-means clustering for each value of k.

    Args:
        X_normalized (np.ndarray): Feature matrix of shape
            (n_samples, n_features).
        sample_indices (np.ndarray): Grid-index values for each row.
        k_values (list[int]): Numbers of clusters to try.
            Defaults to [3, 5, 7].
        sample_size (int or None): Sub-sample size for large datasets.
            Defaults to None.
        calculate_silhouette (bool): Compute silhouette scores.
            Defaults to True.
        silhouette_sample_size (int): Sample size for faster silhouette
            calculation. Defaults to 10000.

    Returns:
        dict: Keyed by k, each value is a dict with 'method', 'model',
        'labels', 'silhouette_score', 'inertia', and 'sample_indices'.
    """
    X_sample, used_sample_indices = _subsample_data(
        X_normalized, sample_indices, sample_size, "K-means"
    )

    if calculate_silhouette and len(X_sample) > silhouette_sample_size:
        logger.info(
            "Silhouette scores calculated on sample of %s points for efficiency",
            f"{silhouette_sample_size:,}"
        )

    results = {}

    for k in k_values:
        logger.info("Testing K-means with k=%d clusters", k)

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_sample)

        sil_score = _calculate_silhouette(
            X_sample, cluster_labels, k, calculate_silhouette, silhouette_sample_size
        )

        inertia = kmeans.inertia_
        if sil_score is not None:
            logger.info("  Silhouette Score: %.3f", sil_score)
        logger.info("  Inertia (WCSS): %.2f", inertia)

        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        logger.info("  Cluster sizes: %s", dict(zip(unique_labels.tolist(), counts.tolist())))

        results[k] = {
            'method': 'kmeans',
            'model': kmeans,
            'labels': cluster_labels,
            'silhouette_score': sil_score,
            'inertia': inertia,
            'sample_indices': used_sample_indices
        }

    return results


def apply_min_cluster_size_filter(cluster_labels, min_cluster_size):
    """Apply min_cluster_size filtering to DBSCAN results as post-processing.

    Clusters smaller than min_cluster_size are reclassified as noise (-1).
    Remaining clusters are renumbered to be consecutive starting from 0.

    Args:
        cluster_labels (np.ndarray): Original cluster labels from DBSCAN.
        min_cluster_size (int): Minimum number of points required for a
            cluster to be kept.

    Returns:
        np.ndarray: Filtered cluster labels with small clusters converted
        to noise and remaining clusters renumbered consecutively.
    """
    if min_cluster_size <= 1:
        return cluster_labels.copy()

    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    filtered_labels = cluster_labels.copy()

    clusters_removed = 0
    points_converted_to_noise = 0

    for label, count in zip(unique_labels, counts):
        if label == -1:
            continue
        if count < min_cluster_size:
            filtered_labels[cluster_labels == label] = -1
            clusters_removed += 1
            points_converted_to_noise += count

    remaining_clusters = np.unique(filtered_labels)
    remaining_clusters = remaining_clusters[remaining_clusters != -1]

    if len(remaining_clusters) > 0:
        label_mapping = {old: new for new, old in enumerate(remaining_clusters)}
        renumbered = filtered_labels.copy()
        for old_label, new_label in label_mapping.items():
            renumbered[filtered_labels == old_label] = new_label
        filtered_labels = renumbered
        logger.info(
            "Min cluster size filter: removed %d small clusters (%d points -> noise), "
            "%d clusters remaining",
            clusters_removed, points_converted_to_noise, len(remaining_clusters)
        )
    else:
        logger.info(
            "Min cluster size filter: removed %d small clusters (%d points -> noise), "
            "0 clusters remaining",
            clusters_removed, points_converted_to_noise
        )

    return filtered_labels


def retrieve_background_cluster(X, sample_indices, min_samples, sample_size, min_cluster_size,
                                cluster_selection_method, feature_threshold=-0.2, min_fraction=0.1):
    """Find a background cluster by iterating over epsilon values.

    Returns the clustering results and the label of the identified
    background cluster. A cluster qualifies as background if:

    1. Its average feature[0] is below feature_threshold.
    2. It contains more than min_fraction of total labelled samples.

    Among qualifying clusters the largest one is selected.

    Args:
        X (np.ndarray): Feature matrix.
        sample_indices (np.ndarray): Grid-index values for each row.
        min_samples (int): Core-point neighbourhood size.
        sample_size (int): Sub-sample size.
        min_cluster_size (int): Minimum cluster size.
        cluster_selection_method (str): HDBSCAN cluster selection method.
        feature_threshold (float): Maximum average feature[0] for a
            cluster to qualify as background. Defaults to -0.2.
        min_fraction (float): Minimum fraction of total labelled samples
            a cluster must contain. Defaults to 0.1.

    Returns:
        tuple: (dbscan_results, background_label).

    Raises:
        ValueError: If no qualifying background cluster is found across
            all epsilon values.
    """
    for epsilon in [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5]:
        logger.info("Trying to find background cluster with eps=%.2f", epsilon)
        dbscan_results = apply_dbscan_clustering(
            X,
            sample_indices=sample_indices,
            eps_values=[epsilon],
            min_samples_values=[min_samples],
            sample_size=sample_size,
            calculate_silhouette=True,
            silhouette_sample_size=100,
            metric="euclidean",
            use_hdbscan=False,
            min_cluster_size=min_cluster_size,
            cluster_selection_method=cluster_selection_method
        )

        first_key = next(iter(dbscan_results))
        labels = dbscan_results[first_key]["labels"]
        used_sample_indices = dbscan_results[first_key]["sample_indices"]
        unique_labels, counts = np.unique(labels, return_counts=True)
        valid_mask = unique_labels >= 0
        valid_labels = unique_labels[valid_mask]
        valid_counts = counts[valid_mask]

        if len(valid_labels) == 0:
            continue

        total_samples = len(labels)
        grid_idx_to_pos = {gi: pos for pos, gi in enumerate(sample_indices)}
        sample_positions = np.array([grid_idx_to_pos[gi] for gi in used_sample_indices])
        X_sample = X[sample_positions]

        candidates = []
        for i, label in enumerate(valid_labels):
            count = valid_counts[i]
            fraction = count / total_samples
            cluster_mask = labels == label
            avg_feature0 = np.mean(X_sample[cluster_mask, 0])
            logger.debug(
                "  Cluster %d: avg feature[0] = %.2f, size = %s (%.1f%% of total)",
                label, avg_feature0, f"{count:,}", fraction * 100
            )
            if avg_feature0 < feature_threshold and fraction > min_fraction:
                candidates.append((label, count, avg_feature0))

        if candidates:
            best = max(candidates, key=lambda c: c[1])
            logger.info(
                "Found background cluster %d with avg feature[0] = %.2f, "
                "size = %s at eps=%.2f",
                best[0], best[2], f"{best[1]:,}", epsilon
            )
            return dbscan_results, best[0]

    raise ValueError(
        f"Could not find background cluster with avg feature[0] < {feature_threshold} "
        f"and > {min_fraction:.0%} of total samples"
    )
