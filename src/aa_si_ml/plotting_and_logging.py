import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .constants import DEFAULT_CLUSTER_COLORS

logger = logging.getLogger(__name__)


def visualize_normalized_data_histogram(X_normalized, feature_names=None, n_bins=200, as_density=True, percentile_range=(.1, 99.9)):
    """Plot overlaid histograms of each feature after normalization.

    Args:
        X_normalized (np.ndarray): 2-D array of shape
            ``(n_samples, n_features)``.
        feature_names (list[str] or None): Labels for each feature.
            Auto-generated when None. Defaults to None.
        n_bins (int): Number of histogram bins. Defaults to 200.
        as_density (bool): Plot as probability density. Defaults to True.
        percentile_range (tuple[float] or None): ``(lower, upper)``
            percentiles for the x-axis range. Use ``None`` for the full
            data range. Defaults to (0.1, 99.9).
    """
    if X_normalized.ndim != 2:
        raise ValueError("X_normalized must be 2D (n_samples, n_features)")
    n_features = X_normalized.shape[1]

    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(n_features)]
    else:
        feature_names = [str(n) for n in feature_names]
        if len(feature_names) != n_features:
            raise ValueError("feature_names length must match number of features")

    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10.colors

    if percentile_range is not None:
        lower_percentile, upper_percentile = percentile_range
        global_min = np.nanpercentile(X_normalized, lower_percentile)
        global_max = np.nanpercentile(X_normalized, upper_percentile)
        logger.debug("Using %sth to %sth percentile range: %.3f to %.3f", lower_percentile, upper_percentile, global_min, global_max)
    else:
        global_min = float(np.nanmin(X_normalized))
        global_max = float(np.nanmax(X_normalized))
        logger.debug("Using full data range: %.3f to %.3f", global_min, global_max)

    bins = np.linspace(global_min, global_max, n_bins + 1)

    for i in range(n_features):
        data = X_normalized[:, i]
        plt.hist(
            data,
            bins=bins,
            density=as_density,
            histtype='step',
            color=colors[i % len(colors)],
            linewidth=2,
            label=feature_names[i]
        )

    plt.title('Histogram of Normalized Data')
    plt.xlabel('Normalized Value')
    plt.ylabel('Density' if as_density else 'Frequency')
    plt.legend(loc='upper right', fontsize=9, frameon=True)
    plt.grid(alpha=0.25)
    y_min, y_max = plt.ylim()
    plt.ylim(y_min, y_max * 1.1)
    plt.tight_layout()
    plt.show()


def print_basic_cluster_stats(cluster_labels, n_clusters, n_noise, sil_score, calculate_silhouette):
    """Print a summary table of cluster sizes and silhouette score.

    Args:
        cluster_labels (np.ndarray): Cluster label per sample.
        n_clusters (int): Number of clusters.
        n_noise (int): Number of noise points.
        sil_score (float): Silhouette score.
        calculate_silhouette (bool): Whether silhouette was requested.
    """
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Noise points: {n_noise} ({n_noise/len(cluster_labels)*100:.1f}%)")
    
    if not calculate_silhouette:
        print(f"  Silhouette Score: Disabled (calculate_silhouette=False)")
    elif n_clusters <= 1:
        print(f"  Silhouette Score: N/A (need >=2 clusters, found {n_clusters})")
    elif n_clusters >= len(cluster_labels):
        print(f"  Silhouette Score: N/A (too many clusters relative to data)")
    elif sil_score is not None:
        print(f"  Silhouette Score: {sil_score:.3f}")
    else:
        print(f"  Silhouette Score: Calculation failed (possibly due to sampling issues)")
    
    if n_clusters > 0:
        cluster_sizes = {}
        for label, count in zip(unique_labels, counts):
            if label == -1:
                cluster_sizes['Noise'] = count
            else:
                cluster_sizes[f'Cluster {label}'] = count
        print(f"  Cluster sizes: {cluster_sizes}")


def print_cluster_statistics(ds_ml_ready, cluster_data_name, dataset_name='ml_data_clean', 
                            normalize_data_name=None, sv_data_var=None, compute_pairwise_diffs=False):
    """Print statistics (mean and standard deviation) for each cluster.

    Args:
        ds_ml_ready (xr.Dataset): Dataset containing cluster labels and
            source data.
        cluster_data_name (str): Name of the cluster labels variable
            (e.g. ``'kmeans_clusters_k_5'``, ``'dbscan_clusters_5'``).
        dataset_name (str): Base dataset name.
            Defaults to 'ml_data_clean'.
        normalize_data_name (str or None): Name of normalized data to use
            for statistics. Defaults to None (uses *dataset_name*).
        sv_data_var (str or None): Name of original Sv variable to use
            for statistics (e.g. ``'Sv'``, ``'Sv_corrected'``). If
            provided, statistics are calculated from gridded Sv data
            instead of flattened normalized data. Defaults to None.
        compute_pairwise_diffs (bool): If True and *sv_data_var* is
            provided, compute pairwise differences between channels.
            Defaults to False.

    Returns:
        list: List of dicts with per-cluster statistics (for backwards
        compatibility).
    """
    from .ml import extract_cluster_statistics

    stats_dict = extract_cluster_statistics(
        ds_ml_ready, cluster_data_name, dataset_name, 
        normalize_data_name, sv_data_var, compute_pairwise_diffs
    )
    
    cluster_stats_list = stats_dict['cluster_stats']
    noise_stats = stats_dict['noise_stats']
    metadata = stats_dict['metadata']
    feature_coords = stats_dict['feature_coords']
    data_description = stats_dict['data_description']
    
    print(f"\n{'='*80}")
    print(f"CLUSTER STATISTICS: {cluster_data_name}")
    print(f"{'='*80}")
    print(data_description)
    print(f"Total samples: {metadata['n_total_samples']:,}")
    print(f"Number of clusters: {metadata['n_clusters']}")
    if metadata['n_noise'] > 0:
        print(f"Noise points: {metadata['n_noise']:,} ({metadata['n_noise']/metadata['n_total_samples']*100:.2f}%)")
    print(f"{'='*80}\n")
    
    for cluster_stats in cluster_stats_list:
        cluster_id = cluster_stats['Cluster']
        n_samples = cluster_stats['N_Samples']
        percentage = cluster_stats['Percentage']
        
        print(f"Cluster {cluster_id:d}:")
        print(f"  Samples: {n_samples:,} ({percentage:.2f}%)")
        print(f"  Feature statistics:")
        
        for feature_name in feature_coords:
            mean_val = cluster_stats[f'{feature_name}_mean']
            std_val = cluster_stats[f'{feature_name}_std']
            min_val = cluster_stats[f'{feature_name}_min']
            max_val = cluster_stats[f'{feature_name}_max']
            
            print(f"    {feature_name}:")
            print(f"      Mean: {mean_val:8.3f}  Std: {std_val:7.3f}")
            print(f"      Min:  {min_val:8.3f}  Max: {max_val:7.3f}")
        
        print()
    
    if noise_stats is not None:
        print(f"Noise (cluster -1):")
        print(f"  Samples: {noise_stats['N_Samples']:,} ({noise_stats['Percentage']:.2f}%)")
        print(f"  Feature statistics:")
        
        for feature_name in feature_coords:
            mean_val = noise_stats[f'{feature_name}_mean']
            std_val = noise_stats[f'{feature_name}_std']
            print(f"    {feature_name}:")
            print(f"      Mean: {mean_val:8.3f}  Std: {std_val:7.3f}")
        print()
    
    return cluster_stats_list


def plot_cluster_statistics(ds_ml_ready, cluster_data_name, dataset_name='ml_data_clean',
                           normalize_data_name=None, sv_data_var=None,
                           stat_type='mean', include_noise=False, 
                           cluster_colors=None, figsize=(12, 6),
                           title=None, save_path=None, compute_pairwise_diffs=False):
    """Plot cluster statistics as bar charts with error bars.

    Features are grouped by type (Sv vs differences) with shared y-axes.
    All Sv features share one y-axis, all difference features share another.

    Args:
        ds_ml_ready (xr.Dataset): Dataset containing cluster labels and
            source data.
        cluster_data_name (str): Name of the cluster labels variable.
        dataset_name (str): Base dataset name.
            Defaults to 'ml_data_clean'.
        normalize_data_name (str or None): Name of normalized data.
            Defaults to None.
        sv_data_var (str or None): Name of original Sv variable.
            Defaults to None.
        stat_type (str): Statistic to plot: ``'mean'``, ``'min'``, or
            ``'max'``. Defaults to 'mean'.
        include_noise (bool): Include noise cluster in the plot.
            Defaults to False.
        cluster_colors (list[str] or None): Hex colour strings.
            Defaults to None (uses built-in palette).
        figsize (tuple): Figure size. Defaults to (12, 6).
        title (str or None): Custom title. Defaults to None.
        save_path (str or None): Path to save figure. Defaults to None.
        compute_pairwise_diffs (bool): If True and *sv_data_var* is
            provided, compute pairwise differences between channels
            and plot them on a separate y-axis. Defaults to False.

    Returns:
        tuple: ``(fig, axes_list, stats_dict)``.
    """
    from .ml import extract_cluster_statistics

    if cluster_colors is None:
        cluster_colors = DEFAULT_CLUSTER_COLORS

    valid_stat_types = ['mean', 'min', 'max']
    if stat_type not in valid_stat_types:
        raise ValueError(f"stat_type must be one of {valid_stat_types}, got '{stat_type}'")

    stats_dict = extract_cluster_statistics(
        ds_ml_ready, cluster_data_name, dataset_name,
        normalize_data_name, sv_data_var, compute_pairwise_diffs
    )

    cluster_stats_list = stats_dict['cluster_stats']
    noise_stats = stats_dict['noise_stats']
    feature_coords = stats_dict['feature_coords']
    data_description = stats_dict['data_description']

    clusters_to_plot = cluster_stats_list.copy()
    if include_noise and noise_stats is not None:
        clusters_to_plot.append(noise_stats)

    n_clusters = len(clusters_to_plot)
    n_features = len(feature_coords)
    cluster_ids = [stats['Cluster'] for stats in clusters_to_plot]

    means = np.zeros((n_clusters, n_features))
    stds = np.zeros((n_clusters, n_features))
    stat_values = np.zeros((n_clusters, n_features))

    for i, cluster_stats in enumerate(clusters_to_plot):
        for j, feature_name in enumerate(feature_coords):
            means[i, j] = cluster_stats[f'{feature_name}_mean']
            stds[i, j] = cluster_stats[f'{feature_name}_std']
            if stat_type == 'mean':
                stat_values[i, j] = means[i, j]
            elif stat_type == 'min':
                stat_values[i, j] = cluster_stats[f'{feature_name}_min']
            elif stat_type == 'max':
                stat_values[i, j] = cluster_stats[f'{feature_name}_max']

    if cluster_colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        default_colors = prop_cycle.by_key()['color']
        cluster_color_list = [default_colors[i % len(default_colors)] for i in range(n_clusters)]
    else:
        cluster_color_list = []
        for cluster_id in cluster_ids:
            if cluster_id == -1:
                cluster_color_list.append('gray')
            elif cluster_id < len(cluster_colors):
                cluster_color_list.append(cluster_colors[cluster_id])
            else:
                cluster_color_list.append('gray')
                print(f"Warning: No color provided for cluster {cluster_id}, using gray")

    # Classify features as Sv or difference
    sv_feature_indices = []
    diff_feature_indices = []
    
    for j, feature_name in enumerate(feature_coords):
        feature_name_lower = str(feature_name).lower()
        is_sv_feature = (
            'sv' in feature_name_lower and 'diff' not in feature_name_lower
        ) or (np.mean(stat_values[:, j]) < -40 and np.mean(stat_values[:, j]) > -100)
        is_sv_diff = (
            'diff' in feature_name_lower or
            (np.min(stat_values[:, j]) > -50 and np.max(stat_values[:, j]) < 50)
        )
        
        if is_sv_feature:
            sv_feature_indices.append(j)
        elif is_sv_diff:
            diff_feature_indices.append(j)

    baseline_axes_y = 0.5
    
    if sv_feature_indices:
        sv_baseline = -100
        sv_values = stat_values[:, sv_feature_indices]
        sv_stds = stds[:, sv_feature_indices]
        sv_y_min = np.min(sv_values - sv_stds)
        sv_y_max = np.max(sv_values + sv_stds)
        sv_y_min = min(sv_y_min, sv_baseline)
        sv_y_max = max(sv_y_max, sv_baseline)
        
        below = sv_baseline - sv_y_min
        above = sv_y_max - sv_baseline
        total_range = max(below / baseline_axes_y, above / (1 - baseline_axes_y))
        sv_y_min_aligned = sv_baseline - baseline_axes_y * total_range
        sv_y_max_aligned = sv_baseline + (1 - baseline_axes_y) * total_range
    else:
        sv_baseline = None
        sv_y_min_aligned = None
        sv_y_max_aligned = None

    if diff_feature_indices:
        diff_baseline = 0
        diff_values = stat_values[:, diff_feature_indices]
        diff_stds = stds[:, diff_feature_indices]
        diff_y_min = np.min(diff_values - diff_stds)
        diff_y_max = np.max(diff_values + diff_stds)
        diff_y_min = min(diff_y_min, diff_baseline)
        diff_y_max = max(diff_y_max, diff_baseline)
        
        below = diff_baseline - diff_y_min
        above = diff_y_max - diff_baseline
        total_range = max(below / baseline_axes_y, above / (1 - baseline_axes_y))
        diff_y_min_aligned = diff_baseline - baseline_axes_y * total_range
        diff_y_max_aligned = diff_baseline + (1 - baseline_axes_y) * total_range
    else:
        diff_baseline = None
        diff_y_min_aligned = None
        diff_y_max_aligned = None

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(n_clusters)
    width = 0.8 / n_features

    axes = []
    ax_sv = None
    ax_diff = None
    
    if sv_feature_indices:
        ax_sv = ax
        axes.append(('sv', ax_sv, sv_feature_indices))
    
    if diff_feature_indices:
        if ax_sv is None:
            ax_diff = ax
        else:
            ax_diff = ax.twinx()
        axes.append(('diff', ax_diff, diff_feature_indices))

    for group_type, ax_feature, feature_indices in axes:
        if group_type == 'sv':
            baseline = sv_baseline
            y_min, y_max = sv_y_min_aligned, sv_y_max_aligned
            ylabel = 'Sv (dB)'
        else:
            baseline = diff_baseline
            y_min, y_max = diff_y_min_aligned, diff_y_max_aligned
            ylabel = 'Sv Difference (dB)'
        
        ax_feature.set_ylim(y_min, y_max)
        ax_feature.set_ylabel(ylabel, fontsize=10)
        
        for j in feature_indices:
            feature_name = feature_coords[j]
            offset = (j - n_features/2 + 0.5) * width
            feature_values = stat_values[:, j]
            feature_stds = stds[:, j]
            
            for i, (cluster_id, color) in enumerate(zip(cluster_ids, cluster_color_list)):
                ax_feature.bar(
                    x[i] + offset,
                    feature_values[i] - baseline,
                    width,
                    yerr=feature_stds[i],
                    bottom=baseline,
                    color=color,
                    alpha=1,
                    capsize=3,
                    edgecolor='black',
                    linewidth=2
                )
        
        ax_feature.axhline(
            y=baseline,
            color='black',
            linestyle='-',
            linewidth=2,
            alpha=0.9,
            zorder=0
        )
        
        if group_type == 'sv':
            ax_feature.axhline(y=-80, color='black', linestyle=':', linewidth=2, alpha=0.7, zorder=1)
            if ax_sv == ax:
                ax_feature.grid(True, alpha=0.3, linestyle='--', axis='y')
        else:
            ax_feature.axhline(y=3, color='blue', linestyle=':', linewidth=2, alpha=0.7, zorder=2)
            ax_feature.axhline(y=-3, color='blue', linestyle=':', linewidth=2, alpha=0.7, zorder=2)

    feature_handles = []
    feature_labels = []
    for j, feature_name in enumerate(feature_coords):
        feature_handles.append(Line2D([0], [0], marker='s', color='w', 
                                     markerfacecolor='gray', markersize=8, 
                                     markeredgecolor='black', linewidth=0))
        feature_labels.append(str(feature_name))
    
    ax.set_xlabel('Cluster', fontsize=12)
    if title is None:
        if stat_type == 'mean':
            title = f'Cluster Statistics (Mean +/-1 SD): {cluster_data_name}\n{data_description}'
        elif stat_type == 'min':
            title = f'Cluster Statistics (Min, +/-1 SD of mean): {cluster_data_name}\n{data_description}'
        elif stat_type == 'max':
            title = f'Cluster Statistics (Max, +/-1 SD of mean): {cluster_data_name}\n{data_description}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Cluster {cid}' if cid != -1 else 'Noise' for cid in cluster_ids], 
                       rotation=45, ha='right')
    
    ax.legend(feature_handles, feature_labels, loc='upper center', bbox_to_anchor=(0.5, -0.15),
             ncol=1, fontsize=8, title='Features', frameon=True)
    
    plt.tight_layout()
    bottom_margin = 0.2 + (n_features * 0.03)
    plt.subplots_adjust(bottom=bottom_margin, right=0.88)
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()
    
    return fig, [ax_sv, ax_diff] if ax_diff else [ax_sv], stats_dict


def plot_dbscan_cluster_hierarchy(model, cluster_colors_by_index=None):
    """Plot HDBSCAN condensed tree with colours matching cluster labels.

    Args:
        model (hdbscan.HDBSCAN): Fitted HDBSCAN model.
        cluster_colors_by_index (list[str] or None): Hex colour strings
            indexed by cluster label. A default palette is used when
            ``None``.

    Returns:
        tuple: ``(label_to_tree, palette_by_tree_order)`` -- mapping from
        final cluster label to condensed-tree cluster id, and colour
        palette ordered by tree cluster id.
    """
    from aa_si_utils import utils

    if cluster_colors_by_index is None:
        cluster_colors_by_index = DEFAULT_CLUSTER_COLORS
    
    final_labels = model.labels_
    selected_clusters = model.condensed_tree_._select_clusters()
    unique_labels = sorted([l for l in set(final_labels) if l != -1])
    
    max_label = max(unique_labels) if unique_labels else 0
    if len(cluster_colors_by_index) <= max_label:
        num_additional_colors = max_label - len(cluster_colors_by_index) + 1
        hue_offset = 0.3
        additional_colors = utils.generate_colors(hue_offset, num_additional_colors)
        cluster_colors_by_index = cluster_colors_by_index + additional_colors
        print(f"Warning: Generated {num_additional_colors} additional colors for clusters beyond base palette")
    
    label_to_tree = {}
    sorted_tree_clusters = sorted(selected_clusters)
    
    for final_label in unique_labels:
        label_to_tree[final_label] = sorted_tree_clusters[final_label]
    
    palette_by_tree_order = []
    for idx, tree_cluster in enumerate(sorted_tree_clusters):
        final_label = list(label_to_tree.keys())[list(label_to_tree.values()).index(tree_cluster)]
        color = cluster_colors_by_index[final_label]
        palette_by_tree_order.append(color)
    
    ax = model.condensed_tree_.plot(select_clusters=True, selection_palette=palette_by_tree_order)
    from matplotlib.patches import Ellipse as _Ellipse
    import numpy as _np
    for patch in ax.patches:
        if isinstance(patch, _Ellipse):
            if isinstance(patch._width, _np.ndarray):
                patch._width = patch._width.item()
            if isinstance(patch._height, _np.ndarray):
                patch._height = patch._height.item()
            if isinstance(getattr(patch, '_center', None), _np.ndarray):
                patch._center = tuple(float(c) for c in patch._center)
    plt.title('HDBSCAN Condensed Tree')
    plt.show()
    
    return label_to_tree, palette_by_tree_order
