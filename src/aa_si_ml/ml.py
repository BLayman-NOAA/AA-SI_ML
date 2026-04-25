# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries

import logging

import numpy as np
import xarray as xr
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler,
    Normalizer, PowerTransformer, QuantileTransformer,
)
from scipy.stats import norm
import umap
import echopype as ep
from aa_si_visualization import echogram
from aa_si_utils import utils

from .constants import DEFAULT_CLUSTER_COLORS, SV_MIN_VALID, SV_MAX_VALID
from .ml_algorithms import (
    apply_dbscan_clustering,
    apply_kmeans_clustering,
    apply_min_cluster_size_filter,
    assign_noise_by_soft_membership,
    _calculate_silhouette,
    retrieve_background_cluster,
)
from .plotting_and_logging import (
    visualize_normalized_data_histogram,
    plot_cluster_statistics,
    plot_dbscan_cluster_hierarchy,
    print_basic_cluster_stats,
    print_cluster_statistics,
)

logger = logging.getLogger(__name__)



def add_cluster_mask(ds_Sv, cluster_labels_gridded, cluster_label=None,
                     use_corrected=True, mask_name='cluster_mask'):
    """Create a mask that excludes a cluster from the dataset.

    When cluster_label is None the largest cluster is identified and
    excluded. Pass an explicit integer label to exclude a specific cluster.

    Args:
        ds_Sv (xr.Dataset): Dataset containing Sv data.
        cluster_labels_gridded (xr.DataArray): 2-D gridded cluster labels
            (ping_time x range_sample).
        cluster_label (int or None): Cluster label to exclude. When None
            the largest cluster (by point count) is used. Defaults to None.
        use_corrected (bool): Use Sv_corrected when available.
            Defaults to True.
        mask_name (str): Name for the stored mask variable.
            Defaults to 'cluster_mask'.

    Returns:
        xr.Dataset: Input dataset with the mask added as a new variable.
    """
    sv_var = 'Sv_corrected' if use_corrected and 'Sv_corrected' in ds_Sv else 'Sv'
    sv_data = ds_Sv[sv_var]

    if cluster_label is None:
        unique_labels, counts = np.unique(cluster_labels_gridded, return_counts=True)
        valid = unique_labels >= 0
        valid_labels = unique_labels[valid]
        valid_counts = counts[valid]
        if len(valid_labels) == 0:
            logger.warning("No valid clusters found (all points are noise)")
            ds_Sv[mask_name] = xr.zeros_like(sv_data, dtype=bool)
            return ds_Sv
        cluster_label = int(valid_labels[np.argmax(valid_counts)])
        logger.info("Largest cluster: label %d with %s points", cluster_label,
                    f"{int(valid_counts[np.argmax(valid_counts)]):,}")

    cluster_2d_mask = cluster_labels_gridded == cluster_label
    logger.info("Masking cluster %d with %s points", cluster_label,
                f"{int(cluster_2d_mask.sum().values):,}")

    valid_mask = xr.ones_like(sv_data, dtype=bool)
    final_mask = valid_mask & ~cluster_2d_mask.broadcast_like(sv_data)

    ds_Sv[mask_name] = final_mask
    ds_Sv[mask_name].attrs['long_name'] = f'Valid data mask excluding cluster {cluster_label}'
    ds_Sv[mask_name].attrs['description'] = f'Mask that excludes cluster label {cluster_label}'
    ds_Sv[mask_name].attrs['source_variable'] = cluster_labels_gridded

    return ds_Sv


def add_largest_cluster_mask(ds_Sv, cluster_labels_gridded, use_corrected=True,
                             mask_name='largest_cluster_mask'):
    """Create a mask excluding the largest cluster. Wraps add_cluster_mask.

    Args:
        ds_Sv (xr.Dataset): Dataset containing Sv data.
        cluster_labels_gridded (xr.DataArray): 2-D gridded cluster labels.
        use_corrected (bool): Use Sv_corrected when available. Defaults to True.
        mask_name (str): Name for the stored mask variable.
            Defaults to 'largest_cluster_mask'.

    Returns:
        xr.Dataset: Input dataset with the mask added.
    """
    return add_cluster_mask(ds_Sv, cluster_labels_gridded, cluster_label=None,
                            use_corrected=use_corrected, mask_name=mask_name)


def add_cluster_label_mask(ds_Sv, cluster_labels_gridded, cluster_label,
                           use_corrected=True, mask_name='cluster_mask'):
    """Create a mask excluding a specific cluster label. Wraps add_cluster_mask.

    Args:
        ds_Sv (xr.Dataset): Dataset containing Sv data.
        cluster_labels_gridded (xr.DataArray): 2-D gridded cluster labels.
        cluster_label (int): The specific cluster label to mask out.
        use_corrected (bool): Use Sv_corrected when available. Defaults to True.
        mask_name (str): Name for the stored mask variable.
            Defaults to 'cluster_mask'.

    Returns:
        xr.Dataset: Dataset with the mask added.
    """
    return add_cluster_mask(ds_Sv, cluster_labels_gridded, cluster_label=cluster_label,
                            use_corrected=use_corrected, mask_name=mask_name)


def get_grid_coordinates(ds_Sv, data_var):
    """Return the two spatial dimension names for a data variable.

    Finds the non-channel, non-feature dimensions of ``data_var`` and
    returns them in ``(time-like, range-like)`` order.

    Args:
        ds_Sv (xr.Dataset): Dataset containing the data variable.
        data_var (str): Name of the data variable to inspect.

    Returns:
        list[str]: Two-element list of spatial coordinate names.

    Raises:
        ValueError: If the number of spatial coordinates is not exactly 2.
    """
    
    # Get the actual dimensions of the data variable (not all associated coordinates)
    all_dims = list(ds_Sv[data_var].dims)
    
    # Filter out 'channel' coordinate and any existing grid_index coordinates
    grid_coords = [dim for dim in all_dims 
                   if dim != 'channel' and not dim.startswith('feature') and not dim.startswith('grid_index')]
    
    # Validate we got exactly 2 spatial coordinates
    if len(grid_coords) != 2:
        raise ValueError(
            f"Expected 2 spatial coordinates (excluding channel), found {len(grid_coords)}: {grid_coords}"
        )
    
    preferred_first_dims = ['time', 'ping_time', 'distance']
    if grid_coords[0] not in preferred_first_dims:
        grid_coords = [grid_coords[1], grid_coords[0]]
        logger.debug("Swapped coordinate order to: %s", grid_coords)

    return grid_coords


def add_valid_data_mask(ds_Sv, remove_nan=True, mask_invalid_values=True, mask_name='valid_mask', custom_mask_name=None, data_var='Sv'):
    """Add a boolean validity mask identifying clean data points for ML.

    Creates a per-element mask over the Sv data, marking points that are
    free of NaNs, extreme artifacts, and (optionally) a user-supplied mask.

    Args:
        ds_Sv (xr.Dataset): Dataset containing acoustic backscatter data.
        remove_nan (bool): Mask NaN values. Defaults to True.
        mask_invalid_values (bool): Mask extreme values outside [-200, 50] dB.
            Defaults to True.
        mask_name (str): Name for the stored mask variable.
            Defaults to 'valid_mask'.
        custom_mask_name (str or None): Name of an existing boolean variable
            in *ds_Sv* to combine with the computed mask. Defaults to None.
        data_var (str): Name of the Sv variable to analyse.
            Defaults to 'Sv'.

    Returns:
        xr.Dataset: Input dataset with the boolean mask added.
    """
    
    ds_with_mask = ds_Sv

    sv_var = data_var
    logger.info("Analyzing '%s' for validity (shape %s)", sv_var, ds_Sv[sv_var].shape)

    sv_data = ds_Sv[sv_var]
    valid_mask = xr.ones_like(sv_data, dtype=bool)

    if remove_nan:
        nan_mask = xr.ufuncs.isnan(sv_data)
        valid_mask = valid_mask & ~nan_mask
        logger.info("Masked %s NaN values", f"{int(nan_mask.sum().values):,}")

    if mask_invalid_values:
        artifact_mask = (sv_data < SV_MIN_VALID) | (sv_data > SV_MAX_VALID)
        valid_mask = valid_mask & ~artifact_mask
        logger.info(
            "Masked %s extreme values (< %d or > %d dB)",
            f"{int(artifact_mask.sum().values):,}", SV_MIN_VALID, SV_MAX_VALID
        )

    if custom_mask_name is not None:
        custom_mask = ds_Sv[custom_mask_name]
        if custom_mask.shape != sv_data.shape:
            raise ValueError("custom_mask must have the same shape as the Sv data")
        valid_mask = valid_mask & custom_mask
        logger.info(
            "Applied custom mask, masking additional %s values",
            f"{int((~custom_mask).sum().values):,}"
        )
    
    ds_with_mask[mask_name] = valid_mask
    ds_with_mask[mask_name].attrs['long_name'] = 'Valid data mask for machine learning'
    ds_with_mask[mask_name].attrs['description'] = 'True where data is valid for ML analysis'
    ds_with_mask[mask_name].attrs['source_variable'] = sv_var

    return ds_with_mask


def create_ml_index_coordinate(ds_with_mask, data_var='Sv', dataset_name='ml_data_clean'):
    """Assign a unique integer index to every grid cell.

    The index is stored as a 2-D coordinate and is used to track
    individual data points through flattening, ML, and regridding.

    Args:
        ds_with_mask (xr.Dataset): Dataset with a valid-data mask.
        data_var (str): Name of the Sv variable whose grid shape to use.
            Defaults to 'Sv'.
        dataset_name (str): Base name used for coordinate naming.
            Defaults to 'ml_data_clean'.

    Returns:
        xr.Dataset: Dataset with a ``grid_index`` coordinate added.
    """
    ds_with_index = ds_with_mask
    grid_coords = get_grid_coordinates(ds_with_mask, data_var)
    logger.info("Creating grid index coordinate based on %s", grid_coords)

    coord_1_size = ds_with_mask.sizes[grid_coords[0]]
    coord_2_size = ds_with_mask.sizes[grid_coords[1]]
    total_points = coord_1_size * coord_2_size
    grid_index_grid = np.arange(total_points).reshape(coord_1_size, coord_2_size)

    if grid_coords[0] == 'ping_time' and grid_coords[1] == 'range_sample':
        grid_index_name = 'grid_index'
    else:
        grid_index_name = f'grid_index_{dataset_name}'

    # Store as coordinate
    ds_with_index.coords[grid_index_name] = ((grid_coords[0], grid_coords[1]), grid_index_grid)
    ds_with_index[grid_index_name].attrs['long_name'] = 'Grid data point index'
    ds_with_index[grid_index_name].attrs['description'] = f'Unique index for each grid point (dims: {grid_coords[0]}, {grid_coords[1]}), preserved in ML operations'
    ds_with_index[grid_index_name].attrs['grid_coordinates'] = grid_coords
    
    print(f"Created {grid_index_name} coordinate with {total_points:,} unique indices")
    print(f"Grid dimensions: {grid_coords[0]} ({coord_1_size}), {grid_coords[1]} ({coord_2_size})")

    return ds_with_index


def extract_ml_data_flattened(ds_ml_ready, data_var='Sv', mask_name='valid_mask', 
                            dataset_name='ml_data_clean', feature_strategy='channels', 
                            baseline_channel=2, **feature_kwargs):
    """Extract valid data points into a flat array with configurable features.

    Reads the validity mask, selects points valid across all channels,
    and constructs a 2-D ``(sample, feature)`` array ready for ML.

    Args:
        ds_ml_ready (xr.Dataset): Dataset with ``grid_index`` coordinate and
            a valid-data mask.
        data_var (str): Name of the Sv variable to extract. Defaults to 'Sv'.
        mask_name (str): Name of the boolean mask variable.
            Defaults to 'valid_mask'.
        dataset_name (str): Base name used for coordinate naming.
            Defaults to 'ml_data_clean'.
        feature_strategy (str): How to construct features from channels.
            'channels': raw channel values (default).
            'baseline_plus_differences': one baseline channel plus
            differences of the remaining channels from the baseline.
            'mean_centered': per-sample mean Sv plus each channel centered
            around that mean (captures intensity and spectral shape).
            'custom': user-provided function via feature_kwargs.
        baseline_channel (int): Channel index used as the baseline when
            *feature_strategy* is ``'baseline_plus_differences'``.
            Defaults to 2.
        **feature_kwargs: Extra arguments forwarded to a ``'custom'`` feature
            function.

    Returns:
        tuple: A tuple of (ml_data_flat, grid_indices) where:
            ml_data_flat (xr.DataArray): Flattened data with shape
            ``(n_samples, n_features)``.
            grid_indices (np.ndarray): Grid index for each sample, used to
            map results back to the original grid.
    """

    data = ds_ml_ready[data_var]
    valid_mask = ds_ml_ready[mask_name]

    grid_coords = get_grid_coordinates(ds_ml_ready, data_var)

    if grid_coords[0] == 'ping_time' and grid_coords[1] == 'range_sample':
        grid_index_name = 'grid_index'
    else:
        grid_index_name = f'grid_index_{dataset_name}'

    if grid_index_name not in ds_ml_ready.coords:
        raise ValueError(f"Dataset must have '{grid_index_name}' coordinate. Run create_ml_index_coordinate() first.")

    grid_index_grid = ds_ml_ready[grid_index_name]

    valid_samples = valid_mask.all(dim='channel')
    ping_indices, range_indices = np.where(valid_samples.values)

    raw_data_values = data.values[:, ping_indices, range_indices].T  # (n_samples, n_channels)
    grid_indices = grid_index_grid.values[ping_indices, range_indices]

    if feature_strategy == 'channels':
        feature_data = raw_data_values
        feature_coords = np.array([str(ch) for ch in data.coords['channel'].values], dtype=str)
        feature_dim_name = f'feature_{dataset_name}'

    elif feature_strategy == 'baseline_plus_differences':
        if baseline_channel >= raw_data_values.shape[1]:
            raise ValueError(f"baseline_channel {baseline_channel} exceeds number of channels {raw_data_values.shape[1]}")
            
        baseline_values = raw_data_values[:, baseline_channel:baseline_channel+1]
        other_channels = [i for i in range(raw_data_values.shape[1]) if i != baseline_channel]
        difference_values = raw_data_values[:, other_channels] - baseline_values
        
        feature_data = np.concatenate([
            baseline_values,
            difference_values
        ], axis=1)

        baseline_name = f"baseline_{data.coords['channel'].values[baseline_channel]}"
        diff_names = [f"diff_{data.coords['channel'].values[i]}_minus_{data.coords['channel'].values[baseline_channel]}" 
                     for i in other_channels]
        feature_coords = np.array([baseline_name] + diff_names, dtype=str)
        feature_dim_name = f'feature_{dataset_name}'

    elif feature_strategy == 'mean_centered':
        sample_means = np.mean(raw_data_values, axis=1, keepdims=True)
        centered_values = raw_data_values - sample_means

        feature_data = np.concatenate([
            sample_means,
            centered_values
        ], axis=1)

        channel_names = data.coords['channel'].values
        feature_coords = np.array(
            ["mean_Sv"] + [f"centered_{ch}" for ch in channel_names], 
            dtype=str
        )
        feature_dim_name = f'feature_{dataset_name}'
        
    elif feature_strategy == 'custom':
        feature_function = feature_kwargs.get('feature_function')
        if feature_function is None:
            raise ValueError("feature_function must be provided for custom strategy")
        feature_data, feature_coords = feature_function(raw_data_values, data.coords['channel'])
        feature_dim_name = feature_kwargs.get('feature_dim_name', f'feature_{dataset_name}')
        
    else:
        raise ValueError(f"Unknown feature_strategy: {feature_strategy}")

    n_samples = len(ping_indices)
    sample_index_coord_name = f'{dataset_name}_sample_index'
    sample_index_coord = np.arange(n_samples)
    
    # Create the flattened DataArray
    ml_data_flat = xr.DataArray(
        feature_data,
        dims=[sample_index_coord_name, feature_dim_name],
        coords={
            sample_index_coord_name: sample_index_coord,
            feature_dim_name: feature_coords
        }
    )
    
    ml_data_flat.attrs['long_name'] = f'Flattened {data_var} ({feature_strategy})'
    ml_data_flat.attrs['description'] = f'Valid data in efficient flattened format using {feature_strategy} features'
    ml_data_flat.attrs['source_variable'] = data_var
    ml_data_flat.attrs['feature_strategy'] = feature_strategy
    ml_data_flat.attrs['grid_name'] = grid_index_name
    if feature_strategy == 'baseline_plus_differences':
        ml_data_flat.attrs['baseline_channel'] = baseline_channel
        ml_data_flat.attrs['baseline_frequency'] = str(data.coords['channel'].values[baseline_channel])
    
    logger.info(
        "Extracted %s valid samples using '%s' strategy, shape %s",
        f"{n_samples:,}", feature_strategy, ml_data_flat.shape
    )

    return ml_data_flat, grid_indices


def store_ml_data_flattened(ds_ml_ready, ml_data_flat, grid_indices, dataset_name):
    """Store flattened ML data in the dataset with a universal index mapping.

    Adds the flattened data array and a ``sample_index -> grid_index``
    mapping variable so that results can later be regridded.

    Args:
        ds_ml_ready (xr.Dataset): Dataset to store data in.
        ml_data_flat (xr.DataArray): Flattened data with
            ``(sample_index, feature)`` dimensions.
        grid_indices (np.ndarray): Grid index for each sample.
        dataset_name (str): Base name used for variable/coordinate naming.

    Returns:
        xr.Dataset: Dataset with the flattened data and mapping added.
    """
    ds_ml_ready[dataset_name] = ml_data_flat

    mapping_name = f'{dataset_name}_sample_index_to_grid_index'
    sample_index_coord_name = f'{dataset_name}_sample_index'

    if mapping_name not in ds_ml_ready:
        ds_ml_ready[mapping_name] = xr.DataArray(
            grid_indices,
            dims=[sample_index_coord_name]
        )
        ds_ml_ready[mapping_name].attrs['long_name'] = f'Sample-index to grid-index mapping for {dataset_name}'
        ds_ml_ready[mapping_name].attrs['description'] = f'Maps {sample_index_coord_name} to original grid_index'
        logger.info("Created %s mapping with %d samples", mapping_name, len(grid_indices))
    else:
        existing_mapping = ds_ml_ready[mapping_name].values
        if not np.array_equal(existing_mapping, grid_indices):
            raise ValueError(f"Grid indices don't match existing mapping for {dataset_name}")
        logger.debug("Using existing %s mapping", mapping_name)

    logger.info("Stored %s in flattened format", dataset_name)

    return ds_ml_ready


def reshape_data_for_ml(ds_Sv, data_var='Sv_corrected', dataset_name='ml_data_clean',
                          remove_nan=True, mask_invalid_values=True, custom_data_mask_name=None,
                          feature_strategy='channels', baseline_channel=0, **feature_kwargs):
    """Prepare an xarray Dataset for ML by masking, indexing, and flattening.

    Orchestrates validity masking, grid-index creation, and feature
    extraction into a single call.  Channels that are entirely NaN are
    automatically dropped.

    Args:
        ds_Sv (xr.Dataset): Dataset containing acoustic backscatter data.
        data_var (str): Sv variable to use. Defaults to 'Sv_corrected'.
        dataset_name (str): Base name for stored ML variables.
            Defaults to 'ml_data_clean'.
        remove_nan (bool): Mask NaN values. Defaults to True.
        mask_invalid_values (bool): Mask extreme artifact values.
            Defaults to True.
        custom_data_mask_name (str or None): Name of a pre-computed mask
            already present in *ds_Sv* to combine with the auto-generated
            mask. Defaults to None.
        feature_strategy (str): Feature extraction strategy forwarded to
            ``extract_ml_data_flattened``. Defaults to 'channels'.
        baseline_channel (int): Channel index for baseline difference
            features. Defaults to 0.
        **feature_kwargs: Extra arguments for feature extraction.

    Returns:
        xr.Dataset: Dataset with flattened ML data, validity mask, and
        index mapping.
    """

    mask_name = f"{dataset_name}_valid_mask"
    if custom_data_mask_name is not None:
        mask_name = f"{dataset_name}_{custom_data_mask_name}"

    logger.info("Creating validity mask...")

    if data_var not in ds_Sv:
        raise ValueError(f"Data variable '{data_var}' not found in dataset")

    ds_with_mask = ds_Sv
    sv_var = data_var
    data = ds_Sv[data_var]

    channels_to_keep = []
    channels_all_nan = []
    
    for i, channel in enumerate(data.coords['channel'].values):
        channel_data = data.isel(channel=i)
        if np.all(np.isnan(channel_data.values)):
            channels_all_nan.append(channel)
        else:
            channels_to_keep.append(channel)
    
    # Notify user and filter if necessary
    if len(channels_all_nan) > 0:
        logger.warning(
            "Found %d channel(s) with all NaN values: %s. Excluding from processing.",
            len(channels_all_nan), channels_all_nan
        )
        logger.info("Keeping %d valid channel(s): %s", len(channels_to_keep), channels_to_keep)

        # Filter dataset to only include valid channels
        ds_Sv = ds_Sv.sel(channel=channels_to_keep)

        # Adjust baseline_channel if necessary
        if baseline_channel >= len(channels_to_keep):
            old_baseline = baseline_channel
            baseline_channel = 0  # Default to first valid channel
            logger.warning(
                "baseline_channel %d is out of range after filtering. Setting to %d (channel: %s).",
                old_baseline, baseline_channel, channels_to_keep[baseline_channel]
            )
    

    if data.dims[0] != 'channel':
        logger.debug("Reordering dimensions from %s to channel-first.", data.dims)
        spatial_dims = [dim for dim in data.dims if dim != 'channel']
        desired_order = ['channel'] + spatial_dims
        data = data.transpose(*desired_order)
        logger.debug("New dimension order: %s", data.dims)
        ds_working = ds_Sv.copy()
        ds_working[data_var] = data
    else:
        ds_working = ds_Sv

    ds_Sv = ds_working

    ds_with_mask = add_valid_data_mask(ds_Sv,
                                    remove_nan=remove_nan,
                                    mask_invalid_values=mask_invalid_values,
                                    mask_name=mask_name,
                                    custom_mask_name=custom_data_mask_name,
                                    data_var=sv_var)
    
    ds_ml_ready = create_ml_index_coordinate(ds_with_mask, data_var=sv_var, dataset_name=dataset_name)

    logger.info("Preparing ML data from '%s' as '%s'.", sv_var, dataset_name)
    logger.debug("Data shape: %s", ds_ml_ready[sv_var].shape)

    ml_data_flat, grid_indices = extract_ml_data_flattened(
        ds_ml_ready, sv_var, mask_name=mask_name, dataset_name=dataset_name,
        feature_strategy=feature_strategy, baseline_channel=baseline_channel, **feature_kwargs
    )
    ds_ml_ready = store_ml_data_flattened(ds_ml_ready, ml_data_flat, grid_indices, dataset_name)

    logger.info("Data stored as '%s'.", dataset_name)

    return ds_ml_ready


def add_auxiliary_features(ds_ml_ready, dataset_name='ml_data_clean', features=None, echodata=None):
    """Append auxiliary coordinate-derived features to the flattened ML dataset.

    Slots between :func:`reshape_data_for_ml` and :func:`normalize_data` in
    the pipeline.  Built-in feature names are ``'depth'``,
    ``'ping_time_seconds'``, ``'seafloor_depth'``, and ``'altitude'``.
    User-defined features are supported via callable dicts.

    Args:
        ds_ml_ready (xr.Dataset): Dataset produced by
            :func:`reshape_data_for_ml`.
        dataset_name (str): Base ML dataset name.  Defaults to
            ``'ml_data_clean'``.
        features (list or None): Features to append.  Each element is
            either a ``str`` naming a built-in feature or a ``dict`` of
            the form ``{'name': str, 'func': callable}`` for a
            user-defined feature.  The callable receives
            ``(ds_ml_ready, ping_indices, range_indices)`` and must
            return a 1-D :class:`numpy.ndarray` of shape
            ``(n_samples,)``.  When ``None`` the function is a no-op.
        echodata: EchoData object providing
            ``echodata["Vendor_specific"]["detected_seafloor_depth"]``
            (dims ``channel × ping_time``).  Required when
            ``'seafloor_depth'`` or ``'altitude'`` is requested.

    Returns:
        xr.Dataset: Dataset with auxiliary features appended to
        ``dataset_name``.

    Raises:
        ValueError: If a requested built-in feature cannot be computed
            from the available data.
    """
    if features is None:
        return ds_ml_ready

    SEAFLOOR_FEATURES = {'seafloor_depth', 'altitude'}
    seafloor_requested = any(
        (f if isinstance(f, str) else f.get('name', '')) in SEAFLOOR_FEATURES
        for f in features
    )
    if seafloor_requested and echodata is None:
        raise ValueError(
            "'seafloor_depth' and 'altitude' require the 'echodata' parameter "
            "to access detected_seafloor_depth from Vendor_specific data."
        )

    # --- Recover ping/range positions from the grid-index mapping ---
    mapping_name = f'{dataset_name}_sample_index_to_grid_index'
    if mapping_name not in ds_ml_ready:
        raise ValueError(
            f"Dataset missing '{mapping_name}'. Run reshape_data_for_ml() first."
        )

    grid_indices_flat = ds_ml_ready[mapping_name].values  # (n_samples,)

    # Determine grid shape from the grid_index coordinate
    grid_index_name = (
        'grid_index'
        if 'grid_index' in ds_ml_ready.coords
        else f'grid_index_{dataset_name}'
    )
    grid_index_coord = ds_ml_ready[grid_index_name]
    grid_coords = list(grid_index_coord.dims)  # [ping_dim, range_dim]
    grid_shape = tuple(ds_ml_ready.sizes[d] for d in grid_coords)

    ping_indices, range_indices = np.unravel_index(grid_indices_flat, grid_shape)

    ping_dim = grid_coords[0]   # e.g. 'ping_time'
    range_dim = grid_coords[1]  # e.g. 'range_sample' or 'echo_range'

    # --- Pre-compute seafloor depth on the dataset's ping grid if needed ---
    seafloor_on_grid = None
    if seafloor_requested:
        raw_sfd = echodata["Vendor_specific"]["detected_seafloor_depth"]
        # raw_sfd dims: (channel, ping_time_raw)
        # Average across channels, then resample to dataset ping times
        sfd_mean = raw_sfd.mean(dim='channel')  # (ping_time_raw,)
        ds_ping_times = ds_ml_ready[ping_dim]   # binned ping times
        # Nearest-neighbour resampling
        seafloor_on_grid = sfd_mean.sel(
            ping_time=ds_ping_times, method='nearest'
        ).values  # (n_pings_in_grid,)

    # --- Build each auxiliary column ---
    new_columns = []   # list of (name, np.ndarray shape (n_samples,))

    for feature_spec in features:
        if isinstance(feature_spec, str):
            name = feature_spec
        else:
            name = feature_spec['name']

        if name == 'depth':
            if 'echo_range' not in ds_ml_ready:
                raise ValueError(
                    "'depth' feature requires 'echo_range' variable in the dataset. "
                    "Ensure compute_Sv() has been called before reshape_data_for_ml()."
                )
            echo_range = ds_ml_ready['echo_range']
            # echo_range dims: (channel, ping_dim, range_dim) — mean across channels
            if 'channel' in echo_range.dims:
                echo_range_2d = echo_range.mean(dim='channel')
            else:
                echo_range_2d = echo_range
            if echo_range_2d.values.ndim == 1:
                # MVBS case: echo_range IS the range coordinate (1D depth values)
                values = echo_range_2d.values[range_indices]
            else:
                # Raw Sv case: echo_range is a (ping_dim, range_dim) variable
                # Explicitly transpose to match grid_coords order before numpy indexing
                echo_range_2d = echo_range_2d.transpose(ping_dim, range_dim)
                values = echo_range_2d.values[ping_indices, range_indices]

        elif name == 'ping_time_seconds':
            ping_times_ns = ds_ml_ready[ping_dim].values.astype('float64')
            t0 = ping_times_ns.min()
            ping_times_sec = (ping_times_ns - t0) / 1e9
            values = ping_times_sec[ping_indices]

        elif name == 'seafloor_depth':
            values = seafloor_on_grid[ping_indices]

        elif name == 'altitude':
            if 'echo_range' not in ds_ml_ready:
                raise ValueError(
                    "'altitude' feature requires 'echo_range' variable in the dataset."
                )
            echo_range = ds_ml_ready['echo_range']
            if 'channel' in echo_range.dims:
                echo_range_2d = echo_range.mean(dim='channel')
            else:
                echo_range_2d = echo_range
            if echo_range_2d.values.ndim == 1:
                depth_values = echo_range_2d.values[range_indices]
            else:
                echo_range_2d = echo_range_2d.transpose(ping_dim, range_dim)
                depth_values = echo_range_2d.values[ping_indices, range_indices]
            values = seafloor_on_grid[ping_indices] - depth_values

        elif isinstance(feature_spec, dict) and 'func' in feature_spec:
            values = feature_spec['func'](ds_ml_ready, ping_indices, range_indices)
            if not isinstance(values, np.ndarray) or values.ndim != 1 or len(values) != len(ping_indices):
                raise ValueError(
                    f"Custom feature function for '{name}' must return a 1-D ndarray "
                    f"of length {len(ping_indices)}, got shape {getattr(values, 'shape', 'unknown')}."
                )
        else:
            raise ValueError(
                f"Unknown built-in feature '{name}'. Supported built-ins are: "
                "'depth', 'ping_time_seconds', 'seafloor_depth', 'altitude'. "
                "For custom features provide a dict with 'name' and 'func' keys."
            )

        new_columns.append((name, values.astype(np.float64)))
        logger.info("Computed auxiliary feature '%s' with shape %s.", name, values.shape)

    # --- Concatenate onto existing DataArray ---
    source_da = ds_ml_ready[dataset_name]
    sample_dim = f'{dataset_name}_sample_index'
    feature_dim = f'feature_{dataset_name}'

    existing_data = source_da.values          # (n_samples, n_existing_features)
    existing_coords = list(source_da.coords[feature_dim].values)

    new_data = np.stack([col for _, col in new_columns], axis=1)   # (n_samples, n_new)
    new_coord_labels = [name for name, _ in new_columns]

    combined_data = np.concatenate([existing_data, new_data], axis=1)
    combined_coords = existing_coords + new_coord_labels

    updated_da = xr.DataArray(
        combined_data,
        dims=[sample_dim, feature_dim],
        coords={
            sample_dim: source_da.coords[sample_dim],
            feature_dim: np.array(combined_coords, dtype=str),
        },
        attrs=source_da.attrs.copy(),
    )

    # Track auxiliary feature names for normalize_data warnings
    existing_aux = list(updated_da.attrs.get('auxiliary_features', []))
    updated_da.attrs['auxiliary_features'] = existing_aux + new_coord_labels

    # Drop the old variable and its stale dimension coordinate before assigning.
    # Without this, xarray silently reindexes the new DataArray against the
    # existing feature coordinate (which has fewer entries), discarding the
    # newly added auxiliary columns.
    if dataset_name in ds_ml_ready:
        ds_ml_ready = ds_ml_ready.drop_vars(dataset_name)
    if feature_dim in ds_ml_ready.coords:
        ds_ml_ready = ds_ml_ready.drop_vars(feature_dim)

    ds_ml_ready[dataset_name] = updated_da

    logger.info(
        "Added %d auxiliary feature(s) to '%s': %s. New shape: %s.",
        len(new_columns), dataset_name, new_coord_labels, updated_da.shape,
    )

    return ds_ml_ready


def _build_scaler(method, X, n_quantiles=100, flatten_weight=1):
    """Fit and apply a named scaler to array X.  Returns (X_normalized, scaler).

    Handles all methods supported by normalize_data except 'l2' and global
    variants (those are handled inline).  ``X`` must already be the column
    subset that this scaler should act on.
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method in ('flatten', 'power', 'flatten_plus_umap'):
        scaler = PowerTransformer(method='yeo-johnson', standardize=True)
    elif method == 'quantile':
        n_q = min(len(X), n_quantiles)
        scaler = QuantileTransformer(output_distribution='uniform', n_quantiles=n_q)
    elif method == 'umap':
        n_components = 2
        scaler = umap.UMAP(
            n_components=n_components,
            n_neighbors=15,
            min_dist=0.1,
            metric='euclidean',
            random_state=42,
            verbose=True,
        )
        X = StandardScaler().fit_transform(X)
    else:
        raise ValueError(f"Unknown normalization method: '{method}'")

    X_out = scaler.fit_transform(X)

    if method in ('flatten', 'flatten_plus_umap'):
        X_flattened = norm.cdf(X_out)
        X_out = (1 - flatten_weight) * X_out + flatten_weight * X_flattened
        if method == 'flatten_plus_umap':
            umap_scaler = umap.UMAP(
                n_components=2,
                n_neighbors=30,
                min_dist=0.1,
                metric='euclidean',
                random_state=42,
                verbose=True,
            )
            X_out = umap_scaler.fit_transform(X_out)

    return X_out, scaler


def normalize_data(ds_ml_ready, method='standard', pre_l2_method='standard',
                  shift_positive=False, per_feature=True, dataset_name='ml_data_clean',
                  normalization_name=None, feature_weights=None, n_quantiles=100,
                  flatten_weight=1, per_group_methods=None):
    """Normalize flattened ML data using a variety of scaling methods.

    Supports per-feature and global normalization, optional L2 row
    normalization with a configurable pre-scaler, power-transform
    flattening, UMAP embedding, feature weighting, and per-group method
    overrides for auxiliary features.

    Args:
        ds_ml_ready (xr.Dataset): Dataset produced by reshape_data_for_ml.
        method (str): Default normalization method applied to all Sv-derived
            features (i.e. features not listed in *per_group_methods*). One
            of 'standard', 'robust', 'minmax', 'flatten', 'power',
            'quantile', 'umap', 'flatten_plus_umap', or 'l2'.
            Defaults to 'standard'.
        pre_l2_method (str): Scaler applied before L2 normalization when
            method is 'l2'. Defaults to 'standard'.
        shift_positive (bool): Shift all values to be positive after
            normalization. Defaults to False.
        per_feature (bool): When True, normalize each Sv-group feature
            independently; when False, use a single global scaler pooled
            across all Sv-group features.  Does not affect per-group
            auxiliary columns, which are always normalized per-column.
            Defaults to True.
        dataset_name (str): Base dataset name. Defaults to 'ml_data_clean'.
        normalization_name (str or None): Suffix for the stored result.
            Defaults to the method name.
        feature_weights (array-like or None): Per-feature multiplicative
            weights applied after normalization. Length must match the total
            number of features. Defaults to None.
        n_quantiles (int): Number of quantiles for 'quantile' method.
            Defaults to 100.
        flatten_weight (float): Blending weight for 'flatten' method's CDF
            transform. Defaults to 1.
        per_group_methods (dict[str, str] or None): Mapping of feature label
            to normalization method for auxiliary features that should be
            normalized differently from the Sv group.  Features not listed
            here fall back to *method*.  A warning is emitted for any
            auxiliary feature (tracked via ``attrs['auxiliary_features']``)
            that is not listed.  Example::

                per_group_methods={
                    'depth': 'minmax',
                    'ping_time_seconds': 'minmax',
                }

            Defaults to None (all features use *method*).

    Returns:
        xr.Dataset: Dataset with the normalized data added.
    """
    # --- Default normalization_name ---
    if normalization_name is None:
        if method == 'l2':
            normalization_name = f'l2_{pre_l2_method}_normalized' if pre_l2_method != 'none' else 'l2_normalized'
        else:
            normalization_name = f'{method}_normalized'

    X_clean, _, _ = extract_valid_samples_for_sklearn(ds_ml_ready, specific_data_name='', dataset_name=dataset_name)

    source_data = ds_ml_ready[dataset_name]
    feature_dim_name = [dim for dim in source_data.dims if dim != f'{dataset_name}_sample_index'][0]
    all_feature_labels = list(source_data.coords[feature_dim_name].values)
    auxiliary_features = list(source_data.attrs.get('auxiliary_features', []))

    # --- Warn for auxiliary features not covered by per_group_methods ---
    if per_group_methods is not None and auxiliary_features:
        for aux_name in auxiliary_features:
            if aux_name not in per_group_methods:
                logger.warning(
                    "Auxiliary feature '%s' is not listed in per_group_methods and will "
                    "be normalized with the default method '%s'. Specify it explicitly "
                    "in per_group_methods to suppress this warning.",
                    aux_name, method,
                )

    # --- Identify column indices for Sv group vs per-group overrides ---
    per_group_methods = per_group_methods or {}
    sv_col_indices = [i for i, lbl in enumerate(all_feature_labels) if lbl not in per_group_methods]
    group_col_map = {lbl: i for i, lbl in enumerate(all_feature_labels) if lbl in per_group_methods}

    X_sv = X_clean[:, sv_col_indices]  # columns handled by default method / global

    # ------------------------------------------------------------------ #
    #  Normalize the Sv group                                             #
    # ------------------------------------------------------------------ #
    if per_feature:
        if method == 'l2':
            if pre_l2_method in ('standard', 'standard_shifted'):
                pre_scaler = StandardScaler()
            elif pre_l2_method == 'robust':
                pre_scaler = RobustScaler()
            elif pre_l2_method == 'minmax':
                pre_scaler = MinMaxScaler()
            elif pre_l2_method == 'none':
                pre_scaler = None
            else:
                raise ValueError(f"Unknown pre_l2_method: {pre_l2_method}")

            pre_normalized = pre_scaler.fit_transform(X_sv) if pre_scaler else X_sv
            if pre_l2_method == 'standard_shifted':
                pre_normalized = pre_normalized + 2
            sv_scaler = Normalizer(norm='l2')
            X_sv_normalized = sv_scaler.fit_transform(pre_normalized)

            normalization_info = {
                'method': method,
                'pre_l2_method': pre_l2_method,
                'per_frequency': per_feature,
                'shift_positive': shift_positive,
            }
        else:
            X_sv_normalized, sv_scaler = _build_scaler(
                method, X_sv, n_quantiles=n_quantiles, flatten_weight=flatten_weight
            )
            normalization_info = {
                'method': method,
                'per_frequency': per_feature,
                'shift_positive': shift_positive,
                'feature_means': sv_scaler.mean_ if hasattr(sv_scaler, 'mean_') else None,
                'feature_scales': sv_scaler.scale_ if hasattr(sv_scaler, 'scale_') else None,
            }
    else:
        # Global normalization — pool all Sv-group columns together
        logger.info("Using global normalization on Sv group (%d columns).", len(sv_col_indices))
        X_flat = X_sv.flatten()

        if method == 'standard':
            global_mean = np.mean(X_flat)
            global_std = np.std(X_flat)
            X_sv_normalized = (X_sv - global_mean) / global_std
            normalization_info = {
                'method': 'standard_global', 'per_frequency': per_feature,
                'shift_positive': shift_positive,
                'global_mean': global_mean, 'global_std': global_std,
            }
        elif method == 'robust':
            global_median = np.median(X_flat)
            global_iqr = np.percentile(X_flat, 75) - np.percentile(X_flat, 25)
            X_sv_normalized = (X_sv - global_median) / global_iqr
            normalization_info = {
                'method': 'robust_global', 'per_frequency': per_feature,
                'shift_positive': shift_positive,
                'global_median': global_median, 'global_iqr': global_iqr,
            }
        elif method == 'minmax':
            global_min = np.min(X_flat)
            global_range = np.max(X_flat) - global_min
            X_sv_normalized = (X_sv - global_min) / global_range
            normalization_info = {
                'method': 'minmax_global', 'per_frequency': per_feature,
                'shift_positive': shift_positive,
                'global_min': global_min, 'global_range': global_range,
            }
        else:
            raise ValueError(
                f"Global normalization (per_feature=False) only supports "
                f"'standard', 'robust', and 'minmax', not '{method}'."
            )

    # ------------------------------------------------------------------ #
    #  Normalize per-group auxiliary columns                              #
    # ------------------------------------------------------------------ #
    # We'll reconstruct X_normalized column-by-column in original order
    X_normalized = np.empty_like(X_clean)
    for out_idx, sv_idx in enumerate(sv_col_indices):
        X_normalized[:, sv_idx] = X_sv_normalized[:, out_idx]

    for feature_label, col_idx in group_col_map.items():
        aux_method = per_group_methods[feature_label]
        col = X_clean[:, col_idx:col_idx + 1]
        col_normalized, _ = _build_scaler(
            aux_method, col, n_quantiles=n_quantiles, flatten_weight=flatten_weight
        )
        X_normalized[:, col_idx] = col_normalized[:, 0]
        logger.info(
            "Normalized auxiliary feature '%s' with method '%s'.",
            feature_label, aux_method,
        )

    # ------------------------------------------------------------------ #
    #  Post-processing: shift and weights                                 #
    # ------------------------------------------------------------------ #
    if shift_positive:
        min_value = X_normalized.min()
        if min_value < 0:
            X_normalized = X_normalized + abs(min_value) + 1e-6
            normalization_info['shift_amount'] = abs(min_value) + 1e-6

    if feature_weights is not None and per_feature:
        if len(feature_weights) != X_normalized.shape[1]:
            raise ValueError(
                f"feature_weights length ({len(feature_weights)}) must match "
                f"number of features ({X_normalized.shape[1]})"
            )
        X_normalized = X_normalized * feature_weights
        normalization_info['feature_weights'] = feature_weights
        logger.info("Applied feature weights: %s", feature_weights)

    if not per_feature and method != 'l2':
        scope = 'global across Sv features'
    elif method != 'l2':
        scope = 'per-feature (each feature independently)'
    else:
        scope = 'per-sample (L2 unit vectors)'
    logger.info("Normalization method: %s, scope: %s", method, scope)
    if group_col_map:
        logger.info("Per-group overrides: %s", {k: per_group_methods[k] for k in group_col_map})
    logger.info("Original data range: %.2f to %.2f", X_clean.min(), X_clean.max())
    logger.info("Normalized data range: %.2f to %.2f", X_normalized.min(), X_normalized.max())
    logger.debug("Normalized mean per feature: %s", X_normalized.mean(axis=0))
    logger.debug("Normalized std per feature: %s", X_normalized.std(axis=0))

    # ------------------------------------------------------------------ #
    #  Store result                                                       #
    # ------------------------------------------------------------------ #
    source_sample_index_coord_name = f'{dataset_name}_sample_index'
    normalized_data_name = f'{dataset_name}_{normalization_name}'

    ml_data_normalized = xr.DataArray(
        X_normalized,
        dims=[source_sample_index_coord_name, feature_dim_name],
        coords={
            source_sample_index_coord_name: ds_ml_ready[dataset_name].coords[source_sample_index_coord_name],
            feature_dim_name: ds_ml_ready[dataset_name].coords[feature_dim_name],
        },
    )

    ds_ml_ready[normalized_data_name] = ml_data_normalized
    logger.info("Stored normalized data as '%s'.", normalized_data_name)

    return ds_ml_ready


def extract_valid_samples_for_sklearn(ds_ml_ready, specific_data_name=None, dataset_name='ml_data_clean'):
    """Extract flattened ML data as a NumPy array for scikit-learn.

    Args:
        ds_ml_ready (xr.Dataset): Dataset containing flattened ML data.
        specific_data_name (str or None): Suffix identifying a particular
            stored result (e.g. ``'standard_normalized'``). Pass ``''``
            or ``None`` for the base dataset. Defaults to None.
        dataset_name (str): Base dataset name. Defaults to 'ml_data_clean'.

    Returns:
        tuple: A tuple of (X, grid_indices, result_sample_indices) where:
            X (np.ndarray): Feature matrix of shape
            ``(n_samples, n_features)``.
            grid_indices (np.ndarray): Corresponding grid indices.
            result_sample_indices (np.ndarray): Sample-index coordinate
            values.
    """
    # Construct full variable name using consistent convention
    full_data_var = f"{dataset_name}_{specific_data_name}" if specific_data_name is not None and specific_data_name != "" and dataset_name != specific_data_name else dataset_name
    
    if full_data_var not in ds_ml_ready:
        raise ValueError(f"Data variable '{full_data_var}' not found in dataset")
    
    data = ds_ml_ready[full_data_var]
    
    # Use dataset_name for coordinate/mapping names (not full_data_var)
    sample_index_coord_name = f'{dataset_name}_sample_index'
    
    # Data should always be in flattened format now
    if sample_index_coord_name in data.dims:
        X = data.values
        # Get grid_indices using universal mapping
        mapping_name = f'{dataset_name}_sample_index_to_grid_index'
        result_sample_indices = data.coords[sample_index_coord_name].values
        grid_indices = ds_ml_ready[mapping_name][result_sample_indices].values
        logger.debug("Using flattened data '%s': %d samples with %d features.", full_data_var, X.shape[0], X.shape[1])
        return X, grid_indices, result_sample_indices
    else:
        raise ValueError(f"Data variable '{full_data_var}' not in expected flattened format with {sample_index_coord_name} dimension")
    

def store_ml_results_flattened(ds_ml_ready, flat_results, specific_data_name, dataset_name='ml_data_clean', result_sample_indices=None):
    """Store 1-D ML results aligned to the flattened sample index.

    Args:
        ds_ml_ready (xr.Dataset): Dataset with the sample-index-to-grid-index mapping.
        flat_results (np.ndarray): 1-D array of ML results (e.g. cluster labels).
        specific_data_name (str): Suffix used for the stored variable name.
        dataset_name (str): Base dataset name. Defaults to 'ml_data_clean'.
        result_sample_indices (np.ndarray or None): Sample indices corresponding to
            flat_results. Must be a subset of the dataset's sample indices.
            If None, assumes all samples are present. Defaults to None.


    Returns:
        xr.Dataset: Dataset with the results added.
    """

    mapping_name = f'{dataset_name}_sample_index_to_grid_index'
    sample_index_coord_name = f'{dataset_name}_sample_index'
    source_sample_indices = ds_ml_ready[dataset_name].coords[sample_index_coord_name].values

    if result_sample_indices is None:
        if len(flat_results) != len(source_sample_indices):
            raise ValueError(
                f"Length mismatch detected!\n"
                f"  ML results length: {len(flat_results)}\n"
                f"  Expected samples from '{dataset_name}': {len(source_sample_indices)}\n"
                f"  This suggests some samples were filtered during ML processing.\n"
                f"  Please provide 'result_sample_indices' parameter to specify which "
                f"samples the results correspond to.\n"
                f"  Example: result_sample_indices = original_sample_indices[valid_mask]"
            )
        
        result_sample_indices = source_sample_indices
        logger.info("Using all %d samples from '%s'.", len(source_sample_indices), dataset_name)
    else:
        if not np.all(np.isin(result_sample_indices, source_sample_indices)):
            invalid_indices = result_sample_indices[~np.isin(result_sample_indices, source_sample_indices)]
            raise ValueError(f"result_sample_indices contains invalid indices not in {dataset_name}: {invalid_indices[:5]}...")
        
        if len(flat_results) != len(result_sample_indices):
            raise ValueError(f"Length mismatch: flat_results has {len(flat_results)} elements "
                            f"but result_sample_indices has {len(result_sample_indices)} elements")

        logger.info(
            "Using custom subset: %d of %d samples from '%s'.",
            len(result_sample_indices), len(source_sample_indices), dataset_name
        )


    full_result_name = f"{dataset_name}_{specific_data_name}"
    
    if mapping_name not in ds_ml_ready:
        raise ValueError(f"Universal {mapping_name} mapping required")
    
    # Ensure flat_results is 1D
    if flat_results.ndim > 1:
        flat_results = flat_results.flatten()
    
    # Store results with subset sample_index coordinates
    ds_ml_ready[full_result_name] = xr.DataArray(
        flat_results,
        dims=[sample_index_coord_name],  # Data-specific dimension name!
        coords={sample_index_coord_name: result_sample_indices}  # Subset of original sample_index values
    )
    ds_ml_ready[full_result_name].attrs['long_name'] = f'ML {specific_data_name} (flattened)'
    ds_ml_ready[full_result_name].attrs['description'] = f'ML results using subset of {sample_index_coord_name} indices'
    
    # Log summary
    unique_values, counts = np.unique(flat_results, return_counts=True)
    logger.info("Stored '%s' using %d %s indices.", specific_data_name, len(result_sample_indices), sample_index_coord_name)
    for value, count in zip(unique_values, counts):
        percentage = count / len(flat_results) * 100
        logger.debug("  %s %s: %d (%.1f%%)", specific_data_name, value, count, percentage)
    
    return ds_ml_ready


def extract_ml_data_gridded(ds_ml_ready, specific_data_name="", dataset_name='ml_data_clean', fill_value=-1, store_in_dataset=False):
    """Regrid flattened ML results back to the original spatial grid.

    Supports both single-valued results (e.g. cluster labels) and
    multi-feature data (e.g. normalized Sv with a channel dimension).

    Args:
        ds_ml_ready (xr.Dataset): Dataset with the
            sample-index-to-grid-index mapping.
        specific_data_name (str): Suffix of the stored result to regrid.
            Defaults to '' (base dataset).
        dataset_name (str): Base dataset name.
            Defaults to 'ml_data_clean'.
        fill_value (int or float): Value used for grid cells without ML
            results. Defaults to -1.
        store_in_dataset (bool): If True, store the gridded result in the
            dataset. Defaults to False.

    Returns:
        xr.DataArray: Gridded result with original spatial coordinates.
    """


    mapping_name = f'{dataset_name}_sample_index_to_grid_index'
    sample_index_coord_name = f'{dataset_name}_sample_index'
    full_result_name = f"{dataset_name}_{specific_data_name}" if (specific_data_name != "" and specific_data_name != dataset_name and specific_data_name is not None) else dataset_name

    if mapping_name not in ds_ml_ready:
        raise ValueError(f"Universal {mapping_name} mapping required")

    if full_result_name not in ds_ml_ready:
        raise ValueError(f"Result '{full_result_name}' not found in dataset")
    
    grid_index_name = 'grid_index'
    unique_grid_index_name = f'grid_index_{dataset_name}'

    if unique_grid_index_name in ds_ml_ready.coords: 
        grid_index_name = unique_grid_index_name
    
    flat_results = ds_ml_ready[full_result_name]
    result_sample_indices = flat_results.coords[sample_index_coord_name].values

    grid_indices = ds_ml_ready[mapping_name][result_sample_indices].values

    grid_coords = get_grid_coordinates(ds_ml_ready, grid_index_name)
    grid_index_shape = ds_ml_ready[grid_index_name].shape
    ping_indices, range_sample_indices = np.unravel_index(grid_indices, grid_index_shape)

    # Determine if this is multi-dimensional (has feature dimension)
    feature_dims = [dim for dim in flat_results.dims if dim != sample_index_coord_name]
    has_features = len(feature_dims) > 0

    if has_features:
        feature_dim_name = feature_dims[0]
        n_features = flat_results.sizes[feature_dim_name]

        grid_shape = (n_features, ds_ml_ready.sizes[grid_coords[0]], ds_ml_ready.sizes[grid_coords[1]])
        result_grid = np.full(grid_shape, fill_value, dtype=np.float64)

        result_grid[:, ping_indices, range_sample_indices] = flat_results.values.T.astype(np.float64)

        result_grid_da = xr.DataArray(
            result_grid,
            dims=[feature_dim_name, grid_coords[0], grid_coords[1]],
            coords={
                feature_dim_name: flat_results.coords[feature_dim_name],  # Use actual feature coords
                grid_coords[0]: ds_ml_ready.coords[grid_coords[0]],
                grid_coords[1]: ds_ml_ready.coords[grid_coords[1]]
            }
        )
        
        logger.info("Regridded %d ML samples with %d features to grid.", len(flat_results), n_features)
        logger.debug("Grid shape: %s, fill value: %s", grid_shape, fill_value)
        
    else:
        # Single-dimensional case (e.g., cluster labels) - existing logic
        grid_shape = (ds_ml_ready.sizes[grid_coords[0]], ds_ml_ready.sizes[grid_coords[1]])
        result_grid = np.full(grid_shape, fill_value, np.float64)
        result_grid[ping_indices, range_sample_indices] = flat_results.values.astype(np.float64)

        result_grid_da = xr.DataArray(
            result_grid,
            dims=[grid_coords[0], grid_coords[1]],
            coords={
                grid_coords[0]: ds_ml_ready.coords[grid_coords[0]],
                grid_coords[1]: ds_ml_ready.coords[grid_coords[1]]
            }
        )

        unique_values, counts = np.unique(flat_results.values, return_counts=True)
        logger.info("Regridded %d results to grid.", len(flat_results))
        logger.debug("Grid shape: %s, fill value: %s", grid_shape, fill_value)
    
    result_grid_da.attrs['long_name'] = f'ML {specific_data_name} (gridded)' if specific_data_name else f'ML {dataset_name} (gridded)'
    result_grid_da.attrs['description'] = f'ML results regridded using lookup arrays'
    result_grid_da.attrs['fill_value'] = fill_value
    result_grid_da.attrs['source_variable'] = full_result_name
    
    # Optionally store in dataset
    if store_in_dataset:
        grid_name = f'{full_result_name}_grid'
        ds_ml_ready[grid_name] = result_grid_da
        logger.info("Stored gridded results as '%s'.", grid_name)

    return result_grid_da


def extract_cluster_statistics(ds_ml_ready, cluster_data_name, dataset_name='ml_data_clean', 
                               normalize_data_name=None, sv_data_var=None, compute_pairwise_diffs=False):
    """Extract statistics for each cluster.

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
            provided, compute pairwise differences between channels in
            addition to original channel values. Defaults to False.

    Returns:
        dict: Dictionary containing:
            - ``'cluster_stats'``: list of dicts with per-cluster statistics.
            - ``'noise_stats'``: dict with noise statistics (if present).
            - ``'metadata'``: dict with overall information.
            - ``'feature_coords'``: array of feature names.
            - ``'feature_dim_name'``: str name of feature dimension.
            - ``'data_description'``: str describing source data.
    """
    
    # Get cluster labels
    full_cluster_name = f"{dataset_name}_{cluster_data_name}"
    if full_cluster_name not in ds_ml_ready:
        raise ValueError(f"Cluster data '{full_cluster_name}' not found in dataset")
    
    cluster_labels = ds_ml_ready[full_cluster_name].values
    
    # Determine source data based on whether sv_data_var is provided
    if sv_data_var is not None:
        # Use original gridded Sv data
        if sv_data_var not in ds_ml_ready:
            raise ValueError(f"Sv data variable '{sv_data_var}' not found in dataset")
        
        # First, regrid the cluster labels to match the original data grid
        cluster_labels_gridded = extract_ml_data_gridded(
            ds_ml_ready, 
            specific_data_name=cluster_data_name,
            dataset_name=dataset_name,
            fill_value=-2,
            store_in_dataset=False
        )
        
        # Get the Sv data
        sv_data = ds_ml_ready[sv_data_var]
        
        # Get grid coordinates
        grid_coords = get_grid_coordinates(ds_ml_ready, sv_data_var)
        
        # Get feature names (channel frequencies)
        feature_coords = sv_data.coords['channel'].values
        feature_dim_name = 'channel'
        
        # Create flattened version for easier indexing
        # Shape: (n_channels, n_ping_time, n_range_sample)
        sv_values = sv_data.values
        cluster_grid_values = cluster_labels_gridded.values
        
        # Flatten spatial dimensions while keeping channel dimension
        n_channels = sv_values.shape[0]
        spatial_shape = sv_values.shape[1:]
        n_spatial = np.prod(spatial_shape)
        
        # Reshape: (n_channels, n_spatial_points)
        sv_flat = sv_values.reshape(n_channels, n_spatial)
        # Reshape cluster labels: (n_spatial_points,)
        cluster_flat = cluster_grid_values.flatten()
        
        # Transpose to match cluster statistics format: (n_spatial_points, n_channels)
        source_data = sv_flat.T
        cluster_labels = cluster_flat
        
        # Optionally compute pairwise differences
        if compute_pairwise_diffs:
            # Compute pairwise differences (each channel minus first channel)
            baseline_channel_data = source_data[:, 0:1]  # Keep 2D for broadcasting
            diff_data = source_data[:, 1:] - baseline_channel_data
            
            # Concatenate original channels with differences
            source_data = np.concatenate([source_data, diff_data], axis=1)
            
            # Update feature names to include differences
            baseline_freq = feature_coords[0]
            diff_names = [f"{freq}-{baseline_freq}_diff" for freq in feature_coords[1:]]
            feature_coords = np.concatenate([feature_coords, diff_names])
            
            data_description = f"Original Sv data: {sv_data_var} (with pairwise differences)"
        else:
            data_description = f"Original Sv data: {sv_data_var}"
        
    else:
        # Use flattened normalized data (original behavior)
        source_data_name = dataset_name
        if normalize_data_name is not None:
            source_data_name = f"{dataset_name}_{normalize_data_name}"
        
        if source_data_name not in ds_ml_ready:
            raise ValueError(f"Source data '{source_data_name}' not found in dataset")
        
        source_data = ds_ml_ready[source_data_name].values
        
        # Get feature names
        feature_dim_name = [dim for dim in ds_ml_ready[source_data_name].dims 
                            if not dim.endswith('_sample_index')][0]
        feature_coords = ds_ml_ready[source_data_name].coords[feature_dim_name].values
        
        data_description = f"Source data: {source_data_name}"
    
    # Get unique clusters (excluding noise if present)
    unique_clusters = np.unique(cluster_labels)
    valid_clusters = unique_clusters[unique_clusters >= 0]
    n_noise = np.sum(cluster_labels == -1)
    valid_cluster_labels_plus_noise = cluster_labels[cluster_labels >= -1]
    
    # Build metadata
    metadata = {
        'cluster_data_name': cluster_data_name,
        'dataset_name': dataset_name,
        'normalize_data_name': normalize_data_name,
        'sv_data_var': sv_data_var,
        'n_total_samples': len(valid_cluster_labels_plus_noise),
        'n_clusters': len(valid_clusters),
        'n_noise': n_noise,
        'valid_clusters': valid_clusters.tolist()
    }
    
    # Calculate statistics for each cluster
    cluster_stats_list = []
    
    for cluster_id in sorted(valid_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_data = source_data[cluster_mask]
        n_samples = len(cluster_data)
        
        # Calculate stats for each feature
        cluster_stats = {
            'Cluster': int(cluster_id),
            'N_Samples': n_samples,
            'Percentage': 100 * n_samples / len(valid_cluster_labels_plus_noise)
        }
        
        for i, feature_name in enumerate(feature_coords):
            feature_data = cluster_data[:, i]
            mean_val = np.mean(feature_data)
            std_val = np.std(feature_data)
            min_val = np.min(feature_data)
            max_val = np.max(feature_data)
            
            cluster_stats[f'{feature_name}_mean'] = mean_val
            cluster_stats[f'{feature_name}_std'] = std_val
            cluster_stats[f'{feature_name}_min'] = min_val
            cluster_stats[f'{feature_name}_max'] = max_val
        
        cluster_stats_list.append(cluster_stats)
    
    # Calculate noise statistics if present
    noise_stats = None
    if n_noise > 0:
        noise_mask = cluster_labels == -1
        noise_data = source_data[noise_mask]
        
        noise_stats = {
            'Cluster': -1,
            'N_Samples': n_noise,
            'Percentage': 100 * n_noise / len(valid_cluster_labels_plus_noise)
        }
        
        for i, feature_name in enumerate(feature_coords):
            feature_data = noise_data[:, i]
            mean_val = np.mean(feature_data)
            std_val = np.std(feature_data)
            min_val = np.min(feature_data)
            max_val = np.max(feature_data)
            
            noise_stats[f'{feature_name}_mean'] = mean_val
            noise_stats[f'{feature_name}_std'] = std_val
            noise_stats[f'{feature_name}_min'] = min_val
            noise_stats[f'{feature_name}_max'] = max_val
    
    # Build return dictionary
    stats_dict = {
        'cluster_stats': cluster_stats_list,
        'noise_stats': noise_stats,
        'metadata': metadata,
        'feature_coords': feature_coords,
        'feature_dim_name': feature_dim_name,
        'data_description': data_description
    }
    
    return stats_dict


def remove_noise(
        ds_Sv,
        noise_range_sample_num=10,
        noise_ping_num=5,
        assign_to_sv=True
        ):
    """Remove background noise from Sv data using echopype.

    Wraps :func:`echopype.clean.remove_background_noise` and optionally
    copies the corrected result into the ``Sv`` variable.

    Args:
        ds_Sv (xr.Dataset): Dataset with ``Sv`` data.
        noise_range_sample_num (int): Number of range samples for noise
            estimation. Defaults to 10.
        noise_ping_num (int): Number of pings for noise estimation.
            Defaults to 5.
        assign_to_sv (bool): Copy ``Sv_corrected`` into ``Sv``.
            Defaults to True.

    Returns:
        xr.Dataset: Dataset with background noise removed.
    """

    ds_Sv_clean = ep.clean.remove_background_noise(
        ds_Sv,
        range_sample_num=noise_range_sample_num,
        ping_num=noise_ping_num,
    )
    if assign_to_sv:
        ds_Sv_clean["Sv"] = ds_Sv_clean["Sv_corrected"]

    return ds_Sv_clean


def compute_mvbs(
        ds_Sv_clean,
        mvbs_range_bin="2m",
        mvbs_ping_time_bin="10s",
        mvbs_nan_threshold=0.9
        ):
    """Compute Mean Volume Backscattering Strength (MVBS).

    Optionally masks sparse bins before computing MVBS via echopype.

    Args:
        ds_Sv_clean (xr.Dataset): Noise-corrected Sv dataset.
        mvbs_range_bin (str): Range bin size. Defaults to '2m'.
        mvbs_ping_time_bin (str): Ping time bin size.
            Defaults to '10s'.
        mvbs_nan_threshold (float or None): Fraction of NaN values
            above which a bin is masked. ``None`` skips this step.
            Defaults to 0.9.

    Returns:
        tuple: ``(ds_Sv_clean, ds_MVBS)`` — input dataset (potentially
        with sparse bins masked) and the MVBS dataset.
    """
    if mvbs_nan_threshold is not None:
        ds_Sv_clean = utils.mask_sparse_bins(
            ds_Sv_clean,
            range_bin=mvbs_range_bin,
            ping_time_bin=mvbs_ping_time_bin,
            nan_threshold=mvbs_nan_threshold
        )

    ds_MVBS = ep.commongrid.compute_MVBS(
        ds_Sv_clean,
        range_bin=mvbs_range_bin,
        ping_time_bin=mvbs_ping_time_bin
    )
    return ds_Sv_clean, ds_MVBS


def data_preprocessing_pipeline(
        ds_Sv, 
        echodata, 
        noise_range_sample_num=10, 
        noise_ping_num=5, 
        mvbs_range_bin="2m", 
        mvbs_ping_time_bin="10s", 
        mvbs_nan_threshold=0.9,
        plot_window= [0, 1200, 0, 600],
        y_to_x_aspect_ratio_override=None,
        remove_background_noise=True,
        overlay_line_var=None,
        overlay_line_path=None
        ):
    """Run the full data preprocessing pipeline: noise removal, MVBS, and echograms.

    Args:
        ds_Sv (xr.Dataset): Raw Sv dataset.
        echodata (ep.EchoData): Source EchoData object.
        noise_range_sample_num (int): Noise estimation range samples.
            Defaults to 10.
        noise_ping_num (int): Noise estimation pings. Defaults to 5.
        mvbs_range_bin (str): MVBS range bin. Defaults to '2m'.
        mvbs_ping_time_bin (str): MVBS ping-time bin.
            Defaults to '10s'.
        mvbs_nan_threshold (float or None): NaN fraction threshold for
            sparse-bin masking. Defaults to 0.9.
        plot_window (list[int]): ``[min_depth, max_depth, ping_min,
            ping_max]`` for echograms.
        y_to_x_aspect_ratio_override (float or None): Override echogram
            aspect ratio.
        remove_background_noise (bool): Whether to remove background
            noise. Defaults to True.
        overlay_line_var (str or None): Variable name for overlay lines.
        overlay_line_path (str or None): Path to overlay line file.

    Returns:
        tuple: ``(ds_Sv_clean, ds_MVBS)``.
    """
    if remove_background_noise:
        ds_Sv_clean = remove_noise(
            ds_Sv,
            noise_range_sample_num=noise_range_sample_num,
            noise_ping_num=noise_ping_num,
        )
    else:
        ds_Sv_clean = ds_Sv

    ds_Sv_clean, ds_MVBS = compute_mvbs(
        ds_Sv_clean,
        mvbs_range_bin=mvbs_range_bin,
        mvbs_ping_time_bin=mvbs_ping_time_bin,
        mvbs_nan_threshold=mvbs_nan_threshold
    )

    overlay_lines = None

    if overlay_line_path is not None and overlay_line_var is not None:
        ds_MVBS = utils.add_dive_profile_to_dataset(
            ds_MVBS,
            overlay_line_path,
            overlay_line_var
        )

        overlay_lines = [
            {'var': f'{overlay_line_var}_fit', 'style': {'color': 'red', 'linewidth': 3.5}}
        ]


    echogram.plot_sv_echogram(
        ds_Sv_clean, 
        min_depth=plot_window[0],
        max_depth=plot_window[1],
        ping_min=plot_window[2],
        ping_max=plot_window[3],
        x_axis_units="seconds",
        y_axis_units="meters",
        echodata=echodata,
        y_to_x_aspect_ratio_override=y_to_x_aspect_ratio_override
    )

    echogram.plot_sv_echogram(
        ds_MVBS, 
        ds_Sv_clean,
        min_depth=plot_window[0],
        max_depth=plot_window[1],
        ping_min=plot_window[2],
        ping_max=plot_window[3],
        x_axis_units="seconds",
        y_axis_units="meters",
        echodata=echodata,
        y_to_x_aspect_ratio_override=y_to_x_aspect_ratio_override,
        overlay_lines=overlay_lines
    )
    return ds_Sv_clean, ds_MVBS


def reshape_and_normalize_data(
        ds_Sv,
        custom_dataset_name, 
        ds_Sv_original=None,
        feature_strategy="baseline_plus_differences", 
        baseline_channel=0,
        data_var="Sv", 
        custom_normalization_name="normalized_data", 
        normalization_strategy="standard", 
        feature_weights=None,
        plot_window=[0, 1200, 0, 600],
        exclude_cluster_data_name=None,
        gridded_results_to_mask=None,
        mask_cluster_label=None,
        y_to_x_aspect_ratio_override=None,
        n_quantiles=100,
        cluster_colors=None,
        per_group_methods=None
        ):
    """Reshape data, normalize, and plot in a single convenience call.

    Optionally excludes a previous clustering result via masking before
    reshaping.  Produces an echogram of the normalized features.

    Args:
        ds_Sv (xr.Dataset): Sv or MVBS dataset.
        custom_dataset_name (str): Name for the ML dataset stored in
            the output.
        ds_Sv_original (xr.Dataset or None): Original (un-binned) Sv
            used for echogram depth reference.
        feature_strategy (str): Feature extraction strategy.
            Defaults to 'baseline_plus_differences'.
        baseline_channel (int): Channel index for baseline features.
            Defaults to 0.
        data_var (str): Sv variable name. Defaults to 'Sv'.
        custom_normalization_name (str): Name suffix for the normalized
            output. Defaults to 'normalized_data'.
        normalization_strategy (str): Normalization method, or ``'none'``
            to skip. Defaults to 'standard'.
        feature_weights (array-like or None): Per-feature multiplicative
            weights.
        plot_window (list[int]): ``[min_depth, max_depth, ping_min,
            ping_max]``.
        exclude_cluster_data_name (str or None): Previous cluster result
            name for masking.
        gridded_results_to_mask (xr.DataArray or None): Gridded cluster
            labels to apply as an exclusion mask.
        mask_cluster_label (int or None): Specific cluster label to
            exclude.  When ``None`` the largest cluster is excluded
            instead.
        y_to_x_aspect_ratio_override (float or None): Echogram aspect
            ratio override.
        n_quantiles (int): Number of quantiles for ``'quantile'``
            normalization. Defaults to 100.
        cluster_colors (list[str] or None): Hex colours for cluster
            display.
        per_group_methods (dict[str, str] or None): Per-feature method
            overrides for auxiliary features. Forwarded to
            :func:`normalize_data`. Defaults to None.

    Returns:
        xr.Dataset: Dataset with reshaped and (optionally) normalized
        data.
    """
    
    if cluster_colors is None:
        cluster_colors = DEFAULT_CLUSTER_COLORS

    cluster_mask_name = None
    # Exclude a specific cluster label (e.g. background from a previous pass)
    if mask_cluster_label is not None and gridded_results_to_mask is not None:
        cluster_mask_name = f'{exclude_cluster_data_name or "cluster"}_mask'
        ds_Sv = add_cluster_mask(ds_Sv, gridded_results_to_mask, cluster_label=mask_cluster_label, mask_name=cluster_mask_name)
    elif exclude_cluster_data_name is not None and gridded_results_to_mask is not None:
        # Fallback: exclude the largest cluster
        cluster_mask_name = f'{exclude_cluster_data_name}_mask'
        ds_Sv = add_cluster_mask(ds_Sv, gridded_results_to_mask, mask_name=cluster_mask_name)

    ds_ml_ready = reshape_data_for_ml(
        ds_Sv, 
        data_var=data_var, 
        dataset_name=custom_dataset_name, 
        feature_strategy=feature_strategy,
        baseline_channel=baseline_channel,
        custom_data_mask_name=cluster_mask_name
        )
    
    if normalization_strategy != "none":
        ds_normalized = normalize_data(
            ds_ml_ready, 
            method=normalization_strategy, 
            shift_positive=False, 
            dataset_name=custom_dataset_name, 
            normalization_name=custom_normalization_name, 
            feature_weights=feature_weights,
            n_quantiles=n_quantiles,
            per_group_methods=per_group_methods,
        )
        normalized_var = f"{custom_dataset_name}_{custom_normalization_name}"
        source_data = ds_normalized[custom_dataset_name]
        feature_dim = [d for d in source_data.dims if d != f'{custom_dataset_name}_sample_index'][0]
        visualize_normalized_data_histogram(
            ds_normalized[normalized_var].values,
            feature_names=source_data.coords[feature_dim].values
        )
    else:
        ds_normalized = ds_ml_ready
        custom_normalization_name = custom_dataset_name

    echogram.plot_flattened_data_echogram(
        ds_normalized, 
        custom_dataset_name,
        ds_Sv_original,
        ml_specific_data_name=custom_normalization_name,
        min_depth=plot_window[0],
        max_depth=plot_window[1],
        ping_min=plot_window[2],
        ping_max=plot_window[3],
        y_to_x_aspect_ratio_override=y_to_x_aspect_ratio_override,
        x_axis_units="seconds",
        y_axis_units="meters"
    )

    return ds_normalized



def extract_data_and_run_hdbscan(
        ds_normalized,
        custom_dataset_name, 
        ds_Sv_original=None,
        custom_normalization_name="normalized_data", 
        ml_result_name="dbscan_clusters",
        plot_window=[0, 1200, 0, 600],
        epsilon=0.004,
        min_samples=2,
        sample_size=1000000,
        min_cluster_size=2000,
        cluster_selection_method="leaf",
        use_hdbscan=True,
        find_background_cluster=False,
        y_to_x_aspect_ratio_override=None,
        soft_membership_threshold=None, 
        cluster_colors=None,
        overlay_line_var=None,
        cluster_stats_sv_data_var="Sv",
        cluster_stats_compute_pairwise_differences=True
        ):
    """Extract normalized data, run HDBSCAN/DBSCAN, store and visualise results.

    Convenience wrapper that chains extraction, clustering, result storage,
    echogram plotting, and cluster-statistics reporting.

    Args:
        ds_normalized (xr.Dataset): Dataset with normalized ML data.
        custom_dataset_name (str): Base ML dataset name.
        ds_Sv_original (xr.Dataset or None): Original Sv dataset for
            echogram depth reference.
        custom_normalization_name (str): Normalization result suffix.
            Defaults to 'normalized_data'.
        ml_result_name (str): Name for the clustering result.
            Defaults to 'dbscan_clusters'.
        plot_window (list[int]): ``[min_depth, max_depth, ping_min,
            ping_max]``.
        epsilon (float): DBSCAN epsilon. Defaults to 0.004.
        min_samples (int): Core-point neighbourhood size.
            Defaults to 2.
        sample_size (int): Sub-sample size. Defaults to 1_000_000.
        min_cluster_size (int): Minimum cluster size.
            Defaults to 2000.
        cluster_selection_method (str): HDBSCAN cluster-selection
            method. Defaults to 'leaf'.
        use_hdbscan (bool): Use HDBSCAN instead of DBSCAN.
            Defaults to True.
        find_background_cluster (bool): Run background-cluster
            detection. Defaults to False.
        y_to_x_aspect_ratio_override (float or None): Echogram
            aspect-ratio override.
        soft_membership_threshold (float or None): Reassign noise
            via soft-membership if set.
        cluster_colors (list[str] or None): Hex colours for cluster
            display.
        overlay_line_var (str or None): Variable name for overlay
            lines.
        cluster_stats_sv_data_var (str): Sv variable for cluster
            statistics. Defaults to 'Sv'.
        cluster_stats_compute_pairwise_differences (bool): Compute
            inter-channel pairwise diffs in stats. Defaults to True.

    Returns:
        tuple: ``(ds_final, gridded_results_dbscan, dbscan_results)``
        or ``(ds_final, gridded_results_dbscan, dbscan_results,
        background_label)`` when *find_background_cluster* is True.
    """

    X, _, sample_indices = extract_valid_samples_for_sklearn(ds_normalized, custom_normalization_name, dataset_name=custom_dataset_name)

    if find_background_cluster:
        dbscan_results, background_label = retrieve_background_cluster(X, sample_indices, min_samples, sample_size, min_cluster_size, cluster_selection_method)

    else:

        dbscan_results = apply_dbscan_clustering(
            X,
            sample_indices=sample_indices,
            eps_values=[epsilon],
            min_samples_values=[min_samples],
            sample_size=sample_size,
            calculate_silhouette=True,
            silhouette_sample_size=100,
            metric="euclidean",
            use_hdbscan=use_hdbscan,
            min_cluster_size=min_cluster_size,
            cluster_selection_method=cluster_selection_method,
            soft_membership_threshold=soft_membership_threshold
        )

    first_key = next(iter(dbscan_results))

    ds_final = store_ml_results_flattened(ds_normalized, dbscan_results[first_key]["labels"], ml_result_name, dataset_name=custom_dataset_name, result_sample_indices=dbscan_results[first_key]["sample_indices"])

    gridded_results_dbscan = extract_ml_data_gridded(ds_final, ml_result_name, dataset_name=custom_dataset_name, fill_value=np.nan, store_in_dataset=True)
    overlay_lines = []

    if overlay_line_var is not None:
        overlay_lines = [
            {'var': f'{overlay_line_var}_fit', 'style': {'color': "#23FFFF", 'linewidth': 7.5}}
        ]

    echogram.plot_cluster_echogram(
        ds_final, 
        dataset_name=custom_dataset_name, 
        specific_data_name=ml_result_name,
        min_depth=plot_window[0],
        max_depth=plot_window[1],
        ping_min=plot_window[2],
        ping_max=plot_window[3],
        ds_Sv_original=ds_Sv_original,
        y_to_x_aspect_ratio_override=y_to_x_aspect_ratio_override,
        cluster_colors=cluster_colors,
        overlay_lines=overlay_lines

    )

    plot_cluster_statistics(ds_final, ml_result_name, dataset_name=custom_dataset_name, cluster_colors=cluster_colors, sv_data_var=cluster_stats_sv_data_var, compute_pairwise_diffs=cluster_stats_compute_pairwise_differences)

    if use_hdbscan and not find_background_cluster:
        plot_dbscan_cluster_hierarchy(dbscan_results[first_key]["model"], cluster_colors_by_index=cluster_colors)

    if find_background_cluster:
        return ds_final, gridded_results_dbscan, dbscan_results, background_label
    return ds_final, gridded_results_dbscan, dbscan_results



def full_dbscan_iteration(
        ds_Sv,
        custom_dataset_name, 
        ds_Sv_original=None,
        feature_strategy="baseline_plus_differences", 
        baseline_channel=0,
        data_var="Sv", 
        custom_normalization_name="normalized_data", 
        ml_result_name="dbscan_clusters",
        normalization_strategy="standard", 
        feature_weights=None,
        plot_window=[0, 1200, 0, 600],
        epsilon=0.004,
        min_samples=2,
        sample_size=1000000,
        min_cluster_size=2000,
        cluster_selection_method="leaf",
        use_hdbscan=True,
        exclude_cluster_data_name=None,
        gridded_results_to_mask=None,
        mask_cluster_label=None,
        find_background_cluster=False,
        y_to_x_aspect_ratio_override=None,
        n_quantiles=100,
        soft_membership_threshold=None, 
        cluster_colors=None,
        overlay_line_var=None,
        cluster_stats_sv_data_var="Sv",
        cluster_stats_compute_pairwise_differences=True,
        per_group_methods=None
        ):
    """End-to-end iteration: reshape, normalize, cluster, and visualise.

    Combines :func:`reshape_and_normalize_data` and
    :func:`extract_data_and_run_hdbscan` into one call, optionally
    masking out a previous clustering result first.

    Args:
        ds_Sv (xr.Dataset): Sv or MVBS dataset.
        custom_dataset_name (str): Name for the ML dataset.
        ds_Sv_original (xr.Dataset or None): Original Sv for echogram
            reference.
        feature_strategy (str): Feature extraction strategy.
            Defaults to 'baseline_plus_differences'.
        baseline_channel (int): Channel index for baseline.
            Defaults to 0.
        data_var (str): Sv variable name. Defaults to 'Sv'.
        custom_normalization_name (str): Normalization suffix.
            Defaults to 'normalized_data'.
        ml_result_name (str): Clustering result name.
            Defaults to 'dbscan_clusters'.
        normalization_strategy (str): Normalization method, or
            ``'none'`` to skip. Defaults to 'standard'.
        feature_weights (array-like or None): Per-feature multiplicative
            weights.
        plot_window (list[int]): ``[min_depth, max_depth, ping_min,
            ping_max]``.
        epsilon (float): DBSCAN epsilon. Defaults to 0.004.
        min_samples (int): Core-point neighbourhood size.
            Defaults to 2.
        sample_size (int): Sub-sample size. Defaults to 1_000_000.
        min_cluster_size (int): Minimum cluster size.
            Defaults to 2000.
        cluster_selection_method (str): HDBSCAN method.
            Defaults to 'leaf'.
        use_hdbscan (bool): Use HDBSCAN. Defaults to True.
        exclude_cluster_data_name (str or None): Previous cluster result
            name for exclusion masking.
        gridded_results_to_mask (xr.DataArray or None): Gridded cluster
            labels for exclusion.
        mask_cluster_label (int or None): Specific cluster label to
            exclude.
        find_background_cluster (bool): Detect background cluster.
            Defaults to False.
        y_to_x_aspect_ratio_override (float or None): Echogram aspect
            ratio override.
        n_quantiles (int): Number of quantiles for ``'quantile'``
            normalization. Defaults to 100.
        soft_membership_threshold (float or None): Reassign noise via
            soft-membership if set.
        cluster_colors (list[str] or None): Hex colours for display.
        overlay_line_var (str or None): Variable name for overlay lines.
        cluster_stats_sv_data_var (str): Sv variable for statistics.
            Defaults to 'Sv'.
        cluster_stats_compute_pairwise_differences (bool): Compute
            pairwise channel diffs in statistics. Defaults to True.
        per_group_methods (dict[str, str] or None): Per-feature method
            overrides for auxiliary features. Forwarded to
            :func:`normalize_data`. Defaults to None.

    Returns:
        tuple: ``(ds_final, gridded_results, dbscan_results)`` or
        ``(ds_final, gridded_results, dbscan_results,
        background_label)`` when *find_background_cluster* is True.
    """
    
    if cluster_colors is None:
        cluster_colors = DEFAULT_CLUSTER_COLORS

    cluster_mask_name = None
    # Exclude a specific cluster label (e.g. background from a previous pass)
    if mask_cluster_label is not None and gridded_results_to_mask is not None:
        cluster_mask_name = f'{exclude_cluster_data_name or "cluster"}_mask'
        ds_Sv = add_cluster_mask(ds_Sv, gridded_results_to_mask, cluster_label=mask_cluster_label, mask_name=cluster_mask_name)
    elif exclude_cluster_data_name is not None and gridded_results_to_mask is not None:
        # Fallback: exclude the largest cluster
        cluster_mask_name = f'{exclude_cluster_data_name}_mask'
        ds_Sv = add_cluster_mask(ds_Sv, gridded_results_to_mask, mask_name=cluster_mask_name)


    ds_ml_ready = reshape_data_for_ml(
        ds_Sv, 
        data_var=data_var, 
        dataset_name=custom_dataset_name, 
        feature_strategy=feature_strategy,
        baseline_channel=baseline_channel,
        custom_data_mask_name=cluster_mask_name
        )
    
    if normalization_strategy != "none":
        ds_normalized = normalize_data(
            ds_ml_ready, 
            method=normalization_strategy, 
            shift_positive=False, 
            dataset_name=custom_dataset_name, 
            normalization_name=custom_normalization_name, 
            feature_weights=feature_weights,
            n_quantiles=n_quantiles,
            per_group_methods=per_group_methods,
        )
        normalized_var = f"{custom_dataset_name}_{custom_normalization_name}"
        source_data = ds_normalized[custom_dataset_name]
        feature_dim = [d for d in source_data.dims if d != f'{custom_dataset_name}_sample_index'][0]
        visualize_normalized_data_histogram(
            ds_normalized[normalized_var].values,
            feature_names=source_data.coords[feature_dim].values
        )
    else:
        ds_normalized = ds_ml_ready
        custom_normalization_name = custom_dataset_name

    echogram.plot_flattened_data_echogram(
        ds_normalized, 
        custom_dataset_name,
        ds_Sv_original,
        ml_specific_data_name=custom_normalization_name,
        min_depth=plot_window[0],
        max_depth=plot_window[1],
        ping_min=plot_window[2],
        ping_max=plot_window[3],
        y_to_x_aspect_ratio_override=y_to_x_aspect_ratio_override,
        x_axis_units="seconds",
        y_axis_units="meters"
    )

    return extract_data_and_run_hdbscan(
        ds_normalized,
        custom_dataset_name, 
        ds_Sv_original=ds_Sv_original,
        custom_normalization_name=custom_normalization_name, 
        ml_result_name=ml_result_name,
        plot_window=plot_window,
        epsilon=epsilon,
        min_samples=min_samples,
        sample_size=sample_size,
        min_cluster_size=min_cluster_size,
        cluster_selection_method=cluster_selection_method,
        use_hdbscan=use_hdbscan,
        find_background_cluster=find_background_cluster,
        y_to_x_aspect_ratio_override=y_to_x_aspect_ratio_override,
        soft_membership_threshold=soft_membership_threshold, 
        cluster_colors=cluster_colors,
        overlay_line_var=overlay_line_var,
        cluster_stats_sv_data_var=cluster_stats_sv_data_var,
        cluster_stats_compute_pairwise_differences=cluster_stats_compute_pairwise_differences
        )


