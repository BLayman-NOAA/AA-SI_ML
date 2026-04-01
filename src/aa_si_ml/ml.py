import time

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler,
    Normalizer, PowerTransformer, QuantileTransformer,
)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.stats import norm
import hdbscan
import umap
import echopype as ep
from aa_si_visualization import echogram
from aa_si_utils import utils



def add_largest_cluster_mask(ds_Sv, cluster_labels_gridded, use_corrected=True, mask_name='largest_cluster_mask'):
    """Create a mask that excludes the largest cluster from the dataset.

    Identifies the cluster with the most points (ignoring noise label -1)
    and produces a boolean mask where those points are ``False``.

    Args:
        ds_Sv (xr.Dataset): Dataset containing Sv data.
        cluster_labels_gridded (xr.DataArray): 2-D gridded cluster labels
            (ping_time x range_sample).
        use_corrected (bool): Use ``Sv_corrected`` when available.
            Defaults to True.
        mask_name (str): Name for the stored mask variable.
            Defaults to 'largest_cluster_mask'.

    Returns:
        xr.Dataset: Input dataset with the mask added as a new variable.
    """
    ds_with_mask = ds_Sv

    sv_var = 'Sv_corrected' if use_corrected and 'Sv_corrected' in ds_Sv else 'Sv'
    sv_data = ds_Sv[sv_var]

    valid_mask = xr.ones_like(sv_data, dtype=bool)

    unique_labels, counts = np.unique(cluster_labels_gridded, return_counts=True)

    valid_label_mask = unique_labels >= 0
    valid_labels = unique_labels[valid_label_mask]
    valid_counts = counts[valid_label_mask]
    
    if len(valid_labels) == 0:
        print("No valid clusters found (all points are noise)")
        largest_cluster_label = -1
        largest_cluster_size = 0
    else:
        # Find the label with maximum count
        max_count_idx = np.argmax(valid_counts)
        largest_cluster_label = valid_labels[max_count_idx]
        largest_cluster_size = valid_counts[max_count_idx]
    
    print(f"Largest cluster: label {largest_cluster_label} with {largest_cluster_size:,} points")

    if largest_cluster_label >= 0:
        largest_cluster_2d_mask = cluster_labels_gridded == largest_cluster_label
        largest_cluster_mask = largest_cluster_2d_mask.broadcast_like(sv_data)
        largest_cluster_mask = valid_mask & ~largest_cluster_mask
    else:
        largest_cluster_mask = xr.zeros_like(sv_data, dtype=bool)

    n_excluded = largest_cluster_mask.sum().values
    print(f"Masked {n_excluded:,} values")

    ds_with_mask[mask_name] = largest_cluster_mask
    ds_with_mask[mask_name].attrs['long_name'] = 'Valid data mask of largest cluster'
    ds_with_mask[mask_name].attrs['description'] = 'Mask excluding all data points in the largest cluster'
    ds_with_mask[mask_name].attrs['source_variable'] = cluster_labels_gridded

    return ds_with_mask


def add_cluster_label_mask(ds_Sv, cluster_labels_gridded, cluster_label, use_corrected=True, mask_name='cluster_mask'):
    """Create a mask that excludes a specific cluster label from the dataset.

    Unlike ``add_largest_cluster_mask`` which always picks the largest, this
    lets you specify exactly which cluster label to mask out (e.g. a background
    cluster identified by ``retrieve_background_cluster``).

    Args:
        ds_Sv (xr.Dataset): The dataset to add the mask to.
        cluster_labels_gridded (xr.DataArray): 2-D gridded cluster labels
            (ping_time x range_sample).
        cluster_label (int): The specific cluster label to mask out.
        use_corrected (bool): Use ``Sv_corrected`` for determining shape.
            Defaults to True.
        mask_name (str): Name for the stored mask variable.
            Defaults to 'cluster_mask'.

    Returns:
        xr.Dataset: Dataset with the mask added.
    """
    ds_with_mask = ds_Sv
    sv_var = 'Sv_corrected' if use_corrected and 'Sv_corrected' in ds_Sv else 'Sv'
    sv_data = ds_Sv[sv_var]
    valid_mask = xr.ones_like(sv_data, dtype=bool)

    cluster_2d_mask = cluster_labels_gridded == cluster_label
    cluster_count = int(cluster_2d_mask.sum().values)
    print(f"Masking cluster {cluster_label} with {cluster_count:,} points")

    cluster_broadcast = cluster_2d_mask.broadcast_like(sv_data)
    final_mask = valid_mask & ~cluster_broadcast

    n_excluded = final_mask.sum().values
    print(f"Masked {n_excluded:,} NaN values")

    ds_with_mask[mask_name] = final_mask
    ds_with_mask[mask_name].attrs['long_name'] = f'Valid data mask excluding cluster {cluster_label}'
    ds_with_mask[mask_name].attrs['description'] = f'Mask that excludes cluster label {cluster_label}'

    return ds_with_mask


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
        # Swap coordinates to put time/distance dimension first
        grid_coords = [grid_coords[1], grid_coords[0]]
        print(f"  Swapped coordinate order to: {grid_coords}")
    
    
    return grid_coords


# called by reshape_data_for_ml
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
    print(f"Analyzing '{sv_var}' for validity")
    print(f"Data shape: {ds_Sv[sv_var].shape}")
    print(f"Channels (frequencies): {len(ds_Sv['channel'])}")

    sv_data = ds_Sv[sv_var]
    valid_mask = xr.ones_like(sv_data, dtype=bool)

    if remove_nan:
        nan_mask = xr.ufuncs.isnan(sv_data)
        valid_mask = valid_mask & ~nan_mask
        n_nan = nan_mask.sum().values
        print(f"Masked {n_nan:,} NaN values")

    if mask_invalid_values:
        # Typical Sv range is roughly -120 to 20 dB; use conservative bounds
        too_low_mask = sv_data < -200
        too_high_mask = sv_data > 50
        artifact_mask = too_low_mask | too_high_mask
        valid_mask = valid_mask & ~artifact_mask
        n_artifacts = artifact_mask.sum().values
        print(f"Masked {n_artifacts:,} extreme values (< -200 or > 50 dB)")

    if custom_mask_name is not None:
        custom_mask = ds_Sv[custom_mask_name]
        if custom_mask.shape != sv_data.shape:
            raise ValueError("custom_mask must have the same shape as the Sv data")
        valid_mask = valid_mask & custom_mask
        n_custom_masked = (~custom_mask).sum().values
        print(f"Applied custom mask, masking additional {n_custom_masked:,} values")
    
    ds_with_mask[mask_name] = valid_mask
    ds_with_mask[mask_name].attrs['long_name'] = 'Valid data mask for machine learning'
    ds_with_mask[mask_name].attrs['description'] = 'True where data is valid for ML analysis'
    ds_with_mask[mask_name].attrs['source_variable'] = sv_var

    return ds_with_mask


# called by reshape_data_for_ml
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
    print(f"Creating grid index coordinate based on {grid_coords}...")

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


# called by reshape_data_for_ml
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
            ``'channels'`` -- raw channel values (default).
            ``'baseline_plus_differences'`` -- one baseline channel plus
            differences of the remaining channels from the baseline.
            ``'mean_centered'`` -- per-sample mean Sv plus each channel
            centred around that mean (captures intensity and spectral shape).
            ``'custom'`` -- user-provided function via *feature_kwargs*.
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
        feature_coords = data.coords['channel']
        feature_dim_name = 'channel'

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
    
    print(f"Extracted {n_samples:,} valid samples using '{feature_strategy}' strategy")
    print(f"Shape: {ml_data_flat.shape} (samples, features)")
    if feature_strategy == 'baseline_plus_differences':
        print(f"Baseline channel: {baseline_channel} ({data.coords['channel'].values[baseline_channel]})")
        print(f"Features: 1 baseline + {len(other_channels)} differences")
    elif feature_strategy == 'mean_centered':
        print(f"Features: 1 mean + {raw_data_values.shape[1]} centered channels (intensity + shape)")
        print(f"Note: Mean Sv encodes intensity, centered values encode spectral shape")
    
    return ml_data_flat, grid_indices


# called by reshape_data_for_ml and normalize_data
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
        print(f"Created {mapping_name} mapping with {len(grid_indices)} samples")
    else:
        existing_mapping = ds_ml_ready[mapping_name].values
        if not np.array_equal(existing_mapping, grid_indices):
            raise ValueError(f"Grid indices don't match existing mapping for {dataset_name}")
        print(f"Using existing {mapping_name} mapping")

    print(f"Stored {dataset_name} in flattened format")
    
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
    
    print("Creating validity mask...")

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
        print(f"WARNING: Found {len(channels_all_nan)} channel(s) with all NaN values: {channels_all_nan}")
        print(f"These channels will be excluded from processing.")
        print(f"Keeping {len(channels_to_keep)} valid channel(s): {channels_to_keep}")
        
        # Filter dataset to only include valid channels
        ds_Sv = ds_Sv.sel(channel=channels_to_keep)
        
        # Adjust baseline_channel if necessary
        if baseline_channel >= len(channels_to_keep):
            old_baseline = baseline_channel
            baseline_channel = 0  # Default to first valid channel
            print(f"WARNING: baseline_channel {old_baseline} is out of range after filtering.")
            print(f"Setting baseline_channel to {baseline_channel} (channel: {channels_to_keep[baseline_channel]})")
    

    if data.dims[0] != 'channel':
        print(f"Reordering dimensions from {data.dims} to channel-first...")
        spatial_dims = [dim for dim in data.dims if dim != 'channel']
        desired_order = ['channel'] + spatial_dims
        data = data.transpose(*desired_order)
        print(f"New dimension order: {data.dims}")
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

    print(f"Preparing ML data from '{sv_var}' as '{dataset_name}'...")
    print(f"Data shape: {ds_ml_ready[sv_var].shape}")
    
    # Extract and store using v2 (universal mapping)
    ml_data_flat, grid_indices = extract_ml_data_flattened(
        ds_ml_ready, sv_var, mask_name=mask_name, dataset_name=dataset_name,
        feature_strategy=feature_strategy, baseline_channel=baseline_channel, **feature_kwargs
    )
    ds_ml_ready = store_ml_data_flattened(ds_ml_ready, ml_data_flat, grid_indices, dataset_name)

    print(f"Data stored as: '{dataset_name}'")
    
    return ds_ml_ready


def normalize_data(ds_ml_ready, method='standard', pre_L2_method='standard', 
                  shift_positive=False, per_feature=True, dataset_name='ml_data_clean', 
                  normalization_name=None, feature_weights=None, n_quantiles=100, flatten_weight=1):
    """Normalize flattened ML data using a variety of scaling methods.

    Supports per-feature and global normalization, optional L2 row
    normalization with a configurable pre-scaler, power-transform
    flattening, UMAP embedding, and feature weighting.

    Parameters
    ----------
    ds_ml_ready : xr.Dataset
        Dataset produced by :func:`reshape_data_for_ml`.
    method : str, optional
        Normalization method.  One of ``'standard'``, ``'robust'``,
        ``'minmax'``, ``'flatten'``, ``'power'``, ``'quantile'``,
        ``'umap'``, ``'flatten_plus_umap'``, or ``'l2'``
        (default ``'standard'``).
    pre_L2_method : str, optional
        Scaler applied before L2 normalization when *method* is ``'l2'``
        (default ``'standard'``).
    shift_positive : bool, optional
        Shift all values to be positive after normalization
        (default ``False``).
    per_feature : bool, optional
        Normalize each feature independently (default ``True``).
    dataset_name : str, optional
        Base dataset name (default ``'ml_data_clean'``).
    normalization_name : str or None, optional
        Suffix for the stored result.  Defaults to *method* name.
    feature_weights : array-like or None, optional
        Per-feature multiplicative weights (default ``None``).
    n_quantiles : int, optional
        Number of quantiles for ``'quantile'`` method (default ``100``).
    flatten_weight : float, optional
        Blending weight for ``'flatten'`` method’s CDF transform
        (default ``1``).

    Returns
    -------
    ds_ml_ready : xr.Dataset
        Dataset with the normalized data added.
    """
    # Set default normalization_name based on method
    if normalization_name is None:
        if method == 'l2':
            normalization_name = f'l2_{pre_L2_method}_normalized' if pre_L2_method != 'none' else 'l2_normalized'
        else:
            normalization_name = f'{method}_normalized'
    
    X_clean, _, _ = extract_valid_samples_for_sklearn(ds_ml_ready, specific_data_name='', dataset_name=dataset_name)

    if per_feature:
        if method == 'l2':
            # L2 normalization with optional pre-normalization
            if pre_L2_method == 'standard' or pre_L2_method == 'standard_shifted':
                pre_scaler = StandardScaler()
            elif pre_L2_method == 'robust':
                pre_scaler = RobustScaler()
            elif pre_L2_method == 'minmax':
                pre_scaler = MinMaxScaler()
            elif pre_L2_method == 'none':
                pre_scaler = None
            else:
                raise ValueError(f"Unknown pre_L2_method: {pre_L2_method}")

            if pre_scaler:
                pre_normalized = pre_scaler.fit_transform(X_clean)
            else:
                pre_normalized = X_clean

            if pre_L2_method == 'standard_shifted':
                pre_normalized = pre_normalized + 2  # Shift 2 SDs
            
            scaler = Normalizer(norm='l2')
            X_normalized = scaler.fit_transform(pre_normalized)
            
            normalization_info = {
                'method': method,
                'pre_L2_method': pre_L2_method,
                'per_frequency': per_feature,
                'shift_positive': shift_positive
            }
        else:
            # Standard scalers - per frequency normalization
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'flatten' or method == "power" or method == 'flatten_plus_umap':
                scaler = PowerTransformer(method='yeo-johnson', standardize=True)
            elif method == 'quantile':
                n_quantiles = min(len(X_clean), n_quantiles)
                scaler = QuantileTransformer(output_distribution='uniform', n_quantiles=n_quantiles)
            elif method == 'umap':

                n_components = 2
                # Initialize and fit UMAP
                scaler = umap.UMAP(
                    n_components=n_components,
                    n_neighbors=15,
                    min_dist=.1,
                    metric="euclidean",
                    random_state=42,
                    verbose=True
                )
                # perform standard normalization before UMAP
                X_clean = StandardScaler().fit_transform(X_clean)
                
            else:
                raise ValueError(f"Unknown normalization method: {method}")
            
            X_normalized = scaler.fit_transform(X_clean)

            if method == 'flatten' or method == 'flatten_plus_umap':
                X_flattened = norm.cdf(X_normalized)
                X_normalized = (1 - flatten_weight) * X_normalized + flatten_weight * X_flattened

                if method == 'flatten_plus_umap':
                    n_components = 2
                    # Initialize and fit UMAP
                    umap_scaler = umap.UMAP(
                        n_components=n_components,
                        n_neighbors=30,
                        min_dist=.1,
                        metric="euclidean",
                        random_state=42,
                        verbose=True
                    )
                    X_normalized = umap_scaler.fit_transform(X_normalized)

            
            normalization_info = {
                'method': method,
                'per_frequency': per_feature,
                'shift_positive': shift_positive,
                'feature_means': scaler.mean_ if hasattr(scaler, 'mean_') else None,
                'feature_scales': scaler.scale_ if hasattr(scaler, 'scale_') else None
            }
    else:
        # Global normalization across all features
        print(f"Using GLOBAL normalization (treating all features together)")
        
        X_flat = X_clean.flatten()
        
        if method == 'standard':
            global_mean = np.mean(X_flat)
            global_std = np.std(X_flat)
            X_normalized = (X_clean - global_mean) / global_std
            normalization_info = {
                'method': 'standard_global',
                'per_frequency': per_feature,
                'shift_positive': shift_positive,
                'global_mean': global_mean,
                'global_std': global_std
            }
        elif method == 'robust':
            global_median = np.median(X_flat)
            global_q25 = np.percentile(X_flat, 25)
            global_q75 = np.percentile(X_flat, 75)
            global_iqr = global_q75 - global_q25
            X_normalized = (X_clean - global_median) / global_iqr
            normalization_info = {
                'method': 'robust_global',
                'per_frequency': per_feature,
                'shift_positive': shift_positive,
                'global_median': global_median,
                'global_iqr': global_iqr
            }
        elif method == 'minmax':
            global_min = np.min(X_flat)
            global_max = np.max(X_flat)
            global_range = global_max - global_min
            X_normalized = (X_clean - global_min) / global_range
            normalization_info = {
                'method': 'minmax_global',
                'per_frequency': per_feature,
                'shift_positive': shift_positive,
                'global_min': global_min,
                'global_range': global_range
            }

    # Apply positive shift if requested
    if shift_positive:
        min_value = X_normalized.min()
        if min_value < 0:
            X_normalized = X_normalized + abs(min_value) + 1e-6
            normalization_info['shift_amount'] = abs(min_value) + 1e-6

    # Apply feature weights
    if feature_weights is not None and per_feature:
        if len(feature_weights) != X_normalized.shape[1]:
            raise ValueError(f"feature_weights length ({len(feature_weights)}) must match number of features ({X_normalized.shape[1]})")
        X_normalized = X_normalized * feature_weights
        normalization_info['feature_weights'] = feature_weights
        print(f"Applied feature weights: {feature_weights}")

    # Print summary
    print(f"Normalization method: {method}")
    if not per_feature and method != 'l2':
        print(f"Normalization scope: GLOBAL (across all features)")
    elif method != 'l2':
        print(f"Normalization scope: PER-FEATURE (each feature independently)")
    else:
        print(f"Normalization scope: PER-SAMPLE (L2 unit vectors)")
    
    print(f"Original data range: {X_clean.min():.2f} to {X_clean.max():.2f}")
    print(f"Normalized data range: {X_normalized.min():.2f} to {X_normalized.max():.2f}")
    print(f"Normalized mean per feature: {X_normalized.mean(axis=0)}")
    print(f"Normalized std per feature: {X_normalized.std(axis=0)}")

    # Store in dataset using the source data's coordinate system
    source_sample_index_coord_name = f'{dataset_name}_sample_index'
    normalized_data_name = f'{dataset_name}_{normalization_name}'
    
    source_data = ds_ml_ready[dataset_name]
    feature_dim_name = [dim for dim in source_data.dims if dim != f'{dataset_name}_sample_index'][0]
    
    # Store normalized data
    ml_data_normalized = xr.DataArray(
        X_normalized,
        dims=[source_sample_index_coord_name, feature_dim_name],
        coords={
            source_sample_index_coord_name: ds_ml_ready[dataset_name].coords[source_sample_index_coord_name],
            feature_dim_name: ds_ml_ready[dataset_name].coords[feature_dim_name]
        }
    )

    ds_ml_ready[normalized_data_name] = ml_data_normalized
    print(f"Stored normalized data as '{normalized_data_name}'")
    
    feature_names = ds_ml_ready[dataset_name].coords[feature_dim_name].values
    visualize_normalized_data_histogram(X_normalized, feature_names=feature_names)

    return ds_ml_ready

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
        # make sure labels are strings and length matches
        feature_names = [str(n) for n in feature_names]
        if len(feature_names) != n_features:
            raise ValueError("feature_names length must match number of features")

    plt.figure(figsize=(10, 8))  # taller so legend fits inside
    colors = plt.cm.tab10.colors

    # Determine bin range: use percentiles if specified, otherwise full range
    if percentile_range is not None:
        lower_percentile, upper_percentile = percentile_range
        global_min = np.nanpercentile(X_normalized, lower_percentile)
        global_max = np.nanpercentile(X_normalized, upper_percentile)
        print(f"Using {lower_percentile}th to {upper_percentile}th percentile range: {global_min:.3f} to {global_max:.3f}")
    else:
        global_min = float(np.nanmin(X_normalized))
        global_max = float(np.nanmax(X_normalized))
        print(f"Using full data range: {global_min:.3f} to {global_max:.3f}")

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
    # place legend inside plot, compact
    plt.legend(loc='upper right', fontsize=9, frameon=True)
    plt.grid(alpha=0.25)
    y_min, y_max = plt.ylim()
    plt.ylim(y_min, y_max * 1.1)
    plt.tight_layout()
    plt.show()


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
        print(f"Using flattened data '{full_data_var}': {X.shape[0]:,} samples with {X.shape[1]} features")
        return X, grid_indices, result_sample_indices
    else:
        raise ValueError(f"Data variable '{full_data_var}' not in expected flattened format with {sample_index_coord_name} dimension")
    

def store_ml_results_flattened(ds_ml_ready, flat_results, specific_data_name, dataset_name='ml_data_clean', result_sample_indices=None):
    """Store 1-D ML results aligned to the flattened sample index.

    Parameters
    ----------
    ds_ml_ready : xr.Dataset
        Dataset with the sample-index-to-grid-index mapping.
    flat_results : np.ndarray
        1-D array of ML results (e.g. cluster labels).
    specific_data_name : str
        Suffix used for the stored variable name.
    dataset_name : str, optional
        Base dataset name (default ``'ml_data_clean'``).
    result_sample_indices : np.ndarray or None, optional
        Sample indices corresponding to *flat_results*.  Must be a
        subset of the dataset’s sample indices.  If ``None``, assumes
        all samples are present (default ``None``).

    Returns
    -------
    ds_ml_ready : xr.Dataset
        Dataset with the results added.
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
        print(f"Using all {len(source_sample_indices)} samples from '{dataset_name}'")
    else:
        if not np.all(np.isin(result_sample_indices, source_sample_indices)):
            invalid_indices = result_sample_indices[~np.isin(result_sample_indices, source_sample_indices)]
            raise ValueError(f"result_sample_indices contains invalid indices not in {dataset_name}: {invalid_indices[:5]}...")
        
        if len(flat_results) != len(result_sample_indices):
            raise ValueError(f"Length mismatch: flat_results has {len(flat_results)} elements "
                            f"but result_sample_indices has {len(result_sample_indices)} elements")
        
        print(f"Using custom subset: {len(result_sample_indices)} of {len(source_sample_indices)} samples from '{dataset_name}'")


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
    
    # Print summary
    unique_values, counts = np.unique(flat_results, return_counts=True)
    print(f"Stored {specific_data_name} using {len(result_sample_indices)} {sample_index_coord_name} indices")
    for value, count in zip(unique_values, counts):
        percentage = count / len(flat_results) * 100
        print(f"  {specific_data_name} {value}: {count:,} ({percentage:.1f}%)")
    
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
        
        print(f"Regridded {len(flat_results):,} ML samples with {n_features} features to grid")
        
        print(f"Grid shape: {grid_shape}, fill value: {fill_value}")
        
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
        print(f"Regridded {len(flat_results):,} results to grid")
        print(f"Grid shape: {grid_shape}, fill value: {fill_value}")
    
    result_grid_da.attrs['long_name'] = f'ML {specific_data_name} (gridded)' if specific_data_name else f'ML {dataset_name} (gridded)'
    result_grid_da.attrs['description'] = f'ML results regridded using lookup arrays'
    result_grid_da.attrs['fill_value'] = fill_value
    result_grid_da.attrs['source_variable'] = full_result_name
    
    # Optionally store in dataset
    if store_in_dataset:
        grid_name = f'{full_result_name}_grid'
        ds_ml_ready[grid_name] = result_grid_da
        print(f"Stored gridded results as '{grid_name}'")
        
    return result_grid_da


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
        useHDBScan=False, 
        min_cluster_size=5, 
        cluster_selection_method='eom',
        soft_membership_threshold=None
        ):
    """Run DBSCAN or HDBSCAN clustering over a parameter grid.

    Iterates over the provided parameter values, fits a model for each
    combination, and returns results including labels and scores.

    Args:
        X_normalized (np.ndarray): Feature matrix of shape
            ``(n_samples, n_features)``.
        sample_indices (np.ndarray): Grid-index values corresponding to
            each row of *X_normalized*.
        eps_values (list[float]): Epsilon values for DBSCAN neighbourhood
            radius (ignored when *useHDBScan* is True).
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
        useHDBScan (bool): Use HDBSCAN instead of DBSCAN.
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
        ``'model'``, ``'labels'``, ``'silhouette_score'``,
        ``'sample_indices'``, ``'n_clusters'``, ``'n_noise'``, etc.
    """
    
    # Use all data or sample based on parameter
    if sample_size is not None and sample_size < len(X_normalized):
        print(f"Using random sample of {sample_size:,} points for {'HDBSCAN' if useHDBScan else 'DBSCAN'} clustering (from {len(X_normalized):,} total)")
        np.random.seed(42)  # For reproducibility
        subsample_mask = np.random.choice(len(X_normalized), size=sample_size, replace=False)
        X_sample = X_normalized[subsample_mask]
        used_sample_indices = sample_indices[subsample_mask]  # Get corresponding indices
    else:
        print(f"Using ALL {len(X_normalized):,} valid data points for {'HDBSCAN' if useHDBScan else 'DBSCAN'} clustering")
        X_sample = X_normalized
        used_sample_indices = sample_indices  # Track the indices we're using
    
    results = {}
    
    if useHDBScan:
        print("\n=== HDBSCAN CLUSTERING RESULTS ===")
        if calculate_silhouette and len(X_sample) > silhouette_sample_size:
            print(f"Note: Silhouette scores calculated on sample of {silhouette_sample_size:,} points for efficiency")
        
        for min_samples in min_samples_values:
            param_key = f"hdbscan_mincluster_{min_cluster_size}_min_{min_samples}"
            print(f"\nTesting HDBSCAN with min_cluster_size={min_cluster_size}, min_samples={min_samples}...")
            
            # Apply HDBSCAN
            start_time = time.time()
            
            if metric == "mahalanobis":
                V = np.cov(X_normalized, rowvar=False)
                model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric, V=V, cluster_selection_method=cluster_selection_method, prediction_data=True)
            else:
                model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric, cluster_selection_method=cluster_selection_method, prediction_data=True)
            
            cluster_labels = model.fit_predict(X_sample)
            clustering_time = time.time() - start_time
            print(f"  HDBSCAN fitting took: {clustering_time:.2f} seconds")
            
            # Get number of clusters (excluding noise points labeled as -1)
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            
            # Calculate silhouette score with sampling for efficiency
            sil_score = _calculate_silhouette(X_sample, cluster_labels, n_clusters, calculate_silhouette, silhouette_sample_size)

            print("Before noise reassignment:")
            print_basic_cluster_stats(cluster_labels, n_clusters, n_noise, sil_score, calculate_silhouette)

            if soft_membership_threshold is not None:
                cluster_labels = assign_noise_by_soft_membership(model, threshold=soft_membership_threshold)

            print("After noise reassignment:")

            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            # Calculate silhouette score with sampling for efficiency
            sil_score = _calculate_silhouette(X_sample, cluster_labels, n_clusters, calculate_silhouette, silhouette_sample_size)
            print_basic_cluster_stats(cluster_labels, n_clusters, n_noise, sil_score, calculate_silhouette)

            # Store results
            results[param_key] = {
                'model': model,
                'labels': cluster_labels,
                'silhouette_score': sil_score,
                'sample_indices': used_sample_indices,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'eps': None,  # HDBSCAN doesn't use eps
                'min_samples': min_samples,
                'min_cluster_size': min_cluster_size,
                'metric': metric,
            }
            
            # Show cluster statistics
            print_basic_cluster_stats(cluster_labels, n_clusters, n_noise, sil_score, calculate_silhouette)
    
    else:
        # Standard DBSCAN: Loop through eps and min_samples
        print("\n=== DBSCAN CLUSTERING RESULTS ===")
        if calculate_silhouette and len(X_sample) > silhouette_sample_size:
            print(f"Note: Silhouette scores calculated on sample of {silhouette_sample_size:,} points for efficiency")
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                param_key = f"eps_{eps}_min_{min_samples}"
                print(f"\nTesting DBSCAN with eps={eps}, min_samples={min_samples}...")
                
                # Apply DBSCAN
                start_time = time.time()
                model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, algorithm=algorithm)
                cluster_labels = model.fit_predict(X_sample)
                
                # Apply min_cluster_size filter as post-processing
                filtered_labels = apply_min_cluster_size_filter(cluster_labels, min_cluster_size)
                cluster_labels = filtered_labels
                
                clustering_time = time.time() - start_time
                print(f"  DBSCAN fitting took: {clustering_time:.2f} seconds")
                
                # Get number of clusters (excluding noise points labeled as -1)
                n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                n_noise = list(cluster_labels).count(-1)
                
                # Calculate silhouette score with sampling for efficiency
                sil_score = _calculate_silhouette(X_sample, cluster_labels, n_clusters, calculate_silhouette, silhouette_sample_size)

                # Show cluster statistics
                print_basic_cluster_stats(cluster_labels, n_clusters, n_noise, sil_score, calculate_silhouette)

                # Store results
                results[param_key] = {
                    'model': model,
                    'labels': cluster_labels,
                    'silhouette_score': sil_score,
                    'sample_indices': used_sample_indices,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'eps': eps,
                    'min_samples': min_samples,
                    'metric': metric,
                }

    return results


def assign_noise_by_soft_membership(clusterer, threshold=0.1):
    """Assign noise points to clusters based on soft membership probabilities.

    Args:
        clusterer (hdbscan.HDBSCAN): Fitted HDBSCAN clusterer with
            ``prediction_data=True``.
        threshold (float): Minimum probability required to assign a noise
            point to a cluster. Defaults to 0.1.

    Returns:
        np.ndarray: Cluster labels with noise points reassigned if their
        max probability exceeds *threshold*.
    """
    print("start reassignment")
    # Get hard cluster labels
    labels = clusterer.labels_.copy()
    # Get soft membership vectors for all points
    soft_clusters = hdbscan.all_points_membership_vectors(clusterer)
    
    print("starting iteration")
    # Vectorized approach: find noise points and their best clusters
    noise_mask = labels == -1
    if np.any(noise_mask):
        # Get max probabilities and best clusters for noise points only
        noise_soft_clusters = soft_clusters[noise_mask]
        # Combined argmax and max in single operation (more cache efficient)
        best_clusters = np.argmax(noise_soft_clusters, axis=1)
        max_probs = noise_soft_clusters[np.arange(len(best_clusters)), best_clusters]
        
        # Find which noise points exceed threshold
        reassign_mask = max_probs > threshold
        
        # Create indices for noise points that should be reassigned
        noise_indices = np.where(noise_mask)[0]
        reassign_indices = noise_indices[reassign_mask]
        
        # Assign new cluster labels
        labels[reassign_indices] = best_clusters[reassign_mask]
        
        print(f"Reassigned {np.sum(reassign_mask)} of {np.sum(noise_mask)} noise points")
    
    print("end reassignment")
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
        float: Silhouette score, or -1 if not computed.
    """
    if calculate_silhouette and n_clusters > 1 and n_clusters < len(X_sample):
        sil_start_time = time.time()
        try:
            if len(X_sample) > silhouette_sample_size:
                # Sample for silhouette calculation to avoid O(n²) slowdown
                np.random.seed(42)
                sil_indices = np.random.choice(len(X_sample), size=silhouette_sample_size, replace=False)
                X_sil_sample = X_sample[sil_indices]
                labels_sil_sample = cluster_labels[sil_indices]
                
                # Only calculate if sampled data still has multiple clusters
                if len(set(labels_sil_sample)) > 1:
                    sil_score = silhouette_score(X_sil_sample, labels_sil_sample)
                else:
                    sil_score = -1
            else:
                sil_score = silhouette_score(X_sample, cluster_labels)
                
            sil_time = time.time() - sil_start_time
            print(f"  Silhouette calculation took: {sil_time:.2f} seconds")
        except Exception:
            sil_score = -1
    else:
        sil_score = -1

    return sil_score


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
    
    # More detailed silhouette score reporting
    if not calculate_silhouette:
        print(f"  Silhouette Score: Disabled (calculate_silhouette=False)")
    elif n_clusters <= 1:
        print(f"  Silhouette Score: N/A (need ≥2 clusters, found {n_clusters})")
    elif n_clusters >= len(cluster_labels):
        print(f"  Silhouette Score: N/A (too many clusters relative to data)")
    elif sil_score > -1:
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


# kmeans clustering
def apply_kmeans_clustering(X_normalized, sample_indices, k_values=[3, 5, 7], sample_size=None, 
                           calculate_silhouette=True, silhouette_sample_size=10000):
    """Run K-means clustering for each value of *k*.

    Args:
        X_normalized (np.ndarray): Feature matrix of shape
            ``(n_samples, n_features)``.
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
        dict: Keyed by *k*, each value is a dict with ``'model'``,
        ``'labels'``, ``'silhouette_score'``, ``'inertia'``, and
        ``'sample_indices'``.
    """
    
    if sample_size is not None and sample_size < len(X_normalized):
        print(f"Using random sample of {sample_size:,} points for clustering (from {len(X_normalized):,} total)")
        np.random.seed(42)  # For reproducibility
        subsample_mask = np.random.choice(len(X_normalized), size=sample_size, replace=False)
        X_sample = X_normalized[subsample_mask]
        used_sample_indices = sample_indices[subsample_mask]  # Get corresponding indices
    # Use all data or subsample
    else:
        print(f"Using ALL {len(X_normalized):,} valid data points for clustering")
        X_sample = X_normalized
        used_sample_indices = sample_indices  # Track the indices we're using
        
    
    results = {}
    
    print("\n=== K-MEANS CLUSTERING RESULTS ===")
    if calculate_silhouette and len(X_sample) > silhouette_sample_size:
        print(f"Note: Silhouette scores calculated on sample of {silhouette_sample_size:,} points for efficiency")
    
    for k in k_values:
        print(f"\nTesting K-means with k={k} clusters...")
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_sample)
        
        # Calculate silhouette score with sampling for efficiency
        if calculate_silhouette:
            if len(X_sample) > silhouette_sample_size:
                np.random.seed(42)
                sil_indices = np.random.choice(len(X_sample), size=silhouette_sample_size, replace=False)
                X_sil_sample = X_sample[sil_indices]
                labels_sil_sample = cluster_labels[sil_indices]
                sil_score = silhouette_score(X_sil_sample, labels_sil_sample)
            else:
                sil_score = silhouette_score(X_sample, cluster_labels)
        else:
            sil_score = -999  # Placeholder when not calculated
        
        # Calculate inertia (within-cluster sum of squares)
        inertia = kmeans.inertia_
        
        # Store results
        results[k] = {
            'model': kmeans,
            'labels': cluster_labels,
            'silhouette_score': sil_score,
            'inertia': inertia,
            'sample_indices': used_sample_indices
        }
        
        # Show cluster statistics
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        if calculate_silhouette and sil_score != -999:
            print(f"  Silhouette Score: {sil_score:.3f}")
        else:
            print(f"  Silhouette Score: Not calculated (use calculate_silhouette=True)")
        print(f"  Inertia (WCSS): {inertia:.2f}")
        print(f"  Cluster sizes: {dict(zip(unique_labels, counts))}")
    
    return results


def apply_min_cluster_size_filter(cluster_labels, min_cluster_size):
    """Apply min_cluster_size filtering to DBSCAN results as post-processing.

    Clusters smaller than *min_cluster_size* are reclassified as noise (-1).
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
    
    # Get unique labels and their counts
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    
    # Create mapping for labels to keep vs convert to noise
    filtered_labels = cluster_labels.copy()
    
    clusters_removed = 0
    points_converted_to_noise = 0
    
    # Step 1: Convert small clusters to noise
    for label, count in zip(unique_labels, counts):
        # Skip noise points (already labeled as -1)
        if label == -1:
            continue
            
        # Convert small clusters to noise
        if count < min_cluster_size:
            filtered_labels[cluster_labels == label] = -1
            clusters_removed += 1
            points_converted_to_noise += count
    
    # Step 2: Renumber remaining clusters to be consecutive
    remaining_labels = np.unique(filtered_labels)
    remaining_clusters = remaining_labels[remaining_labels != -1]  # Exclude noise
    
    if len(remaining_clusters) > 0:
        # Create mapping from old labels to new consecutive labels
        label_mapping = {}
        for new_label, old_label in enumerate(remaining_clusters):
            label_mapping[old_label] = new_label
        
        # Apply the mapping
        renumbered_labels = filtered_labels.copy()
        for old_label, new_label in label_mapping.items():
            renumbered_labels[filtered_labels == old_label] = new_label
        
        filtered_labels = renumbered_labels
        
        print(f"  Min cluster size filter: removed {clusters_removed} small clusters "
              f"({points_converted_to_noise} points → noise), {len(remaining_clusters)} clusters remaining")
        print(f"  Renumbered clusters: {sorted(remaining_clusters)} → {list(range(len(remaining_clusters)))}")
    else:
        print(f"  Min cluster size filter: removed {clusters_removed} small clusters "
              f"({points_converted_to_noise} points → noise), 0 clusters remaining")
    
    return filtered_labels


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
    
    # Extract statistics using the shared function
    stats_dict = extract_cluster_statistics(
        ds_ml_ready, cluster_data_name, dataset_name, 
        normalize_data_name, sv_data_var, compute_pairwise_diffs
    )
    
    # Unpack for easier access
    cluster_stats_list = stats_dict['cluster_stats']
    noise_stats = stats_dict['noise_stats']
    metadata = stats_dict['metadata']
    feature_coords = stats_dict['feature_coords']
    data_description = stats_dict['data_description']
    
    # Print header
    print(f"\n{'='*80}")
    print(f"CLUSTER STATISTICS: {cluster_data_name}")
    print(f"{'='*80}")
    print(data_description)
    print(f"Total samples: {metadata['n_total_samples']:,}")
    print(f"Number of clusters: {metadata['n_clusters']}")
    if metadata['n_noise'] > 0:
        print(f"Noise points: {metadata['n_noise']:,} ({metadata['n_noise']/metadata['n_total_samples']*100:.2f}%)")
    print(f"{'='*80}\n")
    
    # Print statistics for each cluster
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
    
    # Print noise statistics if present
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
    
    # Return cluster_stats_list for backwards compatibility
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
    if cluster_colors is None:
        cluster_colors = [
                "#00F3FC", "#35E200", "#0400FF", "#F943FF", "#F30101", 
                "#EDFF4D", "#4E9200", "#970021", "#5600C7", "#017685FF", "#FFA600FF"
            ]

    # Validate stat_type
    valid_stat_types = ['mean', 'min', 'max']
    if stat_type not in valid_stat_types:
        raise ValueError(f"stat_type must be one of {valid_stat_types}, got '{stat_type}'")

    # Extract statistics using the shared function
    stats_dict = extract_cluster_statistics(
        ds_ml_ready, cluster_data_name, dataset_name,
        normalize_data_name, sv_data_var, compute_pairwise_diffs
    )

    # Unpack for easier access
    cluster_stats_list = stats_dict['cluster_stats']
    noise_stats = stats_dict['noise_stats']
    feature_coords = stats_dict['feature_coords']
    data_description = stats_dict['data_description']

    # Determine which clusters to plot
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

    # Set up colors for clusters (use provided or default)
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

    # --- Classify features as Sv or difference ---
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

    # Determine baselines and ylims for each feature group
    baseline_axes_y = 0.5
    
    # Calculate for Sv features (baseline = -100)
    if sv_feature_indices:
        sv_baseline = -100
        sv_values = stat_values[:, sv_feature_indices]
        sv_stds = stds[:, sv_feature_indices]
        sv_y_min = np.min(sv_values - sv_stds)
        sv_y_max = np.max(sv_values + sv_stds)
        sv_y_min = min(sv_y_min, sv_baseline)
        sv_y_max = max(sv_y_max, sv_baseline)
        
        # Align baseline
        below = sv_baseline - sv_y_min
        above = sv_y_max - sv_baseline
        total_range = max(below / baseline_axes_y, above / (1 - baseline_axes_y))
        sv_y_min_aligned = sv_baseline - baseline_axes_y * total_range
        sv_y_max_aligned = sv_baseline + (1 - baseline_axes_y) * total_range
    else:
        sv_baseline = None
        sv_y_min_aligned = None
        sv_y_max_aligned = None

    # Calculate for difference features (baseline = 0)
    if diff_feature_indices:
        diff_baseline = 0
        diff_values = stat_values[:, diff_feature_indices]
        diff_stds = stds[:, diff_feature_indices]
        diff_y_min = np.min(diff_values - diff_stds)
        diff_y_max = np.max(diff_values + diff_stds)
        diff_y_min = min(diff_y_min, diff_baseline)
        diff_y_max = max(diff_y_max, diff_baseline)
        
        # Align baseline
        below = diff_baseline - diff_y_min
        above = diff_y_max - diff_baseline
        total_range = max(below / baseline_axes_y, above / (1 - baseline_axes_y))
        diff_y_min_aligned = diff_baseline - baseline_axes_y * total_range
        diff_y_max_aligned = diff_baseline + (1 - baseline_axes_y) * total_range
    else:
        diff_baseline = None
        diff_y_min_aligned = None
        diff_y_max_aligned = None

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(n_clusters)
    width = 0.8 / n_features

    # Create axes for each feature group
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

    # Plot each feature group
    for group_type, ax_feature, feature_indices in axes:
        if group_type == 'sv':
            baseline = sv_baseline
            y_min, y_max = sv_y_min_aligned, sv_y_max_aligned
            ylabel = 'Sv (dB)'
        else:  # diff
            baseline = diff_baseline
            y_min, y_max = diff_y_min_aligned, diff_y_max_aligned
            ylabel = 'Sv Difference (dB)'
        
        ax_feature.set_ylim(y_min, y_max)
        ax_feature.set_ylabel(ylabel, fontsize=10)
        
        # Plot bars for all features in this group
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
        
        # Draw baseline
        ax_feature.axhline(
            y=baseline,
            color='black',
            linestyle='-',
            linewidth=2,
            alpha=0.9,
            zorder=0
        )
        
        # Draw reference lines
        if group_type == 'sv':
            ax_feature.axhline(y=-80, color='black', linestyle=':', linewidth=2, alpha=0.7, zorder=1)
            if ax_sv == ax:  # Only add grid to first axis
                ax_feature.grid(True, alpha=0.3, linestyle='--', axis='y')
        else:  # diff
            ax_feature.axhline(y=3, color='blue', linestyle=':', linewidth=2, alpha=0.7, zorder=2)
            ax_feature.axhline(y=-3, color='blue', linestyle=':', linewidth=2, alpha=0.7, zorder=2)


    
      # Feature legend (bar position markers)
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
            title = f'Cluster Statistics (Mean ±1 SD): {cluster_data_name}\n{data_description}'
        elif stat_type == 'min':
            title = f'Cluster Statistics (Min, ±1 SD of mean): {cluster_data_name}\n{data_description}'
        elif stat_type == 'max':
            title = f'Cluster Statistics (Max, ±1 SD of mean): {cluster_data_name}\n{data_description}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Cluster {cid}' if cid != -1 else 'Noise' for cid in cluster_ids], 
                       rotation=45, ha='right')
    
    # Single column legend with full feature names
    ax.legend(feature_handles, feature_labels, loc='upper center', bbox_to_anchor=(0.5, -0.15),
             ncol=1, fontsize=8, title='Features', frameon=True)
    
    plt.tight_layout()
    # Adjust bottom margin to accommodate vertical legend
    bottom_margin = 0.2 + (n_features * 0.03)  # Add space per feature
    plt.subplots_adjust(bottom=bottom_margin, right=0.88)
    
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()
    
    return fig, [ax_sv, ax_diff] if ax_diff else [ax_sv], stats_dict





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
        cluster_colors=None
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

    Returns:
        xr.Dataset: Dataset with reshaped and (optionally) normalized
        data.
    """
    
    if cluster_colors is None:
        cluster_colors = [
                "#5A00CF", "#35E200", "#FF8800", "#F943FF", "#F30101", 
                "#EDFF4D", "#4E9200", "#970021", "#5600C7", "#017685FF", "#0400FFFF"
            ]

    cluster_mask_name = None
    # Exclude a specific cluster label (e.g. background from a previous pass)
    if mask_cluster_label is not None and gridded_results_to_mask is not None:
        cluster_mask_name = f'{exclude_cluster_data_name or "cluster"}_mask'
        ds_Sv = add_cluster_label_mask(ds_Sv, gridded_results_to_mask, mask_cluster_label, mask_name=cluster_mask_name)
    elif exclude_cluster_data_name is not None and gridded_results_to_mask is not None:
        # Fallback: exclude the largest cluster
        cluster_mask_name = f'{exclude_cluster_data_name}_mask'
        ds_Sv = add_largest_cluster_mask(ds_Sv, gridded_results_to_mask, mask_name=cluster_mask_name)

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
            n_quantiles=n_quantiles
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
        useHDBScan=True,
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
        useHDBScan (bool): Use HDBSCAN instead of DBSCAN.
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
            useHDBScan=useHDBScan,
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

    if useHDBScan and not find_background_cluster:
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
        useHDBScan=True,
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
        cluster_stats_compute_pairwise_differences=True
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
        useHDBScan (bool): Use HDBSCAN. Defaults to True.
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

    Returns:
        tuple: ``(ds_final, gridded_results, dbscan_results)`` or
        ``(ds_final, gridded_results, dbscan_results,
        background_label)`` when *find_background_cluster* is True.
    """
    
    if cluster_colors is None:
        cluster_colors = [
                "#5A00CF", "#35E200", "#FF8800", "#F943FF", "#F30101", 
                "#EDFF4D", "#4E9200", "#970021", "#5600C7", "#017685FF", "#0400FFFF"
            ]

    cluster_mask_name = None
    # Exclude a specific cluster label (e.g. background from a previous pass)
    if mask_cluster_label is not None and gridded_results_to_mask is not None:
        cluster_mask_name = f'{exclude_cluster_data_name or "cluster"}_mask'
        ds_Sv = add_cluster_label_mask(ds_Sv, gridded_results_to_mask, mask_cluster_label, mask_name=cluster_mask_name)
    elif exclude_cluster_data_name is not None and gridded_results_to_mask is not None:
        # Fallback: exclude the largest cluster
        cluster_mask_name = f'{exclude_cluster_data_name}_mask'
        ds_Sv = add_largest_cluster_mask(ds_Sv, gridded_results_to_mask, mask_name=cluster_mask_name)
    

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
            n_quantiles=n_quantiles
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
        useHDBScan=useHDBScan,
        find_background_cluster=find_background_cluster,
        y_to_x_aspect_ratio_override=y_to_x_aspect_ratio_override,
        soft_membership_threshold=soft_membership_threshold, 
        cluster_colors=cluster_colors,
        overlay_line_var=overlay_line_var,
        cluster_stats_sv_data_var=cluster_stats_sv_data_var,
        cluster_stats_compute_pairwise_differences=cluster_stats_compute_pairwise_differences
        )




def retrieve_background_cluster(X, sample_indices, min_samples, sample_size, min_cluster_size, cluster_selection_method,
                                feature_threshold=-0.2, min_fraction=0.1):
    """Find a background cluster by iterating over epsilon values.

    Returns the clustering results and the label of the identified
    background cluster.  A cluster qualifies as background if:

    1. Its average ``feature[0]`` is below *feature_threshold*.
    2. It contains more than *min_fraction* of total labelled samples.

    Among qualifying clusters the largest one is selected.

    Args:
        X (np.ndarray): Feature matrix.
        sample_indices (np.ndarray): Grid-index values for each row.
        min_samples (int): Core-point neighbourhood size.
        sample_size (int): Sub-sample size.
        min_cluster_size (int): Minimum cluster size.
        cluster_selection_method (str): HDBSCAN cluster selection method.
        feature_threshold (float): Maximum average ``feature[0]`` for a
            cluster to qualify as background. Defaults to -0.2.
        min_fraction (float): Minimum fraction of total labelled samples
            a cluster must contain. Defaults to 0.1.

    Returns:
        tuple: ``(dbscan_results, background_label)``.

    Raises:
        ValueError: If no qualifying background cluster is found across
            all epsilon values.
    """
    for epsilon in [0.05, .06, .07, .08, .09, .1, .2, .3, .4, .5]:
        print(f"Trying to find background cluster with eps={epsilon}")  
        dbscan_results = apply_dbscan_clustering(
                X,
                sample_indices=sample_indices,
                eps_values=[epsilon], 
                min_samples_values=[min_samples],
                sample_size=sample_size, 
                calculate_silhouette=True,
                silhouette_sample_size=100,
                metric="euclidean",
                useHDBScan=False,
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

        # Map grid indices back to positional indices in X
        grid_idx_to_pos = {gi: pos for pos, gi in enumerate(sample_indices)}
        sample_positions = np.array([grid_idx_to_pos[gi] for gi in used_sample_indices])
        X_sample = X[sample_positions]

        # Iterate over all clusters and collect qualifying background candidates
        candidates = []  # list of (label, count, avg_feature0)
        for i, label in enumerate(valid_labels):
            count = valid_counts[i]
            fraction = count / total_samples
            cluster_mask = labels == label
            avg_feature0 = np.mean(X_sample[cluster_mask, 0])
            print(f"  Cluster {label}: avg feature[0] = {avg_feature0:.2f}, "
                  f"size = {count:,} ({fraction:.1%} of total)")
            if avg_feature0 < feature_threshold and fraction > min_fraction:
                candidates.append((label, count, avg_feature0))

        if candidates:
            # Pick the largest qualifying candidate
            best = max(candidates, key=lambda c: c[1])
            background_label = best[0]
            print(f"Found background cluster {background_label} with avg feature[0] = {best[2]:.2f}, "
                  f"size = {best[1]:,} at eps={epsilon}")
            return dbscan_results, background_label

    raise ValueError(f"Could not find background cluster with avg feature[0] < {feature_threshold} "
                     f"and > {min_fraction:.0%} of total samples")


def plot_dbscan_cluster_hierarchy(model, cluster_colors_by_index=None):
    """Plot HDBSCAN condensed tree with colours matching cluster labels.

    Args:
        model (hdbscan.HDBSCAN): Fitted HDBSCAN model.
        cluster_colors_by_index (list[str] or None): Hex colour strings
            indexed by cluster label. A default palette is used when
            ``None``.

    Returns:
        tuple: ``(label_to_tree, palette_by_tree_order)`` — mapping from
        final cluster label to condensed-tree cluster id, and colour
        palette ordered by tree cluster id.
    """
    if cluster_colors_by_index is None:
        cluster_colors_by_index = [
            "#00F3FC", "#35E200", "#0400FF", "#F943FF", "#F30101", 
            "#EDFF4D", "#4E9200", "#970021", "#5600C7", "#017685FF", "#FFA600FF"
        ]
    
    final_labels = model.labels_
    selected_clusters = model.condensed_tree_._select_clusters()
    unique_labels = sorted([l for l in set(final_labels) if l != -1])
    
    # Ensure we have enough colors for all clusters
    max_label = max(unique_labels) if unique_labels else 0
    if len(cluster_colors_by_index) <= max_label:
        num_additional_colors = max_label - len(cluster_colors_by_index) + 1
        hue_offset = 0.3
        additional_colors = utils.generate_colors(hue_offset, num_additional_colors)
        cluster_colors_by_index = cluster_colors_by_index + additional_colors
        print(f"Warning: Generated {num_additional_colors} additional colors for clusters beyond base palette")
    
    # HDBSCAN assigns final labels in sorted order of selected tree clusters
    label_to_tree = {}
    sorted_tree_clusters = sorted(selected_clusters)
    
    for final_label in unique_labels:
        label_to_tree[final_label] = sorted_tree_clusters[final_label]
    
    # Build palette in tree cluster order (required by condensed_tree_.plot)
    palette_by_tree_order = []
    for idx, tree_cluster in enumerate(sorted_tree_clusters):
        final_label = list(label_to_tree.keys())[list(label_to_tree.values()).index(tree_cluster)]
        color = cluster_colors_by_index[final_label]
        palette_by_tree_order.append(color)
    
    # Plot with correct color mapping
    ax = model.condensed_tree_.plot(select_clusters=True, selection_palette=palette_by_tree_order)
    # Fix HDBSCAN/matplotlib compatibility: Ellipse patches may have array-valued
    # width/height which newer numpy (>=2.0) rejects during rendering.
    from matplotlib.patches import Ellipse as _Ellipse
    for patch in ax.patches:
        if isinstance(patch, _Ellipse):
            if isinstance(patch._width, np.ndarray):
                patch._width = patch._width.item()
            if isinstance(patch._height, np.ndarray):
                patch._height = patch._height.item()
            if isinstance(getattr(patch, '_center', None), np.ndarray):
                patch._center = tuple(float(c) for c in patch._center)
    plt.title('HDBSCAN Condensed Tree')
    plt.show()
    
    return label_to_tree, palette_by_tree_order

