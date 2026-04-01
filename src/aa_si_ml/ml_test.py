import numpy as np
import xarray as xr
from aa_si_ml.ml import (
    add_valid_data_mask,
    create_ml_index_coordinate,
    extract_ml_data_flattened,
    extract_ml_data_gridded,
    extract_valid_samples_for_sklearn,
    normalize_data,
    reshape_data_for_ml,
    store_ml_results_flattened,
)


def create_mock_echodata(ping_time_size=100, range_sample_size=50, n_channels=4, add_nans=True, add_artifacts=True):
    """Create mock echodata with same structure as real data for testing.

    Args:
        ping_time_size (int): Number of ping times. Defaults to 100.
        range_sample_size (int): Number of range samples. Defaults to 50.
        n_channels (int): Number of frequency channels. Defaults to 4.
        add_nans (bool): Whether to add some NaN values. Defaults to True.
        add_artifacts (bool): Whether to add extreme values (artifacts).
            Defaults to True.

    Returns:
        xr.Dataset: Mock dataset with Sv data.
    """
    import pandas as pd
    
    # Create coordinates
    ping_time = pd.date_range('2023-01-01', periods=ping_time_size, freq='1s')  # Use lowercase 's'
    range_sample = np.arange(range_sample_size)
    channel = [f'freq_{i}' for i in range(n_channels)]
    
    # Create realistic Sv data with CORRECT dimension order (channel, ping_time, range_sample)
    np.random.seed(42)  # For reproducible tests
    sv_data = np.random.normal(-60, 20, (n_channels, ping_time_size, range_sample_size))
    
    # Add some structure to make it more realistic
    # Simulate layers at different depths
    for i in range(n_channels):
        # Add a "scattering layer" 
        layer_depth = 10 + i * 5
        layer_width = 8
        layer_start = max(0, layer_depth - layer_width//2)
        layer_end = min(range_sample_size, layer_depth + layer_width//2)
        sv_data[i, :, layer_start:layer_end] += 15  # Stronger backscatter in layer
    
    # Add some NaN values if requested
    if add_nans:
        nan_mask = np.random.random((n_channels, ping_time_size, range_sample_size)) < 0.1
        sv_data[nan_mask] = np.nan
    
    # Add some artifacts if requested  
    if add_artifacts:
        # Extremely low values
        artifact_mask_low = np.random.random((n_channels, ping_time_size, range_sample_size)) < 0.05
        sv_data[artifact_mask_low] = -250
        
        # Extremely high values
        artifact_mask_high = np.random.random((n_channels, ping_time_size, range_sample_size)) < 0.05
        sv_data[artifact_mask_high] = 100
    
    # Create xarray dataset with CORRECT dimension order
    ds_mock = xr.Dataset(
        {
            'Sv': (['channel', 'ping_time', 'range_sample'], sv_data),
        },
        coords={
            'channel': channel,
            'ping_time': ping_time,
            'range_sample': range_sample
        }
    )
    
    # Add attributes like real echodata
    ds_mock['Sv'].attrs['long_name'] = 'Volume backscattering strength'
    ds_mock['Sv'].attrs['units'] = 'dB re 1 m-1'
    
    print(f"Created mock echodata:")
    print(f"  Shape: {ds_mock['Sv'].shape} (channel, ping_time, range_sample)")
    print(f"  Sv range: {ds_mock['Sv'].min().values:.1f} to {ds_mock['Sv'].max().values:.1f} dB")
    print(f"  NaN count: {np.isnan(ds_mock['Sv'].values).sum():,}")
    
    return ds_mock




# Test Functions

def test_ml_index_tracking():
    """Test that grid indices correctly track data points through transformations."""
    print("="*60)
    print("TESTING GRID INDEX TRACKING")
    print("="*60)
    
    # Create small mock dataset for easy verification
    ds_mock = create_mock_echodata(ping_time_size=5, range_sample_size=7, n_channels=4, 
                                  add_nans=True, add_artifacts=True)
    
    print(f"\nOriginal data shape: {ds_mock['Sv'].shape} (channel, ping_time, range_sample)")
    print(f"Original data (first channel):")
    print(ds_mock['Sv'][0, :, :].values)
    print(f"NaN count in original data: {np.isnan(ds_mock['Sv'].values).sum()}")
    print(f"Artifact count in original data: {((ds_mock['Sv'].values < -200) | (ds_mock['Sv'].values > 50)).sum()}")
    
    # Step 1: Add valid mask
    ds_with_mask = add_valid_data_mask(ds_mock, remove_nan=True, mask_invalid_values=True)
    
    # Step 2: Create grid index coordinate  
    ds_with_index = create_ml_index_coordinate(ds_with_mask)
    
    print(f"\nGrid index grid:")
    print(ds_with_index['grid_index'].values)
    
    # Step 3: Extract flattened data
    ml_data_flat, grid_indices = extract_ml_data_flattened(ds_with_index, 'Sv', mask_name='valid_mask', dataset_name="ml_data_clean")

    print(f"\nFlattened data shape: {ml_data_flat.shape}")
    print(f"Grid indices for first 5 samples: {grid_indices[:min(5, len(grid_indices))]}")
    print(f"Flattened data for first 3 samples (all channels):")
    print(ml_data_flat[:min(3, len(grid_indices)), :].values)
    
    # Verify no NaNs in flattened data
    print(f"\nFiltering verification:")
    print(f"NaNs in flattened data: {np.isnan(ml_data_flat.values).sum()}")
    print(f"Values < -200 in flattened: {(ml_data_flat.values < -200).sum()}")
    print(f"Values > 50 in flattened: {(ml_data_flat.values > 50).sum()}")
    
    # Step 4: Verify index tracking by checking specific points
    print(f"\nVERIFYING INDEX TRACKING:")
    for i in range(min(3, len(grid_indices))):
        grid_idx = grid_indices[i]
        
        # Convert grid index back to grid coordinates
        ping_idx = grid_idx // ds_with_index.sizes['range_sample']
        range_idx = grid_idx % ds_with_index.sizes['range_sample']
        
        # Get original data at this point - use correct dimension order
        original_data = ds_mock['Sv'][:, ping_idx, range_idx].values
        
        # Get flattened data at this point
        flat_data = ml_data_flat[i, :].values
        
        print(f"  Sample {i}: grid_idx={grid_idx} -> grid_pos=({ping_idx},{range_idx})")
        print(f"    Original: {original_data}")
        print(f"    Flattened: {flat_data}")
        print(f"    Match: {np.allclose(original_data, flat_data, equal_nan=True)}")
    
    return ds_with_index, ml_data_flat, grid_indices


def test_nan_filtering_specifically():
    """Dedicated test for NaN filtering functionality."""
    print("="*60)
    print("TESTING NaN FILTERING SPECIFICALLY")
    print("="*60)
    
    # Create data with known NaN pattern
    ds_mock = create_mock_echodata(ping_time_size=5, range_sample_size=7, n_channels=4,
                                  add_nans=False, add_artifacts=False)
    
    # Manually add NaNs in specific, predictable locations - use correct dimension order
    ds_mock['Sv'][:, 0, 0] = np.nan  # First point, all channels
    ds_mock['Sv'][0, 1, 1] = np.nan  # Second ping, second range, first channel only
    ds_mock['Sv'][:, 2, 2] = np.nan  # Third point, all channels
    
    print(f"Original data shape: {ds_mock['Sv'].shape} (channel, ping_time, range_sample)")
    print(f"Manually added NaN locations:")
    print(f"  (0,0) -> all channels")
    print(f"  (1,1,0) -> first channel only") 
    print(f"  (2,2) -> all channels")
    print(f"Total NaNs: {np.isnan(ds_mock['Sv'].values).sum()}")
    
    # Test the integrated pipeline (no separate add_valid_data_mask call)
    ds_ml_ready = reshape_data_for_ml(ds_mock, remove_nan=True, mask_invalid_values=False)
    
    # Check what gets filtered
    valid_mask = ds_ml_ready['ml_data_clean_valid_mask']
    valid_across_all_channels = valid_mask.all(dim='channel')
    
    print(f"\nValid mask results:")
    print(f"Points valid in all channels: {valid_across_all_channels.sum().values}")
    print(f"Expected to be filtered out:")
    print(f"  (0,0) -> has NaN in all channels")
    print(f"  (1,1) -> has NaN in one channel")  
    print(f"  (2,2) -> has NaN in all channels")
    
    # Verify filtering worked
    ml_data = ds_ml_ready['ml_data_clean']
    print(f"\nFiltering verification:")
    print(f"Flattened data shape: {ml_data.shape}")
    print(f"Contains any NaN: {np.isnan(ml_data.values).any()}")
    print(f"Expected valid samples: {6*5 - 3} = {6*5 - 3} (total - 3 filtered points)")
    print(f"Actual valid samples: {ml_data.shape[0]}")
    
    return ds_ml_ready


def test_filtering_and_masking():
    """Test that filtering correctly removes NaNs and artifacts."""
    print("="*60)
    print("TESTING FILTERING AND MASKING")
    print("="*60)
    
    # Create mock data with known NaNs and artifacts
    ds_mock = create_mock_echodata(ping_time_size=5, range_sample_size=7, n_channels=4,
                                  add_nans=True, add_artifacts=True)
    
    print(f"Original data shape: {ds_mock['Sv'].shape}")
    print(f"Total NaN count: {np.isnan(ds_mock['Sv'].values).sum():,}")
    print(f"Values < -200: {(ds_mock['Sv'].values < -200).sum():,}")
    print(f"Values > 50: {(ds_mock['Sv'].values > 50).sum():,}")
    
    # Test extraction with integrated filtering (no separate add_valid_data_mask call)
    ds_ml_ready = reshape_data_for_ml(ds_mock, remove_nan=True, mask_invalid_values=True)
    
    # Check mask results
    valid_mask = ds_ml_ready['ml_data_clean_valid_mask']
    
    print(f"\nValid mask results:")
    print(f"Valid points per channel: {valid_mask.sum(dim=['ping_time', 'range_sample']).values}")
    print(f"Points valid across ALL channels: {valid_mask.all(dim='channel').sum().values}")
    
    # Verify no NaNs or artifacts in flattened data
    ml_data = ds_ml_ready['ml_data_clean']
    print(f"\nFlattened data verification:")
    print(f"Shape: {ml_data.shape}")
    print(f"Contains NaN: {np.isnan(ml_data.values).any()}")
    print(f"Min value: {ml_data.min().values:.2f}")
    print(f"Max value: {ml_data.max().values:.2f}")
    print(f"All values in valid range: {((ml_data.values >= -200) & (ml_data.values <= 50)).all()}")
    
    return ds_ml_ready


def test_regridding_accuracy():
    """Test that regridding accurately places results back in original grid positions.

    Tests both single-dimensional results (clusters) and multi-dimensional
    data (normalized ML data).
    """
    print("="*60)
    print("TESTING REGRIDDING ACCURACY")
    print("="*60)
    
    # Create simple mock data
    ds_mock = create_mock_echodata(ping_time_size=5, range_sample_size=7, n_channels=4,
                                  add_nans=True, add_artifacts=False)
    
    # Prepare ML data with integrated masking
    ds_ml_ready = reshape_data_for_ml(ds_mock)
    
    # Add normalization to test multi-dimensional regridding
    ds_normalized = normalize_data(ds_ml_ready, method='standard', dataset_name='ml_data_clean')
    
    print("\n3. EXTRACTING FOR SKLEARN...")
    X, grid_indices, _ = extract_valid_samples_for_sklearn(ds_normalized, specific_data_name='standard_normalized', dataset_name='ml_data_clean')
    
    print(f"Original grid shape: {ds_mock['Sv'].shape}")
    print(f"Valid samples extracted: {X.shape[0]}")
    
    # Create mock ML results (e.g., cluster labels)
    np.random.seed(123)
    mock_clusters = np.random.randint(0, 3, size=X.shape[0])
    
    print(f"Mock cluster labels: {np.unique(mock_clusters, return_counts=True)}")
    
    print("\n5a. REGRIDDING SINGLE-DIMENSIONAL RESULTS (clusters)...")

    # Store flattened results first
    ds_final = store_ml_results_flattened(ds_normalized, mock_clusters, 'kmeans_clusters', dataset_name='ml_data_clean')
    
    # Then create gridded version and store it
    gridded_clusters = extract_ml_data_gridded(ds_final, 'kmeans_clusters', dataset_name='ml_data_clean', 
                                                fill_value=-1, store_in_dataset=True)
    
    # Verify regridding accuracy for clusters
    print(f"\nRegridded cluster results shape: {gridded_clusters.shape}")
    print(f"Fill value (-1) count: {(gridded_clusters.values == -1).sum()}")
    print(f"Valid result count: {(gridded_clusters.values != -1).sum()}")
    
    print("\n5b. REGRIDDING MULTI-DIMENSIONAL DATA (normalized ML data)...")
    
    # Test regridding of normalized ML data (multi-dimensional with channels)
    gridded_normalized = extract_ml_data_gridded(ds_final, 'standard_normalized', dataset_name='ml_data_clean', 
                                                   fill_value=np.nan, store_in_dataset=True)
    
    print(f"Regridded normalized data shape: {gridded_normalized.shape}")
    print(f"Expected shape: {ds_mock['Sv'].shape} (channel, ping_time, range_sample)")
    print(f"Shapes match: {gridded_normalized.shape == ds_mock['Sv'].shape}")
    
    # Check specific points for both types
    print(f"\nVERIFYING REGRIDDING ACCURACY:")
    for i in range(min(3, len(grid_indices))):
        grid_idx = grid_indices[i]
        original_cluster = mock_clusters[i]
        original_normalized = X[i, :]  # All channels for this sample
        
        # Convert to grid coordinates
        ping_idx = grid_idx // ds_final.sizes['range_sample']
        range_idx = grid_idx % ds_final.sizes['range_sample']
        
        # Get regridded values
        regridded_cluster = gridded_clusters[ping_idx, range_idx].values
        regridded_normalized = gridded_normalized[:, ping_idx, range_idx].values
        
        print(f"  Sample {i}: grid_idx={grid_idx} -> ({ping_idx},{range_idx})")
        print(f"    Cluster - Original: {original_cluster}, Regridded: {regridded_cluster}, Match: {original_cluster == regridded_cluster}")
        print(f"    Normalized - Match: {np.allclose(original_normalized, regridded_normalized, equal_nan=True)}")
    
    # Verify that regridded normalized data has same valid pattern as original
    valid_mask = ds_final['ml_data_clean_valid_mask']
    valid_pattern = valid_mask.all(dim='channel')
    normalized_valid_pattern = ~np.isnan(gridded_normalized[0, :, :])  # Use first channel to check pattern
    
    print(f"\nValid pattern verification:")
    print(f"Original valid count: {valid_pattern.sum().values}")
    print(f"Regridded valid count: {normalized_valid_pattern.sum()}")
    print(f"Valid patterns match: {np.array_equal(valid_pattern.values, normalized_valid_pattern)}")
    
    return ds_final


def test_full_ml_pipeline():
    """Test the complete ML pipeline end-to-end."""
    print("="*60)
    print("TESTING FULL ML PIPELINE")
    print("="*60)
    
    # Create realistic mock dataset
    ds_mock = create_mock_echodata(ping_time_size=5, range_sample_size=7, n_channels=4)

    print("\n1. PREPARING ML DATA (with integrated masking)...")
    ds_ml_ready = reshape_data_for_ml(ds_mock)

    print("\n2. NORMALIZING DATA...")
    ds_normalized = normalize_data(ds_ml_ready, method='standard', dataset_name='ml_data_clean')

    print("\n3. EXTRACTING FOR SKLEARN...")
    X, _, _ = extract_valid_samples_for_sklearn(ds_normalized, specific_data_name='standard_normalized', dataset_name='ml_data_clean')

    print("\n4. RUNNING MOCK ML ALGORITHM...")
    # Simulate a simple clustering algorithm
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    print(f"Cluster distribution: {np.unique(cluster_labels, return_counts=True)}")

    print("\n5. REGRIDDING RESULTS...")

    # Store flattened results first (now needs dataset_name parameter)
    ds_final = store_ml_results_flattened(ds_normalized, cluster_labels, 'kmeans_clusters', dataset_name='ml_data_clean')
    
    # Then create gridded version and store it (now needs dataset_name parameter)
    gridded_results = extract_ml_data_gridded(ds_final, 'kmeans_clusters', dataset_name='ml_data_clean', 
                                                fill_value=-1, store_in_dataset=True)

    print("\n6. PIPELINE VERIFICATION...")
    print(f"Final dataset variables: {list(ds_final.data_vars.keys())}")
    print(f"Storage efficiency check:")
    
    # Compare storage sizes using original data and flattened data
    original_total_size = ds_final.sizes['ping_time'] * ds_final.sizes['range_sample'] * ds_final.sizes['channel']
    flattened_size = ds_final['ml_data_clean'].size
    print(f"  Original grid storage: {original_total_size:,} elements")
    print(f"  Flattened storage: {flattened_size:,} elements")
    print(f"  Storage reduction: {(1 - flattened_size/original_total_size)*100:.1f}%")
    
    # Verify data consistency
    valid_mask = ds_final['ml_data_clean_valid_mask']
    expected_valid_count = valid_mask.all(dim='channel').sum().values
    actual_flat_count = ds_final['ml_data_clean'].shape[0]
    print(f"  Expected valid elements: {expected_valid_count:,}")
    print(f"  Actual flat elements: {actual_flat_count:,}")
    print(f"  Counts match: {expected_valid_count == actual_flat_count}")
    
    return ds_final


def run_all_tests():
    """Run all test functions in sequence."""
    print("STARTING ML PIPELINE TESTS")
    print("="*80)
    
    try:
        print("\nTest 1: Grid Index Tracking")
        ds1, data1, indices1 = test_ml_index_tracking()
        print("✓ PASSED")
        print("\nTest 1.5: NaN Filtering Specifically")
        ds1_5 = test_nan_filtering_specifically()
        print("✓ PASSED")
        print("\nTest 2: Filtering and Masking") 
        ds2 = test_filtering_and_masking()
        print("✓ PASSED")
        print("\nTest 3: Regridding Accuracy")
        ds3 = test_regridding_accuracy()
        print("✓ PASSED")
        print("\nTest 4: Full ML Pipeline")
        ds4 = test_full_ml_pipeline()
        print("✓ PASSED")
        print("\n" + "="*80)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*80)
        return ds4
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return None