# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Focused tests for the recipe-facing clustering API."""

import numpy as np
import xarray as xr

from aa_si_ml import ml


def _make_ds_normalized():
    sample_dim = 'ml_data_clean_sample_index'
    feature_dim = 'feature_ml_data_clean'
    data = np.zeros((3, 2), dtype=float)
    return xr.Dataset(
        {
            'ml_data_clean': xr.DataArray(
                data,
                dims=[sample_dim, feature_dim],
                coords={
                    sample_dim: np.array([0, 1, 2]),
                    feature_dim: np.array(['f1', 'f2']),
                },
            )
        }
    )


def test_run_hdbscan_returns_single_clustering_result(monkeypatch):
    ds_normalized = _make_ds_normalized()

    monkeypatch.setattr(
        ml,
        'extract_valid_samples_for_sklearn',
        lambda *_args, **_kwargs: (
            np.array([[0.1, 0.2], [0.2, 0.3]]),
            np.array([0, 1]),
            np.array([10, 11]),
        ),
    )
    monkeypatch.setattr(
        ml,
        'apply_dbscan_clustering',
        lambda *_args, **_kwargs: {
            'run_1': {
                'labels': np.array([0, 1]),
                'sample_indices': np.array([10, 11]),
                'model': 'model',
                'method': 'hdbscan',
                'silhouette_score': 0.5,
            }
        },
    )

    output = ml.run_hdbscan(
        ds_normalized,
        dataset_name='ml_data_clean',
        normalization_name='normalized_data',
        ml_result_name='clusters',
        min_cluster_size=5,
    )

    assert output['background_label'] is None
    assert isinstance(output['clustering_results'], xr.Dataset)
    assert output['clustering_results'].attrs['ml_result_name'] == 'clusters'
    assert output['clustering_results'].attrs['model_type'] == 'builtins.str'
    assert output['clustering_results'].attrs['model_available'] is False
    assert output['clustering_results'].attrs['plot_hierarchy'] is True
    assert output['clustering_model'] == 'model'
    np.testing.assert_array_equal(output['clustering_results']['labels'].values, np.array([0, 1]))


def test_run_hdbscan_defaults_min_cluster_size_to_three_percent(monkeypatch):
    ds_normalized = _make_ds_normalized()
    captured = {}

    monkeypatch.setattr(
        ml,
        'extract_valid_samples_for_sklearn',
        lambda *_args, **_kwargs: (
            np.zeros((101, 2), dtype=float),
            np.arange(101),
            np.arange(101),
        ),
    )

    def _capture_clustering(*_args, **kwargs):
        captured['min_cluster_size'] = kwargs['min_cluster_size']
        return {
            'run_1': {
                'labels': np.zeros(101, dtype=int),
                'sample_indices': np.arange(101),
                'model': 'model',
                'method': 'hdbscan',
                'min_cluster_size': kwargs['min_cluster_size'],
            }
        }

    monkeypatch.setattr(ml, 'apply_dbscan_clustering', _capture_clustering)

    output = ml.run_hdbscan(
        ds_normalized,
        dataset_name='ml_data_clean',
        normalization_name='normalized_data',
        ml_result_name='clusters',
    )

    assert captured['min_cluster_size'] == 4
    assert output['clustering_results'].attrs['min_cluster_size'] == 4


def test_run_hdbscan_accepts_min_cluster_size_fraction(monkeypatch):
    ds_normalized = _make_ds_normalized()
    captured = {}

    monkeypatch.setattr(
        ml,
        'extract_valid_samples_for_sklearn',
        lambda *_args, **_kwargs: (
            np.zeros((101, 2), dtype=float),
            np.arange(101),
            np.arange(101),
        ),
    )

    def _capture_clustering(*_args, **kwargs):
        captured['min_cluster_size'] = kwargs['min_cluster_size']
        return {
            'run_1': {
                'labels': np.zeros(101, dtype=int),
                'sample_indices': np.arange(101),
                'model': 'model',
                'method': 'hdbscan',
                'min_cluster_size': kwargs['min_cluster_size'],
            }
        }

    monkeypatch.setattr(ml, 'apply_dbscan_clustering', _capture_clustering)

    output = ml.run_hdbscan(
        ds_normalized,
        dataset_name='ml_data_clean',
        normalization_name='normalized_data',
        ml_result_name='clusters',
        min_cluster_size_fraction=0.07,
    )

    assert captured['min_cluster_size'] == 8
    assert output['clustering_results'].attrs['min_cluster_size'] == 8


def test_run_hdbscan_rejects_min_cluster_size_and_fraction(monkeypatch):
    ds_normalized = _make_ds_normalized()

    monkeypatch.setattr(
        ml,
        'extract_valid_samples_for_sklearn',
        lambda *_args, **_kwargs: (
            np.zeros((101, 2), dtype=float),
            np.arange(101),
            np.arange(101),
        ),
    )

    try:
        ml.run_hdbscan(
            ds_normalized,
            dataset_name='ml_data_clean',
            normalization_name='normalized_data',
            ml_result_name='clusters',
            min_cluster_size=5,
            min_cluster_size_fraction=0.07,
        )
    except ValueError as exc:
        assert "min_cluster_size or min_cluster_size_fraction" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_run_hdbscan_returns_background_label_when_requested(monkeypatch):
    ds_normalized = _make_ds_normalized()

    monkeypatch.setattr(
        ml,
        'extract_valid_samples_for_sklearn',
        lambda *_args, **_kwargs: (
            np.array([[0.1, 0.2], [0.2, 0.3]]),
            np.array([0, 1]),
            np.array([10, 11]),
        ),
    )
    monkeypatch.setattr(
        ml,
        'retrieve_background_cluster',
        lambda *_args, **_kwargs: (
            {
                'run_1': {
                    'labels': np.array([0, -1]),
                    'sample_indices': np.array([10, 11]),
                    'model': 'model',
                    'method': 'dbscan',
                }
            },
            0,
        ),
    )

    output = ml.run_hdbscan(
        ds_normalized,
        dataset_name='ml_data_clean',
        normalization_name='normalized_data',
        ml_result_name='clusters',
        min_cluster_size=5,
        find_background_cluster=True,
    )

    assert output['background_label'] == 0
    assert output['clustering_results'].attrs['plot_hierarchy'] is False


def test_embed_clustering_results_returns_dataset_and_grid(monkeypatch):
    ds_normalized = _make_ds_normalized()

    def _store(ds_in, flat_results, specific_data_name, dataset_name='ml_data_clean', result_sample_indices=None):
        ds_out = ds_in.copy()
        ds_out[f'{dataset_name}_{specific_data_name}'] = xr.DataArray(
            flat_results,
            dims=[f'{dataset_name}_sample_index'],
            coords={f'{dataset_name}_sample_index': result_sample_indices},
        )
        return ds_out

    monkeypatch.setattr(ml, 'store_ml_results_flattened', _store)
    monkeypatch.setattr(
        ml,
        'extract_ml_data_gridded',
        lambda *_args, **_kwargs: xr.DataArray(np.array([[1.0, np.nan]]), dims=['ping_time', 'range_sample']),
    )

    clustering_result = {
        'labels': np.array([0, 1]),
        'sample_indices': np.array([0, 1]),
        'model': 'model',
        'method': 'hdbscan',
        'plot_hierarchy': True,
        'ml_result_name': 'clusters',
    }

    output = ml.embed_clustering_results(
        ds_normalized,
        clustering_result,
        dataset_name='ml_data_clean',
        ml_result_name='clusters',
    )

    assert 'ml_data_clean_clusters' in output['ds_normalized']
    assert output['gridded_results'].dims == ('ping_time', 'range_sample')


def test_embed_clustering_results_accepts_xarray_dataset(monkeypatch):
    ds_normalized = _make_ds_normalized()

    def _store(ds_in, flat_results, specific_data_name, dataset_name='ml_data_clean', result_sample_indices=None):
        ds_out = ds_in.copy()
        ds_out[f'{dataset_name}_{specific_data_name}'] = xr.DataArray(
            flat_results,
            dims=[f'{dataset_name}_sample_index'],
            coords={f'{dataset_name}_sample_index': result_sample_indices},
        )
        return ds_out

    monkeypatch.setattr(ml, 'store_ml_results_flattened', _store)
    monkeypatch.setattr(
        ml,
        'extract_ml_data_gridded',
        lambda *_args, **_kwargs: xr.DataArray(np.array([[1.0, np.nan]]), dims=['ping_time', 'range_sample']),
    )

    clustering_result = ml._clustering_result_to_dataset(
        {
            'labels': np.array([0, 1]),
            'sample_indices': np.array([0, 1]),
            'model': 'model',
            'method': 'hdbscan',
            'plot_hierarchy': True,
            'ml_result_name': 'clusters',
        }
    )

    output = ml.embed_clustering_results(
        ds_normalized,
        clustering_result,
        dataset_name='ml_data_clean',
        ml_result_name='clusters',
    )

    assert 'ml_data_clean_clusters' in output['ds_normalized']
    np.testing.assert_array_equal(
        output['ds_normalized']['ml_data_clean_clusters'].values[:2],
        np.array([0.0, 1.0]),
    )
    assert np.isnan(output['ds_normalized']['ml_data_clean_clusters'].values[2])


def test_plot_clustering_report_renders_full_report(monkeypatch):
    ds_normalized = _make_ds_normalized()
    plotted = {'echogram': 0, 'stats': 0, 'hierarchy': 0}

    monkeypatch.setattr(
        ml.echogram,
        'plot_cluster_echogram',
        lambda *args, **kwargs: plotted.__setitem__('echogram', plotted['echogram'] + 1),
    )
    monkeypatch.setattr(
        ml,
        'plot_cluster_statistics',
        lambda *args, **kwargs: plotted.__setitem__('stats', plotted['stats'] + 1),
    )
    monkeypatch.setattr(
        ml,
        'plot_dbscan_cluster_hierarchy',
        lambda *args, **kwargs: plotted.__setitem__('hierarchy', plotted['hierarchy'] + 1),
    )

    clustering_result = {
        'labels': np.array([0, 1]),
        'sample_indices': np.array([0, 1]),
        'model': 'model',
        'method': 'hdbscan',
        'plot_hierarchy': True,
        'ml_result_name': 'clusters',
    }

    ml.plot_clustering_report(
        ds_normalized,
        clustering_result,
        dataset_name='ml_data_clean',
        ml_result_name='clusters',
    )

    assert plotted == {'echogram': 1, 'stats': 1, 'hierarchy': 1}


def test_plot_clustering_report_supports_explicit_ml_feature_stats_source(monkeypatch):
    ds_normalized = _make_ds_normalized()
    captured = {}

    monkeypatch.setattr(ml.echogram, 'plot_cluster_echogram', lambda *args, **kwargs: None)
    monkeypatch.setattr(ml, 'plot_dbscan_cluster_hierarchy', lambda *args, **kwargs: None)

    def _capture_stats(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(ml, 'plot_cluster_statistics', _capture_stats)

    clustering_result = {
        'labels': np.array([0, 1]),
        'sample_indices': np.array([0, 1]),
        'model': None,
        'method': 'hdbscan',
        'plot_hierarchy': False,
        'ml_result_name': 'clusters',
        'normalization_name': 'normalized_data',
    }

    ml.plot_clustering_report(
        ds_normalized,
        clustering_result,
        dataset_name='ml_data_clean',
        ml_result_name='clusters',
        cluster_stats_sv_data_var='ml_features',
        cluster_stats_compute_pairwise_differences=False,
    )

    assert captured['sv_data_var'] == 'ml_features'
    assert captured['normalize_data_name'] == 'normalized_data'
    assert captured['compute_pairwise_diffs'] is False


def test_plot_clustering_report_uses_separate_model_for_dataset_results(monkeypatch):
    ds_normalized = _make_ds_normalized()
    plotted = {'hierarchy_model': None}

    monkeypatch.setattr(ml.echogram, 'plot_cluster_echogram', lambda *args, **kwargs: None)
    monkeypatch.setattr(ml, 'plot_cluster_statistics', lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ml,
        'plot_dbscan_cluster_hierarchy',
        lambda model, **kwargs: plotted.__setitem__('hierarchy_model', model),
    )

    clustering_result = ml._clustering_result_to_dataset(
        {
            'labels': np.array([0, 1]),
            'sample_indices': np.array([0, 1]),
            'model': 'model',
            'method': 'hdbscan',
            'plot_hierarchy': True,
            'ml_result_name': 'clusters',
        }
    )

    ml.plot_clustering_report(
        ds_normalized,
        clustering_result,
        dataset_name='ml_data_clean',
        ml_result_name='clusters',
        clustering_model='model',
    )

    assert plotted['hierarchy_model'] == 'model'


def test_embed_clustering_results_supports_list_input(monkeypatch):
    ds_normalized = _make_ds_normalized()

    def _store(ds_in, flat_results, specific_data_name, dataset_name='ml_data_clean', result_sample_indices=None):
        ds_out = ds_in.copy()
        ds_out[f'{dataset_name}_{specific_data_name}'] = xr.DataArray(
            flat_results,
            dims=[f'{dataset_name}_sample_index'],
            coords={f'{dataset_name}_sample_index': result_sample_indices},
        )
        return ds_out

    monkeypatch.setattr(ml, 'store_ml_results_flattened', _store)
    monkeypatch.setattr(
        ml,
        'extract_ml_data_gridded',
        lambda ds_in, specific_data_name, **_kwargs: xr.DataArray(
            np.array([[len(specific_data_name)]]), dims=['ping_time', 'range_sample']
        ),
    )

    output = ml.embed_clustering_results(
        ds_normalized,
        [
            {'labels': np.array([0]), 'sample_indices': np.array([0]), 'ml_result_name': 'clusters_a'},
            {'labels': np.array([1]), 'sample_indices': np.array([1]), 'ml_result_name': 'clusters_b'},
        ],
        dataset_name='ml_data_clean',
        ml_result_name='clusters',
    )

    assert set(output['gridded_results'].keys()) == {'clusters_a', 'clusters_b'}
    assert 'ml_data_clean_clusters_a' in output['ds_normalized']
    assert 'ml_data_clean_clusters_b' in output['ds_normalized']


def test_extract_data_and_run_hdbscan_preserves_wrapper_tuple(monkeypatch):
    ds_normalized = _make_ds_normalized()
    embedded_dataset = ds_normalized.copy()
    gridded = xr.DataArray(np.array([[1.0]]), dims=['ping_time', 'range_sample'])
    raw_dbscan_results = {
        'run_1': {'labels': np.array([0]), 'sample_indices': np.array([0])}
    }
    plotted = {'count': 0}

    monkeypatch.setattr(
        ml,
        'run_hdbscan',
        lambda *_args, **_kwargs: {
            'clustering_results': {'labels': np.array([0]), 'sample_indices': np.array([0])},
            'dbscan_results': raw_dbscan_results,
            'background_label': None,
        },
    )
    monkeypatch.setattr(
        ml,
        'embed_clustering_results',
        lambda *_args, **_kwargs: {
            'ds_normalized': embedded_dataset,
            'gridded_results': gridded,
        },
    )
    monkeypatch.setattr(
        ml,
        'plot_clustering_report',
        lambda *args, **kwargs: plotted.__setitem__('count', plotted['count'] + 1),
    )

    ds_final, gridded_results, clustering_results = ml.extract_data_and_run_hdbscan(
        ds_normalized,
        'ml_data_clean',
    )

    assert ds_final is embedded_dataset
    assert gridded_results is gridded
    assert clustering_results is raw_dbscan_results
    assert clustering_results['run_1']['labels'][0] == 0
    assert plotted['count'] == 1