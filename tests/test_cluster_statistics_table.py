# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Tests for the JSON cluster statistics table."""

import json

import numpy as np
import pytest
import xarray as xr

from aa_si_ml import ml

DATASET = 'ml_data_clean'
RESULT = 'hdbscan_results'
CHANNELS = [
    'GPT  38 kHz 0002 2-1 ES38B',
    'GPT  18 kHz 0001 1-1 ES18-11',
    'GPT 120 kHz 0003 3-1 ES120-7C',
]


def _labelled_dataset(sv=None, labels=None, **reshape_kwargs):
    rng = np.random.default_rng(0)
    if sv is None:
        sv = rng.normal(-70.0, 5.0, size=(3, 4, 5))
    ds = xr.Dataset(
        {'Sv': (('channel', 'ping_time', 'range_sample'), sv)},
        coords={
            'channel': CHANNELS,
            'ping_time': np.arange(4),
            'range_sample': np.arange(5),
        },
    )
    ds = ml.reshape_data_for_ml(ds, data_var='Sv', dataset_name=DATASET, **reshape_kwargs)
    n_samples = ds.sizes[f'{DATASET}_sample_index']
    if labels is None:
        labels = np.array([0, 1, -1] * (n_samples // 3 + 1))[:n_samples]
    return ml.store_ml_results_flattened(ds, np.asarray(labels), RESULT, dataset_name=DATASET)


def test_table_has_the_documented_shape_and_is_json_serialisable():
    table = ml.cluster_statistics_table(_labelled_dataset(), RESULT, DATASET)

    json.dumps(table, allow_nan=False)
    assert set(table) == {
        'schema', 'title', 'description', 'group_column', 'columns',
        'features', 'rows', 'source',
    }
    assert table['schema'] == 'aa-table/1'
    assert table['group_column'] == 'group'
    assert table['source']['ml_result_name'] == RESULT
    assert table['source']['n_clusters'] == 2

    names = [column['name'] for column in table['columns']]
    assert names[:4] == ['group', 'group_id', 'n_samples', 'percentage']
    for column in table['columns'][4:]:
        assert column['unit'] == 'dB'
        assert column['stat'] in ('mean', 'std', 'min', 'max')
        assert column['feature'] in {feature['name'] for feature in table['features']}
    assert all(column['name'] in table['rows'][0] for column in table['columns'])


def test_features_are_ordered_by_frequency_with_differences_last():
    table = ml.cluster_statistics_table(_labelled_dataset(), RESULT, DATASET)

    kinds = [feature['kind'] for feature in table['features']]
    labels = [feature['label'] for feature in table['features']]
    assert kinds == ['sv', 'sv', 'sv', 'difference', 'difference']
    assert labels[:3] == ['18 kHz', '38 kHz', '120 kHz']
    assert labels[3:] == ['18 kHz - 38 kHz', '120 kHz - 38 kHz']


def test_rows_cover_each_cluster_and_noise_with_counts():
    ds = _labelled_dataset()
    table = ml.cluster_statistics_table(ds, RESULT, DATASET)
    n_samples = ds.sizes[f'{DATASET}_sample_index']

    groups = [(row['group'], row['group_id']) for row in table['rows']]
    assert groups == [('Cluster 0', 0), ('Cluster 1', 1), ('Noise', -1)]
    assert sum(row['n_samples'] for row in table['rows']) == n_samples
    assert sum(row['percentage'] for row in table['rows']) == pytest.approx(100.0)
    for row in table['rows']:
        assert all(type(row[key]) is int for key in ('group_id', 'n_samples'))
        assert type(row['percentage']) is float


def test_statistics_match_extract_cluster_statistics():
    ds = _labelled_dataset()
    table = ml.cluster_statistics_table(ds, RESULT, DATASET)
    stats = ml.extract_cluster_statistics(
        ds, RESULT, dataset_name=DATASET, sv_data_var='Sv', compute_pairwise_diffs=True
    )

    first = stats['cluster_stats'][0]
    feature = table['features'][0]['name']
    assert table['rows'][0][f'{feature}_mean'] == pytest.approx(first[f'{feature}_mean'])
    assert table['rows'][0][f'{feature}_std'] == pytest.approx(first[f'{feature}_std'])


def test_nan_statistics_become_null():
    sv = np.random.default_rng(0).normal(-70.0, 5.0, size=(3, 4, 5))
    sv[0, 0, 0] = np.nan
    ds = _labelled_dataset(sv=sv, remove_nan=False, mask_invalid_values=False)

    table = ml.cluster_statistics_table(ds, RESULT, DATASET)

    json.dumps(table, allow_nan=False)
    nulls = [value for row in table['rows'] for value in row.values() if value is None]
    assert nulls


def test_differences_can_be_left_out():
    table = ml.cluster_statistics_table(
        _labelled_dataset(), RESULT, DATASET, compute_pairwise_diffs=False
    )

    assert [feature['kind'] for feature in table['features']] == ['sv', 'sv', 'sv']


def test_ml_feature_source_has_no_unit():
    table = ml.cluster_statistics_table(
        _labelled_dataset(), RESULT, DATASET, sv_data_var='ml_features'
    )

    json.dumps(table, allow_nan=False)
    assert all(column['unit'] is None for column in table['columns'][4:])
    assert table['source']['sv_data_var'] == 'ml_features'
