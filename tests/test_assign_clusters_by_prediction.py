# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Tests for extending a subsampled clustering to every point."""

import hdbscan
import numpy as np
import pytest
import xarray as xr

from aa_si_ml import ml, ml_algorithms


def _ds_normalized(n_samples=6):
    sample_dim = 'ml_data_clean_sample_index'
    feature_dim = 'feature_ml_data_clean'
    return xr.Dataset(
        {
            'ml_data_clean': xr.DataArray(
                np.zeros((n_samples, 2), dtype=float),
                dims=[sample_dim, feature_dim],
                coords={
                    sample_dim: np.arange(n_samples),
                    feature_dim: np.array(['f1', 'f2']),
                },
            )
        }
    )


def _fitted_result(labels, sample_indices, ml_result_name='clusters'):
    return ml._clustering_result_to_dataset({
        'labels': np.asarray(labels),
        'sample_indices': np.asarray(sample_indices),
        'method': 'hdbscan',
        'ml_result_name': ml_result_name,
        'dataset_name': 'ml_data_clean',
        'normalization_name': 'normalized_data',
    })


class _StubModel:
    """Stands in for a fitted HDBSCAN, returning a fixed label per row."""

    def __init__(self, labels_by_row):
        self.labels_by_row = labels_by_row
        self.prediction_data_ = object()
        self.calls = []


def _patch_extract(monkeypatch, X, sample_indices):
    monkeypatch.setattr(
        ml,
        'extract_valid_samples_for_sklearn',
        lambda *_a, **_kw: (X, np.arange(len(X)), np.asarray(sample_indices)),
    )


def _patch_predict(monkeypatch, labels):
    seen = {}

    def fake_predict(model, X, batch_size=200000):
        seen['n_rows'] = len(X)
        seen['batch_size'] = batch_size
        return np.asarray(labels)

    monkeypatch.setattr(ml, 'predict_cluster_labels', fake_predict)
    return seen


# ---------------------------------------------------------------------------
# Coverage: every sample index ends up labelled
# ---------------------------------------------------------------------------


def test_result_covers_every_sample_index(monkeypatch):
    X = np.arange(12, dtype=float).reshape(6, 2)
    _patch_extract(monkeypatch, X, [10, 11, 12, 13, 14, 15])
    _patch_predict(monkeypatch, [1, 1, -1, 0])

    out = ml.assign_clusters_by_prediction(
        _ds_normalized(),
        _fitted_result([0, 1], [11, 14]),
        clustering_model=_StubModel(None),
        dataset_name='ml_data_clean',
        normalization_name='normalized_data',
    )

    result = out['clustering_results']
    np.testing.assert_array_equal(
        result['sample_indices'].values, [10, 11, 12, 13, 14, 15]
    )
    assert result['labels'].values.shape == (6,)


def test_fitted_points_keep_their_fitted_labels(monkeypatch):
    """The fit is authoritative where it exists; only the rest is filled in."""
    X = np.arange(12, dtype=float).reshape(6, 2)
    _patch_extract(monkeypatch, X, [10, 11, 12, 13, 14, 15])
    # Four unfitted rows (positions 0, 2, 3, 5) all predicted as cluster 7.
    _patch_predict(monkeypatch, [7, 7, 7, 7])

    out = ml.assign_clusters_by_prediction(
        _ds_normalized(),
        _fitted_result([3, 5], [11, 14]),
        clustering_model=_StubModel(None),
        dataset_name='ml_data_clean',
        normalization_name='normalized_data',
    )

    labels = out['clustering_results']['labels'].values
    # positions 1 and 4 are sample indices 11 and 14, the fitted ones
    assert labels[1] == 3
    assert labels[4] == 5
    np.testing.assert_array_equal(labels[[0, 2, 3, 5]], [7, 7, 7, 7])


def test_only_the_unfitted_rows_are_predicted(monkeypatch):
    X = np.arange(12, dtype=float).reshape(6, 2)
    _patch_extract(monkeypatch, X, [10, 11, 12, 13, 14, 15])
    seen = _patch_predict(monkeypatch, [7, 7, 7, 7])

    ml.assign_clusters_by_prediction(
        _ds_normalized(),
        _fitted_result([3, 5], [11, 14]),
        clustering_model=_StubModel(None),
        dataset_name='ml_data_clean',
        normalization_name='normalized_data',
        batch_size=1234,
    )

    assert seen['n_rows'] == 4
    assert seen['batch_size'] == 1234


def test_a_complete_fit_predicts_nothing(monkeypatch):
    X = np.arange(6, dtype=float).reshape(3, 2)
    _patch_extract(monkeypatch, X, [10, 11, 12])

    def explode(*_a, **_kw):
        raise AssertionError("predict must not run when the fit already covers all")

    monkeypatch.setattr(ml, 'predict_cluster_labels', explode)

    out = ml.assign_clusters_by_prediction(
        _ds_normalized(3),
        _fitted_result([0, 1, 0], [10, 11, 12]),
        clustering_model=_StubModel(None),
        dataset_name='ml_data_clean',
        normalization_name='normalized_data',
    )

    np.testing.assert_array_equal(out['clustering_results']['labels'].values, [0, 1, 0])


# ---------------------------------------------------------------------------
# Metadata and error handling
# ---------------------------------------------------------------------------


def test_counts_and_provenance_land_in_attrs(monkeypatch):
    X = np.arange(12, dtype=float).reshape(6, 2)
    _patch_extract(monkeypatch, X, [10, 11, 12, 13, 14, 15])
    _patch_predict(monkeypatch, [0, -1, -1, 1])

    out = ml.assign_clusters_by_prediction(
        _ds_normalized(),
        _fitted_result([0, 1], [11, 14]),
        clustering_model=_StubModel(None),
        dataset_name='ml_data_clean',
        normalization_name='normalized_data',
    )

    attrs = out['clustering_results'].attrs
    assert attrs['n_fitted'] == 2
    assert attrs['n_predicted'] == 4
    assert attrs['n_noise'] == 2
    assert attrs['n_clusters'] == 2
    assert attrs['label_source'] == 'fit + approximate_predict'
    assert out['cluster_labels'] == [0, 1]


def test_ml_result_name_can_be_renamed(monkeypatch):
    X = np.arange(12, dtype=float).reshape(6, 2)
    _patch_extract(monkeypatch, X, [10, 11, 12, 13, 14, 15])
    _patch_predict(monkeypatch, [0, 0, 0, 0])

    out = ml.assign_clusters_by_prediction(
        _ds_normalized(),
        _fitted_result([0, 1], [11, 14], ml_result_name='pass1'),
        clustering_model=_StubModel(None),
        dataset_name='ml_data_clean',
        normalization_name='normalized_data',
        ml_result_name='pass1_full',
    )

    assert out['clustering_results'].attrs['ml_result_name'] == 'pass1_full'


def test_missing_model_raises_with_a_pointed_message():
    with pytest.raises(ValueError, match="does not carry the model"):
        ml.assign_clusters_by_prediction(
            _ds_normalized(),
            _fitted_result([0, 1], [11, 14]),
            clustering_model=None,
            dataset_name='ml_data_clean',
            normalization_name='normalized_data',
        )


def test_mismatched_sample_indices_raise(monkeypatch):
    X = np.arange(12, dtype=float).reshape(6, 2)
    _patch_extract(monkeypatch, X, [10, 11, 12, 13, 14, 15])

    with pytest.raises(ValueError, match="do not belong together"):
        ml.assign_clusters_by_prediction(
            _ds_normalized(),
            _fitted_result([0, 1], [11, 999]),
            clustering_model=_StubModel(None),
            dataset_name='ml_data_clean',
            normalization_name='normalized_data',
        )


# ---------------------------------------------------------------------------
# predict_cluster_labels against a real HDBSCAN fit
# ---------------------------------------------------------------------------


def _two_blobs(n_per_blob=150, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=(0.0, 0.0), scale=0.15, size=(n_per_blob, 2))
    b = rng.normal(loc=(5.0, 5.0), scale=0.15, size=(n_per_blob, 2))
    return np.vstack([a, b])


def test_predict_recovers_the_fitted_structure_on_held_out_points():
    """The real check: a point near a blob gets that blob's label."""
    X = _two_blobs()
    fit_rows = np.arange(0, len(X), 2)
    model = hdbscan.HDBSCAN(min_cluster_size=15, prediction_data=True)
    model.fit(X[fit_rows])

    held_out = np.setdiff1d(np.arange(len(X)), fit_rows)
    predicted = ml_algorithms.predict_cluster_labels(model, X[held_out])

    assert predicted.shape == (len(held_out),)
    # Held-out points sit inside the blobs, so almost all should be assigned.
    assert (predicted >= 0).mean() > 0.9
    # The two blobs are far apart, so each half must land in one label.
    first_half = predicted[held_out < len(X) // 2]
    second_half = predicted[held_out >= len(X) // 2]
    assert len(set(first_half[first_half >= 0])) == 1
    assert len(set(second_half[second_half >= 0])) == 1
    assert set(first_half[first_half >= 0]) != set(second_half[second_half >= 0])


def test_predict_batches_do_not_change_the_answer():
    X = _two_blobs()
    model = hdbscan.HDBSCAN(min_cluster_size=15, prediction_data=True)
    model.fit(X)

    whole = ml_algorithms.predict_cluster_labels(model, X, batch_size=len(X))
    batched = ml_algorithms.predict_cluster_labels(model, X, batch_size=37)

    np.testing.assert_array_equal(whole, batched)


def test_predict_without_prediction_data_raises():
    X = _two_blobs(n_per_blob=40)
    model = hdbscan.HDBSCAN(min_cluster_size=5)
    model.fit(X)

    with pytest.raises(ValueError, match="prediction_data=True"):
        ml_algorithms.predict_cluster_labels(model, X)
