# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Tests for reapplying a fitted normalization to new data."""

import pickle

import numpy as np
import pytest
import xarray as xr

from aa_si_ml import ml

DATASET = 'ml_data_clean'
SAMPLE_DIM = f'{DATASET}_sample_index'
FEATURE_DIM = f'feature_{DATASET}'
FEATURES = ['18kHz', '38kHz', 'depth']


def _ml_dataset(X, features=FEATURES, aux=('depth',), start_index=0):
    n_samples = len(X)
    data = xr.DataArray(
        X,
        dims=[SAMPLE_DIM, FEATURE_DIM],
        coords={
            SAMPLE_DIM: np.arange(start_index, start_index + n_samples),
            FEATURE_DIM: np.array(features, dtype=str),
        },
        attrs={'auxiliary_features': list(aux)},
    )
    ds = xr.Dataset({DATASET: data})
    ds[f'{DATASET}_sample_index_to_grid_index'] = xr.DataArray(
        np.arange(n_samples), dims=[SAMPLE_DIM]
    )
    return ds


def _features(n_samples, seed, loc=(-70.0, -65.0, 30.0), scale=(5.0, 4.0, 10.0)):
    rng = np.random.default_rng(seed)
    return rng.normal(loc, scale, size=(n_samples, 3))


FIT_KWARGS = dict(
    dataset_name=DATASET,
    normalization_name='norm',
    per_group_methods={'depth': 'minmax'},
    shift_positive=True,
    feature_weights=[1.0, 2.0, 0.5],
)


# ---------------------------------------------------------------------------
# Fit then transform reproduces the fit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('method', ['flatten', 'standard', 'minmax', 'robust', 'power', 'quantile'])
def test_transform_on_the_fitted_data_matches_the_fit_output(method):
    X = _features(200, seed=0)
    ds_fit, normalizer = ml.normalize_data(
        _ml_dataset(X), method=method, return_normalizer=True, **FIT_KWARGS
    )
    expected = ds_fit[f'{DATASET}_norm'].values

    np.testing.assert_allclose(normalizer.transform(X), expected, rtol=1e-9, atol=1e-12)

    ds_again = ml.normalize_data(_ml_dataset(X), normalizer=normalizer, **FIT_KWARGS)
    np.testing.assert_allclose(ds_again[f'{DATASET}_norm'].values, expected, rtol=1e-9, atol=1e-12)


def test_shift_and_weights_are_carried_by_the_normalizer():
    X = _features(200, seed=0)
    _, normalizer = ml.normalize_data(
        _ml_dataset(X), method='standard', return_normalizer=True, **FIT_KWARGS
    )

    assert normalizer.shift_amount > 0
    np.testing.assert_array_equal(normalizer.feature_weights, [1.0, 2.0, 0.5])
    assert normalizer.feature_labels == FEATURES
    assert normalizer.sv_columns == [0, 1]
    assert list(normalizer.group_scalers) == ['depth']


# ---------------------------------------------------------------------------
# Transform on new data uses no new statistics
# ---------------------------------------------------------------------------


def test_transform_on_new_data_uses_the_fitted_statistics():
    X_a = _features(200, seed=0)
    X_b = _features(150, seed=1, loc=(-60.0, -50.0, 80.0))
    _, normalizer = ml.normalize_data(
        _ml_dataset(X_a), method='standard', dataset_name=DATASET,
        normalization_name='norm', per_group_methods={'depth': 'minmax'},
        return_normalizer=True,
    )

    out_b = ml.normalize_data(
        _ml_dataset(X_b, start_index=1000), normalizer=normalizer,
        dataset_name=DATASET, normalization_name='norm',
    )[f'{DATASET}_norm'].values

    scaler = normalizer.sv_scaler
    expected_sv = (X_b[:, :2] - scaler.mean_) / scaler.scale_
    np.testing.assert_allclose(out_b[:, :2], expected_sv)
    # A refit would centre segment B on zero; the fitted scale does not.
    assert np.abs(out_b[:, :2].mean(axis=0)).min() > 0.5
    # minmax fitted on A: B's deeper values land above 1 instead of being
    # rescaled into [0, 1].
    assert out_b[:, 2].max() > 1.0


def test_transform_is_deterministic_and_leaves_the_normalizer_unchanged():
    X_a = _features(100, seed=0)
    X_b = _features(80, seed=2)
    _, normalizer = ml.normalize_data(
        _ml_dataset(X_a), method='flatten', dataset_name=DATASET,
        normalization_name='norm', per_group_methods={'depth': 'minmax'},
        return_normalizer=True,
    )
    before = pickle.dumps(normalizer)

    first = normalizer.transform(X_b)
    second = normalizer.transform(X_b)

    np.testing.assert_array_equal(first, second)
    assert pickle.dumps(normalizer) == before


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_feature_label_mismatch_names_both_lists():
    X = _features(50, seed=0)
    _, normalizer = ml.normalize_data(
        _ml_dataset(X), method='standard', dataset_name=DATASET,
        normalization_name='norm', return_normalizer=True,
    )
    other = _ml_dataset(X, features=['18kHz', '38kHz', 'altitude'], aux=('altitude',))

    with pytest.raises(ValueError, match=r"\['18kHz', '38kHz', 'depth'\].*\['18kHz', '38kHz', 'altitude'\]"):
        ml.normalize_data(other, normalizer=normalizer, dataset_name=DATASET)


def test_wrong_column_count_is_refused_by_transform():
    X = _features(50, seed=0)
    _, normalizer = ml.normalize_data(
        _ml_dataset(X), method='standard', dataset_name=DATASET,
        normalization_name='norm', return_normalizer=True,
    )

    with pytest.raises(ValueError, match="Expected 3 feature columns"):
        normalizer.transform(X[:, :2])


def test_l2_and_pooled_fits_cannot_be_captured():
    X = _features(50, seed=0)

    with pytest.raises(ValueError, match="cannot be reapplied"):
        ml.normalize_data(
            _ml_dataset(X), method='l2', dataset_name=DATASET,
            normalization_name='norm', return_normalizer=True,
        )
    with pytest.raises(ValueError, match="Pooled"):
        ml.normalize_data(
            _ml_dataset(X), method='standard', per_feature=False,
            dataset_name=DATASET, normalization_name='norm', return_normalizer=True,
        )


def test_umap_methods_are_refused_at_construction():
    for method in ('umap', 'flatten_plus_umap'):
        with pytest.raises(ValueError, match="cannot be reapplied"):
            ml.FittedNormalizer(
                method=method, flatten_weight=1, feature_labels=FEATURES,
                sv_columns=[0, 1, 2], sv_scaler=object(), group_scalers={},
                shift_amount=0.0, feature_weights=None,
            )


def test_existing_callers_still_get_a_dataset_for_l2():
    X = _features(50, seed=0)

    out = ml.normalize_data(
        _ml_dataset(X), method='l2', dataset_name=DATASET, normalization_name='norm'
    )

    assert isinstance(out, xr.Dataset)
    assert f'{DATASET}_norm' in out


# ---------------------------------------------------------------------------
# Pickling and the recipe step
# ---------------------------------------------------------------------------


def test_normalizer_survives_a_pickle_round_trip():
    X = _features(100, seed=0)
    _, normalizer = ml.normalize_data(
        _ml_dataset(X), method='flatten', return_normalizer=True, **FIT_KWARGS
    )

    restored = pickle.loads(pickle.dumps(normalizer))

    np.testing.assert_array_equal(restored.transform(X), normalizer.transform(X))
    assert restored.feature_labels == normalizer.feature_labels


def test_step_fits_then_reuses_the_normalizer():
    X_a = _features(100, seed=0)
    X_b = _features(60, seed=3)

    fit = ml.normalize_ml_data_step(
        _ml_dataset(X_a), method='flatten', dataset_name=DATASET,
        normalization_name='norm', per_group_methods={'depth': 'minmax'},
    )
    reuse = ml.normalize_ml_data_step(
        _ml_dataset(X_b), normalizer=fit['normalizer'], method='flatten',
        dataset_name=DATASET, normalization_name='norm',
        per_group_methods={'depth': 'minmax'},
    )

    assert set(fit) == {'ds_normalized', 'normalizer'}
    assert isinstance(fit['normalizer'], ml.FittedNormalizer)
    assert isinstance(fit['ds_normalized'], xr.Dataset)
    assert reuse['normalizer'] is fit['normalizer']
    np.testing.assert_array_equal(
        reuse['ds_normalized'][f'{DATASET}_norm'].values,
        fit['normalizer'].transform(X_b),
    )
