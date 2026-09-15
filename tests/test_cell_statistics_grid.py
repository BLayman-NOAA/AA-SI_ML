# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Pinning the cell grid so per-file statistics can be merged.

compute_per_cell_statistics builds its cell grid out to the deepest value in
the data it is handed. Run once on one merged dataset that is right. Mapped
over a survey's per-file Sv it is the same bug the survey tier fixes on
range_sample: each file bins to its own extent, and the fan-in outer-joins
cell_echo_range and NaN-pads the short ones. range_var_max pins the extent so
every file produces the same grid.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from aa_si_ml.ml import compute_per_cell_statistics


def _ds_sv(deepest_m, n_pings=40, n_range=20):
    """Sv whose echo_range reaches *deepest_m*, on a 1 s ping cadence."""
    ping_time = np.datetime64("2016-06-27T00:00:00") + np.arange(
        n_pings, dtype="timedelta64[s]"
    )
    echo_range = np.linspace(0.0, deepest_m, n_range)
    shape = (1, n_pings, n_range)
    return xr.Dataset(
        {
            "Sv": (("channel", "ping_time", "range_sample"),
                   np.full(shape, -70.0) + np.random.default_rng(0).normal(0, 1, shape)),
            "echo_range": (("channel", "ping_time", "range_sample"),
                           np.broadcast_to(echo_range, shape).copy()),
        },
        coords={
            "channel": ["ch0"],
            "ping_time": ping_time,
            "range_sample": np.arange(n_range),
        },
    )


def _cells(ds):
    return ds["cell_cv"].sizes["cell_echo_range"]


def test_unpinned_grids_differ_between_datasets():
    """The behaviour that makes a per-file fan-in unsafe."""
    shallow = compute_per_cell_statistics(_ds_sv(100.0), range_bin="2m")
    deep = compute_per_cell_statistics(_ds_sv(400.0), range_bin="2m")

    assert _cells(shallow) != _cells(deep)


def test_pinning_gives_every_dataset_the_same_grid():
    shallow = compute_per_cell_statistics(
        _ds_sv(100.0), range_bin="2m", range_var_max="400m"
    )
    deep = compute_per_cell_statistics(
        _ds_sv(400.0), range_bin="2m", range_var_max="400m"
    )

    assert _cells(shallow) == _cells(deep)
    # And the pin, not the data, sets the length: 0..400 at 2 m.
    assert _cells(shallow) == 200


def test_pinned_results_concatenate_without_padding():
    """The point of the pin: cell_echo_range lines up, so the merge is clean."""
    parts = [
        compute_per_cell_statistics(
            _ds_sv(depth), range_bin="2m", range_var_max="400m"
        )["cell_cv"]
        for depth in (100.0, 250.0, 400.0)
    ]

    merged = xr.concat(parts, dim="cell_ping_time", join="exact")

    assert merged.sizes["cell_echo_range"] == 200


def test_pinning_does_not_disturb_the_statistic_where_data_exists():
    ds = _ds_sv(100.0)
    unpinned = compute_per_cell_statistics(ds, range_bin="2m")
    pinned = compute_per_cell_statistics(ds, range_bin="2m", range_var_max="400m")

    overlap = unpinned["cell_cv"].sizes["cell_echo_range"]
    np.testing.assert_allclose(
        pinned["cell_cv"].isel(cell_echo_range=slice(0, overlap)).values,
        unpinned["cell_cv"].values,
        equal_nan=True,
    )


def test_a_unit_string_without_the_suffix_is_accepted():
    pinned = compute_per_cell_statistics(
        _ds_sv(100.0), range_bin="2m", range_var_max="400"
    )
    assert _cells(pinned) == 200
