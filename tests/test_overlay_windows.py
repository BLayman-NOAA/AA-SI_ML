# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Tests for per-window line overlays on the clustering report."""

import numpy as np
import pandas as pd
import xarray as xr

from aa_si_ml import ml


def _write_evl(tmp_path, name, points, date="20240101"):
    lines = ["EVBD 3 15.1.65.0", str(len(points))]
    lines += [f"{date} {when}  {depth} 3" for when, depth in points]
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return path.as_posix()


def _dataset(n_pings=10):
    ping_time = pd.date_range("2024-01-01T00:00:00", periods=n_pings, freq="1s").values
    return xr.Dataset(
        {"grid": (("ping_time", "range_sample"), np.zeros((n_pings, 3)))},
        coords={
            "ping_time": ping_time,
            "range_sample": np.arange(3),
            "depth": ("range_sample", np.array([0.0, 2.0, 4.0])),
        },
    )


def _window(label, start_s, end_s, **lines):
    return {
        "label": label,
        "start": f"2024-01-01T00:00:{start_s:02d}",
        "end": f"2024-01-01T00:00:{end_s:02d}",
        **lines,
    }


def test_each_window_gets_its_own_variable_nan_outside_its_span(tmp_path):
    ds = _dataset()
    fit_a = _write_evl(tmp_path, "a.evl", [("0000000000", 40.0), ("0000020000", 60.0)])
    fit_b = _write_evl(tmp_path, "b.evl", [("0000060000", 100.0), ("0000080000", 120.0)])
    windows = [
        _window("A", 0, 2, dive_fit_evl=fit_a),
        _window("B", 6, 8, dive_fit_evl=fit_b),
    ]

    out, overlays = ml._attach_window_overlays(ds, windows, ["dive_fit_evl"])

    assert [o["var"] for o in overlays] == ["overlay_dive_fit_evl_A", "overlay_dive_fit_evl_B"]
    assert len({o["style"]["color"] for o in overlays}) == 1
    a = out["overlay_dive_fit_evl_A"].values
    b = out["overlay_dive_fit_evl_B"].values
    np.testing.assert_allclose(a[:3], [40.0, 50.0, 60.0])
    assert np.isnan(a[3:]).all()
    assert np.isnan(b[:6]).all()
    np.testing.assert_allclose(b[6:9], [100.0, 110.0, 120.0])
    assert np.isnan(b[9])


def test_overlapping_windows_each_keep_their_own_depth(tmp_path):
    ds = _dataset()
    fit_a = _write_evl(tmp_path, "a.evl", [("0000000000", 40.0), ("0000050000", 40.0)])
    fit_b = _write_evl(tmp_path, "b.evl", [("0000030000", 90.0), ("0000080000", 90.0)])
    windows = [
        _window("A", 0, 5, dive_fit_evl=fit_a),
        _window("B", 3, 8, dive_fit_evl=fit_b),
    ]

    out, _ = ml._attach_window_overlays(ds, windows, ["dive_fit_evl"])

    shared = slice(3, 6)
    np.testing.assert_allclose(out["overlay_dive_fit_evl_A"].values[shared], 40.0)
    np.testing.assert_allclose(out["overlay_dive_fit_evl_B"].values[shared], 90.0)


def test_missing_keys_and_empty_windows_are_skipped(tmp_path):
    ds = _dataset()
    fit = _write_evl(tmp_path, "a.evl", [("0000000000", 40.0), ("0000020000", 60.0)])
    windows = [
        _window("A", 0, 2, dive_fit_evl=fit),
        _window("late", 30, 40, dive_fit_evl=fit),
    ]

    out, overlays = ml._attach_window_overlays(
        ds, windows, ["dive_fit_evl", "dive_u99_evl"]
    )

    assert [o["var"] for o in overlays] == ["overlay_dive_fit_evl_A"]
    assert set(out.data_vars) == {"grid", "overlay_dive_fit_evl_A"}


def test_no_windows_is_a_no_op():
    ds = _dataset()
    out, overlays = ml._attach_window_overlays(ds, [], ["dive_fit_evl"])
    assert out is ds
    assert overlays == []


def test_report_passes_window_overlays_to_the_echogram(tmp_path, monkeypatch):
    ds = _dataset()
    fit = _write_evl(tmp_path, "a.evl", [("0000000000", 40.0), ("0000020000", 60.0)])
    seen = {}

    def fake_echogram(ds_plot, *args, **kwargs):
        seen["ds"] = ds_plot
        seen["overlay_lines"] = kwargs["overlay_lines"]

    monkeypatch.setattr(ml.echogram, "plot_cluster_echogram", fake_echogram)
    monkeypatch.setattr(ml, "plot_cluster_statistics", lambda *a, **k: None)
    monkeypatch.setattr(ml, "plot_dbscan_cluster_hierarchy", lambda *a, **k: None)

    clustering_result = {
        "labels": np.array([0, 1]),
        "sample_indices": np.array([0, 1]),
        "method": "hdbscan",
        "ml_result_name": "clusters",
    }
    ml.plot_clustering_report(
        ds,
        clustering_result,
        dataset_name="ml",
        ml_result_name="clusters",
        overlay_windows=[_window("A", 0, 2, dive_fit_evl=fit)],
    )

    assert [o["var"] for o in seen["overlay_lines"]] == ["overlay_dive_fit_evl_A"]
    assert "overlay_dive_fit_evl_A" in seen["ds"]
    assert "overlay_dive_fit_evl_A" not in ds


def test_bounds_are_dotted_and_a_different_color_from_the_fit(tmp_path):
    ds = _dataset()
    fit = _write_evl(tmp_path, "fit.evl", [("0000000000", 40.0), ("0000020000", 60.0)])
    upper = _write_evl(tmp_path, "u.evl", [("0000000000", 30.0), ("0000020000", 50.0)])
    lower = _write_evl(tmp_path, "l.evl", [("0000000000", 50.0), ("0000020000", 70.0)])
    windows = [_window("A", 0, 2, dive_fit_evl=fit, dive_u99_evl=upper, dive_l99_evl=lower)]

    _, overlays = ml._attach_window_overlays(
        ds, windows, ["dive_fit_evl", "dive_u99_evl", "dive_l99_evl"]
    )

    fit_style, upper_style, lower_style = (o["style"] for o in overlays)
    assert "linestyle" not in fit_style
    assert upper_style["linestyle"] == ":" and lower_style["linestyle"] == ":"
    assert upper_style["color"] == lower_style["color"] != fit_style["color"]
    assert upper_style is not lower_style
