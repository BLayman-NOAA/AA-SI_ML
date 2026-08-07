# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries

"""Channel frequency lookups shared by cluster statistics and plotting."""

import re

import numpy as np

_KHZ_IN_NAME = re.compile(r"(\d+(?:\.\d+)?)\s*kHz", re.IGNORECASE)
_TRANSDUCER_IN_NAME = re.compile(r"\bES(\d+(?:\.\d+)?)", re.IGNORECASE)


def channel_frequencies_khz(ds, channel_values):
    """Look up the nominal frequency of each channel in kHz.

    Uses the ``frequency_nominal`` variable when the dataset carries one and
    falls back to parsing the frequency out of the channel name.

    Args:
        ds (xr.Dataset or None): Dataset that may hold ``frequency_nominal``.
        channel_values (sequence): Channel coordinate values.

    Returns:
        list: Frequency in kHz per channel, with ``None`` for any channel
        whose frequency could not be determined.
    """
    channel_values = list(channel_values)
    frequencies = _frequencies_from_dataset(ds, channel_values)
    if frequencies is not None:
        return frequencies
    return [_frequency_from_name(value) for value in channel_values]


def frequency_sort_order(frequencies_khz):
    """Order channel indices by increasing frequency.

    Args:
        frequencies_khz (sequence): Frequency per channel, ``None`` where
            unknown.

    Returns:
        list or None: Channel indices sorted by frequency, or ``None`` when
        any frequency is unknown.
    """
    frequencies_khz = list(frequencies_khz)
    if any(freq is None for freq in frequencies_khz):
        return None
    return sorted(range(len(frequencies_khz)), key=lambda index: frequencies_khz[index])


def second_lowest_frequency_index(frequencies_khz):
    """Find the channel with the second-lowest frequency.

    Args:
        frequencies_khz (sequence): Frequency per channel, ``None`` where
            unknown.

    Returns:
        int or None: Channel index, or ``None`` when any frequency is unknown
        or there are fewer than two channels.
    """
    order = frequency_sort_order(frequencies_khz)
    if order is None or len(order) < 2:
        return None
    return order[1]


def frequency_label(frequency_khz):
    """Format a frequency in kHz for display.

    Args:
        frequency_khz (float or None): Frequency in kHz.

    Returns:
        str or None: Label such as ``'38 kHz'``, or ``None`` when the
        frequency is unknown.
    """
    if frequency_khz is None:
        return None
    return f"{frequency_khz:g} kHz"


def _frequencies_from_dataset(ds, channel_values):
    if ds is None or "frequency_nominal" not in ds:
        return None

    frequency_nominal = ds["frequency_nominal"]
    if "channel" in frequency_nominal.dims:
        try:
            frequency_nominal = frequency_nominal.sel(channel=channel_values)
        except (KeyError, ValueError):
            pass

    values = np.asarray(frequency_nominal.values, dtype=float).ravel()
    if values.size != len(channel_values) or np.isnan(values).any():
        return None
    return [float(value) / 1000.0 for value in values]


def _frequency_from_name(channel_value):
    name = str(channel_value)
    for pattern in (_KHZ_IN_NAME, _TRANSDUCER_IN_NAME):
        match = pattern.search(name)
        if match:
            return float(match.group(1))
    return None
