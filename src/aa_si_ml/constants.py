# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries

"""Shared constants for the aa_si_ml package."""

DEFAULT_CLUSTER_COLORS = [
    "#5A00CF", "#35E200", "#FF8800", "#F943FF", "#F30101",
    "#EDFF4D", "#4E9200", "#970021", "#5600C7", "#017685FF", "#0400FFFF"
]

# Typical valid Sv range in dB re 1 m-1. Values outside this window are
# treated as artifacts by add_valid_data_mask.
SV_MIN_VALID = -200
SV_MAX_VALID = 50
