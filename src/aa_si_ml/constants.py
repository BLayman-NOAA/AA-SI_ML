# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries

"""Shared constants for the aa_si_ml package."""

DEFAULT_CLUSTER_COLORS = [
    "#5A00CF", "#35E200", "#FF8800", "#F943FF", "#F30101",
    "#EDFF4D", "#4E9200", "#970021", "#5600C7", "#017685FF", "#0400FFFF"
]

# Bar outline colors used to tell features apart in the cluster statistics
# plot, where the bar fill is already taken by the cluster color.
FEATURE_EDGE_COLORS = [
    "#000000", "#0072B2", "#009E73", "#D55E00", "#CC79A7",
    "#56B4E9", "#8C564B", "#7F3FBF", "#BF9000", "#4D4D4D"
]

# Typical valid Sv range in dB re 1 m-1. Values outside this window are
# treated as artifacts by add_valid_data_mask.
SV_MIN_VALID = -200
SV_MAX_VALID = 50
