# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""
mypackagename - A Python package for NOAA Fisheries AA-SI.

This is the main package module. Import your public API here.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mypackagename")
except PackageNotFoundError:
    # Package is not installed (e.g., running from source without install)
    __version__ = "0.0.0.dev"

__all__ = ["__version__"]
