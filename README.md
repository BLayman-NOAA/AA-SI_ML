<!-- markdownlint-disable MD033 MD041 -->

<div align="center">

# AA-SI ML

**Machine learning tools for NOAA Fisheries Active Acoustics Strategic Initiative sonar data analysis**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Features](#features) •
[Getting Started](#getting-started) •
[Usage](#usage) •
[Development](#development)

</div>

---

## Features

- **Data Preprocessing** — Validity masking, NaN/artifact filtering, and grid index tracking for acoustic backscatter (Sv) data
- **Flexible Normalization** — Standard, robust, min-max, power transform, quantile, and L2 normalization methods
- **Clustering** — KMeans, DBSCAN, and HDBSCAN clustering with parameter sweeps and silhouette scoring
- **ML Pipeline** — End-to-end pipeline from raw xarray Datasets to clustered results regridded back to original coordinates
- **Index Tracking** — Persistent grid indices that track data points through flattening, filtering, ML, and regridding

## Getting Started

### Requirements

- Python 3.10 or higher
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/nmfs-ost/AA-SI_ML.git
cd AA-SI_ML

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

---

## Usage

```python
import aa_si_ml

# Prepare data for ML (masking, indexing, flattening)
ds_ml_ready = aa_si_ml.reshape_data_for_ml(ds_Sv)

# Normalize
ds_normalized = aa_si_ml.normalize_data(ds_ml_ready, method='standard')

# Extract for sklearn
X, grid_indices, sample_indices = aa_si_ml.extract_valid_samples_for_sklearn(
    ds_normalized, specific_data_name='standard_normalized'
)

# Cluster and regrid results back to original coordinates
ds_final = aa_si_ml.store_ml_results_flattened(ds_normalized, cluster_labels, 'clusters')
gridded = aa_si_ml.extract_ml_data_gridded(ds_final, 'clusters', fill_value=-1)
```

---

## Development

### Running Tests

```bash
pytest
pytest --cov=aa_si_ml
```

### Code Quality

```bash
black src/ tests/
pylint src/aa_si_ml
pre-commit run --all-files
```

### Building

```bash
pip install build
python -m build
```

---

## Project Structure

```
├── pyproject.toml
├── README.md
├── src/
│   └── aa_si_ml/
│       ├── __init__.py
│       ├── ml.py
│       └── ml_test.py
└── tests/
    ├── conftest.py
    └── test_package.py
```

---

## License

This project uses the Apache License 2.0. See [LICENSE](LICENSE) for details.

---

## Disclaimer

This repository is a scientific product and is not official communication of the National Oceanic and Atmospheric Administration, or the United States Department of Commerce. All NOAA GitHub project code is provided on an ‘as is’ basis and the user assumes responsibility for its use. Any claims against the Department of Commerce or Department of Commerce bureaus stemming from the use of this GitHub project will be governed by all applicable Federal law. Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by the Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to imply endorsement of any commercial product or activity by DOC or the United States Government.
