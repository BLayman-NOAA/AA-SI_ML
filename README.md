<!-- markdownlint-disable MD033 MD041 -->

<div align="center">

# NOAA Fisheries AA-SI Python Package Template

**A modern Python package template for NOAA Fisheries Active Acoustics Strategic Initiative projects**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Getting Started](#getting-started) •
[Customization Checklist](#customization-checklist) •
[Development](#development) •
[Project Structure](#project-structure)

</div>

---

## Getting Started

### Use This Template

1. Click the **"Use this template"** button on GitHub
2. Name your new repository
3. Clone your new repository locally
4. Follow the [Customization Checklist](#customization-checklist) below

### Requirements

- Python 3.10 or higher
- pip

---

## Customization Checklist

After creating your repository from this template, complete the following steps:

### 1. Rename the Package

Replace all instances of `mypackagename` with your actual package name:

| Location | Action |
|----------|--------|
| `src/mypackagename/` | Rename the folder |
| `pyproject.toml` | Update `name`, URLs, and tool paths |
| `src/mypackagename/__init__.py` | Update imports and docstring |
| `tests/test_package.py` | Update import statements |

### 2. Update Project Metadata

Edit `pyproject.toml`:

- `name` - Your package name
- `version` - Start with `0.1.0`
- `description` - Brief description of your package
- `authors` / `maintainers` - Your information
- `keywords` - Relevant search terms

### 3. Update URLs

In `pyproject.toml`, update `[project.urls]` with your repository information.

### 4. Add Dependencies

Add your package dependencies to the `dependencies` list in `pyproject.toml`.

### 5. Update Documentation

| File | Action |
|------|--------|
| `README.md` | Replace with your project documentation |
| `CHANGELOG.md` | Update links to your repository |
| `NOTICE` | Update copyright information |
| `LICENSE` | Verify Apache 2.0 meets your needs |

### 6. Clean Up

- Delete this checklist section after completing setup
- Make your first commit

---

## Development

### Installation

```bash
# Clone the repository
git clone https://github.com/nmfs-ost/your-repo-name.git
cd your-repo-name

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
```

### Running Tests

```bash
pytest
pytest --cov=mypackagename
```

### Code Quality

```bash
black src/ tests/
pylint src/mypackagename
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
├── .gitignore
├── .pre-commit-config.yaml
├── .pylintrc
├── CHANGELOG.md
├── LICENSE
├── NOTICE
├── pyproject.toml
├── README.md
├── src/
│   └── mypackagename/
│       └── __init__.py
└── tests/
    ├── conftest.py
    └── test_package.py
```

---

## License

This template uses the Apache License 2.0. Verify this license meets your project requirements before use.

---

## Disclaimer

This repository is a scientific product and is not official communication of the National Oceanic and Atmospheric Administration, or the United States Department of Commerce. All NOAA GitHub project code is provided on an ‘as is’ basis and the user assumes responsibility for its use. Any claims against the Department of Commerce or Department of Commerce bureaus stemming from the use of this GitHub project will be governed by all applicable Federal law. Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by the Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to imply endorsement of any commercial product or activity by DOC or the United States Government.
