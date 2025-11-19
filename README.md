# scikit-downscale

Statistical downscaling and postprocessing models for climate and weather model simulations.

[![CI](https://github.com/jhamman/scikit-downscale/workflows/CI/badge.svg)](https://github.com/jhamman/scikit-downscale/actions?query=workflow%3ACI+branch%3Amain+) [![Documentation Status](https://readthedocs.org/projects/scikit-downscale/badge/?version=latest)](https://scikit-downscale.readthedocs.io/en/latest/?badge=latest) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pangeo-data/scikit-downscale/HEAD)

[![](https://img.shields.io/pypi/v/scikit-downscale.svg)](https://pypi.org/pypi/name/)
![Conda (channel only)](https://img.shields.io/conda/vn/conda-forge/scikit-downscale)

## Development

Sync the development environment with `uv`:

```bash
uv sync --all-groups
```

Run unit tests with:

```bash
uv run pytest
```

## Build Documentation

```bash
# Install documentation dependencies
uv sync --group docs

# Build HTML documentation
uv run sphinx-build docs docs/_build/html

# View the built docs
open docs/_build/html/index.html
```
