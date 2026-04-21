# Alitra

WIP Library for ALIgnment and TRAnsformation between fixed coordinate frames. The transform
is described by a translation and a homogeneous rotation.

Developed for transforming between the fixed local coordinate-frame and the asset-fixed
coordinate-frame.

## Installation

### Installation from pip

```
pip install alitra
```

```python
import alitra
help(alitra)
```

### Installation from source

```
git clone https://github.com/equinor/alitra
cd alitra
uv sync --extra dev
```

You can test whether installation was successful with pytest

```
uv run pytest .
```

## Dependencies

The dependencies used for this package are listed in `pyproject.toml` and pinned in `uv.lock`. This ensures our builds are predictable and deterministic. This project uses [uv](https://docs.astral.sh/uv/) for dependency management:

```
uv lock
```

To update the dependencies to the latest versions, run:

```
uv lock --upgrade
```

### Contributing

We welcome all kinds of contributions, including code, bug reports, issues, feature requests, and documentation. The
preferred way of submitting a contribution is to either make an [issue](https://github.com/equinor/isar/issues) on
GitHub or by forking the project on GitHub and making a pull requests.

### How to use

The tests in this repository can be used as examples
of how to use the different models and functions. The
[test_example.py](tests/test_example.py) is a good place to start.
