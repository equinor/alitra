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
pip install .[dev]
```

You can test whether installation was successfull with pytest

```
pytest .
```

### Local development

```
pip install -e /path/to/package
```

This will install package in _editable_ mode. Convenient for local development

### How to use

The tests in this repository can be used as examples
of how to use the different models and functions. The
[test_example.py](tests/test_example.py) is a good place to start.
