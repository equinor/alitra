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
pip install .
```

You can test whether installation was successfull with pytest

```
pip install pytest
pytest .
```

### Local development

```
pip install -e /path/to/package
```

This will install package in _editable_ mode. Convenient for local development

## Components

### Frame transform

Class for transforming coordinates between two coordinates frames. Use custom
dataclasses for conveniency, and to ensure that no mistakes are made in the transform.

### Align frames

Finds the rotations and translations between two coordinate systems by minimizing the
matching error given a set of points described in both coordinate frames.
Run `python examples/example_manual_alignment.py` for a demonstration of its use.
