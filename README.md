# Alitra

Library for ALIgnment and TRAnsformation between fixed coordinate frames. The transform
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

Image alignment (see [Image alignment](#image-alignment) below) requires the optional
`image-alignment` extra:

```
pip install alitra[image-alignment]
```

### Installation from source

```
git clone https://github.com/equinor/alitra
cd alitra
uv sync --extra dev
```

You can test whether installation was successful with pytest

```
uv run pytest
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

## Image alignment

When two photos of the same scene are taken at different times, small differences in camera position mean a
region-of-interest (ROI) polygon drawn on one photo no longer lines up with the same spot
in the other. Alitra provides two functions to re-align a reference photo (and its ROI
polygon) to a new source photo, so the polygon can be reused without redrawing it:

- `align_two_images_translation_cv2`: estimates a simple (dx, dy) pixel shift using phase
  correlation. Fast and robust, but only correct when the camera hasn't rotated or changed
  perspective between the two photos (e.g. a fixed camera with minor positional drift).
- `align_two_images_orb_bf_cv2`: detects ORB keypoints and estimates a full homography
  (rotation, scale and perspective change included). Handles a moved/rotated camera, but
  needs enough distinct visual features to match between the two photos.

Both functions return the ROI polygon transformed into the new photo's coordinates. If the
transformed polygon has no overlap with the new photo at all, `None` is returned instead.
If it partially overlaps, the returned coordinates are clamped to valid pixel indices.

### Contributing

We welcome all kinds of contributions, including code, bug reports, issues, feature requests, and documentation. The
preferred way of submitting a contribution is to either make an [issue](https://github.com/equinor/alitra/issues) on
GitHub or by forking the project on GitHub and making a pull request.

### How to use

The tests in this repository can be used as examples
of how to use the different models and functions. The
[test_example.py](tests/test_example.py) is a good place to start.
