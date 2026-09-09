from unittest import mock

import numpy as np
import pytest

from alitra import align_two_images_orb_bf_cv2, align_two_images_translation_cv2


def _make_textured_image(size: int = 200) -> np.ndarray:
    """Create a reproducible, texture-rich uint8 grayscale image for feature matching."""
    rng = np.random.default_rng(seed=42)
    image = rng.integers(0, 255, size=(size, size), dtype=np.uint8)
    return image


def _shift_image(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Shift an image by (dx, dy) pixels, padding with zeros. Works for grayscale and color."""
    shifted = np.zeros_like(image)
    height, width = image.shape[:2]
    src_x_start = max(0, -dx)
    src_y_start = max(0, -dy)
    dst_x_start = max(0, dx)
    dst_y_start = max(0, dy)
    copy_width = width - abs(dx)
    copy_height = height - abs(dy)
    shifted[
        dst_y_start : dst_y_start + copy_height,
        dst_x_start : dst_x_start + copy_width,
        ...,
    ] = image[
        src_y_start : src_y_start + copy_height,
        src_x_start : src_x_start + copy_width,
        ...,
    ]
    return shifted


def _make_textured_color_image(size: int = 200) -> np.ndarray:
    """Create a reproducible, texture-rich uint8 BGR image for feature matching."""
    rng = np.random.default_rng(seed=7)
    image = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    return image


@pytest.fixture()
def reference_image() -> np.ndarray:
    return _make_textured_image()


@pytest.fixture()
def roi_polygon() -> list[tuple[int, int]]:
    return [(50, 50), (100, 50), (100, 100), (50, 100)]


def test_align_two_images_translation_cv2_estimates_shift(reference_image, roi_polygon):
    dx, dy = 5, -3
    source_image = _shift_image(reference_image, dx, dy)

    translated_polygon, translated_reference_image, alignment_score = (
        align_two_images_translation_cv2(reference_image, source_image, roi_polygon)
    )

    assert translated_reference_image.shape == source_image.shape
    assert translated_reference_image.dtype == np.uint8
    assert len(translated_polygon) == len(roi_polygon)
    for (x, y), (expected_x, expected_y) in zip(
        translated_polygon, [(px + dx, py + dy) for px, py in roi_polygon]
    ):
        assert x == pytest.approx(expected_x, abs=1)
        assert y == pytest.approx(expected_y, abs=1)
    assert alignment_score > 0


def test_align_two_images_translation_cv2_estimates_shift_for_color_images(
    roi_polygon,
):
    reference_image = _make_textured_color_image()
    dx, dy = 5, -3
    source_image = _shift_image(reference_image, dx, dy)

    translated_polygon, translated_reference_image, alignment_score = (
        align_two_images_translation_cv2(reference_image, source_image, roi_polygon)
    )

    assert translated_reference_image.shape == source_image.shape
    assert translated_reference_image.dtype == np.uint8
    for (x, y), (expected_x, expected_y) in zip(
        translated_polygon, [(px + dx, py + dy) for px, py in roi_polygon]
    ):
        assert x == pytest.approx(expected_x, abs=1)
        assert y == pytest.approx(expected_y, abs=1)
    assert alignment_score > 0


def test_align_two_images_orb_bf_cv2_estimates_homography(reference_image, roi_polygon):
    dx, dy = 5, -3
    source_image = _shift_image(reference_image, dx, dy)

    warped_polygon, aligned_reference_image = align_two_images_orb_bf_cv2(
        reference_image, source_image, roi_polygon
    )

    assert aligned_reference_image.shape == source_image.shape
    assert aligned_reference_image.dtype == np.uint8
    assert len(warped_polygon) == len(roi_polygon)
    for (x, y), (expected_x, expected_y) in zip(
        warped_polygon, [(px + dx, py + dy) for px, py in roi_polygon]
    ):
        assert x == pytest.approx(expected_x, abs=2)
        assert y == pytest.approx(expected_y, abs=2)


def test_align_two_images_orb_bf_cv2_falls_back_on_featureless_images(roi_polygon):
    reference_image = np.zeros((100, 100), dtype=np.uint8)
    source_image = np.zeros((100, 100), dtype=np.uint8)

    warped_polygon, aligned_reference_image = align_two_images_orb_bf_cv2(
        reference_image, source_image, roi_polygon
    )

    assert warped_polygon == roi_polygon
    assert aligned_reference_image is reference_image


def test_align_two_images_translation_cv2_returns_none_when_polygon_leaves_frame(
    reference_image, roi_polygon
):
    source_image = reference_image.copy()

    # Force a shift estimate large enough to push the polygon out of frame,
    # independent of what real phase correlation would estimate here.
    with mock.patch(
        "alitra.image_alignment.cv2.phaseCorrelate",
        return_value=((1000.0, 1000.0), 0.9),
    ):
        translated_polygon, _translated_reference_image, _alignment_score = (
            align_two_images_translation_cv2(reference_image, source_image, roi_polygon)
        )

    assert translated_polygon is None


def test_align_two_images_orb_bf_cv2_returns_none_when_polygon_leaves_frame(
    reference_image, roi_polygon
):
    dx, dy = 5, -3
    source_image = _shift_image(reference_image, dx, dy)
    out_of_bounds = np.full((len(roi_polygon), 1, 2), 1000.0, dtype=np.float32)

    # Force the warped polygon out of frame, independent of the real homography.
    with mock.patch(
        "alitra.image_alignment.cv2.perspectiveTransform", return_value=out_of_bounds
    ):
        warped_polygon, _aligned_reference_image = align_two_images_orb_bf_cv2(
            reference_image, source_image, roi_polygon
        )

    assert warped_polygon is None
