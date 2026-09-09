"""
Alignment of two images (e.g. two photos of the same scene taken at different
times) using either a translation-only estimate (phase correlation) or a full
homography estimate (ORB feature matching). Both approaches also transform a
region-of-interest polygon defined in the reference image into the source
image's coordinates.
"""

import logging
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def _polygon_overlaps_image(
    polygon: list[tuple[int, int]], width: int, height: int
) -> bool:
    """True if the polygon's bounding box overlaps the image bounds at all."""
    xs = [x for x, _ in polygon]
    ys = [y for _, y in polygon]
    return max(xs) >= 0 and min(xs) < width and max(ys) >= 0 and min(ys) < height


def _clip_polygon_to_image(
    polygon: list[tuple[int, int]], width: int, height: int
) -> list[tuple[int, int]]:
    """Clamp polygon points into valid pixel indices, avoiding negative-index wraparound."""
    return [(min(max(x, 0), width - 1), min(max(y, 0), height - 1)) for x, y in polygon]


def _finalize_polygon(
    polygon: list[tuple[int, int]], width: int, height: int
) -> list[tuple[int, int]] | None:
    """Clamp polygon into image bounds, or None if it has no overlap with the image at all."""
    if not _polygon_overlaps_image(polygon, width, height):
        return None
    return _clip_polygon_to_image(polygon, width, height)


def _to_grayscale(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert a color (BGR) image to grayscale; pass single-channel images through unchanged."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def align_two_images_translation_cv2(
    reference_image: NDArray[np.uint8],
    source_image: NDArray[np.uint8],
    roi_polygon: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]] | None, NDArray[np.uint8], float]:
    """
    Align reference image to source image using translation only, and
    transform the polygon accordingly.

    Uses phase correlation to estimate the (dx, dy) shift between images.
    This is appropriate when the camera moves only slightly between captures
    and no rotation or perspective change is expected.

    :param reference_image: The reference image (source of polygon)
    :param source_image: The source image (target alignment)
    :param roi_polygon: Polygon points defined in reference image coordinates
    :return: tuple (translated_polygon, translated_reference_image, alignment_score)
        translated_polygon: Polygon shifted to source image coordinates, clamped to
            valid pixel indices, or None if it falls entirely outside the source image
        translated_reference_image: Reference image shifted to match source image
        alignment_score: Phase correlation response (confidence of the match)
    """
    # Phase correlation only supports single-channel images, so color (e.g. JPEG)
    # images are converted to grayscale for the shift estimation. The warp is
    # still applied to the original reference image further down.
    gray_reference = _to_grayscale(reference_image)
    gray_source = _to_grayscale(source_image)

    # Phase correlation requires float32 images
    reference_float = gray_reference.astype(np.float32)
    source_float = gray_source.astype(np.float32)

    # The Hann window reduces edge effects by weighting the center of the image the most.
    hann = cv2.createHanningWindow(gray_reference.shape, cv2.CV_32F).T
    # Estimate translation: phaseCorrelate returns (dx, dy) such that
    # the source image is shifted by (dx, dy) relative to the reference.
    (dx, dy), phase_correlation = cv2.phaseCorrelate(
        reference_float, source_float, hann
    )

    logger.info(
        f"Estimated translation: dx={dx:.2f}, dy={dy:.2f}, response={phase_correlation:.3f}"
    )

    # Apply the translation to the reference image to align it with the source
    translation_matrix = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    translated_reference_image: NDArray[np.uint8] = cv2.warpAffine(
        reference_image,
        translation_matrix,
        (source_image.shape[1], source_image.shape[0]),
    ).astype(np.uint8)

    # Shift the polygon by the same (dx, dy) to move it into source image coordinates
    translated_polygon: list[tuple[int, int]] = [
        (int(round(x + dx)), int(round(y + dy))) for x, y in roi_polygon
    ]

    translated_polygon = _finalize_polygon(
        translated_polygon, source_image.shape[1], source_image.shape[0]
    )
    if translated_polygon is None:
        logger.warning("Translated polygon falls entirely outside the source image")

    return translated_polygon, translated_reference_image, phase_correlation


def align_two_images_orb_bf_cv2(
    reference_image: NDArray[np.uint8],
    source_image: NDArray[np.uint8],
    roi_polygon: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]] | None, NDArray[np.uint8]]:
    """
    Align reference image to source image and transform the polygon accordingly.

    Uses ORB keypoint detection and brute-force matching to estimate a
    homography between the two images. This is appropriate when the camera
    may have moved, rotated or changed perspective between captures.

    :param reference_image: The reference image (source of polygon)
    :param source_image: The source image (target alignment)
    :param roi_polygon: Polygon points defined in reference image coordinates
    :return: tuple (warped_polygon, aligned_reference_image)
        warped_polygon: Polygon transformed to source image coordinates, clamped to
            valid pixel indices, or None if it falls entirely outside the source image
        aligned_reference_image: Reference image warped to match source image geometry
    """
    # Define the polygon points from reference image
    polygon_points = np.array(roi_polygon, dtype=np.float32)

    # Convert images to grayscale if they are not already
    gray_reference = _to_grayscale(reference_image)
    gray_source = _to_grayscale(source_image)

    # Detect ORB keypoints and compute descriptors
    orb = cv2.ORB_create(nfeatures=1000)  # type: ignore
    keypoints_ref, descriptors_ref = orb.detectAndCompute(gray_reference, None)
    keypoints_src, descriptors_src = orb.detectAndCompute(gray_source, None)

    if descriptors_ref is None or descriptors_src is None:
        logger.error("Could not find features in one or both images")
        # Return original polygon and reference image (fallback)
        return roi_polygon, reference_image

    # Create a BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(descriptors_ref, descriptors_src)

    if len(matches) < 4:
        logger.error(f"Insufficient matches found: {len(matches)}. Need at least 4.")
        return roi_polygon, reference_image

    # Sort them in order of distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    # Take only the best matches (top 50% or max 100)
    num_good_matches = min(len(matches), max(len(matches) // 2, 100))
    good_matches = matches[:num_good_matches]

    logger.info(
        f"Using {num_good_matches} good matches out of {len(matches)} total matches"
    )

    # Extract location of good matches
    points_ref = np.zeros((len(good_matches), 2), dtype=np.float32)
    points_src = np.zeros((len(good_matches), 2), dtype=np.float32)

    for i, match in enumerate(good_matches):
        points_ref[i, :] = keypoints_ref[match.queryIdx].pt
        points_src[i, :] = keypoints_src[match.trainIdx].pt

    # Calculate homography to transform REFERENCE image TO SOURCE image coordinates
    # Note: points_ref -> points_src
    H, mask = cv2.findHomography(
        points_ref, points_src, cv2.RANSAC, ransacReprojThreshold=3.0, confidence=0.99
    )

    # Check if homography is valid
    if H is None:
        logger.error("Could not compute homography")
        return roi_polygon, reference_image

    num_inliers = np.sum(mask) if mask is not None else 0
    logger.info(
        f"Homography computed with {num_inliers} inliers out of {len(good_matches)} matches"
    )

    if num_inliers < 10:
        logger.warning(f"Poor homography quality - only {num_inliers} inliers")

    # Check if homography is reasonable (not too distorted)
    try:
        det = np.linalg.det(cast(NDArray[np.float64], H[:2, :2]))
        if abs(det) < 0.1 or abs(det) > 10:
            logger.warning(f"Homography appears distorted (determinant: {det:.3f})")
    except np.linalg.LinAlgError:
        logger.warning("Could not validate homography determinant")

    # Warp reference image to align with source image
    aligned_reference_image = cv2.warpPerspective(
        reference_image, H, (source_image.shape[1], source_image.shape[0])
    )

    # Warp the polygon points from Reference to Source
    warped_polygon_array = cv2.perspectiveTransform(polygon_points.reshape(-1, 1, 2), H)

    # Convert warped polygon array to list of tuples
    warped_polygon_list = [
        (int(pt[0]), int(pt[1])) for pt in warped_polygon_array.reshape(-1, 2)
    ]

    warped_polygon_list = _finalize_polygon(
        warped_polygon_list, source_image.shape[1], source_image.shape[0]
    )
    if warped_polygon_list is None:
        logger.warning("Warped polygon falls entirely outside the source image")

    return warped_polygon_list, cast(NDArray[np.uint8], aligned_reference_image)
