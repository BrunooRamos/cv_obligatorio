"""
HOG (Histogram of Oriented Gradients) features for person re-identification.
"""

import numpy as np
from typing import Optional, Tuple
from skimage.feature import hog


def extract_hog_stripe(
    image: np.ndarray,
    pixels_per_cell: Tuple[int, int] = (8, 8),
    cells_per_block: Tuple[int, int] = (2, 2),
    orientations: int = 9,
    block_norm: str = 'L2-Hys'
) -> np.ndarray:
    """
    Extract HOG features from a single stripe or image.

    Args:
        image: RGB image (H, W, 3) with values in [0, 255]
        pixels_per_cell: Size of a cell (height, width) in pixels
        cells_per_block: Number of cells in each block (height, width)
        orientations: Number of orientation bins
        block_norm: Block normalization method ('L1', 'L1-sqrt', 'L2', 'L2-Hys')

    Returns:
        HOG feature descriptor (1D array)
    """
    # Convert to grayscale using standard weights
    if len(image.shape) == 3:
        gray_image = np.dot(image[...,:3], [0.2989, 0.587, 0.114])
    else:
        gray_image = image

    # Ensure image is in valid range [0, 255]
    gray_image = np.clip(gray_image, 0, 255).astype(np.uint8)

    # Extract HOG features
    features = hog(
        gray_image,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm=block_norm,
        visualize=False,
        feature_vector=True,
        channel_axis=None
    )

    return features


def extract_hog_features(
    image: np.ndarray,
    n_stripes: int = 6,
    pixels_per_cell: Tuple[int, int] = (8, 8),
    cells_per_block: Tuple[int, int] = (2, 2),
    orientations: int = 9,
    block_norm: str = 'L2-Hys'
) -> np.ndarray:
    """
    Extract HOG features from image using horizontal stripes.

    Args:
        image: RGB image (H, W, 3)
        n_stripes: Number of horizontal stripes
        pixels_per_cell: Size of a cell in pixels
        cells_per_block: Number of cells in each block
        orientations: Number of orientation bins
        block_norm: Block normalization method

    Returns:
        HOG features (n_stripes × hog_dim,)
    """
    h, w = image.shape[:2]
    stripe_height = h // n_stripes

    features = []
    for i in range(n_stripes):
        start_h = i * stripe_height
        end_h = (i + 1) * stripe_height if i < n_stripes - 1 else h

        stripe = image[start_h:end_h, :]

        # Extract HOG for this stripe
        hog_features = extract_hog_stripe(
            stripe,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            orientations=orientations,
            block_norm=block_norm
        )

        features.append(hog_features)

    return np.concatenate(features)


def compute_hog_dimension(
    image_shape: Tuple[int, int],
    n_stripes: int,
    pixels_per_cell: Tuple[int, int] = (8, 8),
    cells_per_block: Tuple[int, int] = (2, 2),
    orientations: int = 9
) -> int:
    """
    Compute the expected dimension of HOG features.

    Args:
        image_shape: (height, width) of input image
        n_stripes: Number of horizontal stripes
        pixels_per_cell: Size of a cell in pixels
        cells_per_block: Number of cells in each block
        orientations: Number of orientation bins

    Returns:
        Total HOG feature dimension
    """
    h, w = image_shape
    stripe_height = h // n_stripes

    # Compute for a single stripe
    n_cells_h = stripe_height // pixels_per_cell[0]
    n_cells_w = w // pixels_per_cell[1]

    # Number of blocks in each direction
    n_blocks_h = n_cells_h - cells_per_block[0] + 1
    n_blocks_w = n_cells_w - cells_per_block[1] + 1

    # Features per stripe
    features_per_stripe = (n_blocks_h * n_blocks_w *
                          cells_per_block[0] * cells_per_block[1] *
                          orientations)

    return features_per_stripe * n_stripes
