"""
HOG (Histogram of Oriented Gradients) feature extraction for BILP.

Adds shape/gradient features to complement color and texture descriptors.
"""

import numpy as np
from typing import Tuple
from skimage.feature import hog
from skimage import color


def extract_hog_stripe(
    image_stripe: np.ndarray,
    orientations: int = 8,
    pixels_per_cell: Tuple[int, int] = (8, 8),
    cells_per_block: Tuple[int, int] = (2, 2),
    visualize: bool = False
) -> np.ndarray:
    """
    Extract HOG features from a single image stripe.

    Args:
        image_stripe: RGB image stripe (H, W, 3)
        orientations: Number of orientation bins
        pixels_per_cell: Size of cells (height, width)
        cells_per_block: Number of cells per block
        visualize: Whether to return visualization (for debugging)

    Returns:
        HOG feature vector
    """
    # Convert to grayscale for HOG
    if len(image_stripe.shape) == 3:
        gray_stripe = color.rgb2gray(image_stripe)
    else:
        gray_stripe = image_stripe

    # Extract HOG features
    hog_features = hog(
        gray_stripe,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        visualize=visualize,
        feature_vector=True,
        channel_axis=None
    )

    return hog_features.astype(np.float32)


def extract_hog_features(
    image: np.ndarray,
    n_stripes: int = 6,
    orientations: int = 8,
    pixels_per_cell: Tuple[int, int] = (8, 8),
    cells_per_block: Tuple[int, int] = (2, 2)
) -> np.ndarray:
    """
    Extract HOG features from image with horizontal stripe partitioning.

    Args:
        image: RGB image (H, W, 3)
        n_stripes: Number of horizontal stripes
        orientations: Number of orientation bins for HOG
        pixels_per_cell: Size of cells for HOG
        cells_per_block: Number of cells per block for HOG

    Returns:
        Concatenated HOG features for all stripes (n_stripes * hog_dim,)
    """
    height = image.shape[0]
    stripe_height = height // n_stripes

    hog_features_list = []

    for i in range(n_stripes):
        # Extract stripe
        y_start = i * stripe_height
        y_end = (i + 1) * stripe_height if i < n_stripes - 1 else height
        stripe = image[y_start:y_end, :, :]

        # Extract HOG from stripe
        stripe_hog = extract_hog_stripe(
            stripe,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block
        )

        hog_features_list.append(stripe_hog)

    # Concatenate all stripe features
    hog_features = np.concatenate(hog_features_list)

    return hog_features


def compute_hog_dimensionality(
    image_height: int,
    image_width: int,
    n_stripes: int,
    orientations: int = 8,
    pixels_per_cell: Tuple[int, int] = (8, 8),
    cells_per_block: Tuple[int, int] = (2, 2)
) -> int:
    """
    Compute expected HOG feature dimensionality.

    Args:
        image_height: Height of input image
        image_width: Width of input image
        n_stripes: Number of horizontal stripes
        orientations: Number of orientation bins
        pixels_per_cell: Size of cells
        cells_per_block: Number of cells per block

    Returns:
        Total HOG feature dimension
    """
    stripe_height = image_height // n_stripes

    # Compute cells per stripe
    cells_h = stripe_height // pixels_per_cell[0]
    cells_w = image_width // pixels_per_cell[1]

    # Compute blocks per stripe
    blocks_h = cells_h - cells_per_block[0] + 1
    blocks_w = cells_w - cells_per_block[1] + 1

    # Features per block
    features_per_block = orientations * cells_per_block[0] * cells_per_block[1]

    # Total features per stripe
    features_per_stripe = blocks_h * blocks_w * features_per_block

    # Total for all stripes
    total_features = features_per_stripe * n_stripes

    return total_features
