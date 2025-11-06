"""
Color features: Log-chromaticity for brightness invariance
"""

import numpy as np
from typing import Tuple, Optional


def rgb_to_log_chromaticity(image: np.ndarray, epsilon: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert RGB image to log-chromaticity space (u, v).

    u = log((R + ε) / (G + ε))
    v = log((B + ε) / (G + ε))

    Args:
        image: RGB image (H, W, 3) with values in [0, 255]
        epsilon: Small constant to avoid log(0)

    Returns:
        u, v: Log-chromaticity channels (H, W)
    """
    # Normalize to [0, 1] and add epsilon
    rgb = image.astype(np.float32) / 255.0 + epsilon

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    u = np.log(r / g)
    v = np.log(b / g)

    return u, v


def compute_luminance(image: np.ndarray) -> np.ndarray:
    """
    Compute luminance from RGB image.
    L = 0.299*R + 0.587*G + 0.114*B

    Args:
        image: RGB image (H, W, 3)

    Returns:
        Luminance channel (H, W)
    """
    weights = np.array([0.299, 0.587, 0.114])
    luminance = np.dot(image, weights)
    return luminance


def extract_color_histogram_stripe(
    u: np.ndarray,
    v: np.ndarray,
    luminance: np.ndarray,
    u_range: Tuple[float, float],
    v_range: Tuple[float, float],
    n_bins_uv: int = 16,
    n_bins_lum: int = 16
) -> np.ndarray:
    """
    Extract color histogram for a single stripe.

    Args:
        u, v: Log-chromaticity channels
        luminance: Luminance channel
        u_range: (min, max) range for u channel
        v_range: (min, max) range for v channel
        n_bins_uv: Number of bins for u and v (creates n_bins_uv × n_bins_uv 2D histogram)
        n_bins_lum: Number of bins for luminance

    Returns:
        Color histogram (n_bins_uv² + n_bins_lum,)
    """
    # 2D histogram for (u, v)
    hist_uv, _, _ = np.histogram2d(
        u.ravel(),
        v.ravel(),
        bins=[n_bins_uv, n_bins_uv],
        range=[u_range, v_range]
    )

    # 1D histogram for luminance
    hist_lum, _ = np.histogram(
        luminance.ravel(),
        bins=n_bins_lum,
        range=(0, 255)
    )

    # Concatenate and normalize
    hist = np.concatenate([hist_uv.ravel(), hist_lum])

    return hist


def extract_color_features(
    image: np.ndarray,
    n_stripes: int = 6,
    u_range: Optional[Tuple[float, float]] = None,
    v_range: Optional[Tuple[float, float]] = None,
    n_bins_uv: int = 16,
    n_bins_lum: int = 16
) -> np.ndarray:
    """
    Extract color features from image using horizontal stripes.

    Args:
        image: RGB image (H, W, 3)
        n_stripes: Number of horizontal stripes
        u_range: (min, max) range for u channel (calibrated from training data)
        v_range: (min, max) range for v channel (calibrated from training data)
        n_bins_uv: Number of bins for u and v
        n_bins_lum: Number of bins for luminance

    Returns:
        Color features (n_stripes × (n_bins_uv² + n_bins_lum),)
    """
    h, w = image.shape[:2]
    stripe_height = h // n_stripes

    # Convert to log-chromaticity
    u, v = rgb_to_log_chromaticity(image)
    luminance = compute_luminance(image)

    # If ranges not provided, compute from data (for calibration)
    if u_range is None:
        u_range = (np.percentile(u, 1), np.percentile(u, 99))
    if v_range is None:
        v_range = (np.percentile(v, 1), np.percentile(v, 99))

    # Extract features per stripe
    features = []
    for i in range(n_stripes):
        start_h = i * stripe_height
        end_h = (i + 1) * stripe_height if i < n_stripes - 1 else h

        u_stripe = u[start_h:end_h, :]
        v_stripe = v[start_h:end_h, :]
        lum_stripe = luminance[start_h:end_h, :]

        hist = extract_color_histogram_stripe(
            u_stripe, v_stripe, lum_stripe,
            u_range, v_range,
            n_bins_uv, n_bins_lum
        )

        features.append(hist)

    return np.concatenate(features)


def calibrate_color_ranges(
    images: list,
    percentile_low: float = 1,
    percentile_high: float = 99
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Calibrate u and v ranges from training images.

    Args:
        images: List of RGB images
        percentile_low: Lower percentile for range
        percentile_high: Upper percentile for range

    Returns:
        u_range, v_range: Calibrated ranges
    """
    all_u = []
    all_v = []

    for image in images:
        u, v = rgb_to_log_chromaticity(image)
        all_u.append(u.ravel())
        all_v.append(v.ravel())

    all_u = np.concatenate(all_u)
    all_v = np.concatenate(all_v)

    u_range = (np.percentile(all_u, percentile_low), np.percentile(all_u, percentile_high))
    v_range = (np.percentile(all_v, percentile_low), np.percentile(all_v, percentile_high))

    return u_range, v_range
