"""
Color features extraction for Liu et al. 2012.

8 color channels: R, G, B, H, S, V, Cb, Cr
Each channel: 16-bin histogram
Total per stripe: 8 × 16 = 128 dimensions
"""

import numpy as np
import cv2
from typing import Tuple


def extract_color_features_stripe(
    stripe: np.ndarray,
    n_bins: int = 16
) -> np.ndarray:
    """
    Extract color features from a single stripe.
    
    Args:
        stripe: RGB image stripe (H, W, 3) in float32 [0, 1]
        n_bins: Number of bins per histogram (default 16)
    
    Returns:
        Color features (128,) = 8 channels × 16 bins
        Order: [R, G, B, H, S, V, Cb, Cr]
    """
    h, w = stripe.shape[:2]
    
    # Ensure stripe is in [0, 1] range
    if stripe.max() > 1.0:
        stripe = stripe.astype(np.float32) / 255.0
    
    # Extract RGB channels (already in [0, 1])
    r = stripe[:, :, 0].ravel()
    g = stripe[:, :, 1].ravel()
    b = stripe[:, :, 2].ravel()
    
    # Convert to HSV
    # OpenCV expects uint8 [0, 255] for conversion
    stripe_uint8 = (stripe * 255.0).astype(np.uint8)
    hsv = cv2.cvtColor(stripe_uint8, cv2.COLOR_RGB2HSV)
    hsv_float = hsv.astype(np.float32) / 255.0
    
    h_channel = hsv_float[:, :, 0].ravel()  # H in [0, 1]
    s_channel = hsv_float[:, :, 1].ravel()  # S in [0, 1]
    v_channel = hsv_float[:, :, 2].ravel()  # V in [0, 1]
    
    # Convert to YCbCr
    ycbcr = cv2.cvtColor(stripe_uint8, cv2.COLOR_RGB2YCrCb)
    ycbcr_float = ycbcr.astype(np.float32) / 255.0
    
    cb_channel = ycbcr_float[:, :, 1].ravel()  # Cb in [0, 1]
    cr_channel = ycbcr_float[:, :, 2].ravel()  # Cr in [0, 1]
    
    # Compute histograms for each channel
    channels = [
        r, g, b,
        h_channel, s_channel, v_channel,
        cb_channel, cr_channel
    ]
    
    histograms = []
    for channel in channels:
        hist, _ = np.histogram(
            channel,
            bins=n_bins,
            range=(0.0, 1.0)
        )
        # Normalize to sum = 1 (probability distribution)
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist = hist.astype(np.float32) / hist_sum
        else:
            hist = np.zeros(n_bins, dtype=np.float32)
        histograms.append(hist)
    
    # Concatenate all histograms
    features = np.concatenate(histograms)
    
    return features.astype(np.float32)


def extract_color_features(
    image: np.ndarray,
    n_stripes: int = 6,
    n_bins: int = 16
) -> np.ndarray:
    """
    Extract color features from image using horizontal stripes.
    
    Args:
        image: RGB image (H, W, 3) in uint8 [0, 255] or float32 [0, 1]
        n_stripes: Number of horizontal stripes (default 6)
        n_bins: Number of bins per histogram (default 16)
    
    Returns:
        Color features (n_stripes × 128,) = (768,) for n_stripes=6
    """
    # Validate and convert input
    if image.dtype != np.uint8 and image.dtype != np.float32:
        raise ValueError(f"Image must be uint8 or float32, got {image.dtype}")
    
    # Convert to float32 [0, 1] if needed
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    
    h, w = image.shape[:2]
    
    # Handle very small images
    if h < n_stripes:
        # Resize to minimum height
        min_height = max(128, n_stripes)
        aspect_ratio = w / h if h > 0 else 1.0
        new_w = int(min_height * aspect_ratio)
        image = cv2.resize(image, (new_w, min_height))
        h = min_height
    
    stripe_height = h // n_stripes
    
    features = []
    
    for i in range(n_stripes):
        start_h = i * stripe_height
        end_h = (i + 1) * stripe_height if i < n_stripes - 1 else h
        
        if end_h <= start_h:
            # Empty stripe, return zeros
            stripe_features = np.zeros(8 * n_bins, dtype=np.float32)
        else:
            stripe = image[start_h:end_h, :, :]
            stripe_features = extract_color_features_stripe(stripe, n_bins)
        
        features.append(stripe_features)
    
    return np.concatenate(features).astype(np.float32)

