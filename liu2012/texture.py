"""
Texture features extraction for Liu et al. 2012.

21 texture filters: 8 Gabor + 13 Schmid
Applied to luminance channel
Each filter response: 16-bin histogram
Total per stripe: 21 × 16 = 336 dimensions
"""

import numpy as np
from scipy import ndimage
from typing import List, Tuple, Optional
import cv2

# Import GPU utilities from bilp
try:
    from bilp.gpu_utils import convolve_gpu, get_array_module, to_gpu, to_cpu
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    convolve_gpu = None
    get_array_module = None
    to_gpu = None
    to_cpu = None


def compute_luminance(image: np.ndarray) -> np.ndarray:
    """
    Compute luminance channel from RGB image.
    
    Uses Y from YCbCr conversion (standard ITU-R BT.601).
    
    Args:
        image: RGB image (H, W, 3) in float32 [0, 1]
    
    Returns:
        Luminance channel (H, W) in float32 [0, 1]
    """
    if len(image.shape) == 3:
        # Convert RGB to YCbCr
        image_uint8 = (image * 255.0).astype(np.uint8)
        ycbcr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2YCrCb)
        y_channel = ycbcr[:, :, 0].astype(np.float32) / 255.0
        return y_channel
    else:
        # Already grayscale
        return image.astype(np.float32)


def create_gabor_filter(
    ksize: int,
    sigma: float,
    theta: float,
    lambd: float,
    gamma: float = 0.5,
    psi: float = 0
) -> np.ndarray:
    """
    Create a Gabor filter kernel.
    
    Args:
        ksize: Kernel size (should be odd)
        sigma: Standard deviation of Gaussian envelope
        theta: Orientation (radians)
        lambd: Wavelength
        gamma: Spatial aspect ratio
        psi: Phase offset
    
    Returns:
        Gabor kernel (ksize, ksize)
    """
    sigma_x = sigma
    sigma_y = sigma / gamma
    
    # Create coordinate grid
    x = np.arange(-ksize // 2, ksize // 2 + 1, dtype=np.float32)
    y = np.arange(-ksize // 2, ksize // 2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    
    # Rotate coordinates
    x_theta = xx * np.cos(theta) + yy * np.sin(theta)
    y_theta = -xx * np.sin(theta) + yy * np.cos(theta)
    
    # Gabor function
    gaussian = np.exp(-0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2))
    sinusoid = np.cos(2 * np.pi * x_theta / lambd + psi)
    
    gabor = gaussian * sinusoid
    
    return gabor


def build_gabor_bank_liu2012() -> List[np.ndarray]:
    """
    Build bank of 8 Gabor filters as specified in Liu et al. 2012.
    
    Frequencies: [0.10, 0.14, 0.20, 0.28]
    Orientations: [0, π/2]
    sigma = 0.56 / freq
    
    Returns:
        List of 8 Gabor filters
    """
    filters = []
    
    freqs = [0.10, 0.14, 0.20, 0.28]
    thetas = [0, np.pi / 2]
    
    # Kernel size: choose based on largest sigma
    # sigma_max ≈ 0.56 / 0.10 = 5.6, so use ksize = 21 (covers ~3.5 sigma)
    ksize = 21
    
    for freq in freqs:
        sigma = 0.56 / freq
        lambd = 1.0 / freq  # Wavelength
        
        for theta in thetas:
            gabor = create_gabor_filter(
                ksize=ksize,
                sigma=sigma,
                theta=theta,
                lambd=lambd,
                gamma=0.5,
                psi=0
            )
            filters.append(gabor)
    
    return filters


def create_schmid_filter(
    ksize: int,
    sigma: float,
    tau: float
) -> np.ndarray:
    """
    Create a Schmid filter (rotationally invariant).
    
    Schmid filters are Gaussian-modulated radial filters:
    F(r) = cos(2πτr) * exp(-r²/(2σ²))
    
    Args:
        ksize: Kernel size (should be odd)
        sigma: Scale parameter
        tau: Frequency parameter
    
    Returns:
        Schmid kernel (ksize, ksize)
    """
    center = ksize // 2
    x = np.arange(ksize, dtype=np.float32) - center
    y = np.arange(ksize, dtype=np.float32) - center
    xx, yy = np.meshgrid(x, y)
    
    # Radial distance
    r = np.sqrt(xx**2 + yy**2)
    
    # Avoid division by zero at center
    r = np.maximum(r, 1e-6)
    
    # Schmid filter formula
    gaussian = np.exp(-r**2 / (2 * sigma**2))
    cosine = np.cos(2 * np.pi * tau * r)
    
    schmid = gaussian * cosine
    
    # Normalize to zero mean
    schmid = schmid - schmid.mean()
    
    return schmid


def build_schmid_bank_liu2012() -> List[np.ndarray]:
    """
    Build bank of 13 Schmid filters as specified in Liu et al. 2012.
    
    Standard Schmid filter parameters (σ, τ):
    (2, 1), (4, 1), (4, 2), (6, 1), (6, 2), (6, 3),
    (8, 1), (8, 2), (8, 3), (10, 1), (10, 2), (10, 3), (10, 4)
    
    Returns:
        List of 13 Schmid filters
    """
    filters = []
    
    # Standard Schmid filter parameters (sigma, tau)
    params = [
        (2, 1), (4, 1), (4, 2), (6, 1), (6, 2), (6, 3),
        (8, 1), (8, 2), (8, 3), (10, 1), (10, 2), (10, 3), (10, 4)
    ]
    
    # Kernel size: choose based on largest sigma
    # sigma_max = 10, so use ksize = 41 (covers ~2 sigma)
    ksize = 41
    
    for sigma, tau in params:
        schmid = create_schmid_filter(ksize, sigma, tau)
        filters.append(schmid)
    
    return filters


def extract_texture_features_stripe(
    stripe_luminance: np.ndarray,
    gabor_bank: List[np.ndarray],
    schmid_bank: List[np.ndarray],
    n_bins: int = 16,
    device: Optional = None
) -> np.ndarray:
    """
    Extract texture features from a single stripe.
    
    Args:
        stripe_luminance: Luminance channel stripe (H, W) in float32 [0, 1]
        gabor_bank: List of 8 Gabor filters
        schmid_bank: List of 13 Schmid filters
        n_bins: Number of bins per histogram (default 16)
        device: GPU device (CuPy) or None for CPU
    
    Returns:
        Texture features (336,) = 21 filters × 16 bins
        Order: [8 Gabor histograms, 13 Schmid histograms]
    """
    histograms = []
    
    # Apply Gabor filters
    for gabor in gabor_bank:
        # Convolve filter with stripe (use GPU if available)
        if device is not None and GPU_AVAILABLE:
            filtered = convolve_gpu(stripe_luminance, gabor, device)
        else:
            filtered = ndimage.convolve(stripe_luminance, gabor, mode='reflect')
        
        # Compute magnitude (absolute value)
        magnitude = np.abs(filtered)
        
        # Normalize to [0, 1] using min-max per stripe
        mag_min = magnitude.min()
        mag_max = magnitude.max()
        if mag_max > mag_min:
            magnitude = (magnitude - mag_min) / (mag_max - mag_min)
        else:
            magnitude = np.zeros_like(magnitude)
        
        # Compute histogram
        hist, _ = np.histogram(
            magnitude.ravel(),
            bins=n_bins,
            range=(0.0, 1.0)
        )
        
        # Normalize to sum = 1
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist = hist.astype(np.float32) / hist_sum
        else:
            hist = np.zeros(n_bins, dtype=np.float32)
        
        histograms.append(hist)
    
    # Apply Schmid filters
    for schmid in schmid_bank:
        # Convolve filter with stripe (use GPU if available)
        if device is not None and GPU_AVAILABLE:
            filtered = convolve_gpu(stripe_luminance, schmid, device)
        else:
            filtered = ndimage.convolve(stripe_luminance, schmid, mode='reflect')
        
        # Compute magnitude (absolute value)
        magnitude = np.abs(filtered)
        
        # Normalize to [0, 1] using min-max per stripe
        mag_min = magnitude.min()
        mag_max = magnitude.max()
        if mag_max > mag_min:
            magnitude = (magnitude - mag_min) / (mag_max - mag_min)
        else:
            magnitude = np.zeros_like(magnitude)
        
        # Compute histogram
        hist, _ = np.histogram(
            magnitude.ravel(),
            bins=n_bins,
            range=(0.0, 1.0)
        )
        
        # Normalize to sum = 1
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist = hist.astype(np.float32) / hist_sum
        else:
            hist = np.zeros(n_bins, dtype=np.float32)
        
        histograms.append(hist)
    
    # Concatenate all histograms
    features = np.concatenate(histograms)
    
    return features.astype(np.float32)


def extract_texture_features(
    image: np.ndarray,
    n_stripes: int = 6,
    n_bins: int = 16,
    gabor_bank: List[np.ndarray] = None,
    schmid_bank: List[np.ndarray] = None,
    device: Optional = None
) -> np.ndarray:
    """
    Extract texture features from image using horizontal stripes.
    
    Args:
        image: RGB image (H, W, 3) in uint8 [0, 255] or float32 [0, 1]
        n_stripes: Number of horizontal stripes (default 6)
        n_bins: Number of bins per histogram (default 16)
        gabor_bank: Pre-computed Gabor filters (optional)
        schmid_bank: Pre-computed Schmid filters (optional)
        device: GPU device (CuPy) or None for CPU
    
    Returns:
        Texture features (n_stripes × 336,) = (2016,) for n_stripes=6
    """
    # Validate and convert input
    if image.dtype != np.uint8 and image.dtype != np.float32:
        raise ValueError(f"Image must be uint8 or float32, got {image.dtype}")
    
    # Convert to float32 [0, 1] if needed
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    
    # Compute luminance
    luminance = compute_luminance(image)
    
    h, w = luminance.shape
    
    # Handle very small images
    if h < n_stripes:
        # Resize to minimum height
        min_height = max(128, n_stripes)
        aspect_ratio = w / h if h > 0 else 1.0
        new_w = int(min_height * aspect_ratio)
        luminance = cv2.resize(luminance, (new_w, min_height))
        h = min_height
    
    # Build filter banks if not provided
    if gabor_bank is None:
        gabor_bank = build_gabor_bank_liu2012()
    
    if schmid_bank is None:
        schmid_bank = build_schmid_bank_liu2012()
    
    stripe_height = h // n_stripes
    
    features = []
    
    for i in range(n_stripes):
        start_h = i * stripe_height
        end_h = (i + 1) * stripe_height if i < n_stripes - 1 else h
        
        if end_h <= start_h:
            # Empty stripe, return zeros
            stripe_features = np.zeros(21 * n_bins, dtype=np.float32)
        else:
            stripe_lum = luminance[start_h:end_h, :]
            stripe_features = extract_texture_features_stripe(
                stripe_lum,
                gabor_bank,
                schmid_bank,
                n_bins,
                device=device
            )
        
        features.append(stripe_features)
    
    return np.concatenate(features).astype(np.float32)

