"""
Texture features: Gabor filters, periodicity, and spectral analysis
"""

import numpy as np
from scipy import ndimage
from typing import Tuple, List, Optional
from .gpu_utils import get_device, convolve_gpu


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
        ksize: Kernel size
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
    x = np.arange(-ksize // 2 + 1, ksize // 2 + 1)
    y = np.arange(-ksize // 2 + 1, ksize // 2 + 1)
    xx, yy = np.meshgrid(x, y)

    # Rotate coordinates
    x_theta = xx * np.cos(theta) + yy * np.sin(theta)
    y_theta = -xx * np.sin(theta) + yy * np.cos(theta)

    # Gabor function
    gaussian = np.exp(-0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2))
    sinusoid = np.cos(2 * np.pi * x_theta / lambd + psi)

    gabor = gaussian * sinusoid

    return gabor


def build_gabor_bank(
    n_scales: int = 5,
    n_orientations: int = 8,
    ksize: int = 31
) -> List[np.ndarray]:
    """
    Build a bank of Gabor filters.

    Args:
        n_scales: Number of scales (wavelengths)
        n_orientations: Number of orientations
        ksize: Kernel size

    Returns:
        List of Gabor filters
    """
    filters = []

    # Scales: wavelengths from ~4 to ~16 pixels
    wavelengths = np.logspace(np.log2(4), np.log2(16), n_scales, base=2)

    # Orientations: 0 to π
    orientations = np.linspace(0, np.pi, n_orientations, endpoint=False)

    for lambd in wavelengths:
        sigma = 0.56 * lambd  # Common ratio
        for theta in orientations:
            gabor = create_gabor_filter(ksize, sigma, theta, lambd)
            filters.append(gabor)

    return filters


def apply_gabor_bank(
    image: np.ndarray,
    gabor_bank: List[np.ndarray],
    device: Optional = None
) -> np.ndarray:
    """
    Apply Gabor filter bank to image and compute energy.

    Args:
        image: Grayscale image (H, W)
        gabor_bank: List of Gabor filters
        device: GPU device (CuPy) or None for CPU

    Returns:
        Gabor energies (n_filters,)
    """
    energies = []

    for gabor in gabor_bank:
        # Convolve (use GPU if available)
        if device is not None:
            filtered = convolve_gpu(image, gabor, device)
        else:
            filtered = ndimage.convolve(image, gabor, mode='reflect')
        # Compute energy (L2 norm)
        energy = np.sqrt(np.mean(filtered**2))
        energies.append(energy)

    return np.array(energies)


def compute_fft_features(image: np.ndarray) -> Tuple[float, float]:
    """
    Compute FFT-based texture features.

    Args:
        image: Grayscale image (H, W)

    Returns:
        peak_frequency: Dominant radial frequency
        spectral_entropy: Entropy of frequency spectrum
    """
    # Compute 2D FFT
    fft = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)

    # Radial frequency profile
    h, w = image.shape
    center = (h // 2, w // 2)

    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)

    # Radial average
    r_max = min(center)
    radial_profile = np.zeros(r_max)

    for radius in range(r_max):
        mask = (r == radius)
        if mask.sum() > 0:
            radial_profile[radius] = magnitude[mask].mean()

    # Peak frequency (ignore DC component)
    if len(radial_profile) > 1:
        peak_frequency = np.argmax(radial_profile[1:]) + 1
    else:
        peak_frequency = 0

    # Spectral entropy
    spectrum = magnitude.ravel()
    spectrum = spectrum / (spectrum.sum() + 1e-10)
    spectrum = spectrum[spectrum > 0]
    spectral_entropy = -np.sum(spectrum * np.log(spectrum + 1e-10))

    return float(peak_frequency), float(spectral_entropy)


def extract_texture_features(
    image: np.ndarray,
    n_stripes: int = 6,
    gabor_bank: List[np.ndarray] = None,
    n_scales: int = 5,
    n_orientations: int = 8,
    device: Optional = None
) -> np.ndarray:
    """
    Extract texture features from image using horizontal stripes.

    Args:
        image: RGB or grayscale image
        n_stripes: Number of horizontal stripes
        gabor_bank: Pre-computed Gabor filter bank (optional)
        n_scales: Number of Gabor scales
        n_orientations: Number of Gabor orientations
        device: GPU device (CuPy) or None for CPU

    Returns:
        Texture features (n_stripes × n_features,)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        # Simple grayscale conversion
        gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    else:
        gray = image.copy()

    gray = gray.astype(np.float32)

    # Build Gabor bank if not provided
    if gabor_bank is None:
        gabor_bank = build_gabor_bank(n_scales, n_orientations)

    h, w = gray.shape
    stripe_height = h // n_stripes

    features = []

    for i in range(n_stripes):
        start_h = i * stripe_height
        end_h = (i + 1) * stripe_height if i < n_stripes - 1 else h

        stripe = gray[start_h:end_h, :]

        # Gabor energies (use GPU if available)
        gabor_energies = apply_gabor_bank(stripe, gabor_bank, device)

        # FFT features
        peak_freq, spec_entropy = compute_fft_features(stripe)

        # Concatenate features
        stripe_features = np.concatenate([
            gabor_energies,
            [peak_freq, spec_entropy]
        ])

        features.append(stripe_features)

    return np.concatenate(features)
