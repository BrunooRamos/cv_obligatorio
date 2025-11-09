"""
Utility functions for BILP feature extraction and normalization
"""

import numpy as np
import json
import os
from typing import Tuple, Optional, Dict
from .color import extract_color_features
from .texture import extract_texture_features, build_gabor_bank
from .gpu_utils import get_device


def normalize_l1(features: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    L1 normalization of feature vector.

    Args:
        features: Feature vector
        epsilon: Small constant to avoid division by zero

    Returns:
        L1-normalized feature vector
    """
    norm = np.sum(np.abs(features)) + epsilon
    return features / norm


def power_normalize(features: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Power normalization (signed square root).

    Args:
        features: Feature vector
        alpha: Power parameter (0.5 for square root)

    Returns:
        Power-normalized feature vector
    """
    return np.sign(features) * np.abs(features) ** alpha


def normalize_features(
    features: np.ndarray,
    method: str = 'l1',
    alpha: float = 0.5
) -> np.ndarray:
    """
    Normalize features using specified method.

    Args:
        features: Feature vector or matrix
        method: 'l1', 'l2', 'power', or 'l1_power'
        alpha: Power parameter for power normalization

    Returns:
        Normalized features
    """
    if method == 'l1':
        return normalize_l1(features)
    elif method == 'l2':
        norm = np.linalg.norm(features) + 1e-10
        return features / norm
    elif method == 'power':
        return power_normalize(features, alpha)
    elif method == 'l1_power':
        # Power normalization followed by L1
        features = power_normalize(features, alpha)
        return normalize_l1(features)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def load_calibrated_color_ranges(filepath: str = 'data/color_ranges.json') -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Load calibrated (u,v) color ranges from file.

    Args:
        filepath: Path to calibration file

    Returns:
        u_range, v_range: Tuples with (min, max) ranges
    """
    if not os.path.exists(filepath):
        # Return default ranges if file doesn't exist
        print(f"Warning: Calibration file not found at {filepath}, using default ranges")
        return (-0.66, 1.07), (-0.66, 0.48)

    with open(filepath, 'r') as f:
        data = json.load(f)

    u_range = tuple(data['u_range'])
    v_range = tuple(data['v_range'])

    return u_range, v_range


def normalize_per_stripe(
    features: np.ndarray,
    n_stripes: int,
    features_per_stripe: int,
    method: str = 'l1'
) -> np.ndarray:
    """
    Normalize features independently per stripe.

    Args:
        features: Concatenated feature vector (n_stripes × features_per_stripe,)
        n_stripes: Number of stripes
        features_per_stripe: Number of features per stripe
        method: Normalization method

    Returns:
        Normalized features
    """
    normalized = np.zeros_like(features)

    for i in range(n_stripes):
        start_idx = i * features_per_stripe
        end_idx = (i + 1) * features_per_stripe
        stripe_features = features[start_idx:end_idx]
        normalized[start_idx:end_idx] = normalize_features(stripe_features, method)

    return normalized


def extract_bilp_descriptor(
    image: np.ndarray,
    n_stripes: int = 6,
    color_params: Optional[Dict] = None,
    texture_params: Optional[Dict] = None,
    normalize: bool = True,
    normalize_method: str = 'l1',
    device: Optional = None
) -> Dict[str, np.ndarray]:
    """
    Extract complete BILP descriptor from image.

    Args:
        image: RGB image (H, W, 3)
        n_stripes: Number of horizontal stripes
        color_params: Parameters for color feature extraction
        texture_params: Parameters for texture feature extraction
        normalize: Whether to normalize features per stripe
        normalize_method: Normalization method ('l1', 'l2', 'power', 'l1_power')
        device: GPU device (CuPy) or None for CPU

    Returns:
        Dictionary containing:
            - 'color': Color features
            - 'texture': Texture features
            - 'combined': Concatenated features (optional)
    """
    # Default parameters
    if color_params is None:
        # Load calibrated ranges if available
        u_range, v_range = load_calibrated_color_ranges()

        color_params = {
            'n_bins_uv': 16,
            'n_bins_lum': 16,
            'u_range': u_range,
            'v_range': v_range
        }

    if texture_params is None:
        texture_params = {
            'n_scales': 5,
            'n_orientations': 8
        }

    # Extract color features
    color_features = extract_color_features(
        image,
        n_stripes=n_stripes,
        **color_params
    )

    # Extract texture features (use GPU if available)
    texture_features = extract_texture_features(
        image,
        n_stripes=n_stripes,
        device=device,
        **texture_params
    )

    # Normalize per stripe if requested
    if normalize:
        # Color: 16*16 + 16 = 272 per stripe
        color_per_stripe = color_params['n_bins_uv']**2 + color_params['n_bins_lum']
        color_features = normalize_per_stripe(
            color_features,
            n_stripes,
            color_per_stripe,
            normalize_method
        )

        # Texture: n_scales * n_orientations + 2 per stripe
        texture_per_stripe = (texture_params['n_scales'] *
                             texture_params['n_orientations'] + 2)
        texture_features = normalize_per_stripe(
            texture_features,
            n_stripes,
            texture_per_stripe,
            normalize_method
        )

    return {
        'color': color_features,
        'texture': texture_features,
        'n_stripes': n_stripes
    }


def extract_bilp_batch(
    images: list,
    n_stripes: int = 6,
    color_params: Optional[Dict] = None,
    texture_params: Optional[Dict] = None,
    normalize: bool = True,
    normalize_method: str = 'l1',
    verbose: bool = False,
    use_gpu: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract BILP descriptors from batch of images.

    Args:
        images: List of RGB images
        n_stripes: Number of horizontal stripes
        color_params: Parameters for color feature extraction
        texture_params: Parameters for texture feature extraction
        normalize: Whether to normalize features per stripe
        normalize_method: Normalization method
        verbose: Print progress
        use_gpu: Whether to use GPU if available

    Returns:
        color_features: (n_images, color_dim)
        texture_features: (n_images, texture_dim)
    """
    # Get GPU device if requested
    is_gpu, device = get_device(use_gpu)
    if is_gpu and verbose:
        print(f"Using GPU for feature extraction")
    elif use_gpu and not is_gpu and verbose:
        print(f"GPU requested but not available, using CPU")
    
    color_features_list = []
    texture_features_list = []

    for i, image in enumerate(images):
        if verbose and (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(images)} images")

        descriptor = extract_bilp_descriptor(
            image,
            n_stripes=n_stripes,
            color_params=color_params,
            texture_params=texture_params,
            normalize=normalize,
            normalize_method=normalize_method,
            device=device
        )

        color_features_list.append(descriptor['color'])
        texture_features_list.append(descriptor['texture'])

    color_features = np.array(color_features_list)
    texture_features = np.array(texture_features_list)

    return color_features, texture_features


def compute_feature_stats(features: np.ndarray) -> Dict[str, float]:
    """
    Compute statistics of feature vectors.

    Args:
        features: Feature matrix (n_samples, n_features)

    Returns:
        Dictionary with statistics
    """
    return {
        'mean': float(np.mean(features)),
        'std': float(np.std(features)),
        'min': float(np.min(features)),
        'max': float(np.max(features)),
        'median': float(np.median(features)),
        'sparsity': float(np.sum(features == 0) / features.size)
    }


def save_features(
    filepath: str,
    color_features: np.ndarray,
    texture_features: np.ndarray,
    metadata: Optional[Dict] = None
):
    """
    Save extracted features to file.

    Args:
        filepath: Path to save file (.npz)
        color_features: Color feature matrix
        texture_features: Texture feature matrix
        metadata: Optional metadata dictionary
    """
    save_dict = {
        'color': color_features,
        'texture': texture_features
    }

    if metadata is not None:
        save_dict['metadata'] = metadata

    np.savez_compressed(filepath, **save_dict)


def load_features(filepath: str) -> Tuple[np.ndarray, np.ndarray, Optional[Dict]]:
    """
    Load extracted features from file.

    Args:
        filepath: Path to .npz file

    Returns:
        color_features, texture_features, metadata
    """
    data = np.load(filepath, allow_pickle=True)

    color_features = data['color']
    texture_features = data['texture']
    metadata = data.get('metadata', None)

    if metadata is not None:
        metadata = metadata.item()  # Convert numpy array to dict

    return color_features, texture_features, metadata
