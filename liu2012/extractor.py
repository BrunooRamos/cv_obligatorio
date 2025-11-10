"""
Main feature extractor for Liu et al. 2012.

Combines color and texture features into a single 2784-dimensional vector.
"""

import numpy as np
from typing import Optional
from .color import extract_color_features
from .texture import extract_texture_features, build_gabor_bank_liu2012, build_schmid_bank_liu2012

# Import GPU utilities
try:
    from bilp.gpu_utils import get_device
except ImportError:
    def get_device(use_gpu: bool = False):
        return False, None


# Constants from the paper
N_STRIPES = 6
N_COLOR_CHANNELS = 8  # R, G, B, H, S, V, Cb, Cr
N_BINS = 16
N_GABOR = 8
N_SCHMID = 13
DIM_COLOR_PER_STRIPE = N_COLOR_CHANNELS * N_BINS  # 128
DIM_TEXTURE_PER_STRIPE = (N_GABOR + N_SCHMID) * N_BINS  # 336
DIM_PER_STRIPE = DIM_COLOR_PER_STRIPE + DIM_TEXTURE_PER_STRIPE  # 464
DIM_TOTAL = N_STRIPES * DIM_PER_STRIPE  # 2784


def extract_liu2012_features(
    image: np.ndarray,
    n_stripes: int = N_STRIPES,
    n_bins: int = N_BINS,
    gabor_bank: Optional[list] = None,
    schmid_bank: Optional[list] = None,
    device: Optional = None
) -> np.ndarray:
    """
    Extract Liu et al. 2012 features from an image.
    
    Input:
        image: RGB image in uint8 (H×W×3), any size
    
    Output:
        feat: np.ndarray of shape (2784,) and dtype float32
    
    Features:
        - 6 horizontal stripes
        - Color: 8 channels (RGB, HSV, YCbCr) × 16 bins = 128 dims/stripe
        - Texture: 21 filters (8 Gabor + 13 Schmid) × 16 bins = 336 dims/stripe
        - Total: 6 × (128 + 336) = 2784 dimensions
    
    Args:
        image: RGB image (H, W, 3) in uint8 [0, 255] or float32 [0, 1]
        n_stripes: Number of horizontal stripes (default 6)
        n_bins: Number of bins per histogram (default 16)
        gabor_bank: Pre-computed Gabor filters (optional, for efficiency)
        schmid_bank: Pre-computed Schmid filters (optional, for efficiency)
        device: GPU device (CuPy) or None for CPU
    
    Returns:
        Feature vector (2784,) of dtype float32
    """
    # Validate input
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError(f"Image must be RGB (H, W, 3), got shape {image.shape}")
    
    # Ensure uint8 input
    if image.dtype != np.uint8:
        if image.dtype == np.float32:
            # Assume [0, 1] range, convert to uint8
            image = (np.clip(image, 0, 1) * 255.0).astype(np.uint8)
        else:
            raise ValueError(f"Image must be uint8 or float32, got {image.dtype}")
    
    # Extract color features
    color_features = extract_color_features(
        image,
        n_stripes=n_stripes,
        n_bins=n_bins
    )
    
    # Extract texture features
    texture_features = extract_texture_features(
        image,
        n_stripes=n_stripes,
        n_bins=n_bins,
        gabor_bank=gabor_bank,
        schmid_bank=schmid_bank,
        device=device
    )
    
    # Concatenate color and texture features
    # Order: stripe 1 (color + texture), stripe 2 (color + texture), ..., stripe 6
    features = []
    dim_color_per_stripe = N_COLOR_CHANNELS * n_bins
    dim_texture_per_stripe = (N_GABOR + N_SCHMID) * n_bins
    
    for i in range(n_stripes):
        start_color = i * dim_color_per_stripe
        end_color = (i + 1) * dim_color_per_stripe
        start_texture = i * dim_texture_per_stripe
        end_texture = (i + 1) * dim_texture_per_stripe
        
        stripe_color = color_features[start_color:end_color]
        stripe_texture = texture_features[start_texture:end_texture]
        
        # Concatenate: color first, then texture
        stripe_features = np.concatenate([stripe_color, stripe_texture])
        features.append(stripe_features)
    
    # Final concatenation
    feat = np.concatenate(features).astype(np.float32)
    
    # Validate output dimensions
    expected_dim = n_stripes * (dim_color_per_stripe + dim_texture_per_stripe)
    if feat.shape[0] != expected_dim:
        raise ValueError(
            f"Feature dimension mismatch: expected {expected_dim}, got {feat.shape[0]}"
        )
    
    # Validate no NaN or Inf
    if np.any(np.isnan(feat)) or np.any(np.isinf(feat)):
        raise ValueError("Feature vector contains NaN or Inf values")
    
    return feat


def extract_liu2012_batch(
    images: list,
    n_stripes: int = N_STRIPES,
    n_bins: int = N_BINS,
    verbose: bool = False,
    use_gpu: bool = False
) -> np.ndarray:
    """
    Extract Liu et al. 2012 features from a batch of images.
    
    Args:
        images: List of RGB images (each H×W×3, uint8)
        n_stripes: Number of horizontal stripes (default 6)
        n_bins: Number of bins per histogram (default 16)
        verbose: Print progress
        use_gpu: Whether to use GPU if available
    
    Returns:
        Feature matrix (n_images, 2784) of dtype float32
    """
    # Get GPU device if requested
    is_gpu, device = get_device(use_gpu)
    if is_gpu and verbose:
        print(f"Using GPU for feature extraction")
    elif use_gpu and not is_gpu and verbose:
        print(f"GPU requested but not available, using CPU")
    
    # Pre-compute filter banks once for efficiency
    gabor_bank = build_gabor_bank_liu2012()
    schmid_bank = build_schmid_bank_liu2012()
    
    features = []
    
    for i, image in enumerate(images):
        if verbose and (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(images)} images")
        
        feat = extract_liu2012_features(
            image,
            n_stripes=n_stripes,
            n_bins=n_bins,
            gabor_bank=gabor_bank,
            schmid_bank=schmid_bank,
            device=device
        )
        
        features.append(feat)
    
    return np.vstack(features).astype(np.float32)

