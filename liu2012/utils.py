"""
Utility functions for Liu et al. 2012 feature extraction.
"""

import numpy as np
import os
from typing import Optional, Dict, Tuple


def save_features(
    filepath: str,
    features: np.ndarray,
    metadata: Optional[Dict] = None
):
    """
    Save extracted features to file.
    
    Args:
        filepath: Path to save file (.npz)
        features: Feature matrix (n_samples, 2784)
        metadata: Optional metadata dictionary
    """
    save_dict = {
        'features': features
    }
    
    if metadata is not None:
        save_dict['metadata'] = metadata
    
    np.savez_compressed(filepath, **save_dict)


def load_features(filepath: str) -> Tuple[np.ndarray, Optional[Dict]]:
    """
    Load extracted features from file.
    
    Args:
        filepath: Path to .npz file
    
    Returns:
        features, metadata
    """
    data = np.load(filepath, allow_pickle=True)
    
    features = data['features']
    metadata = data.get('metadata', None)
    
    if metadata is not None:
        metadata = metadata.item()  # Convert numpy array to dict
    
    return features, metadata


def validate_features(features: np.ndarray, expected_dim: int = 2784) -> bool:
    """
    Validate feature vector/matrix.
    
    Args:
        features: Feature vector (2784,) or matrix (n, 2784)
        expected_dim: Expected feature dimension (default 2784)
    
    Returns:
        True if valid, raises ValueError otherwise
    """
    if features.ndim == 1:
        if features.shape[0] != expected_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {expected_dim}, got {features.shape[0]}"
            )
    elif features.ndim == 2:
        if features.shape[1] != expected_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {expected_dim}, got {features.shape[1]}"
            )
    else:
        raise ValueError(f"Features must be 1D or 2D array, got {features.ndim}D")
    
    # Check for NaN or Inf
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        raise ValueError("Features contain NaN or Inf values")
    
    # Check dtype
    if features.dtype != np.float32:
        raise ValueError(f"Features must be float32, got {features.dtype}")
    
    return True

