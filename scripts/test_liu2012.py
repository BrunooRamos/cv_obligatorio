"""Quick test script for Liu et al. 2012 feature extraction."""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import cv2
from liu2012.extractor import extract_liu2012_features


def test_feature_extraction():
    """Test feature extraction on a sample image."""
    
    # Create a dummy RGB image (64x128x3)
    dummy_image = np.random.randint(0, 255, (64, 128, 3), dtype=np.uint8)
    
    print("Testing Liu et al. 2012 feature extraction...")
    print(f"Input image shape: {dummy_image.shape}")
    print(f"Input image dtype: {dummy_image.dtype}")
    
    # Extract features
    features = extract_liu2012_features(dummy_image)
    
    print(f"\nExtracted features shape: {features.shape}")
    print(f"Extracted features dtype: {features.dtype}")
    print(f"Expected dimension: 2784")
    
    # Validate
    assert features.shape == (2784,), f"Expected shape (2784,), got {features.shape}"
    assert features.dtype == np.float32, f"Expected float32, got {features.dtype}"
    assert not np.any(np.isnan(features)), "Features contain NaN"
    assert not np.any(np.isinf(features)), "Features contain Inf"
    
    # Check histogram normalization (each histogram should sum to ~1)
    n_stripes = 6
    dim_color_per_stripe = 8 * 16  # 128
    dim_texture_per_stripe = 21 * 16  # 336
    
    print("\nValidating histogram normalization per stripe...")
    for i in range(n_stripes):
        start = i * (dim_color_per_stripe + dim_texture_per_stripe)
        end = start + dim_color_per_stripe
        
        # Check color histograms (8 channels × 16 bins)
        color_stripe = features[start:end]
        for j in range(8):
            hist_start = j * 16
            hist_end = hist_start + 16
            hist = color_stripe[hist_start:hist_end]
            hist_sum = hist.sum()
            assert abs(hist_sum - 1.0) < 1e-5 or hist_sum == 0, \
                f"Stripe {i}, color channel {j}: histogram sum = {hist_sum}, expected ~1.0 or 0"
        
        # Check texture histograms (21 filters × 16 bins)
        texture_start = end
        texture_end = texture_start + dim_texture_per_stripe
        texture_stripe = features[texture_start:texture_end]
        for j in range(21):
            hist_start = j * 16
            hist_end = hist_start + 16
            hist = texture_stripe[hist_start:hist_end]
            hist_sum = hist.sum()
            assert abs(hist_sum - 1.0) < 1e-5 or hist_sum == 0, \
                f"Stripe {i}, texture filter {j}: histogram sum = {hist_sum}, expected ~1.0 or 0"
    
    print("✓ All histograms are properly normalized")
    
    print("\n✓ Feature extraction test passed!")
    print(f"Feature statistics:")
    print(f"  Min: {features.min():.6f}")
    print(f"  Max: {features.max():.6f}")
    print(f"  Mean: {features.mean():.6f}")
    print(f"  Std: {features.std():.6f}")
    print(f"  Non-zero: {np.count_nonzero(features)}/{len(features)} ({100*np.count_nonzero(features)/len(features):.1f}%)")


if __name__ == '__main__':
    test_feature_extraction()

