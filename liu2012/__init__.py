"""
Liu et al. 2012 feature extraction for Person Re-identification.

Implementation based on:
"Person Re-identification: What Features Are Important?"
Liu et al., ECCV Workshops 2012

Features:
- 6 horizontal stripes
- Color: 8 channels (RGB, HSV, YCbCr) with 16-bin histograms → 128 dims/stripe
- Texture: 21 filters (8 Gabor + 13 Schmid) on luminance with 16-bin histograms → 336 dims/stripe
- Total: 6 × (128 + 336) = 2784 dimensions
"""

from .extractor import extract_liu2012_features

__all__ = ['extract_liu2012_features']

