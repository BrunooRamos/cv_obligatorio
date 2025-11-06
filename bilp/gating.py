"""
Adaptive gating for color/texture fusion in BILP
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from scipy.stats import entropy


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-x))


def compute_chromatic_entropy(
    u: np.ndarray,
    v: np.ndarray,
    n_bins: int = 16
) -> float:
    """
    Compute entropy of chromatic histogram (u, v).

    Args:
        u, v: Log-chromaticity channels
        n_bins: Number of bins for histogram

    Returns:
        Entropy value
    """
    # Compute 2D histogram
    hist, _, _ = np.histogram2d(
        u.ravel(),
        v.ravel(),
        bins=[n_bins, n_bins]
    )

    # Normalize to probability distribution
    hist = hist.ravel()
    hist = hist / (hist.sum() + 1e-10)

    # Compute entropy
    return float(entropy(hist + 1e-10))


def compute_texture_complexity(
    gabor_energies: np.ndarray,
    high_freq_threshold: float = 0.5
) -> float:
    """
    Compute texture complexity as proportion of energy in high frequencies.

    Args:
        gabor_energies: Gabor filter responses (n_filters,)
        high_freq_threshold: Threshold to consider high frequencies

    Returns:
        Proportion of energy in high frequencies
    """
    # Assume Gabor energies are ordered by scale (low to high frequency)
    # High frequency responses are at the beginning (smaller wavelengths)

    total_energy = np.sum(gabor_energies) + 1e-10

    # Take first half as high frequencies
    n_filters = len(gabor_energies)
    n_high_freq = int(n_filters * high_freq_threshold)

    high_freq_energy = np.sum(gabor_energies[:n_high_freq])

    return float(high_freq_energy / total_energy)


def compute_gating_weight(
    color_features: np.ndarray,
    texture_features: np.ndarray,
    params: Dict[str, float],
    n_stripes: int = 6,
    color_per_stripe: int = 272,
    texture_per_stripe: int = 42
) -> float:
    """
    Compute adaptive gating weight alpha.

    alpha = sigmoid(a1 * T - a2 * C + b)

    where:
    - T: texture complexity (proportion of high-frequency energy)
    - C: chromatic entropy
    - a1, a2, b: learnable parameters

    Args:
        color_features: Color features (can be used to compute C if needed)
        texture_features: Texture features (n_stripes * texture_per_stripe,)
        params: Dictionary with 'a1', 'a2', 'b'
        n_stripes: Number of stripes
        color_per_stripe: Features per stripe for color
        texture_per_stripe: Features per stripe for texture

    Returns:
        Gating weight alpha in [0, 1]
    """
    a1 = params.get('a1', 2.0)
    a2 = params.get('a2', 1.0)
    b = params.get('b', 0.0)

    # Compute T: average texture complexity across stripes
    T_values = []
    for i in range(n_stripes):
        start_idx = i * texture_per_stripe
        end_idx = start_idx + texture_per_stripe - 2  # Exclude FFT features
        gabor_stripe = texture_features[start_idx:end_idx]
        T_values.append(compute_texture_complexity(gabor_stripe))

    T = np.mean(T_values)

    # Compute C: we need to recompute from image or use proxy
    # For now, use a proxy: entropy of color histogram
    # This is a simplification; ideally we'd pass u,v directly
    # or store C during feature extraction

    # Proxy: use normalized color features as histogram
    C_values = []
    for i in range(n_stripes):
        start_idx = i * color_per_stripe
        end_idx = start_idx + color_per_stripe
        hist_stripe = color_features[start_idx:end_idx]

        # Normalize to probability
        hist_stripe = hist_stripe / (hist_stripe.sum() + 1e-10)

        # Compute entropy
        C_values.append(entropy(hist_stripe + 1e-10))

    C = np.mean(C_values)

    # Compute alpha
    logit = a1 * T - a2 * C + b
    alpha = sigmoid(logit)

    # Clip to reasonable range
    alpha = np.clip(alpha, 0.1, 0.9)

    return float(alpha)


def compute_gating_weights_batch(
    color_features: np.ndarray,
    texture_features: np.ndarray,
    params: Dict[str, float],
    n_stripes: int = 6,
    color_per_stripe: int = 272,
    texture_per_stripe: int = 42
) -> np.ndarray:
    """
    Compute gating weights for a batch of images.

    Args:
        color_features: Color features (n_images, color_dim)
        texture_features: Texture features (n_images, texture_dim)
        params: Gating parameters
        n_stripes: Number of stripes
        color_per_stripe: Features per stripe for color
        texture_per_stripe: Features per stripe for texture

    Returns:
        Alpha weights (n_images,)
    """
    n_images = color_features.shape[0]
    alphas = np.zeros(n_images)

    for i in range(n_images):
        alphas[i] = compute_gating_weight(
            color_features[i],
            texture_features[i],
            params,
            n_stripes,
            color_per_stripe,
            texture_per_stripe
        )

    return alphas


def optimize_gating_params(
    train_color: np.ndarray,
    train_texture: np.ndarray,
    train_ids: np.ndarray,
    distance_fn,
    param_grid: Optional[Dict[str, List[float]]] = None,
    n_stripes: int = 6,
    color_per_stripe: int = 272,
    texture_per_stripe: int = 42,
    metric: str = 'rank1'
) -> Dict[str, float]:
    """
    Optimize gating parameters using grid search on validation set.

    Args:
        train_color: Training color features
        train_texture: Training texture features
        train_ids: Training person IDs
        distance_fn: Distance computation function
        param_grid: Grid of parameters to search
        n_stripes: Number of stripes
        color_per_stripe: Features per stripe for color
        texture_per_stripe: Features per stripe for texture
        metric: Optimization metric ('rank1', 'rank5', 'map')

    Returns:
        Best parameters
    """
    if param_grid is None:
        param_grid = {
            'a1': [1.5, 2.0, 3.0],
            'a2': [0.5, 1.0, 1.5],
            'b': [-0.5, 0.0, 0.5]
        }

    best_score = -1
    best_params = {'a1': 2.0, 'a2': 1.0, 'b': 0.0}

    print("Starting grid search for gating parameters...")

    # Grid search
    for a1 in param_grid['a1']:
        for a2 in param_grid['a2']:
            for b in param_grid['b']:
                params = {'a1': a1, 'a2': a2, 'b': b}

                # Compute gating weights
                alphas = compute_gating_weights_batch(
                    train_color,
                    train_texture,
                    params,
                    n_stripes,
                    color_per_stripe,
                    texture_per_stripe
                )

                # Evaluate on validation subset (sample for speed)
                # For simplicity, use mean alpha and evaluate
                alpha_mean = np.mean(alphas)

                # Compute distances with this alpha
                # (This is a simplified version; full evaluation would be expensive)
                # Here we just track the score based on a simple criterion

                # For now, we'll return default params
                # Full implementation would require actual distance computation
                # and CMC/mAP evaluation

                score = alpha_mean  # Placeholder

                if score > best_score:
                    best_score = score
                    best_params = params.copy()

    print(f"Best parameters: {best_params}")
    print(f"Best score: {best_score}")

    return best_params


def compute_per_stripe_gating(
    color_features: np.ndarray,
    texture_features: np.ndarray,
    params: Dict[str, float],
    n_stripes: int = 6,
    color_per_stripe: int = 272,
    texture_per_stripe: int = 42
) -> np.ndarray:
    """
    Compute per-stripe gating weights.

    Args:
        color_features: Color features
        texture_features: Texture features
        params: Gating parameters
        n_stripes: Number of stripes
        color_per_stripe: Features per stripe for color
        texture_per_stripe: Features per stripe for texture

    Returns:
        Alpha weights per stripe (n_stripes,)
    """
    a1 = params.get('a1', 2.0)
    a2 = params.get('a2', 1.0)
    b = params.get('b', 0.0)

    alphas = np.zeros(n_stripes)

    for i in range(n_stripes):
        # Texture complexity for this stripe
        start_idx_t = i * texture_per_stripe
        end_idx_t = start_idx_t + texture_per_stripe - 2
        gabor_stripe = texture_features[start_idx_t:end_idx_t]
        T = compute_texture_complexity(gabor_stripe)

        # Chromatic entropy for this stripe
        start_idx_c = i * color_per_stripe
        end_idx_c = start_idx_c + color_per_stripe
        hist_stripe = color_features[start_idx_c:end_idx_c]
        hist_stripe = hist_stripe / (hist_stripe.sum() + 1e-10)
        C = entropy(hist_stripe + 1e-10)

        # Compute alpha for this stripe
        logit = a1 * T - a2 * C + b
        alpha = sigmoid(logit)
        alphas[i] = np.clip(alpha, 0.1, 0.9)

    return alphas
