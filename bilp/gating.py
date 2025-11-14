"""
Adaptive gating for color/texture fusion in BILP
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from scipy.stats import entropy


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-x))




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
    texture_per_stripe: int = 42,
    t_stats: Optional[Dict[str, float]] = None,
    c_stats: Optional[Dict[str, float]] = None
) -> float:
    """
    Compute adaptive gating weight alpha.

    alpha = sigmoid(a1 * T_norm - a2 * C_norm + b)

    where:
    - T_norm: normalized texture complexity
    - C_norm: normalized chromatic entropy
    - a1, a2, b: learnable parameters

    Args:
        color_features: Color features (can be used to compute C if needed)
        texture_features: Texture features (n_stripes * texture_per_stripe,)
        params: Dictionary with 'a1', 'a2', 'b'
        n_stripes: Number of stripes
        color_per_stripe: Features per stripe for color
        texture_per_stripe: Features per stripe for texture
        t_stats: Statistics for T normalization (mean, std) - optional
        c_stats: Statistics for C normalization (mean, std) - optional

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

    # Normalize T and C to same scale (simplified approach)
    if t_stats is not None and c_stats is not None:
        # Simple z-score normalization - keep it simple
        T_norm = (T - t_stats['mean']) / (t_stats['std'] + 1e-10)
        C_norm = (C - c_stats['mean']) / (c_stats['std'] + 1e-10)
    else:
        # No normalization (use raw values)
        T_norm = T
        C_norm = C

    # Compute alpha
    logit = a1 * T_norm - a2 * C_norm + b
    alpha = sigmoid(logit)

    # Clip to reasonable range (less restrictive to allow more variation)
    # The sigmoid already bounds it to [0, 1], so we just prevent extreme values
    alpha = np.clip(alpha, 0.05, 0.95)

    return float(alpha)


def compute_gating_weights_batch(
    color_features: np.ndarray,
    texture_features: np.ndarray,
    params: Dict[str, float],
    n_stripes: int = 6,
    color_per_stripe: int = 272,
    texture_per_stripe: int = 42,
    t_stats: Optional[Dict[str, float]] = None,
    c_stats: Optional[Dict[str, float]] = None
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
        t_stats: Statistics for T normalization (mean, std) - optional
        c_stats: Statistics for C normalization (mean, std) - optional

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
            texture_per_stripe,
            t_stats,
            c_stats
        )

    return alphas


def optimize_gating_params(
    query_color: np.ndarray,
    query_texture: np.ndarray,
    query_ids: np.ndarray,
    gallery_color: np.ndarray,
    gallery_texture: np.ndarray,
    gallery_ids: np.ndarray,
    query_cams: Optional[np.ndarray] = None,
    gallery_cams: Optional[np.ndarray] = None,
    param_grid: Optional[Dict[str, List[float]]] = None,
    n_stripes: int = 6,
    color_per_stripe: int = 272,
    texture_per_stripe: int = 42,
    metric: str = 'rank1',
    distance_metric: str = 'cityblock',
    verbose: bool = True
) -> Dict[str, float]:
    """
    Optimize gating parameters using grid search with real evaluation.

    Args:
        query_color: Query color features (n_query, color_dim)
        query_texture: Query texture features (n_query, texture_dim)
        query_ids: Query person IDs (n_query,)
        gallery_color: Gallery color features (n_gallery, color_dim)
        gallery_texture: Gallery texture features (n_gallery, texture_dim)
        gallery_ids: Gallery person IDs (n_gallery,)
        query_cams: Query camera IDs (optional)
        gallery_cams: Gallery camera IDs (optional)
        param_grid: Grid of parameters to search
        n_stripes: Number of stripes
        color_per_stripe: Features per stripe for color
        texture_per_stripe: Features per stripe for texture
        metric: Optimization metric ('rank1', 'rank5', 'map')
        distance_metric: Distance metric for features ('cityblock', 'euclidean', 'cosine')
        verbose: Print progress

    Returns:
        Best parameters
    """
    from bilp.distance import compute_distance_matrix_fast
    from eval.cmc_map import evaluate_reid
    
    if param_grid is None:
        # Grid with wider bias range - bias seems to be the main driver of performance
        param_grid = {
            'a1': [0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
            'a2': [0.5, 1.0, 1.5, 2.0, 3.0],
            'b': [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]  # Wider range including higher values
        }

    best_score = -1
    best_params = {'a1': 2.0, 'a2': 1.0, 'b': 0.0}

    # Compute T and C statistics for normalization
    from scipy.stats import entropy
    
    T_all = []
    C_all = []
    
    # Sample from query to compute statistics
    n_sample = min(100, query_color.shape[0])
    for i in range(n_sample):
        # T: texture complexity
        T_vals = []
        for s in range(n_stripes):
            start_idx = s * texture_per_stripe
            end_idx = start_idx + texture_per_stripe - 2
            gabor_stripe = query_texture[i, start_idx:end_idx]
            total_energy = np.sum(gabor_stripe) + 1e-10
            n_high_freq = int(len(gabor_stripe) * 0.5)
            high_freq_energy = np.sum(gabor_stripe[:n_high_freq])
            T_vals.append(high_freq_energy / total_energy)
        T_all.append(np.mean(T_vals))
        
        # C: chromatic entropy
        C_vals = []
        for s in range(n_stripes):
            start_idx = s * color_per_stripe
            end_idx = start_idx + color_per_stripe
            hist_stripe = query_color[i, start_idx:end_idx]
            hist_stripe = hist_stripe / (hist_stripe.sum() + 1e-10)
            C_vals.append(entropy(hist_stripe + 1e-10))
        C_all.append(np.mean(C_vals))
    
    T_all = np.array(T_all)
    C_all = np.array(C_all)
    
    # Compute statistics including min/max for min-max normalization
    t_stats = {
        'mean': float(np.mean(T_all)),
        'std': float(np.std(T_all)),
        'min': float(np.min(T_all)),
        'max': float(np.max(T_all))
    }
    c_stats = {
        'mean': float(np.mean(C_all)),
        'std': float(np.std(C_all)),
        'min': float(np.min(C_all)),
        'max': float(np.max(C_all))
    }
    
    if verbose:
        print("Starting grid search for gating parameters...")
        print(f"Grid size: {len(param_grid['a1']) * len(param_grid['a2']) * len(param_grid['b'])} combinations")
        print(f"Optimization metric: {metric}")
        print(f"T stats: mean={t_stats['mean']:.4f}, std={t_stats['std']:.4f}, range=[{t_stats['min']:.4f}, {t_stats['max']:.4f}]")
        print(f"C stats: mean={c_stats['mean']:.4f}, std={c_stats['std']:.4f}, range=[{c_stats['min']:.4f}, {c_stats['max']:.4f}]")
        print(f"Using z-score normalization for T and C")

    total_combinations = len(param_grid['a1']) * len(param_grid['a2']) * len(param_grid['b'])
    current_combination = 0

    # Grid search
    for a1 in param_grid['a1']:
        for a2 in param_grid['a2']:
            for b in param_grid['b']:
                current_combination += 1
                params = {'a1': a1, 'a2': a2, 'b': b}

                # Compute adaptive gating weights for queries (with normalization)
                query_alphas = compute_gating_weights_batch(
                    query_color,
                    query_texture,
                    params,
                    n_stripes,
                    color_per_stripe,
                    texture_per_stripe,
                    t_stats,
                    c_stats
                )

                # Compute distance matrix with adaptive alpha
                distance_matrix = compute_distance_matrix_fast(
                    query_color,
                    query_texture,
                    gallery_color,
                    gallery_texture,
                    alpha=query_alphas,  # Use adaptive alpha per query
                    metric=distance_metric
                )

                # Evaluate metrics
                results = evaluate_reid(
                    distance_matrix,
                    query_ids,
                    gallery_ids,
                    query_cams,
                    gallery_cams,
                    max_rank=20,
                    remove_junk=(query_cams is not None and gallery_cams is not None)
                )

                # Get score based on metric
                if metric == 'rank1':
                    score = results['rank1']
                elif metric == 'rank5':
                    score = results['rank5']
                elif metric == 'map':
                    score = results.get('mAP', 0.0)
                else:
                    score = results.get('rank1', 0.0)

                # Debug: show logit and alpha statistics for first few combinations
                # NOTE: Evaluation uses ALL query images, this is just for debugging
                if verbose and current_combination <= 5:
                    # Compute logits for a sample to see what's happening (just for debugging)
                    logits = []
                    n_sample = min(10, query_color.shape[0])
                    for i in range(n_sample):
                        # Compute T and C for this image
                        T_vals = []
                        for s in range(n_stripes):
                            start_idx = s * texture_per_stripe
                            end_idx = start_idx + texture_per_stripe - 2
                            gabor_stripe = query_texture[i, start_idx:end_idx]
                            total_energy = np.sum(gabor_stripe) + 1e-10
                            n_high_freq = int(len(gabor_stripe) * 0.5)
                            high_freq_energy = np.sum(gabor_stripe[:n_high_freq])
                            T_vals.append(high_freq_energy / total_energy)
                        T = np.mean(T_vals)
                        
                        C_vals = []
                        for s in range(n_stripes):
                            start_idx = s * color_per_stripe
                            end_idx = start_idx + color_per_stripe
                            hist_stripe = query_color[i, start_idx:end_idx]
                            hist_stripe = hist_stripe / (hist_stripe.sum() + 1e-10)
                            C_vals.append(entropy(hist_stripe + 1e-10))
                        C = np.mean(C_vals)
                        
                        # Z-score normalization
                        T_norm = (T - t_stats['mean']) / (t_stats['std'] + 1e-10)
                        C_norm = (C - c_stats['mean']) / (c_stats['std'] + 1e-10)
                        logit = a1 * T_norm - a2 * C_norm + b
                        logits.append(logit)
                    
                    logits = np.array(logits)
                    alpha_min = float(np.min(query_alphas))
                    alpha_max = float(np.max(query_alphas))
                    alpha_mean = float(np.mean(query_alphas))
                    alpha_std = float(np.std(query_alphas))
                    logit_mean = float(np.mean(logits))
                    logit_std = float(np.std(logits))
                    print(f"  [DEBUG - sample of {n_sample} images] Logit stats: mean={logit_mean:.3f}, std={logit_std:.3f}")
                    print(f"  [ALL {query_color.shape[0]} query images] Alpha range: [{alpha_min:.3f}, {alpha_max:.3f}], "
                          f"mean={alpha_mean:.3f}, std={alpha_std:.3f}")

                if verbose and current_combination % 10 == 0:
                    alpha_min = float(np.min(query_alphas))
                    alpha_max = float(np.max(query_alphas))
                    alpha_mean = float(np.mean(query_alphas))
                    print(f"Progress: {current_combination}/{total_combinations} | "
                          f"Params: a1={a1:.1f}, a2={a2:.1f}, b={b:.1f} | "
                          f"Score: {score:.4f} | "
                          f"Alpha range: [{alpha_min:.3f}, {alpha_max:.3f}] (mean={alpha_mean:.3f})")

                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    if verbose:
                        alpha_min = float(np.min(query_alphas))
                        alpha_max = float(np.max(query_alphas))
                        alpha_mean = float(np.mean(query_alphas))
                        print(f"  ✓ New best! Params: {best_params}, {metric}={best_score:.4f} "
                              f"(alpha: [{alpha_min:.3f}, {alpha_max:.3f}], mean={alpha_mean:.3f})")

    if verbose:
        print(f"\nBest parameters: {best_params}")
        print(f"Best {metric}: {best_score:.4f}")

    return best_params