"""
Distance computation for BILP features.

By default, we decided to use the L1 distance as it is more robust to outliers and noise.
We also tried l2 distance, chi2 distance, and bhattacharyya distance.
"""

import numpy as np
from typing import Optional, Union
from scipy.spatial.distance import cdist
from .gpu_utils import cdist_gpu

def _l1_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute L1 (Manhattan) distance between two vectors.

    Args:
        x, y: Feature vectors

    Returns:
        L1 distance
    """
    return np.sum(np.abs(x - y))


def _l2_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute L2 (Euclidean) distance between two vectors.

    Args:
        x, y: Feature vectors

    Returns:
        L2 distance
    """
    return np.sqrt(np.sum((x - y) ** 2))


def _chi_square_distance(x: np.ndarray, y: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute Chi-square distance between two histograms.

    Args:
        x, y: Histogram vectors
        epsilon: Small constant to avoid division by zero

    Returns:
        Chi-square distance
    """
    numerator = (x - y) ** 2
    denominator = x + y + epsilon
    return np.sum(numerator / denominator)


def _bhattacharyya_distance(x: np.ndarray, y: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute Bhattacharyya distance between two histograms.

    Args:
        x, y: Normalized histogram vectors
        epsilon: Small constant to avoid numerical issues

    Returns:
        Bhattacharyya distance
    """
    bc = np.sum(np.sqrt(x * y + epsilon))  
    return -np.log(bc + epsilon)


def bilp_distance(
    query_color: np.ndarray,
    query_texture: np.ndarray,
    gallery_color: np.ndarray,
    gallery_texture: np.ndarray,
    alpha: Union[float, np.ndarray] = 0.5,
    metric: str = 'l1'
) -> float:
    """
    Compute BILP distance with adaptive gating.

    d = alpha * d_texture + (1 - alpha) * d_color

    Args:
        query_color: Query color features
        query_texture: Query texture features
        gallery_color: Gallery color features
        gallery_texture: Gallery texture features
        alpha: Gating weight (scalar or per-stripe weights)
        metric: Distance metric ('l1', 'l2', 'chi2', 'bhattacharyya')

    Returns:
        Combined distance
    """
    # Select distance function
    if metric == 'l1':
        dist_fn = _l1_distance
    elif metric == 'l2':
        dist_fn = _l2_distance
    elif metric == 'chi2':
        dist_fn = _chi_square_distance
    elif metric == 'bhattacharyya':
        dist_fn = _bhattacharyya_distance
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Compute individual distances
    d_color = dist_fn(query_color, gallery_color)
    d_texture = dist_fn(query_texture, gallery_texture)

    # Combine with gating
    if isinstance(alpha, (int, float)):
        # Scalar alpha
        distance = alpha * d_texture + (1 - alpha) * d_color
    else:
        # Per-stripe alpha (not implemented for now, use mean)
        alpha_mean = np.mean(alpha)
        distance = alpha_mean * d_texture + (1 - alpha_mean) * d_color

    return float(distance)


def compute_distance_matrix(
    query_color: np.ndarray,
    query_texture: np.ndarray,
    gallery_color: np.ndarray,
    gallery_texture: np.ndarray,
    alpha: Union[float, np.ndarray] = 0.5,
    metric: str = 'l1',
    verbose: bool = False
) -> np.ndarray:
    """
    Compute distance matrix between query and gallery sets.

    Args:
        query_color: Query color features (n_query, color_dim)
        query_texture: Query texture features (n_query, texture_dim)
        gallery_color: Gallery color features (n_gallery, color_dim)
        gallery_texture: Gallery texture features (n_gallery, texture_dim)
        alpha: Gating weight
        metric: Distance metric
        verbose: Print progress

    Returns:
        Distance matrix (n_query, n_gallery)
    """
    n_query = query_color.shape[0]
    n_gallery = gallery_color.shape[0]

    dist_matrix = np.zeros((n_query, n_gallery))

    for i in range(n_query):
        if verbose and (i + 1) % 100 == 0:
            print(f"Computing distances for query {i + 1}/{n_query}")

        for j in range(n_gallery):
            dist_matrix[i, j] = bilp_distance(
                query_color[i],
                query_texture[i],
                gallery_color[j],
                gallery_texture[j],
                alpha=alpha,
                metric=metric
            )

    return dist_matrix


def compute_distance_matrix_fast(
    query_color: np.ndarray,
    query_texture: np.ndarray,
    gallery_color: np.ndarray,
    gallery_texture: np.ndarray,
    alpha: float = 0.5,
    metric: str = 'cityblock',
    device: Optional = None
) -> np.ndarray:
    """
    Fast computation of distance matrix using scipy or GPU.

    Args:
        query_color: Query color features (n_query, color_dim)
        query_texture: Query texture features (n_query, texture_dim)
        gallery_color: Gallery color features (n_gallery, color_dim)
        gallery_texture: Gallery texture features (n_gallery, texture_dim)
        alpha: Gating weight (scalar only)
        metric: Distance metric ('cityblock', 'euclidean', 'cosine')
        device: GPU device (CuPy) or None for CPU

    Returns:
        Distance matrix (n_query, n_gallery)
    """
    # Compute separate distance matrices (use GPU if available)
    if device is not None:
        dist_color = cdist_gpu(query_color, gallery_color, metric=metric, device=device)
        dist_texture = cdist_gpu(query_texture, gallery_texture, metric=metric, device=device)
    else:
        dist_color = cdist(query_color, gallery_color, metric=metric)
        dist_texture = cdist(query_texture, gallery_texture, metric=metric)

    # Combine with gating
    if isinstance(alpha, (int, float)):
        # Scalar alpha: same weight for all queries
        dist_matrix = alpha * dist_texture + (1 - alpha) * dist_color
    else:
        # Adaptive alpha: different weight per query
        # alpha is (n_query,) array
        alpha = np.asarray(alpha)
        if alpha.ndim == 0:
            alpha = alpha.item()
            dist_matrix = alpha * dist_texture + (1 - alpha) * dist_color
        else:
            # Reshape alpha to (n_query, 1) for broadcasting
            alpha = alpha.reshape(-1, 1)
            dist_matrix = alpha * dist_texture + (1 - alpha) * dist_color

    return dist_matrix


def compute_pairwise_distances_per_stripe(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    n_stripes: int,
    features_per_stripe: int,
    metric: str = 'l1'
) -> np.ndarray:
    """
    Compute distances per stripe and return as matrix.

    Args:
        query_features: Query features (n_query, n_stripes * features_per_stripe)
        gallery_features: Gallery features (n_gallery, n_stripes * features_per_stripe)
        n_stripes: Number of stripes
        features_per_stripe: Features per stripe
        metric: Distance metric

    Returns:
        Distance tensor (n_query, n_gallery, n_stripes)
    """
    n_query = query_features.shape[0]
    n_gallery = gallery_features.shape[0]

    dist_tensor = np.zeros((n_query, n_gallery, n_stripes))

    for s in range(n_stripes):
        start_idx = s * features_per_stripe
        end_idx = (s + 1) * features_per_stripe

        query_stripe = query_features[:, start_idx:end_idx]
        gallery_stripe = gallery_features[:, start_idx:end_idx]

        if metric == 'l1':
            dist_tensor[:, :, s] = cdist(query_stripe, gallery_stripe, metric='cityblock')
        elif metric == 'l2':
            dist_tensor[:, :, s] = cdist(query_stripe, gallery_stripe, metric='euclidean')
        else:
            # For other metrics, use custom computation
            for i in range(n_query):
                for j in range(n_gallery):
                    if metric == 'chi2':
                        dist_tensor[i, j, s] = _chi_square_distance(
                            query_stripe[i], gallery_stripe[j]
                        )
                    elif metric == 'bhattacharyya':
                        dist_tensor[i, j, s] = _bhattacharyya_distance(
                            query_stripe[i], gallery_stripe[j]
                        )

    return dist_tensor
