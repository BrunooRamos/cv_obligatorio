"""
Distance computation for BILP features
"""

import numpy as np
from typing import Optional, Union
from scipy.spatial.distance import cdist
from .gpu_utils import cdist_gpu


def l1_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute L1 (Manhattan) distance between two vectors.

    Args:
        x, y: Feature vectors

    Returns:
        L1 distance
    """
    return np.sum(np.abs(x - y))


def l2_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute L2 (Euclidean) distance between two vectors.

    Args:
        x, y: Feature vectors

    Returns:
        L2 distance
    """
    return np.sqrt(np.sum((x - y) ** 2))


def chi_square_distance(x: np.ndarray, y: np.ndarray, epsilon: float = 1e-10) -> float:
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


def bhattacharyya_distance(x: np.ndarray, y: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute Bhattacharyya distance between two histograms.

    Args:
        x, y: Normalized histogram vectors
        epsilon: Small constant to avoid numerical issues

    Returns:
        Bhattacharyya distance
    """
    bc = np.sum(np.sqrt(x * y + epsilon))  # Bhattacharyya coefficient
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
        dist_fn = l1_distance
    elif metric == 'l2':
        dist_fn = l2_distance
    elif metric == 'chi2':
        dist_fn = chi_square_distance
    elif metric == 'bhattacharyya':
        dist_fn = bhattacharyya_distance
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


def compute_distance_matrix_fast_with_hog(
    query_color: np.ndarray,
    query_texture: np.ndarray,
    query_hog: np.ndarray,
    gallery_color: np.ndarray,
    gallery_texture: np.ndarray,
    gallery_hog: np.ndarray,
    alpha: float = 0.5,
    alpha_hog: float = 0.33,
    metric: str = 'cityblock',
    device: Optional = None
) -> np.ndarray:
    """
    Fast computation of distance matrix using color, texture, and HOG features.

    Args:
        query_color: Query color features (n_query, color_dim)
        query_texture: Query texture features (n_query, texture_dim)
        query_hog: Query HOG features (n_query, hog_dim)
        gallery_color: Gallery color features (n_gallery, color_dim)
        gallery_texture: Gallery texture features (n_gallery, texture_dim)
        gallery_hog: Gallery HOG features (n_gallery, hog_dim)
        alpha: Gating weight for texture vs color (default: 0.5)
        alpha_hog: Weight for HOG features in 3-way fusion (default: 0.33)
        metric: Distance metric ('cityblock', 'euclidean', 'cosine')
        device: GPU device (CuPy) or None for CPU

    Returns:
        Distance matrix (n_query, n_gallery)

    Note:
        Final distance = alpha_hog * D_hog + (1 - alpha_hog) * [alpha * D_texture + (1 - alpha) * D_color]
    """
    # Compute separate distance matrices (use GPU if available)
    if device is not None:
        dist_color = cdist_gpu(query_color, gallery_color, metric=metric, device=device)
        dist_texture = cdist_gpu(query_texture, gallery_texture, metric=metric, device=device)
        dist_hog = cdist_gpu(query_hog, gallery_hog, metric=metric, device=device)
    else:
        dist_color = cdist(query_color, gallery_color, metric=metric)
        dist_texture = cdist(query_texture, gallery_texture, metric=metric)
        dist_hog = cdist(query_hog, gallery_hog, metric=metric)

    # Combine color and texture with gating
    if isinstance(alpha, (int, float)):
        # Scalar alpha: same weight for all queries
        dist_color_texture = alpha * dist_texture + (1 - alpha) * dist_color
    else:
        # Adaptive alpha: different weight per query
        alpha = np.asarray(alpha)
        if alpha.ndim == 0:
            alpha = alpha.item()
            dist_color_texture = alpha * dist_texture + (1 - alpha) * dist_color
        else:
            # Reshape alpha to (n_query, 1) for broadcasting
            alpha = alpha.reshape(-1, 1)
            dist_color_texture = alpha * dist_texture + (1 - alpha) * dist_color

    # Combine with HOG
    dist_matrix = alpha_hog * dist_hog + (1 - alpha_hog) * dist_color_texture

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
                        dist_tensor[i, j, s] = chi_square_distance(
                            query_stripe[i], gallery_stripe[j]
                        )
                    elif metric == 'bhattacharyya':
                        dist_tensor[i, j, s] = bhattacharyya_distance(
                            query_stripe[i], gallery_stripe[j]
                        )

    return dist_tensor


def rank_gallery(
    distance_vector: np.ndarray,
    gallery_ids: np.ndarray,
    query_id: int,
    query_cam: Optional[int] = None,
    gallery_cams: Optional[np.ndarray] = None,
    remove_same_cam: bool = True
) -> np.ndarray:
    """
    Rank gallery images for a single query.

    Args:
        distance_vector: Distances to all gallery images (n_gallery,)
        gallery_ids: Person IDs for gallery images
        query_id: Person ID of query
        query_cam: Camera ID of query (optional)
        gallery_cams: Camera IDs for gallery images (optional)
        remove_same_cam: Remove same camera images (for Market-1501)

    Returns:
        Ranked indices (sorted by distance)
    """
    # Create mask for valid gallery images
    valid_mask = np.ones(len(distance_vector), dtype=bool)

    # Remove same ID images (if any in gallery)
    # Actually, for evaluation we keep them but mark them

    # Remove same camera images if requested
    if remove_same_cam and query_cam is not None and gallery_cams is not None:
        same_cam_mask = (gallery_cams == query_cam)
        same_id_same_cam = same_cam_mask & (gallery_ids == query_id)
        valid_mask &= ~same_id_same_cam

    # Get valid distances
    valid_distances = distance_vector.copy()
    valid_distances[~valid_mask] = np.inf

    # Sort by distance
    ranked_indices = np.argsort(valid_distances)

    return ranked_indices


def compute_ap(
    distance_vector: np.ndarray,
    gallery_ids: np.ndarray,
    query_id: int,
    query_cam: Optional[int] = None,
    gallery_cams: Optional[np.ndarray] = None
) -> float:
    """
    Compute Average Precision for a single query.

    Args:
        distance_vector: Distances to all gallery images
        gallery_ids: Person IDs for gallery images
        query_id: Person ID of query
        query_cam: Camera ID of query
        gallery_cams: Camera IDs for gallery images

    Returns:
        Average Precision
    """
    # Rank gallery
    ranked_indices = rank_gallery(
        distance_vector,
        gallery_ids,
        query_id,
        query_cam,
        gallery_cams,
        remove_same_cam=True
    )

    # Find matches
    matches = (gallery_ids[ranked_indices] == query_id)

    # Remove same camera matches (junk images)
    if query_cam is not None and gallery_cams is not None:
        same_cam = (gallery_cams[ranked_indices] == query_cam)
        matches = matches & ~same_cam

    # Compute AP
    if not np.any(matches):
        return 0.0

    # Cumulative precision at each recall point
    cum_tp = np.cumsum(matches)
    cum_fp = np.cumsum(~matches)

    recall = cum_tp / np.sum(matches)
    precision = cum_tp / (cum_tp + cum_fp)

    # Average precision
    ap = np.sum(precision[matches]) / np.sum(matches)

    return float(ap)
