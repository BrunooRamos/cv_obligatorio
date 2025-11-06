"""
CMC and mAP metrics for Person Re-Identification evaluation
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


def compute_cmc(
    distance_matrix: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    query_cams: Optional[np.ndarray] = None,
    gallery_cams: Optional[np.ndarray] = None,
    max_rank: int = 50,
    remove_junk: bool = True
) -> np.ndarray:
    """
    Compute Cumulative Matching Characteristic (CMC) curve.

    Args:
        distance_matrix: Distance matrix (n_query, n_gallery)
        query_ids: Person IDs for queries (n_query,)
        gallery_ids: Person IDs for gallery (n_gallery,)
        query_cams: Camera IDs for queries (optional)
        gallery_cams: Camera IDs for gallery (optional)
        max_rank: Maximum rank to compute (default 50)
        remove_junk: Remove same camera images for Market-1501

    Returns:
        CMC curve (max_rank,)
    """
    n_query = distance_matrix.shape[0]

    # Initialize CMC
    cmc = np.zeros(max_rank)

    for q_idx in range(n_query):
        q_id = query_ids[q_idx]
        q_cam = query_cams[q_idx] if query_cams is not None else None

        # Get distances for this query
        distances = distance_matrix[q_idx]

        # Create validity mask
        valid_mask = np.ones(len(distances), dtype=bool)

        # Find good gallery images (same ID, different camera)
        good_indices = (gallery_ids == q_id)

        if remove_junk and q_cam is not None and gallery_cams is not None:
            # Remove same camera images (junk)
            same_cam_mask = (gallery_cams == q_cam)
            junk_mask = same_cam_mask & (gallery_ids == q_id)

            # Also remove different IDs from same camera if needed
            # For Market-1501, we typically only remove same ID + same camera
            valid_mask &= ~junk_mask

            # Update good indices to exclude junk
            good_indices = good_indices & ~junk_mask

        if not np.any(good_indices):
            # No valid matches for this query
            continue

        # Set invalid gallery images to max distance
        invalid_distances = distances.copy()
        invalid_distances[~valid_mask] = np.inf

        # Sort gallery by distance
        sorted_indices = np.argsort(invalid_distances)

        # Find ranks of good matches
        good_sorted = good_indices[sorted_indices]

        # Find first occurrence of a good match
        if np.any(good_sorted):
            first_match_rank = np.where(good_sorted)[0][0]

            # Update CMC (all ranks >= first match get +1)
            if first_match_rank < max_rank:
                cmc[first_match_rank:] += 1

    # Normalize by number of queries
    cmc = cmc / n_query

    return cmc


def compute_map(
    distance_matrix: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    query_cams: Optional[np.ndarray] = None,
    gallery_cams: Optional[np.ndarray] = None,
    remove_junk: bool = True
) -> float:
    """
    Compute mean Average Precision (mAP).

    Args:
        distance_matrix: Distance matrix (n_query, n_gallery)
        query_ids: Person IDs for queries (n_query,)
        gallery_ids: Person IDs for gallery (n_gallery,)
        query_cams: Camera IDs for queries (optional)
        gallery_cams: Camera IDs for gallery (optional)
        remove_junk: Remove same camera images for Market-1501

    Returns:
        mAP score
    """
    n_query = distance_matrix.shape[0]

    ap_list = []

    for q_idx in range(n_query):
        ap = compute_single_ap(
            distance_matrix[q_idx],
            query_ids[q_idx],
            gallery_ids,
            query_cams[q_idx] if query_cams is not None else None,
            gallery_cams,
            remove_junk
        )

        if ap is not None:
            ap_list.append(ap)

    if len(ap_list) == 0:
        return 0.0

    return float(np.mean(ap_list))


def compute_single_ap(
    distances: np.ndarray,
    query_id: int,
    gallery_ids: np.ndarray,
    query_cam: Optional[int] = None,
    gallery_cams: Optional[np.ndarray] = None,
    remove_junk: bool = True
) -> Optional[float]:
    """
    Compute Average Precision for a single query.

    Args:
        distances: Distances to gallery images (n_gallery,)
        query_id: Person ID of query
        gallery_ids: Person IDs for gallery
        query_cam: Camera ID of query
        gallery_cams: Camera IDs for gallery
        remove_junk: Remove same camera images

    Returns:
        Average Precision or None if no valid matches
    """
    # Find matches
    matches = (gallery_ids == query_id)

    # Create junk mask (same ID, same camera)
    junk_mask = np.zeros(len(distances), dtype=bool)

    if remove_junk and query_cam is not None and gallery_cams is not None:
        same_cam = (gallery_cams == query_cam)
        junk_mask = same_cam & matches

    # Good matches (same ID, different camera or no junk removal)
    good_matches = matches & ~junk_mask

    if not np.any(good_matches):
        return None

    # Sort by distance
    sorted_indices = np.argsort(distances)
    matches_sorted = good_matches[sorted_indices]
    junk_sorted = junk_mask[sorted_indices]

    # Remove junk from consideration
    valid_sorted = ~junk_sorted
    matches_valid = matches_sorted[valid_sorted]

    # Compute precision at each recall level
    n_good = np.sum(matches_valid)

    if n_good == 0:
        return None

    # Cumulative true positives
    tp = np.cumsum(matches_valid)

    # Precision = TP / (TP + FP) = TP / rank
    ranks = np.arange(1, len(matches_valid) + 1)
    precision = tp / ranks

    # Average precision: mean of precisions at recall points
    ap = np.sum(precision[matches_valid]) / n_good

    return float(ap)


def evaluate_reid(
    distance_matrix: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    query_cams: Optional[np.ndarray] = None,
    gallery_cams: Optional[np.ndarray] = None,
    max_rank: int = 50,
    remove_junk: bool = True
) -> Dict[str, float]:
    """
    Complete evaluation: CMC + mAP.

    Args:
        distance_matrix: Distance matrix (n_query, n_gallery)
        query_ids: Person IDs for queries
        gallery_ids: Person IDs for gallery
        query_cams: Camera IDs for queries (optional)
        gallery_cams: Camera IDs for gallery (optional)
        max_rank: Maximum rank for CMC
        remove_junk: Remove same camera images

    Returns:
        Dictionary with metrics
    """
    # Compute CMC
    cmc = compute_cmc(
        distance_matrix,
        query_ids,
        gallery_ids,
        query_cams,
        gallery_cams,
        max_rank,
        remove_junk
    )

    # Compute mAP
    mAP = compute_map(
        distance_matrix,
        query_ids,
        gallery_ids,
        query_cams,
        gallery_cams,
        remove_junk
    )

    results = {
        'mAP': mAP,
        'rank1': cmc[0],
        'rank5': cmc[4] if len(cmc) > 4 else 0.0,
        'rank10': cmc[9] if len(cmc) > 9 else 0.0,
        'rank20': cmc[19] if len(cmc) > 19 else 0.0,
        'cmc': cmc
    }

    return results


def evaluate_market1501(
    distance_matrix: np.ndarray,
    query_data: List[Dict],
    gallery_data: List[Dict],
    max_rank: int = 50
) -> Dict[str, float]:
    """
    Evaluate on Market-1501 dataset.

    Args:
        distance_matrix: Distance matrix (n_query, n_gallery)
        query_data: List of query dictionaries (with person_id, camera_id)
        gallery_data: List of gallery dictionaries
        max_rank: Maximum rank for CMC

    Returns:
        Dictionary with metrics
    """
    query_ids = np.array([d['person_id'] for d in query_data])
    gallery_ids = np.array([d['person_id'] for d in gallery_data])
    query_cams = np.array([d['camera_id'] for d in query_data])
    gallery_cams = np.array([d['camera_id'] for d in gallery_data])

    return evaluate_reid(
        distance_matrix,
        query_ids,
        gallery_ids,
        query_cams,
        gallery_cams,
        max_rank,
        remove_junk=True
    )


def evaluate_ilids_vid(
    distance_matrix: np.ndarray,
    query_data: List[Dict],
    gallery_data: List[Dict],
    max_rank: int = 50
) -> Dict[str, float]:
    """
    Evaluate on iLIDS-VID dataset.

    Args:
        distance_matrix: Distance matrix (n_query, n_gallery)
        query_data: List of query sequence dictionaries
        gallery_data: List of gallery sequence dictionaries
        max_rank: Maximum rank for CMC

    Returns:
        Dictionary with CMC metrics (no mAP for video)
    """
    query_ids = np.array([d['person_id'] for d in query_data])
    gallery_ids = np.array([d['person_id'] for d in gallery_data])
    query_cams = np.array([d['camera_id'] for d in query_data])
    gallery_cams = np.array([d['camera_id'] for d in gallery_data])

    # For iLIDS-VID, typically report CMC only
    cmc = compute_cmc(
        distance_matrix,
        query_ids,
        gallery_ids,
        query_cams,
        gallery_cams,
        max_rank,
        remove_junk=False  # Usually no junk removal for video datasets
    )

    results = {
        'rank1': cmc[0],
        'rank5': cmc[4] if len(cmc) > 4 else 0.0,
        'rank10': cmc[9] if len(cmc) > 9 else 0.0,
        'rank20': cmc[19] if len(cmc) > 19 else 0.0,
        'cmc': cmc
    }

    return results


def print_results(results: Dict[str, float], dataset: str = "Market-1501"):
    """
    Print evaluation results in a nice format.

    Args:
        results: Dictionary with metrics
        dataset: Dataset name
    """
    print("=" * 60)
    print(f"Evaluation Results - {dataset}")
    print("=" * 60)

    if 'mAP' in results:
        print(f"mAP: {results['mAP']:.4f} ({results['mAP']*100:.2f}%)")

    print(f"\nCMC Scores:")
    print(f"  Rank-1:  {results['rank1']:.4f} ({results['rank1']*100:.2f}%)")
    print(f"  Rank-5:  {results['rank5']:.4f} ({results['rank5']*100:.2f}%)")
    print(f"  Rank-10: {results['rank10']:.4f} ({results['rank10']*100:.2f}%)")
    print(f"  Rank-20: {results['rank20']:.4f} ({results['rank20']*100:.2f}%)")
    print("=" * 60)


def save_results(results: Dict[str, float], filepath: str):
    """
    Save evaluation results to file.

    Args:
        results: Dictionary with metrics
        filepath: Path to save file (.npz)
    """
    np.savez(filepath, **results)


def load_results(filepath: str) -> Dict[str, float]:
    """
    Load evaluation results from file.

    Args:
        filepath: Path to .npz file

    Returns:
        Dictionary with metrics
    """
    data = np.load(filepath, allow_pickle=True)
    results = {key: float(data[key]) if data[key].ndim == 0 else data[key]
               for key in data.files}
    return results
