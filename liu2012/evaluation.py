"""
Liu et al. 2012 evaluation protocol for iLIDS-VID.

Protocol:
- Select p=50 persons randomly
- For each trial (10 trials):
  - Gallery: 1 random image per person (50 images)
  - Probes: all remaining images from those 50 persons
  - Compute CMC
- Average CMC over 10 trials
- Use L1 distance (cityblock)
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy.spatial.distance import cdist
from collections import defaultdict


def evaluate_liu2012_ilids_vid(
    features: np.ndarray,
    metadata: Dict[str, List],
    n_persons: int = 50,
    n_trials: int = 10,
    max_rank: int = 50,
    metric: str = 'cityblock',
    random_seed: int = 42,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Evaluate using Liu et al. 2012 protocol for iLIDS-VID.
    
    Args:
        features: Feature matrix (n_images, 2784)
        metadata: Metadata dict with 'person_ids', 'camera_ids', etc.
        n_persons: Number of persons to select (default 50)
        n_trials: Number of trials to average over (default 10)
        max_rank: Maximum rank for CMC
        metric: Distance metric (default 'cityblock' for L1)
        random_seed: Random seed for reproducibility
        verbose: Print progress
    
    Returns:
        Dictionary with averaged CMC metrics
    """
    person_ids = np.array(metadata['person_ids'])
    camera_ids = np.array(metadata['camera_ids'])
    
    # Group images by person ID
    person_to_indices = defaultdict(list)
    for idx, person_id in enumerate(person_ids):
        person_to_indices[person_id].append(idx)
    
    unique_persons = sorted(person_to_indices.keys())
    
    if len(unique_persons) < n_persons:
        raise ValueError(
            f"Dataset has only {len(unique_persons)} persons, "
            f"but protocol requires {n_persons} persons"
        )
    
    # Initialize random seed
    rng = np.random.RandomState(random_seed)
    
    # Store CMC curves from all trials
    all_cmc_curves = []
    
    for trial in range(n_trials):
        if verbose:
            print(f"Trial {trial + 1}/{n_trials}...")
        
        # Sample n_persons randomly (without replacement)
        selected_persons = rng.choice(unique_persons, size=n_persons, replace=False)
        
        # Build gallery: 1 random image per selected person
        gallery_indices = []
        probe_indices = []
        
        for person_id in selected_persons:
            person_images = person_to_indices[person_id]
            
            if len(person_images) < 2:
                # Skip persons with only 1 image (need at least 1 for gallery + 1 for probe)
                continue
            
            # Randomly select 1 image for gallery
            gallery_idx = rng.choice(person_images)
            gallery_indices.append(gallery_idx)
            
            # All other images go to probes
            for img_idx in person_images:
                if img_idx != gallery_idx:
                    probe_indices.append(img_idx)
        
        if len(gallery_indices) == 0 or len(probe_indices) == 0:
            if verbose:
                print(f"Warning: Trial {trial + 1} has no valid gallery/probe split, skipping")
            continue
        
        # Extract features for gallery and probes
        gallery_features = features[gallery_indices]
        probe_features = features[probe_indices]
        
        gallery_person_ids = person_ids[gallery_indices]
        probe_person_ids = person_ids[probe_indices]
        
        # Compute distance matrix
        distance_matrix = cdist(probe_features, gallery_features, metric=metric)
        
        # Compute CMC for this trial
        cmc = compute_cmc_liu2012(
            distance_matrix,
            probe_person_ids,
            gallery_person_ids,
            max_rank=max_rank
        )
        
        all_cmc_curves.append(cmc)
    
    if len(all_cmc_curves) == 0:
        raise RuntimeError("No valid trials completed. Check dataset and parameters.")
    
    # Average CMC over all trials
    avg_cmc = np.mean(all_cmc_curves, axis=0)
    
    results = {
        'rank1': avg_cmc[0],
        'rank5': avg_cmc[4] if len(avg_cmc) > 4 else 0.0,
        'rank10': avg_cmc[9] if len(avg_cmc) > 9 else 0.0,
        'rank20': avg_cmc[19] if len(avg_cmc) > 19 else 0.0,
        'cmc': avg_cmc,
        'n_trials': len(all_cmc_curves),
        'n_persons': n_persons,
    }
    
    return results


def compute_cmc_liu2012(
    distance_matrix: np.ndarray,
    probe_ids: np.ndarray,
    gallery_ids: np.ndarray,
    max_rank: int = 50
) -> np.ndarray:
    """
    Compute CMC curve for Liu et al. 2012 protocol.
    
    Args:
        distance_matrix: Distance matrix (n_probes, n_gallery)
        probe_ids: Person IDs for probes
        gallery_ids: Person IDs for gallery
        max_rank: Maximum rank
    
    Returns:
        CMC curve (max_rank,)
    """
    n_probes = distance_matrix.shape[0]
    
    # Initialize CMC
    cmc = np.zeros(max_rank)
    
    for p_idx in range(n_probes):
        p_id = probe_ids[p_idx]
        
        # Get distances for this probe
        distances = distance_matrix[p_idx]
        
        # Find matches (same person ID)
        matches = (gallery_ids == p_id)
        
        if not np.any(matches):
            # No matches for this probe
            continue
        
        # Sort gallery by distance
        sorted_indices = np.argsort(distances)
        
        # Find ranks of matches
        matches_sorted = matches[sorted_indices]
        
        # Find first occurrence of a match
        if np.any(matches_sorted):
            first_match_rank = np.where(matches_sorted)[0][0]
            
            # Update CMC (all ranks >= first match get +1)
            if first_match_rank < max_rank:
                cmc[first_match_rank:] += 1
    
    # Normalize by number of probes
    cmc = cmc / n_probes
    
    return cmc

