"""Diagnose why BILP features have low performance on iLIDS-VID."""

import argparse
import os
import sys

sys.path.append('/app')

import numpy as np
from scipy.spatial.distance import cdist

from bilp.utils import load_features
from bilp.distance import compute_distance_matrix_fast
from eval.cmc_map import evaluate_ilids_vid


def analyze_features(color_features, texture_features, name="Features"):
    """Analyze feature statistics."""
    print(f"\n{'='*60}")
    print(f"Analyzing {name}")
    print(f"{'='*60}")
    
    print(f"\nShape: color={color_features.shape}, texture={texture_features.shape}")
    
    # Check for NaN or Inf
    color_nan = np.isnan(color_features).sum()
    color_inf = np.isinf(color_features).sum()
    texture_nan = np.isnan(texture_features).sum()
    texture_inf = np.isinf(texture_features).sum()
    
    print(f"\nNaN/Inf check:")
    print(f"  Color: {color_nan} NaN, {color_inf} Inf")
    print(f"  Texture: {texture_nan} NaN, {texture_inf} Inf")
    
    # Check if features are all zeros
    color_zeros = (color_features == 0).all(axis=1).sum()
    texture_zeros = (texture_features == 0).all(axis=1).sum()
    
    print(f"\nZero vectors:")
    print(f"  Color: {color_zeros}/{color_features.shape[0]} are all zeros")
    print(f"  Texture: {texture_zeros}/{texture_features.shape[0]} are all zeros")
    
    # Check feature variation
    color_std = np.std(color_features, axis=0)
    texture_std = np.std(texture_features, axis=0)
    
    print(f"\nFeature variation (std across samples):")
    print(f"  Color: mean_std={color_std.mean():.6f}, min_std={color_std.min():.6f}, max_std={color_std.max():.6f}")
    print(f"  Texture: mean_std={texture_std.mean():.6f}, min_std={texture_std.min():.6f}, max_std={texture_std.max():.6f}")
    
    # Check if features are normalized
    color_norms = np.linalg.norm(color_features, axis=1)
    texture_norms = np.linalg.norm(texture_features, axis=1)
    
    print(f"\nFeature norms (L2):")
    print(f"  Color: mean={color_norms.mean():.6f}, std={color_norms.std():.6f}, range=[{color_norms.min():.6f}, {color_norms.max():.6f}]")
    print(f"  Texture: mean={texture_norms.mean():.6f}, std={texture_norms.std():.6f}, range=[{texture_norms.min():.6f}, {texture_norms.max():.6f}]")
    
    # Check pairwise distances (sample)
    n_sample = min(20, color_features.shape[0])
    color_sample = color_features[:n_sample]
    texture_sample = texture_features[:n_sample]
    
    color_dist = cdist(color_sample, color_sample, metric='cityblock')
    texture_dist = cdist(texture_sample, texture_sample, metric='cityblock')
    
    # Remove diagonal (self-distances)
    color_dist_no_diag = color_dist[~np.eye(n_sample, dtype=bool)].reshape(n_sample, n_sample-1)
    texture_dist_no_diag = texture_dist[~np.eye(n_sample, dtype=bool)].reshape(n_sample, n_sample-1)
    
    print(f"\nPairwise distances (sample of {n_sample} images):")
    print(f"  Color: mean={color_dist_no_diag.mean():.6f}, std={color_dist_no_diag.std():.6f}, range=[{color_dist_no_diag.min():.6f}, {color_dist_no_diag.max():.6f}]")
    print(f"  Texture: mean={texture_dist_no_diag.mean():.6f}, std={texture_dist_no_diag.std():.6f}, range=[{texture_dist_no_diag.min():.6f}, {texture_dist_no_diag.max():.6f}]")
    
    # Check if distances are too small (features too similar)
    if color_dist_no_diag.mean() < 0.01:
        print(f"  ⚠ WARNING: Color features are very similar (mean distance < 0.01)")
    if texture_dist_no_diag.mean() < 0.01:
        print(f"  ⚠ WARNING: Texture features are very similar (mean distance < 0.01)")


def test_alpha_values(query_color, query_texture, gallery_color, gallery_texture, 
                     query_data, gallery_data):
    """Test different alpha values to see if gating is the problem."""
    print(f"\n{'='*60}")
    print("Testing different alpha values (fixed, not adaptive)")
    print(f"{'='*60}")
    
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    best_alpha = 0.5
    best_rank1 = 0.0
    
    for alpha in alphas:
        distance_matrix = compute_distance_matrix_fast(
            query_color,
            query_texture,
            gallery_color,
            gallery_texture,
            alpha=alpha,
            metric='cityblock'
        )
        
        results = evaluate_ilids_vid(
            distance_matrix,
            query_data,
            gallery_data,
            max_rank=20
        )
        
        rank1 = results['rank1']
        print(f"  Alpha={alpha:.1f}: Rank-1={rank1:.4f} ({rank1*100:.2f}%)")
        
        if rank1 > best_rank1:
            best_rank1 = rank1
            best_alpha = alpha
    
    print(f"\n  Best alpha: {best_alpha:.1f} with Rank-1={best_rank1:.4f} ({best_rank1*100:.2f}%)")
    return best_alpha, best_rank1


def check_matching(query_ids, gallery_ids, distance_matrix):
    """Check if correct matches have lower distances."""
    print(f"\n{'='*60}")
    print("Checking if correct matches have lower distances")
    print(f"{'='*60}")
    
    n_query = len(query_ids)
    correct_distances = []
    incorrect_distances = []
    
    for i in range(n_query):
        query_id = query_ids[i]
        distances = distance_matrix[i]
        
        # Find matches
        matches = (gallery_ids == query_id)
        
        if matches.sum() > 0:
            correct_dist = distances[matches].min()
            incorrect_dist = distances[~matches].min() if (~matches).sum() > 0 else np.inf
            
            correct_distances.append(correct_dist)
            if not np.isinf(incorrect_dist):
                incorrect_distances.append(incorrect_dist)
    
    if correct_distances and incorrect_distances:
        correct_mean = np.mean(correct_distances)
        incorrect_mean = np.mean(incorrect_distances)
        
        print(f"\nCorrect match distances: mean={correct_mean:.6f}, std={np.std(correct_distances):.6f}")
        print(f"Incorrect match distances: mean={incorrect_mean:.6f}, std={np.std(incorrect_distances):.6f}")
        
        if correct_mean < incorrect_mean:
            print(f"  ✓ Correct matches have lower distances (good!)")
            print(f"  Ratio (incorrect/correct): {incorrect_mean/correct_mean:.3f}")
        else:
            print(f"  ✗ PROBLEM: Correct matches have HIGHER distances!")
            print(f"  Ratio (incorrect/correct): {incorrect_mean/correct_mean:.3f}")
    else:
        print("  Could not compute statistics")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose BILP features on iLIDS-VID"
    )
    parser.add_argument(
        '--query-features',
        type=str,
        default='data/features/ilidsvid_query.npz',
        help='Path to query features'
    )
    parser.add_argument(
        '--gallery-features',
        type=str,
        default='data/features/ilidsvid_gallery.npz',
        help='Path to gallery features'
    )
    args = parser.parse_args()
    
    print("="*60)
    print("BILP Features Diagnosis for iLIDS-VID")
    print("="*60)
    
    # Load features
    print(f"\nLoading query features from {args.query_features}")
    query_color, query_texture, query_metadata = load_features(args.query_features)
    
    print(f"Loading gallery features from {args.gallery_features}")
    gallery_color, gallery_texture, gallery_metadata = load_features(args.gallery_features)
    
    if query_metadata is None or gallery_metadata is None:
        raise ValueError("Features must have metadata")
    
    # Build metadata lists
    query_data = []
    for person_id, camera_id in zip(query_metadata['person_ids'], query_metadata['camera_ids']):
        query_data.append({
            'person_id': int(person_id),
            'camera_id': int(camera_id),
            'filename': '',
            'path': ''
        })
    
    gallery_data = []
    for person_id, camera_id in zip(gallery_metadata['person_ids'], gallery_metadata['camera_ids']):
        gallery_data.append({
            'person_id': int(person_id),
            'camera_id': int(camera_id),
            'filename': '',
            'path': ''
        })
    
    query_ids = np.array([d['person_id'] for d in query_data])
    gallery_ids = np.array([d['person_id'] for d in gallery_data])
    
    # Analyze features
    analyze_features(query_color, query_texture, "Query Features")
    analyze_features(gallery_color, gallery_texture, "Gallery Features")
    
    # Test different alpha values
    best_alpha, best_rank1 = test_alpha_values(
        query_color, query_texture,
        gallery_color, gallery_texture,
        query_data, gallery_data
    )
    
    # Check matching with best alpha
    distance_matrix = compute_distance_matrix_fast(
        query_color,
        query_texture,
        gallery_color,
        gallery_texture,
        alpha=best_alpha,
        metric='cityblock'
    )
    
    check_matching(query_ids, gallery_ids, distance_matrix)
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("Final Evaluation with best alpha")
    print(f"{'='*60}")
    
    results = evaluate_ilids_vid(
        distance_matrix,
        query_data,
        gallery_data,
        max_rank=20
    )
    
    print(f"\nRank-1: {results['rank1']:.4f} ({results['rank1']*100:.2f}%)")
    print(f"Rank-5: {results['rank5']:.4f} ({results['rank5']*100:.2f}%)")
    print(f"Rank-10: {results['rank10']:.4f} ({results['rank10']*100:.2f}%)")
    print(f"Rank-20: {results['rank20']:.4f} ({results['rank20']*100:.2f}%)")


if __name__ == '__main__':
    main()

