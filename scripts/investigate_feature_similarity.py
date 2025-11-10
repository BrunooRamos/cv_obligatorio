"""Investigate why BILP features are so similar between different people."""

import argparse
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import cv2
from scipy.spatial.distance import cdist
from scipy.stats import entropy

from bilp.utils import load_features, extract_bilp_descriptor, load_calibrated_color_ranges
from eval.loaders import load_ilids_vid


def analyze_histogram_distribution(features, name="Features"):
    """Analyze how concentrated or spread out the histogram values are."""
    print(f"\n{'='*60}")
    print(f"Histogram Distribution Analysis: {name}")
    print(f"{'='*60}")
    
    # Check sparsity (how many bins are non-zero)
    n_samples, n_features = features.shape
    non_zero_per_sample = (features > 1e-10).sum(axis=1)
    
    print(f"\nSparsity (non-zero bins per sample):")
    print(f"  Mean: {non_zero_per_sample.mean():.1f}/{n_features}")
    print(f"  Std: {non_zero_per_sample.std():.1f}")
    print(f"  Min: {non_zero_per_sample.min()}/{n_features}")
    print(f"  Max: {non_zero_per_sample.max()}/{n_features}")
    print(f"  Percentage: {non_zero_per_sample.mean()/n_features*100:.1f}%")
    
    # Check entropy of histograms (higher = more uniform, lower = more concentrated)
    entropies = []
    for i in range(min(100, n_samples)):  # Sample first 100
        hist = features[i]
        hist_norm = hist / (hist.sum() + 1e-10)
        hist_norm = hist_norm[hist_norm > 0]
        ent = entropy(hist_norm)
        entropies.append(ent)
    
    print(f"\nHistogram Entropy (sample of {len(entropies)}):")
    print(f"  Mean: {np.mean(entropies):.4f}")
    print(f"  Std: {np.std(entropies):.4f}")
    print(f"  Range: [{np.min(entropies):.4f}, {np.max(entropies):.4f}]")
    print(f"  (Higher = more uniform, Lower = more concentrated)")
    
    # Check if most values are very small
    small_values = (features < 0.001).sum(axis=1)
    print(f"\nVery small values (< 0.001):")
    print(f"  Mean: {small_values.mean():.1f}/{n_features} ({small_values.mean()/n_features*100:.1f}%)")


def compare_features_before_after_normalization():
    """Compare features before and after normalization."""
    print(f"\n{'='*60}")
    print("Comparing Features Before and After Normalization")
    print(f"{'='*60}")
    
    # Load a few images
    sequences = load_ilids_vid(
        dataset_path='datasets/iLIDS-VID',
        num_frames=1,  # Just one frame per sequence
        sampling_strategy='uniform',
        return_images=True,
        verbose=False,
    )
    
    # Get first 10 sequences
    images = [seq['frames'][0] for seq in sequences[:10]]
    
    u_range, v_range = load_calibrated_color_ranges('data/color_ranges_market.json')
    color_params = {
        'n_bins_uv': 16,
        'n_bins_lum': 16,
        'u_range': u_range,
        'v_range': v_range,
    }
    texture_params = {
        'n_scales': 5,
        'n_orientations': 8,
    }
    
    # Extract WITHOUT normalization
    color_features_no_norm = []
    texture_features_no_norm = []
    
    for img in images:
        descriptor = extract_bilp_descriptor(
            img,
            n_stripes=6,
            color_params=color_params,
            texture_params=texture_params,
            normalize=False,  # NO normalization
            normalize_method='l1',
        )
        color_features_no_norm.append(descriptor['color'])
        texture_features_no_norm.append(descriptor['texture'])
    
    color_features_no_norm = np.array(color_features_no_norm)
    texture_features_no_norm = np.array(texture_features_no_norm)
    
    # Extract WITH normalization
    color_features_norm = []
    texture_features_norm = []
    
    for img in images:
        descriptor = extract_bilp_descriptor(
            img,
            n_stripes=6,
            color_params=color_params,
            texture_params=texture_params,
            normalize=True,  # WITH normalization
            normalize_method='l1',
        )
        color_features_norm.append(descriptor['color'])
        texture_features_norm.append(descriptor['texture'])
    
    color_features_norm = np.array(color_features_norm)
    texture_features_norm = np.array(texture_features_norm)
    
    # Compare distances
    print("\n=== Color Features ===")
    dist_no_norm = cdist(color_features_no_norm, color_features_no_norm, metric='cityblock')
    dist_norm = cdist(color_features_norm, color_features_norm, metric='cityblock')
    
    # Remove diagonal
    mask = ~np.eye(len(dist_no_norm), dtype=bool)
    dist_no_norm_no_diag = dist_no_norm[mask]
    dist_norm_no_diag = dist_norm[mask]
    
    print(f"Without normalization:")
    print(f"  Mean distance: {dist_no_norm_no_diag.mean():.6f}")
    print(f"  Std distance: {dist_no_norm_no_diag.std():.6f}")
    print(f"  Range: [{dist_no_norm_no_diag.min():.6f}, {dist_no_norm_no_diag.max():.6f}]")
    
    print(f"\nWith normalization:")
    print(f"  Mean distance: {dist_norm_no_diag.mean():.6f}")
    print(f"  Std distance: {dist_norm_no_diag.std():.6f}")
    print(f"  Range: [{dist_norm_no_diag.min():.6f}, {dist_norm_no_diag.max():.6f}]")
    
    print(f"\nRatio (norm/no_norm): {dist_norm_no_diag.mean() / dist_no_norm_no_diag.mean():.3f}")
    
    print("\n=== Texture Features ===")
    dist_no_norm = cdist(texture_features_no_norm, texture_features_no_norm, metric='cityblock')
    dist_norm = cdist(texture_features_norm, texture_features_norm, metric='cityblock')
    
    dist_no_norm_no_diag = dist_no_norm[mask]
    dist_norm_no_diag = dist_norm[mask]
    
    print(f"Without normalization:")
    print(f"  Mean distance: {dist_no_norm_no_diag.mean():.6f}")
    print(f"  Std distance: {dist_no_norm_no_diag.std():.6f}")
    print(f"  Range: [{dist_no_norm_no_diag.min():.6f}, {dist_no_norm_no_diag.max():.6f}]")
    
    print(f"\nWith normalization:")
    print(f"  Mean distance: {dist_norm_no_diag.mean():.6f}")
    print(f"  Std distance: {dist_norm_no_diag.std():.6f}")
    print(f"  Range: [{dist_norm_no_diag.min():.6f}, {dist_norm_no_diag.max():.6f}]")
    
    print(f"\nRatio (norm/no_norm): {dist_norm_no_diag.mean() / dist_no_norm_no_diag.mean():.3f}")


def analyze_pooling_effect():
    """Analyze if pooling frames is causing information loss."""
    print(f"\n{'='*60}")
    print("Analyzing Pooling Effect (Single Frame vs Multiple Frames)")
    print(f"{'='*60}")
    
    sequences = load_ilids_vid(
        dataset_path='datasets/iLIDS-VID',
        num_frames=10,
        sampling_strategy='uniform',
        return_images=True,
        verbose=False,
    )
    
    # Get first 5 sequences with multiple frames
    selected_sequences = [seq for seq in sequences if len(seq.get('frames', [])) >= 10][:5]
    
    u_range, v_range = load_calibrated_color_ranges('data/color_ranges_market.json')
    color_params = {
        'n_bins_uv': 16,
        'n_bins_lum': 16,
        'u_range': u_range,
        'v_range': v_range,
    }
    texture_params = {
        'n_scales': 5,
        'n_orientations': 8,
    }
    
    # Extract features from single frames
    single_frame_features_color = []
    single_frame_features_texture = []
    
    for seq in selected_sequences:
        frames = seq['frames'][:10]
        for frame in frames:
            descriptor = extract_bilp_descriptor(
                frame,
                n_stripes=6,
                color_params=color_params,
                texture_params=texture_params,
                normalize=True,
                normalize_method='l1',
            )
            single_frame_features_color.append(descriptor['color'])
            single_frame_features_texture.append(descriptor['texture'])
    
    single_frame_features_color = np.array(single_frame_features_color)
    single_frame_features_texture = np.array(single_frame_features_texture)
    
    # Extract features with pooling (average)
    pooled_features_color = []
    pooled_features_texture = []
    
    for seq in selected_sequences:
        frames = seq['frames'][:10]
        frame_features_color = []
        frame_features_texture = []
        
        for frame in frames:
            descriptor = extract_bilp_descriptor(
                frame,
                n_stripes=6,
                color_params=color_params,
                texture_params=texture_params,
                normalize=True,
                normalize_method='l1',
            )
            frame_features_color.append(descriptor['color'])
            frame_features_texture.append(descriptor['texture'])
        
        # Average pooling
        pooled_color = np.mean(frame_features_color, axis=0)
        pooled_texture = np.mean(frame_features_texture, axis=0)
        
        pooled_features_color.append(pooled_color)
        pooled_features_texture.append(pooled_texture)
    
    pooled_features_color = np.array(pooled_features_color)
    pooled_features_texture = np.array(pooled_features_texture)
    
    # Compare variation
    print("\n=== Color Features ===")
    print(f"Single frames (n={len(single_frame_features_color)}):")
    print(f"  Std across samples: {np.std(single_frame_features_color, axis=0).mean():.6f}")
    print(f"  Mean pairwise distance: {cdist(single_frame_features_color[:20], single_frame_features_color[:20], metric='cityblock')[~np.eye(20, dtype=bool)].mean():.6f}")
    
    print(f"\nPooled (n={len(pooled_features_color)}):")
    print(f"  Std across samples: {np.std(pooled_features_color, axis=0).mean():.6f}")
    print(f"  Mean pairwise distance: {cdist(pooled_features_color, pooled_features_color, metric='cityblock')[~np.eye(len(pooled_features_color), dtype=bool)].mean():.6f}")
    
    print("\n=== Texture Features ===")
    print(f"Single frames (n={len(single_frame_features_texture)}):")
    print(f"  Std across samples: {np.std(single_frame_features_texture, axis=0).mean():.6f}")
    print(f"  Mean pairwise distance: {cdist(single_frame_features_texture[:20], single_frame_features_texture[:20], metric='cityblock')[~np.eye(20, dtype=bool)].mean():.6f}")
    
    print(f"\nPooled (n={len(pooled_features_texture)}):")
    print(f"  Std across samples: {np.std(pooled_features_texture, axis=0).mean():.6f}")
    print(f"  Mean pairwise distance: {cdist(pooled_features_texture, pooled_features_texture, metric='cityblock')[~np.eye(len(pooled_features_texture), dtype=bool)].mean():.6f}")


def analyze_same_person_vs_different_person():
    """Compare features from same person vs different people."""
    print(f"\n{'='*60}")
    print("Same Person vs Different People Comparison")
    print(f"{'='*60}")
    
    # Load features
    query_color, query_texture, query_meta = load_features('data/features/ilidsvid_query.npz')
    gallery_color, gallery_texture, gallery_meta = load_features('data/features/ilidsvid_gallery.npz')
    
    query_ids = np.array(query_meta['person_ids'])
    gallery_ids = np.array(gallery_meta['person_ids'])
    
    # Find pairs
    same_person_distances_color = []
    same_person_distances_texture = []
    different_person_distances_color = []
    different_person_distances_texture = []
    
    n_samples = min(50, len(query_color))
    
    for i in range(n_samples):
        query_id = query_ids[i]
        
        # Same person matches
        same_person_mask = (gallery_ids == query_id)
        if same_person_mask.sum() > 0:
            same_person_indices = np.where(same_person_mask)[0]
            for j in same_person_indices[:3]:  # Sample up to 3 matches
                dist_color = np.sum(np.abs(query_color[i] - gallery_color[j]))
                dist_texture = np.sum(np.abs(query_texture[i] - gallery_texture[j]))
                same_person_distances_color.append(dist_color)
                same_person_distances_texture.append(dist_texture)
        
        # Different person (sample random)
        different_person_mask = (gallery_ids != query_id)
        if different_person_mask.sum() > 0:
            different_person_indices = np.where(different_person_mask)[0]
            # Sample 3 random different people
            sample_indices = np.random.choice(different_person_indices, size=min(3, len(different_person_indices)), replace=False)
            for j in sample_indices:
                dist_color = np.sum(np.abs(query_color[i] - gallery_color[j]))
                dist_texture = np.sum(np.abs(query_texture[i] - gallery_texture[j]))
                different_person_distances_color.append(dist_color)
                different_person_distances_texture.append(dist_texture)
    
    print("\n=== Color Features ===")
    print(f"Same person distances:")
    print(f"  Mean: {np.mean(same_person_distances_color):.6f}")
    print(f"  Std: {np.std(same_person_distances_color):.6f}")
    print(f"  Range: [{np.min(same_person_distances_color):.6f}, {np.max(same_person_distances_color):.6f}]")
    
    print(f"\nDifferent person distances:")
    print(f"  Mean: {np.mean(different_person_distances_color):.6f}")
    print(f"  Std: {np.std(different_person_distances_color):.6f}")
    print(f"  Range: [{np.min(different_person_distances_color):.6f}, {np.max(different_person_distances_color):.6f}]")
    
    ratio = np.mean(different_person_distances_color) / np.mean(same_person_distances_color)
    print(f"\nRatio (different/same): {ratio:.3f}")
    if ratio < 1.0:
        print("  ⚠ PROBLEM: Different people are CLOSER than same person!")
    else:
        print(f"  ✓ Different people are {ratio:.2f}x farther (good)")
    
    print("\n=== Texture Features ===")
    print(f"Same person distances:")
    print(f"  Mean: {np.mean(same_person_distances_texture):.6f}")
    print(f"  Std: {np.std(same_person_distances_texture):.6f}")
    print(f"  Range: [{np.min(same_person_distances_texture):.6f}, {np.max(same_person_distances_texture):.6f}]")
    
    print(f"\nDifferent person distances:")
    print(f"  Mean: {np.mean(different_person_distances_texture):.6f}")
    print(f"  Std: {np.std(different_person_distances_texture):.6f}")
    print(f"  Range: [{np.min(different_person_distances_texture):.6f}, {np.max(different_person_distances_texture):.6f}]")
    
    ratio = np.mean(different_person_distances_texture) / np.mean(same_person_distances_texture)
    print(f"\nRatio (different/same): {ratio:.3f}")
    if ratio < 1.0:
        print("  ⚠ PROBLEM: Different people are CLOSER than same person!")
    else:
        print(f"  ✓ Different people are {ratio:.2f}x farther (good)")


def main():
    parser = argparse.ArgumentParser(
        description="Investigate why BILP features are so similar"
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
    print("BILP Feature Similarity Investigation")
    print("="*60)
    
    # Load features
    query_color, query_texture, query_meta = load_features(args.query_features)
    gallery_color, gallery_texture, gallery_meta = load_features(args.gallery_features)
    
    # 1. Analyze histogram distributions
    analyze_histogram_distribution(query_color, "Query Color Features")
    analyze_histogram_distribution(query_texture, "Query Texture Features")
    
    # 2. Compare before/after normalization
    compare_features_before_after_normalization()
    
    # 3. Analyze pooling effect
    analyze_pooling_effect()
    
    # 4. Compare same vs different person
    analyze_same_person_vs_different_person()


if __name__ == '__main__':
    main()

