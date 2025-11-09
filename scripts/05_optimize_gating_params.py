"""Optimize gating parameters (a1, a2, b) for BILP on iLIDS-VID."""

import argparse
import os
import sys
import json
from typing import Dict, List

sys.path.append('/app')

import numpy as np

from bilp.utils import load_features
from bilp.gating import optimize_gating_params, compute_gating_weights_batch
from bilp.distance import compute_distance_matrix_fast
from eval.cmc_map import evaluate_ilids_vid, print_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize gating parameters (a1, a2, b) for BILP on iLIDS-VID",
    )
    parser.add_argument(
        '--query-features',
        type=str,
        default='data/features/ilidsvid_query.npz',
        help='Path to the .npz file containing query sequence features.',
    )
    parser.add_argument(
        '--gallery-features',
        type=str,
        default='data/features/ilidsvid_gallery.npz',
        help='Path to the .npz file containing gallery sequence features.',
    )
    parser.add_argument(
        '--output-params',
        type=str,
        default='data/gating_params_ilidsvid.json',
        help='Path to save optimized parameters.',
    )
    parser.add_argument(
        '--metric',
        choices=['rank1', 'rank5', 'rank10'],
        default='rank1',
        help='Metric to optimize (rank1, rank5, or rank10).',
    )
    parser.add_argument(
        '--distance-metric',
        choices=['cityblock', 'euclidean', 'cosine'],
        default='cityblock',
        help='Distance metric for features.',
    )
    parser.add_argument(
        '--param-grid',
        type=str,
        default=None,
        help='JSON file with custom parameter grid (optional).',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress.',
    )
    return parser.parse_args()


def build_metadata_list(metadata: Dict[str, List]) -> List[Dict]:
    required_keys = ['person_ids', 'camera_ids', 'filenames', 'paths']
    for key in required_keys:
        if key not in metadata:
            raise KeyError(f"Metadata missing key '{key}'. Available keys: {list(metadata.keys())}")

    entries: List[Dict] = []
    for person_id, camera_id, filename, path in zip(
        metadata['person_ids'],
        metadata['camera_ids'],
        metadata['filenames'],
        metadata['paths'],
    ):
        entries.append(
            {
                'person_id': int(person_id),
                'camera_id': int(camera_id),
                'filename': filename,
                'path': path,
            }
        )

    return entries


def analyze_t_c_scales(
    color_features: np.ndarray,
    texture_features: np.ndarray,
    n_stripes: int = 6,
    color_per_stripe: int = 272,
    texture_per_stripe: int = 42
) -> Dict[str, float]:
    """
    Analyze the scales of T (texture complexity) and C (chromatic entropy).
    
    Returns statistics to help understand if they need normalization.
    """
    from scipy.stats import entropy
    
    T_values = []
    C_values = []
    
    n_images = color_features.shape[0]
    for i in range(min(100, n_images)):  # Sample first 100 images
        # T: texture complexity
        T_vals = []
        for s in range(n_stripes):
            start_idx = s * texture_per_stripe
            end_idx = start_idx + texture_per_stripe - 2
            gabor_stripe = texture_features[i, start_idx:end_idx]
            
            total_energy = np.sum(gabor_stripe) + 1e-10
            n_high_freq = int(len(gabor_stripe) * 0.5)
            high_freq_energy = np.sum(gabor_stripe[:n_high_freq])
            T_vals.append(high_freq_energy / total_energy)
        T_values.append(np.mean(T_vals))
        
        # C: chromatic entropy
        C_vals = []
        for s in range(n_stripes):
            start_idx = s * color_per_stripe
            end_idx = start_idx + color_per_stripe
            hist_stripe = color_features[i, start_idx:end_idx]
            hist_stripe = hist_stripe / (hist_stripe.sum() + 1e-10)
            C_vals.append(entropy(hist_stripe + 1e-10))
        C_values.append(np.mean(C_vals))
    
    T_values = np.array(T_values)
    C_values = np.array(C_values)
    
    stats = {
        'T_mean': float(np.mean(T_values)),
        'T_std': float(np.std(T_values)),
        'T_min': float(np.min(T_values)),
        'T_max': float(np.max(T_values)),
        'C_mean': float(np.mean(C_values)),
        'C_std': float(np.std(C_values)),
        'C_min': float(np.min(C_values)),
        'C_max': float(np.max(C_values)),
    }
    
    return stats


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.query_features):
        raise FileNotFoundError(f"Query features file not found: {args.query_features}")
    if not os.path.exists(args.gallery_features):
        raise FileNotFoundError(f"Gallery features file not found: {args.gallery_features}")

    print("=" * 60)
    print("Optimizing Gating Parameters for iLIDS-VID")
    print("=" * 60)

    # Load features
    print(f"\nLoading query features from {args.query_features}")
    query_color, query_texture, query_metadata = load_features(args.query_features)
    if query_metadata is None:
        raise ValueError('Query features file does not contain metadata.')

    print(f"Loading gallery features from {args.gallery_features}")
    gallery_color, gallery_texture, gallery_metadata = load_features(args.gallery_features)
    if gallery_metadata is None:
        raise ValueError('Gallery features file does not contain metadata.')

    print(f"\nQuery: {query_color.shape[0]} images")
    print(f"Gallery: {gallery_color.shape[0]} images")

    # Build metadata
    query_data = build_metadata_list(query_metadata)
    gallery_data = build_metadata_list(gallery_metadata)

    query_ids = np.array([d['person_id'] for d in query_data])
    gallery_ids = np.array([d['person_id'] for d in gallery_data])
    query_cams = np.array([d['camera_id'] for d in query_data])
    gallery_cams = np.array([d['camera_id'] for d in gallery_data])

    # Analyze T and C scales
    print("\n" + "=" * 60)
    print("Analyzing T (texture complexity) and C (chromatic entropy) scales...")
    print("=" * 60)
    
    stats = analyze_t_c_scales(query_color, query_texture)
    print(f"\nT (Texture Complexity) Statistics:")
    print(f"  Mean: {stats['T_mean']:.4f}, Std: {stats['T_std']:.4f}")
    print(f"  Range: [{stats['T_min']:.4f}, {stats['T_max']:.4f}]")
    print(f"\nC (Chromatic Entropy) Statistics:")
    print(f"  Mean: {stats['C_mean']:.4f}, Std: {stats['C_std']:.4f}")
    print(f"  Range: [{stats['C_min']:.4f}, {stats['C_max']:.4f}]")
    
    scale_ratio = stats['C_mean'] / stats['T_mean'] if stats['T_mean'] > 0 else 1.0
    print(f"\nScale ratio (C/T): {scale_ratio:.4f}")
    if abs(scale_ratio - 1.0) > 0.5:
        print(f"  ⚠ Warning: T and C are in different scales!")
        print(f"  Consider adjusting a1 and a2 accordingly.")

    # Load custom parameter grid if provided
    param_grid = None
    if args.param_grid and os.path.exists(args.param_grid):
        print(f"\nLoading custom parameter grid from {args.param_grid}")
        with open(args.param_grid, 'r') as f:
            param_grid = json.load(f)

    # Optimize parameters
    print("\n" + "=" * 60)
    print("Starting parameter optimization...")
    print("=" * 60)

    best_params = optimize_gating_params(
        query_color=query_color,
        query_texture=query_texture,
        query_ids=query_ids,
        gallery_color=gallery_color,
        gallery_texture=gallery_texture,
        gallery_ids=gallery_ids,
        query_cams=query_cams,
        gallery_cams=gallery_cams,
        param_grid=param_grid,
        metric=args.metric,
        distance_metric=args.distance_metric,
        verbose=args.verbose
    )

    # Evaluate with best parameters
    print("\n" + "=" * 60)
    print("Evaluating with optimized parameters...")
    print("=" * 60)

    # Use the same statistics for evaluation
    from bilp.gating import compute_gating_weights_batch
    
    # Compute T and C statistics (same as in optimization)
    from scipy.stats import entropy
    
    T_all = []
    C_all = []
    
    n_sample = min(100, query_color.shape[0])
    for i in range(n_sample):
        T_vals = []
        for s in range(6):  # n_stripes
            start_idx = s * 42  # texture_per_stripe
            end_idx = start_idx + 42 - 2
            gabor_stripe = query_texture[i, start_idx:end_idx]
            total_energy = np.sum(gabor_stripe) + 1e-10
            n_high_freq = int(len(gabor_stripe) * 0.5)
            high_freq_energy = np.sum(gabor_stripe[:n_high_freq])
            T_vals.append(high_freq_energy / total_energy)
        T_all.append(np.mean(T_vals))
        
        C_vals = []
        for s in range(6):  # n_stripes
            start_idx = s * 272  # color_per_stripe
            end_idx = start_idx + 272
            hist_stripe = query_color[i, start_idx:end_idx]
            hist_stripe = hist_stripe / (hist_stripe.sum() + 1e-10)
            C_vals.append(entropy(hist_stripe + 1e-10))
        C_all.append(np.mean(C_vals))
    
    T_all = np.array(T_all)
    C_all = np.array(C_all)
    
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

    query_alphas = compute_gating_weights_batch(
        query_color,
        query_texture,
        best_params,
        t_stats=t_stats,
        c_stats=c_stats
    )

    distance_matrix = compute_distance_matrix_fast(
        query_color,
        query_texture,
        gallery_color,
        gallery_texture,
        alpha=query_alphas,
        metric=args.distance_metric
    )

    results = evaluate_ilids_vid(
        distance_matrix,
        query_data,
        gallery_data,
        max_rank=20
    )

    print_results(results, dataset=f'iLIDS-VID (optimized params)')
    print(f"\nOptimized parameters: {best_params}")

    # Save parameters
    output_dir = os.path.dirname(args.output_params)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_params, 'w') as f:
        json.dump({
            'a1': best_params['a1'],
            'a2': best_params['a2'],
            'b': best_params['b'],
            'metric': args.metric,
            'distance_metric': args.distance_metric,
            'results': {
                'rank1': float(results['rank1']),
                'rank5': float(results['rank5']),
                'rank10': float(results['rank10']),
            },
            't_c_stats': stats,
            'normalization': {
                't_stats': t_stats,
                'c_stats': c_stats
            }
        }, f, indent=2)

    print(f"\nSaved optimized parameters to {args.output_params}")


if __name__ == '__main__':
    main()

