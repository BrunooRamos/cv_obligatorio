"""
Sweep over different alpha and alpha_hog values to find optimal HOG fusion.
"""

import argparse
import sys
import numpy as np

sys.path.append('/app')

from bilp.utils import load_features
from bilp.distance import compute_distance_matrix_fast, compute_distance_matrix_fast_with_hog
from bilp.gpu_utils import get_device
from eval.cmc_map import evaluate_ilids_vid


def main():
    parser = argparse.ArgumentParser(description="Sweep HOG fusion parameters")
    parser.add_argument('--query-features', type=str, required=True)
    parser.add_argument('--gallery-features', type=str, required=True)
    parser.add_argument('--metric', type=str, default='cityblock')
    parser.add_argument('--use-gpu', action='store_true')
    args = parser.parse_args()

    # Load features
    print("Loading features...")
    query_color, query_texture, query_hog, query_metadata = load_features(args.query_features)
    gallery_color, gallery_texture, gallery_hog, gallery_metadata = load_features(args.gallery_features)

    use_hog = (query_hog is not None and gallery_hog is not None and
               len(query_hog) > 0 and len(gallery_hog) > 0)

    if not use_hog:
        print("ERROR: HOG features not found in input files")
        return

    # Build metadata
    query_data = []
    for pid, cid, fname, fpath in zip(
        query_metadata['person_ids'],
        query_metadata['camera_ids'],
        query_metadata['filenames'],
        query_metadata['paths']
    ):
        query_data.append({
            'person_id': int(pid),
            'camera_id': int(cid),
            'filename': fname,
            'path': fpath
        })

    gallery_data = []
    for pid, cid, fname, fpath in zip(
        gallery_metadata['person_ids'],
        gallery_metadata['camera_ids'],
        gallery_metadata['filenames'],
        gallery_metadata['paths']
    ):
        gallery_data.append({
            'person_id': int(pid),
            'camera_id': int(cid),
            'filename': fname,
            'path': fpath
        })

    # Get GPU device if requested
    is_gpu, device = get_device(args.use_gpu)
    if is_gpu:
        print("Using GPU for distance computation")

    # Test different alpha_hog values with fixed alpha=0.5
    print("\n" + "="*80)
    print("Parameter Sweep: HOG Fusion Weight (alpha=0.5 fixed)")
    print("="*80)
    print(f"{'alpha_hog':>10} | {'Rank-1':>8} | {'Rank-5':>8} | {'Rank-10':>8} | {'Rank-20':>8}")
    print("-"*80)

    best_rank1 = 0.0
    best_alpha_hog = 0.0

    for alpha_hog in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        dist_matrix = compute_distance_matrix_fast_with_hog(
            query_color, query_texture, query_hog,
            gallery_color, gallery_texture, gallery_hog,
            alpha=0.5, alpha_hog=alpha_hog,
            metric=args.metric, device=device
        )

        results = evaluate_ilids_vid(dist_matrix, query_data, gallery_data, max_rank=20)

        r1 = results['cmc'][0]
        r5 = results['cmc'][4]
        r10 = results['cmc'][9]
        r20 = results['cmc'][19]

        print(f"{alpha_hog:10.1f} | {r1:8.4f} | {r5:8.4f} | {r10:8.4f} | {r20:8.4f}")

        if r1 > best_rank1:
            best_rank1 = r1
            best_alpha_hog = alpha_hog

    print("="*80)
    print(f"Best Rank-1: {best_rank1:.4f} at alpha_hog={best_alpha_hog:.1f}")
    print("="*80)

    # Now sweep alpha with best alpha_hog
    print("\n" + "="*80)
    print(f"Parameter Sweep: Color/Texture Weight (alpha_hog={best_alpha_hog:.1f} fixed)")
    print("="*80)
    print(f"{'alpha':>10} | {'Rank-1':>8} | {'Rank-5':>8} | {'Rank-10':>8} | {'Rank-20':>8}")
    print("-"*80)

    best_rank1_alpha = 0.0
    best_alpha = 0.0

    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        dist_matrix = compute_distance_matrix_fast_with_hog(
            query_color, query_texture, query_hog,
            gallery_color, gallery_texture, gallery_hog,
            alpha=alpha, alpha_hog=best_alpha_hog,
            metric=args.metric, device=device
        )

        results = evaluate_ilids_vid(dist_matrix, query_data, gallery_data, max_rank=20)

        r1 = results['cmc'][0]
        r5 = results['cmc'][4]
        r10 = results['cmc'][9]
        r20 = results['cmc'][19]

        print(f"{alpha:10.1f} | {r1:8.4f} | {r5:8.4f} | {r10:8.4f} | {r20:8.4f}")

        if r1 > best_rank1_alpha:
            best_rank1_alpha = r1
            best_alpha = alpha

    print("="*80)
    print(f"Best Rank-1: {best_rank1_alpha:.4f} at alpha={best_alpha:.1f}, alpha_hog={best_alpha_hog:.1f}")
    print("="*80)

    # Test baseline (no HOG)
    print("\n" + "="*80)
    print("Baseline Comparison (no HOG)")
    print("="*80)

    dist_matrix_no_hog = compute_distance_matrix_fast(
        query_color, query_texture,
        gallery_color, gallery_texture,
        alpha=0.5, metric=args.metric, device=device
    )

    results_no_hog = evaluate_ilids_vid(dist_matrix_no_hog, query_data, gallery_data, max_rank=20)

    print(f"Color + Texture only (alpha=0.5):")
    print(f"  Rank-1:  {results_no_hog['cmc'][0]:.4f}")
    print(f"  Rank-5:  {results_no_hog['cmc'][4]:.4f}")
    print(f"  Rank-10: {results_no_hog['cmc'][9]:.4f}")
    print(f"  Rank-20: {results_no_hog['cmc'][19]:.4f}")
    print("="*80)


if __name__ == '__main__':
    main()
