"""
End-to-end evaluation on Market-1501 subset
"""

import sys
sys.path.append('/app')

import numpy as np
import time
from eval.loaders import load_market1501
from eval.cmc_map import evaluate_market1501, print_results, save_results
from bilp import extract_bilp_batch, compute_distance_matrix_fast
import argparse


def eval_market1501_subset(
    dataset_path: str,
    gallery_size: int = 5000,
    alpha: float = 0.5,
    save_features: bool = True,
    results_path: str = 'data/results_subset.npz'
):
    """
    Evaluate BILP on Market-1501 subset.

    Args:
        dataset_path: Path to Market-1501 dataset
        gallery_size: Number of gallery images (None for all)
        alpha: Gating weight for color/texture fusion
        save_features: Whether to save extracted features
        results_path: Path to save results
    """
    print("=" * 80)
    print("MARKET-1501 EVALUATION (SUBSET)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Gallery size: {gallery_size if gallery_size else 'ALL'}")
    print(f"  Alpha (gating): {alpha}")
    print(f"  Dataset: {dataset_path}")
    print()

    # Load data
    print("-" * 80)
    print("Step 1: Loading data...")
    print("-" * 80)

    start_time = time.time()

    print("\nLoading query set...")
    query_data = load_market1501(dataset_path, split='query', return_images=True)
    print(f"  Loaded {len(query_data)} queries")

    print("\nLoading test/gallery set...")
    gallery_data = load_market1501(dataset_path, split='test', return_images=True)
    print(f"  Loaded {len(gallery_data)} gallery images")

    # Sample gallery if needed
    if gallery_size is not None and gallery_size < len(gallery_data):
        print(f"\nSampling {gallery_size} gallery images...")
        indices = np.random.choice(len(gallery_data), gallery_size, replace=False)
        gallery_data = [gallery_data[i] for i in indices]
        print(f"  Sampled gallery size: {len(gallery_data)}")

    load_time = time.time() - start_time
    print(f"\nData loading complete in {load_time:.2f}s")

    # Extract features
    print("\n" + "-" * 80)
    print("Step 2: Extracting BILP features...")
    print("-" * 80)

    start_time = time.time()

    print("\nExtracting query features...")
    query_images = [d['image'] for d in query_data]
    query_color, query_texture = extract_bilp_batch(
        query_images,
        normalize=True,
        verbose=True
    )
    print(f"  Query features: color={query_color.shape}, texture={query_texture.shape}")

    print("\nExtracting gallery features...")
    gallery_images = [d['image'] for d in gallery_data]
    gallery_color, gallery_texture = extract_bilp_batch(
        gallery_images,
        normalize=True,
        verbose=True
    )
    print(f"  Gallery features: color={gallery_color.shape}, texture={gallery_texture.shape}")

    extraction_time = time.time() - start_time
    print(f"\nFeature extraction complete in {extraction_time:.2f}s")
    print(f"  Average time per image: {extraction_time / (len(query_data) + len(gallery_data)):.3f}s")

    # Save features if requested
    if save_features:
        features_path = 'data/features_subset.npz'
        print(f"\nSaving features to {features_path}...")
        np.savez_compressed(
            features_path,
            query_color=query_color,
            query_texture=query_texture,
            gallery_color=gallery_color,
            gallery_texture=gallery_texture
        )
        print("  Features saved!")

    # Compute distance matrix
    print("\n" + "-" * 80)
    print("Step 3: Computing distance matrix...")
    print("-" * 80)

    start_time = time.time()

    print(f"\nComputing distances (alpha={alpha})...")
    distance_matrix = compute_distance_matrix_fast(
        query_color,
        query_texture,
        gallery_color,
        gallery_texture,
        alpha=alpha,
        metric='cityblock'
    )

    distance_time = time.time() - start_time
    print(f"  Distance matrix: {distance_matrix.shape}")
    print(f"  Computed in {distance_time:.2f}s")

    # Evaluate
    print("\n" + "-" * 80)
    print("Step 4: Evaluating...")
    print("-" * 80)

    start_time = time.time()

    results = evaluate_market1501(
        distance_matrix,
        query_data,
        gallery_data,
        max_rank=50
    )

    eval_time = time.time() - start_time
    print(f"\nEvaluation complete in {eval_time:.2f}s")

    # Display results
    print("\n" + "=" * 80)
    print_results(results, dataset=f"Market-1501 Subset (Gallery={len(gallery_data)})")
    print("=" * 80)

    # Save results
    print(f"\nSaving results to {results_path}...")
    save_results(results, results_path)

    # Summary
    total_time = load_time + extraction_time + distance_time + eval_time

    print("\n" + "=" * 80)
    print("TIMING SUMMARY")
    print("=" * 80)
    print(f"  Data loading:        {load_time:7.2f}s ({100*load_time/total_time:5.1f}%)")
    print(f"  Feature extraction:  {extraction_time:7.2f}s ({100*extraction_time/total_time:5.1f}%)")
    print(f"  Distance computation:{distance_time:7.2f}s ({100*distance_time/total_time:5.1f}%)")
    print(f"  Evaluation:          {eval_time:7.2f}s ({100*eval_time/total_time:5.1f}%)")
    print(f"  {'Total:':<21}{total_time:7.2f}s")
    print("=" * 80)

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate BILP on Market-1501 subset')
    parser.add_argument('--dataset', type=str, default='/datasets/Market-1501-v15.09.15',
                        help='Path to Market-1501 dataset')
    parser.add_argument('--gallery-size', type=int, default=5000,
                        help='Number of gallery images (None for all)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Gating weight (0=color only, 1=texture only)')
    parser.add_argument('--no-save-features', action='store_true',
                        help='Do not save extracted features')
    parser.add_argument('--results-path', type=str, default='data/results_subset.npz',
                        help='Path to save results')

    args = parser.parse_args()

    results = eval_market1501_subset(
        dataset_path=args.dataset,
        gallery_size=args.gallery_size,
        alpha=args.alpha,
        save_features=not args.no_save_features,
        results_path=args.results_path
    )

    print("\nEvaluation complete!")
    print(f"\nFinal Results:")
    print(f"  mAP:     {results['mAP']:.4f} ({results['mAP']*100:.2f}%)")
    print(f"  Rank-1:  {results['rank1']:.4f} ({results['rank1']*100:.2f}%)")
    print(f"  Rank-5:  {results['rank5']:.4f} ({results['rank5']*100:.2f}%)")
    print(f"  Rank-10: {results['rank10']:.4f} ({results['rank10']*100:.2f}%)")


if __name__ == '__main__':
    main()
