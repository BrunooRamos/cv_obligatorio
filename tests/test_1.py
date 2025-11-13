"""
Debug script to analyze BILP features on a small subset of iLIDS-VID.

This script:
1. Loads pre-extracted features from .npz files
2. Selects a small subset (5-10 person IDs)
3. Computes distance matrices for:
   - Color only
   - Texture only
   - Combined (weighted)
4. Analyzes same-ID vs different-ID distance distributions
5. Visualizes results with histograms and statistics
6. ADDITIONAL DIAGNOSTICS:
   - Variance analysis per dimension
   - Direct comparison of 2-3 specific IDs
   - Visual inspection of color histograms
"""

import argparse
import os
import sys

sys.path.append('/app')

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from bilp.utils import load_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug BILP features on small iLIDS-VID subset",
    )
    parser.add_argument(
        '--query-features',
        type=str,
        default='data/features/ilidsvid_query.npz',
        help='Path to query features .npz file',
    )
    parser.add_argument(
        '--gallery-features',
        type=str,
        default='data/features/ilidsvid_gallery.npz',
        help='Path to gallery features .npz file',
    )
    parser.add_argument(
        '--num-persons',
        type=int,
        default=10,
        help='Number of person IDs to analyze (default: 10)',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='debug_output',
        help='Directory to save debug plots and results.',
    )
    parser.add_argument(
        '--color-weight',
        type=float,
        default=0.5,
        help='Weight for color features in combined distance (default: 0.5)',
    )
    parser.add_argument(
        '--texture-weight',
        type=float,
        default=0.5,
        help='Weight for texture features in combined distance (default: 0.5)',
    )
    return parser.parse_args()


def analyze_distance_distribution(
    dist_matrix: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    title: str = "Distance Distribution"
):
    """
    Analyze distance distribution between same-ID and different-ID pairs.

    Args:
        dist_matrix: Distance matrix (query x gallery)
        query_ids: Person IDs for query samples
        gallery_ids: Person IDs for gallery samples
        title: Title for the analysis

    Returns:
        Dictionary with statistics and distance lists
    """
    same_id_distances = []
    diff_id_distances = []

    for i, q_id in enumerate(query_ids):
        for j, g_id in enumerate(gallery_ids):
            dist = dist_matrix[i, j]
            if q_id == g_id:
                same_id_distances.append(dist)
            else:
                diff_id_distances.append(dist)

    same_id_distances = np.array(same_id_distances)
    diff_id_distances = np.array(diff_id_distances)

    # Handle case where there might be no matches
    if len(same_id_distances) == 0:
        same_id_distances = np.array([0.0])

    if len(diff_id_distances) == 0:
        diff_id_distances = np.array([1.0])

    stats = {
        'title': title,
        'same_id': {
            'count': len(same_id_distances),
            'mean': np.mean(same_id_distances),
            'median': np.median(same_id_distances),
            'std': np.std(same_id_distances),
            'min': np.min(same_id_distances),
            'max': np.max(same_id_distances),
            'distances': same_id_distances,
        },
        'diff_id': {
            'count': len(diff_id_distances),
            'mean': np.mean(diff_id_distances),
            'median': np.median(diff_id_distances),
            'std': np.std(diff_id_distances),
            'min': np.min(diff_id_distances),
            'max': np.max(diff_id_distances),
            'distances': diff_id_distances,
        }
    }

    return stats


def print_statistics(stats):
    """Print distance statistics in a readable format."""
    print(f"\n{'='*60}")
    print(f"{stats['title']}")
    print(f"{'='*60}")

    print(f"\nSame-ID distances (n={stats['same_id']['count']}):")
    print(f"  Mean:   {stats['same_id']['mean']:.4f}")
    print(f"  Median: {stats['same_id']['median']:.4f}")
    print(f"  Std:    {stats['same_id']['std']:.4f}")
    print(f"  Min:    {stats['same_id']['min']:.4f}")
    print(f"  Max:    {stats['same_id']['max']:.4f}")

    print(f"\nDifferent-ID distances (n={stats['diff_id']['count']}):")
    print(f"  Mean:   {stats['diff_id']['mean']:.4f}")
    print(f"  Median: {stats['diff_id']['median']:.4f}")
    print(f"  Std:    {stats['diff_id']['std']:.4f}")
    print(f"  Min:    {stats['diff_id']['min']:.4f}")
    print(f"  Max:    {stats['diff_id']['max']:.4f}")

    # Calculate separation
    separation = stats['diff_id']['mean'] - stats['same_id']['mean']

    # Avoid division by zero
    combined_std = np.sqrt(stats['same_id']['std']**2 + stats['diff_id']['std']**2)
    if combined_std > 0:
        separation_std = separation / combined_std
    else:
        separation_std = 0.0

    print(f"\nSeparation Analysis:")
    print(f"  Diff mean - Same mean: {separation:.4f}")
    print(f"  Standardized separation: {separation_std:.4f}")
    overlap = max(stats['same_id']['max'] - stats['diff_id']['min'], 0)
    print(f"  Overlap (Same max - Diff min): {overlap:.4f}")

    # Calculate how well separated the distributions are
    if stats['same_id']['max'] < stats['diff_id']['min']:
        print(f"  ✓ Perfect separation! (Same max < Diff min)")
    elif separation > 0:
        print(f"  ✓ Good: Diff distances are higher on average")
    else:
        print(f"  ✗ Poor: Same-ID distances are higher or equal to Diff-ID")


def plot_distance_histograms(
    stats_list,
    output_path: str,
):
    """Plot histograms comparing same-ID vs different-ID distances."""
    n_plots = len(stats_list)
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))

    if n_plots == 1:
        axes = [axes]

    for ax, stats in zip(axes, stats_list):
        same_dist = stats['same_id']['distances']
        diff_dist = stats['diff_id']['distances']

        # Determine bins
        all_dist = np.concatenate([same_dist, diff_dist])
        bins = np.linspace(all_dist.min(), all_dist.max(), 30)

        # Plot histograms
        ax.hist(same_dist, bins=bins, alpha=0.6, label='Same ID', color='green', density=True)
        ax.hist(diff_dist, bins=bins, alpha=0.6, label='Different ID', color='red', density=True)

        # Add vertical lines for means
        ax.axvline(stats['same_id']['mean'], color='green', linestyle='--', linewidth=2,
                   label=f"Same mean: {stats['same_id']['mean']:.3f}")
        ax.axvline(stats['diff_id']['mean'], color='red', linestyle='--', linewidth=2,
                   label=f"Diff mean: {stats['diff_id']['mean']:.3f}")

        ax.set_xlabel('Distance')
        ax.set_ylabel('Density')
        ax.set_title(stats['title'])
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved histogram plot to: {output_path}")
    plt.close()


def plot_distance_matrices(
    dist_color: np.ndarray,
    dist_texture: np.ndarray,
    dist_combined: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    output_path: str,
):
    """Plot distance matrices as heatmaps."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    matrices = [
        (dist_color, "Color Distance"),
        (dist_texture, "Texture Distance"),
        (dist_combined, "Combined Distance"),
    ]

    for ax, (matrix, title) in zip(axes, matrices):
        im = ax.imshow(matrix, cmap='hot', aspect='auto')
        ax.set_xlabel('Gallery Index')
        ax.set_ylabel('Query Index')
        ax.set_title(title)

        # Add colorbar
        plt.colorbar(im, ax=ax)

        # Add person IDs as tick labels (if not too many)
        if len(query_ids) <= 20:
            ax.set_yticks(range(len(query_ids)))
            ax.set_yticklabels([f"P{pid}" for pid in query_ids], fontsize=8)

        if len(gallery_ids) <= 20:
            ax.set_xticks(range(len(gallery_ids)))
            ax.set_xticklabels([f"P{pid}" for pid in gallery_ids], fontsize=8, rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved distance matrix plot to: {output_path}")
    plt.close()


def analyze_feature_variance(
    features: np.ndarray,
    feature_name: str = "Features"
):
    """
    Analyze variance per dimension across all samples.

    Args:
        features: Feature matrix (N x D)
        feature_name: Name for the features (for display)
    """
    print(f"\n{'='*60}")
    print(f"VARIANCE ANALYSIS: {feature_name}")
    print(f"{'='*60}")

    # Variance per dimension
    feat_var = np.var(features, axis=0)

    print(f"\nFeature matrix shape: {features.shape}")
    print(f"Mean variance per dimension: {feat_var.mean():.6f}")
    print(f"Std of variances: {feat_var.std():.6f}")
    print(f"Min variance: {feat_var.min():.6f}")
    print(f"Max variance: {feat_var.max():.6f}")

    # Top and bottom dimensions
    sorted_var = np.sort(feat_var)
    print(f"\nTop 10 most variable dimensions:")
    print(f"  {sorted_var[-10:]}")
    print(f"\nTop 10 least variable dimensions:")
    print(f"  {sorted_var[:10]}")

    # Count near-zero variance dimensions
    near_zero = np.sum(feat_var < 1e-6)
    print(f"\nDimensions with near-zero variance (<1e-6): {near_zero}/{len(feat_var)}")

    return feat_var


def compare_specific_ids(
    features: np.ndarray,
    ids: np.ndarray,
    id_A: int,
    id_B: int,
    feature_name: str = "Features"
):
    """
    Compare features between two specific person IDs.

    Args:
        features: Feature matrix (N x D)
        ids: Person ID array (N,)
        id_A: First person ID
        id_B: Second person ID
        feature_name: Name for the features (for display)
    """
    print(f"\n{'='*60}")
    print(f"DIRECT ID COMPARISON: {feature_name}")
    print(f"Person {id_A} vs Person {id_B}")
    print(f"{'='*60}")

    # Get features for each person
    feat_A = features[ids == id_A]
    feat_B = features[ids == id_B]

    if len(feat_A) == 0:
        print(f"Warning: No features found for ID {id_A}")
        return
    if len(feat_B) == 0:
        print(f"Warning: No features found for ID {id_B}")
        return

    print(f"\nPerson {id_A}: {len(feat_A)} sample(s)")
    print(f"Person {id_B}: {len(feat_B)} sample(s)")

    # L2 norms
    norm_A = np.linalg.norm(feat_A, axis=1)
    norm_B = np.linalg.norm(feat_B, axis=1)

    print(f"\nL2 norm statistics:")
    print(f"  Person {id_A}: mean={norm_A.mean():.4f}, std={norm_A.std():.4f}")
    print(f"  Person {id_B}: mean={norm_B.mean():.4f}, std={norm_B.std():.4f}")

    # Centroids
    cent_A = feat_A.mean(axis=0)
    cent_B = feat_B.mean(axis=0)

    # Inter-person distance (centroid to centroid)
    dist_AB_L1 = np.linalg.norm(cent_A - cent_B, ord=1)
    dist_AB_L2 = np.linalg.norm(cent_A - cent_B, ord=2)

    print(f"\nInter-person distance (centroid A to centroid B):")
    print(f"  L1 distance: {dist_AB_L1:.4f}")
    print(f"  L2 distance: {dist_AB_L2:.4f}")

    # Intra-person distances (if multiple samples)
    if len(feat_A) >= 2:
        # All pairwise distances within A
        dists_AA = cdist(feat_A, feat_A, metric='euclidean')
        # Get upper triangle (excluding diagonal)
        intra_A = dists_AA[np.triu_indices_from(dists_AA, k=1)]
        print(f"\nIntra-person distance for {id_A}:")
        print(f"  Mean: {intra_A.mean():.4f}, Std: {intra_A.std():.4f}")
        print(f"  Range: [{intra_A.min():.4f}, {intra_A.max():.4f}]")
    else:
        print(f"\nIntra-person distance for {id_A}: N/A (only 1 sample)")
        intra_A = None

    if len(feat_B) >= 2:
        dists_BB = cdist(feat_B, feat_B, metric='euclidean')
        intra_B = dists_BB[np.triu_indices_from(dists_BB, k=1)]
        print(f"\nIntra-person distance for {id_B}:")
        print(f"  Mean: {intra_B.mean():.4f}, Std: {intra_B.std():.4f}")
        print(f"  Range: [{intra_B.min():.4f}, {intra_B.max():.4f}]")
    else:
        print(f"\nIntra-person distance for {id_B}: N/A (only 1 sample)")
        intra_B = None

    # Ratio analysis
    if intra_A is not None and len(intra_A) > 0:
        ratio_A = dist_AB_L2 / intra_A.mean()
        print(f"\nRatio (Inter-person / Intra-person for {id_A}): {ratio_A:.2f}")
        if ratio_A < 1.5:
            print(f"  ⚠️  WARNING: Inter-person distance is too close to intra-person!")

    if intra_B is not None and len(intra_B) > 0:
        ratio_B = dist_AB_L2 / intra_B.mean()
        print(f"Ratio (Inter-person / Intra-person for {id_B}): {ratio_B:.2f}")
        if ratio_B < 1.5:
            print(f"  ⚠️  WARNING: Inter-person distance is too close to intra-person!")


def visualize_color_histograms(
    color_features: np.ndarray,
    ids: np.ndarray,
    id_list: list,
    output_path: str,
    n_stripes: int = 6
):
    """
    Visualize color histograms for specific person IDs.

    Args:
        color_features: Color feature matrix (N x D)
        ids: Person ID array (N,)
        id_list: List of person IDs to visualize
        output_path: Path to save the plot
        n_stripes: Number of stripes (default: 6)
    """
    print(f"\n{'='*60}")
    print(f"COLOR HISTOGRAM VISUALIZATION")
    print(f"{'='*60}")

    n_persons = len(id_list)
    fig, axes = plt.subplots(n_persons, n_stripes, figsize=(3*n_stripes, 2*n_persons))

    if n_persons == 1:
        axes = axes.reshape(1, -1)

    # Assuming color features are: 272 features per stripe (256 UV + 16 lum)
    # For 6 stripes: 272 * 6 = 1632
    dims_per_stripe = color_features.shape[1] // n_stripes

    for person_idx, person_id in enumerate(id_list):
        feat = color_features[ids == person_id]

        if len(feat) == 0:
            print(f"Warning: No features for person {person_id}")
            continue

        # Take first sample for this person
        feat_sample = feat[0]

        print(f"\nPerson {person_id}:")
        print(f"  L1 norm: {np.linalg.norm(feat_sample, ord=1):.4f}")
        print(f"  L2 norm: {np.linalg.norm(feat_sample, ord=2):.4f}")
        print(f"  Min value: {feat_sample.min():.6f}")
        print(f"  Max value: {feat_sample.max():.6f}")
        print(f"  Mean value: {feat_sample.mean():.6f}")

        for stripe_idx in range(n_stripes):
            ax = axes[person_idx, stripe_idx]

            # Extract stripe features
            start_idx = stripe_idx * dims_per_stripe
            end_idx = start_idx + dims_per_stripe
            stripe_feat = feat_sample[start_idx:end_idx]

            # Plot as bar chart
            ax.bar(range(len(stripe_feat)), stripe_feat, width=1.0)
            ax.set_title(f"P{person_id} S{stripe_idx+1}", fontsize=8)
            ax.set_ylim([0, max(0.01, stripe_feat.max() * 1.1)])

            if stripe_idx == 0:
                ax.set_ylabel(f"Person {person_id}", fontsize=9)

            if person_idx == n_persons - 1:
                ax.set_xlabel("Bin", fontsize=8)

            ax.tick_params(labelsize=6)

    plt.suptitle("Color Features per Stripe (First Sample)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved color histogram visualization to: {output_path}")
    plt.close()


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*60)
    print("BILP Feature Debug Script - Small iLIDS-VID Subset")
    print("="*60)

    # Load pre-extracted features
    print(f"\nLoading query features from: {args.query_features}")
    color_query, texture_query, meta_query = load_features(args.query_features)

    print(f"Loading gallery features from: {args.gallery_features}")
    color_gallery, texture_gallery, meta_gallery = load_features(args.gallery_features)

    # Get person IDs
    ids_query = np.array(meta_query['person_ids'])
    ids_gallery = np.array(meta_gallery['person_ids'])

    print(f"\nLoaded features:")
    print(f"  Query: {len(ids_query)} sequences, IDs: {sorted(set(ids_query))}")
    print(f"  Gallery: {len(ids_gallery)} sequences, IDs: {sorted(set(ids_gallery))}")

    # Select subset of person IDs
    all_person_ids = sorted(set(ids_query) & set(ids_gallery))  # Only IDs present in both
    selected_ids = all_person_ids[:args.num_persons]

    print(f"\nSelected person IDs ({len(selected_ids)}): {selected_ids}")

    # Filter to subset
    query_mask = np.isin(ids_query, selected_ids)
    gallery_mask = np.isin(ids_gallery, selected_ids)

    color_query_sub = color_query[query_mask]
    texture_query_sub = texture_query[query_mask]
    ids_query_sub = ids_query[query_mask]

    color_gallery_sub = color_gallery[gallery_mask]
    texture_gallery_sub = texture_gallery[gallery_mask]
    ids_gallery_sub = ids_gallery[gallery_mask]

    print(f"\nSubset sizes:")
    print(f"  Query: {len(ids_query_sub)} sequences")
    print(f"  Gallery: {len(ids_gallery_sub)} sequences")
    print(f"  Color features shape: {color_query_sub.shape}")
    print(f"  Texture features shape: {texture_query_sub.shape}")

    # Compute distance matrices
    print("\nComputing distance matrices...")
    dist_color = cdist(color_query_sub, color_gallery_sub, metric='euclidean')
    dist_texture = cdist(texture_query_sub, texture_gallery_sub, metric='euclidean')
    dist_combined = (args.color_weight * dist_color +
                     args.texture_weight * dist_texture)

    print(f"  Distance matrix shape: {dist_color.shape}")

    # Analyze distributions
    print("\nAnalyzing distance distributions...")
    stats_color = analyze_distance_distribution(
        dist_color, ids_query_sub, ids_gallery_sub, "Color Distance"
    )
    stats_texture = analyze_distance_distribution(
        dist_texture, ids_query_sub, ids_gallery_sub, "Texture Distance"
    )
    stats_combined = analyze_distance_distribution(
        dist_combined, ids_query_sub, ids_gallery_sub,
        f"Combined Distance (w_color={args.color_weight}, w_texture={args.texture_weight})"
    )

    # Print statistics
    print_statistics(stats_color)
    print_statistics(stats_texture)
    print_statistics(stats_combined)

    # ========================================================================
    # ADDITIONAL DIAGNOSTICS
    # ========================================================================

    print("\n" + "="*60)
    print("ADDITIONAL DIAGNOSTICS")
    print("="*60)

    # 2.1. Variance analysis per dimension
    # Combine query and gallery for better statistics
    all_color = np.vstack([color_query_sub, color_gallery_sub])
    all_texture = np.vstack([texture_query_sub, texture_gallery_sub])
    all_ids = np.concatenate([ids_query_sub, ids_gallery_sub])

    var_color = analyze_feature_variance(all_color, "Color Features")
    var_texture = analyze_feature_variance(all_texture, "Texture Features")

    # 2.2. Compare specific IDs
    # Select first 3 IDs for detailed comparison
    if len(selected_ids) >= 3:
        id_A, id_B, id_C = selected_ids[0], selected_ids[1], selected_ids[2]

        # Compare A vs B
        compare_specific_ids(all_color, all_ids, id_A, id_B, "Color Features")
        compare_specific_ids(all_texture, all_ids, id_A, id_B, "Texture Features")

        # Compare A vs C
        compare_specific_ids(all_color, all_ids, id_A, id_C, "Color Features")
        compare_specific_ids(all_texture, all_ids, id_A, id_C, "Texture Features")

    # 2.3. Visualize color histograms
    # Pick 3 persons to visualize
    id_list_to_viz = selected_ids[:3] if len(selected_ids) >= 3 else selected_ids
    visualize_color_histograms(
        all_color,
        all_ids,
        id_list_to_viz,
        os.path.join(args.output_dir, 'color_histogram_comparison.png'),
        n_stripes=6
    )

    # ========================================================================
    # END ADDITIONAL DIAGNOSTICS
    # ========================================================================

    # Plot histograms
    print("\nGenerating plots...")
    plot_distance_histograms(
        [stats_color, stats_texture, stats_combined],
        os.path.join(args.output_dir, 'distance_histograms.png')
    )

    # Plot distance matrices
    plot_distance_matrices(
        dist_color,
        dist_texture,
        dist_combined,
        ids_query_sub,
        ids_gallery_sub,
        os.path.join(args.output_dir, 'distance_matrices.png')
    )

    # Save numerical results
    results_path = os.path.join(args.output_dir, 'debug_results.npz')
    np.savez(
        results_path,
        color_query=color_query_sub,
        texture_query=texture_query_sub,
        color_gallery=color_gallery_sub,
        texture_gallery=texture_gallery_sub,
        ids_query=ids_query_sub,
        ids_gallery=ids_gallery_sub,
        dist_color=dist_color,
        dist_texture=dist_texture,
        dist_combined=dist_combined,
        selected_ids=np.array(selected_ids),
    )
    print(f"Saved numerical results to: {results_path}")

    print("\n" + "="*60)
    print("Debug analysis complete!")
    print("="*60)


if __name__ == '__main__':
    main()
