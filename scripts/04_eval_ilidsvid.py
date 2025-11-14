"""Evaluate BILP descriptors on the iLIDS-VID dataset."""

import argparse
import os
import sys
from typing import Dict, List

sys.path.append('/app')

import numpy as np

from bilp.utils import load_features
from bilp.distance import compute_distance_matrix_fast, compute_distance_matrix_fast_with_hog
from bilp.gpu_utils import get_device
from eval.cmc_map import evaluate_ilids_vid, print_results, save_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute CMC scores on iLIDS-VID using precomputed BILP features",
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
        '--alpha',
        type=float,
        default=None,
        help='Gating weight between texture (alpha) and color (1-alpha). If None, uses adaptive gating with default params.',
    )
    parser.add_argument(
        '--alpha-hog',
        type=float,
        default=0.33,
        help='Weight for HOG features when using 3-way fusion (color + texture + HOG). Default: 0.33.',
    )
    parser.add_argument(
        '--gating-params',
        type=str,
        default=None,
        help='Path to JSON file with optimized gating parameters (a1, a2, b). If provided, uses adaptive gating.',
    )
    parser.add_argument(
        '--metric',
        choices=['cityblock', 'euclidean', 'cosine'],
        default='cityblock',
        help='Distance metric for combining color and texture descriptors.',
    )
    parser.add_argument(
        '--max-rank',
        type=int,
        default=20,
        help='Maximum rank to report in the CMC curve.',
    )
    parser.add_argument(
        '--save-results',
        type=str,
        default=None,
        help='Optional path to save evaluation results (.npz).',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print additional debug information.',
    )
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        help='Use GPU for distance computation if available (requires CuPy).',
    )
    return parser.parse_args()


def ensure_file(path: str, description: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{description} not found: {path}")


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


def main() -> None:
    args = parse_args()

    ensure_file(args.query_features, 'Query features file')
    ensure_file(args.gallery_features, 'Gallery features file')

    if args.verbose:
        print(f"Loading query features from {args.query_features}")
    query_color, query_texture, query_hog, query_metadata = load_features(args.query_features)
    if query_metadata is None:
        raise ValueError('Query features file does not contain metadata. Re-run extraction script with metadata enabled.')

    if args.verbose:
        print(f"Loading gallery features from {args.gallery_features}")
    gallery_color, gallery_texture, gallery_hog, gallery_metadata = load_features(args.gallery_features)
    if gallery_metadata is None:
        raise ValueError('Gallery features file does not contain metadata. Re-run extraction script with metadata enabled.')

    # Check if HOG features are available
    use_hog = (query_hog is not None and gallery_hog is not None and
               len(query_hog) > 0 and len(gallery_hog) > 0)

    if args.verbose:
        print(f"Query color shape: {query_color.shape}, texture shape: {query_texture.shape}")
        print(f"Gallery color shape: {gallery_color.shape}, texture shape: {gallery_texture.shape}")
        if use_hog:
            print(f"Query HOG shape: {query_hog.shape}")
            print(f"Gallery HOG shape: {gallery_hog.shape}")
        else:
            print("HOG features not available, using color + texture only")

    if query_color.shape[1] != gallery_color.shape[1]:
        raise ValueError('Color feature dimensions between query and gallery do not match.')
    if query_texture.shape[1] != gallery_texture.shape[1]:
        raise ValueError('Texture feature dimensions between query and gallery do not match.')
    if use_hog and query_hog.shape[1] != gallery_hog.shape[1]:
        raise ValueError('HOG feature dimensions between query and gallery do not match.')

    if query_color.shape[0] == 0 or gallery_color.shape[0] == 0:
        raise ValueError('Empty query or gallery feature matrices. Check the feature files.')

    query_data = build_metadata_list(query_metadata)
    gallery_data = build_metadata_list(gallery_metadata)

    if len(query_data) != query_color.shape[0]:
        raise ValueError('Mismatch between query metadata entries and feature rows.')
    if len(gallery_data) != gallery_color.shape[0]:
        raise ValueError('Mismatch between gallery metadata entries and feature rows.')

    if args.verbose:
        print('Computing distance matrix...')

    # Get GPU device if requested
    is_gpu, device = get_device(args.use_gpu)
    if is_gpu and args.verbose:
        print(f"Using GPU for distance computation")
    elif args.use_gpu and not is_gpu and args.verbose:
        print(f"GPU requested but not available, using CPU")
    # Determine alpha: use adaptive gating if params provided, else use fixed alpha
    if args.gating_params and os.path.exists(args.gating_params):
        import json
        with open(args.gating_params, 'r') as f:
            gating_data = json.load(f)
        
        from bilp.gating import compute_gating_weights_batch
        
        gating_params = {
            'a1': gating_data['a1'],
            'a2': gating_data['a2'],
            'b': gating_data['b']
        }
        
        # Get normalization stats if available
        t_stats = gating_data.get('normalization', {}).get('t_stats', None)
        c_stats = gating_data.get('normalization', {}).get('c_stats', None)
        
        if args.verbose:
            print(f"Using adaptive gating with params: {gating_params}")
            if t_stats and c_stats:
                print(f"Using normalization: T(mean={t_stats['mean']:.4f}, std={t_stats['std']:.4f}), "
                      f"C(mean={c_stats['mean']:.4f}, std={c_stats['std']:.4f})")
        
        # Compute adaptive alpha for each query
        query_alphas = compute_gating_weights_batch(
            query_color,
            query_texture,
            gating_params,
            t_stats=t_stats,
            c_stats=c_stats
        )
        
        if args.verbose:
            print(f"Alpha range: [{query_alphas.min():.3f}, {query_alphas.max():.3f}], mean: {query_alphas.mean():.3f}")
        
        alpha = query_alphas
    else:
        # Use fixed alpha
        alpha = args.alpha if args.alpha is not None else 0.5
        if args.verbose:
            print(f"Using fixed alpha: {alpha}")

    # Compute distance matrix with or without HOG
    if use_hog:
        if args.verbose:
            print(f"Using 3-way fusion: Color + Texture + HOG (alpha_hog={args.alpha_hog})")
        distance_matrix = compute_distance_matrix_fast_with_hog(
            query_color,
            query_texture,
            query_hog,
            gallery_color,
            gallery_texture,
            gallery_hog,
            alpha=alpha,
            alpha_hog=args.alpha_hog,
            metric=args.metric,
            device=device,
        )
    else:
        if args.verbose:
            print(f"Using 2-way fusion: Color + Texture only")
        distance_matrix = compute_distance_matrix_fast(
            query_color,
            query_texture,
            gallery_color,
            gallery_texture,
            alpha=alpha,
            metric=args.metric,
            device=device,
        )

    if args.verbose:
        print('Evaluating CMC...')

    results = evaluate_ilids_vid(
        distance_matrix,
        query_data,
        gallery_data,
        max_rank=args.max_rank,
    )

    print_results(results, dataset=f'iLIDS-VID (alpha={args.alpha}, metric={args.metric})')

    if args.save_results:
        save_dir = os.path.dirname(args.save_results)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        save_results(results, args.save_results)
        if args.verbose:
            print(f"Saved results to {args.save_results}")


if __name__ == '__main__':
    main()

