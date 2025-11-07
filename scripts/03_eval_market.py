"""Evaluate BILP features end-to-end on Market-1501."""

import argparse
import os
import sys
from typing import Dict, List

sys.path.append('/app')

import numpy as np

from bilp.utils import load_features
from bilp.distance import compute_distance_matrix_fast
from bilp.gpu_utils import get_device
from eval.cmc_map import evaluate_market1501, print_results, save_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute CMC and mAP on Market-1501 using precomputed BILP features",
    )
    parser.add_argument(
        '--query-features',
        type=str,
        default='data/features/market1501_query.npz',
        help='Path to the .npz file containing query features.',
    )
    parser.add_argument(
        '--gallery-features',
        type=str,
        default='data/features/market1501_test.npz',
        help='Path to the .npz file containing gallery features (bounding_box_test).',
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.5,
        help='Gating weight between texture (alpha) and color (1-alpha).',
    )
    parser.add_argument(
        '--metric',
        choices=['cityblock', 'euclidean', 'cosine'],
        default='cityblock',
        help='Distance metric used for color and texture descriptors.',
    )
    parser.add_argument(
        '--max-rank',
        type=int,
        default=50,
        help='Maximum rank evaluated in the CMC curve.',
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
        help='Print additional information.',
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
    query_color, query_texture, query_metadata = load_features(args.query_features)
    if query_metadata is None:
        raise ValueError('Query features file does not contain metadata. Re-run extraction script with metadata enabled.')

    if args.verbose:
        print(f"Loading gallery features from {args.gallery_features}")
    gallery_color, gallery_texture, gallery_metadata = load_features(args.gallery_features)
    if gallery_metadata is None:
        raise ValueError('Gallery features file does not contain metadata. Re-run extraction script with metadata enabled.')

    if args.verbose:
        print(f"Query color shape: {query_color.shape}, texture shape: {query_texture.shape}")
        print(f"Gallery color shape: {gallery_color.shape}, texture shape: {gallery_texture.shape}")

    if query_color.shape[1] != gallery_color.shape[1]:
        raise ValueError('Color feature dimensions between query and gallery do not match.')
    if query_texture.shape[1] != gallery_texture.shape[1]:
        raise ValueError('Texture feature dimensions between query and gallery do not match.')

    if query_color.shape[0] == 0 or gallery_color.shape[0] == 0:
        raise ValueError('Empty query or gallery features. Check the feature files.')

    if args.verbose:
        print('Building metadata lists...')

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

    distance_matrix = compute_distance_matrix_fast(
        query_color,
        query_texture,
        gallery_color,
        gallery_texture,
        alpha=args.alpha,
        metric=args.metric,
        device=device,
    )

    if args.verbose:
        print('Evaluating metrics...')

    results = evaluate_market1501(
        distance_matrix,
        query_data,
        gallery_data,
        max_rank=args.max_rank,
    )

    print_results(results, dataset=f'Market-1501 (alpha={args.alpha}, metric={args.metric})')

    if args.save_results:
        save_dir = os.path.dirname(args.save_results)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        save_results(results, args.save_results)
        if args.verbose:
            print(f"Saved results to {args.save_results}")


if __name__ == '__main__':
    main()

