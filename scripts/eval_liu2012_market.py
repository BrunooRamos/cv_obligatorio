"""Evaluate Liu et al. 2012 features on Market-1501."""

import argparse
import os
import sys
from typing import Dict, List

sys.path.append('/app')

import numpy as np
from scipy.spatial.distance import cdist

from liu2012.utils import load_features
from eval.cmc_map import evaluate_market1501, print_results, save_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute CMC and mAP on Market-1501 using Liu et al. 2012 features",
    )
    parser.add_argument(
        '--query-features',
        type=str,
        default='data/features/liu2012_market1501_query.npz',
        help='Path to the .npz file containing query features.',
    )
    parser.add_argument(
        '--gallery-features',
        type=str,
        default='data/features/liu2012_market1501_test.npz',
        help='Path to the .npz file containing gallery features (bounding_box_test).',
    )
    parser.add_argument(
        '--metric',
        choices=['cityblock', 'euclidean', 'cosine'],
        default='cityblock',
        help='Distance metric (default: cityblock/L1 as in the paper).',
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
    query_features, query_metadata = load_features(args.query_features)
    if query_metadata is None:
        raise ValueError('Query features file does not contain metadata. Re-run extraction script with metadata enabled.')

    if args.verbose:
        print(f"Loading gallery features from {args.gallery_features}")
    gallery_features, gallery_metadata = load_features(args.gallery_features)
    if gallery_metadata is None:
        raise ValueError('Gallery features file does not contain metadata. Re-run extraction script with metadata enabled.')

    if args.verbose:
        print(f"Query features shape: {query_features.shape}")
        print(f"Gallery features shape: {gallery_features.shape}")

    if query_features.shape[1] != gallery_features.shape[1]:
        raise ValueError('Feature dimensions between query and gallery do not match.')
    
    if query_features.shape[1] != 2784:
        print(f"Warning: Expected feature dimension 2784, got {query_features.shape[1]}")

    if query_features.shape[0] == 0 or gallery_features.shape[0] == 0:
        raise ValueError('Empty query or gallery features. Check the feature files.')

    if args.verbose:
        print('Building metadata lists...')

    query_data = build_metadata_list(query_metadata)
    gallery_data = build_metadata_list(gallery_metadata)

    if len(query_data) != query_features.shape[0]:
        raise ValueError('Mismatch between query metadata entries and feature rows.')
    if len(gallery_data) != gallery_features.shape[0]:
        raise ValueError('Mismatch between gallery metadata entries and feature rows.')

    if args.verbose:
        print(f'Computing distance matrix using {args.metric} metric...')

    # Compute distance matrix
    distance_matrix = cdist(query_features, gallery_features, metric=args.metric)

    if args.verbose:
        print('Evaluating metrics...')

    results = evaluate_market1501(
        distance_matrix,
        query_data,
        gallery_data,
        max_rank=args.max_rank,
    )

    print_results(results, dataset=f'Market-1501 (Liu et al. 2012, metric={args.metric})')

    if args.save_results:
        save_dir = os.path.dirname(args.save_results)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        save_results(results, args.save_results)
        if args.verbose:
            print(f"Saved results to {args.save_results}")


if __name__ == '__main__':
    main()

