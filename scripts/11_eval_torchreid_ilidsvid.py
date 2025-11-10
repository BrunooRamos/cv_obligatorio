"""Evaluate torchreid ResNet50 features on the iLIDS-VID dataset."""

import argparse
import os
import sys
from typing import Dict, List

sys.path.append('/app')

import numpy as np

from bilp.distance import compute_distance_matrix_torchreid
from bilp.gpu_utils import get_device
from eval.cmc_map import evaluate_ilids_vid, print_results, save_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute CMC scores on iLIDS-VID using precomputed torchreid features",
    )
    parser.add_argument(
        '--query-features',
        type=str,
        default='data/features/ilidsvid_query_torchreid.npz',
        help='Path to the .npz file containing query sequence features.',
    )
    parser.add_argument(
        '--gallery-features',
        type=str,
        default='data/features/ilidsvid_gallery_torchreid.npz',
        help='Path to the .npz file containing gallery sequence features.',
    )
    parser.add_argument(
        '--metric',
        choices=['cosine', 'euclidean', 'cityblock'],
        default='cosine',
        help='Distance metric (cosine is standard for deep learning features).',
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


def load_torchreid_features(filepath: str) -> tuple:
    """Load torchreid features from .npz file."""
    data = np.load(filepath, allow_pickle=True)
    
    features = data['features']
    metadata = data.get('metadata', None)
    
    if metadata is not None:
        metadata = metadata.item()  # Convert numpy array to dict
    
    return features, metadata


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
    query_features, query_metadata = load_torchreid_features(args.query_features)
    if query_metadata is None:
        raise ValueError('Query features file does not contain metadata. Re-run extraction script with metadata enabled.')

    if args.verbose:
        print(f"Loading gallery features from {args.gallery_features}")
    gallery_features, gallery_metadata = load_torchreid_features(args.gallery_features)
    if gallery_metadata is None:
        raise ValueError('Gallery features file does not contain metadata. Re-run extraction script with metadata enabled.')

    if args.verbose:
        print(f"Query features shape: {query_features.shape}")
        print(f"Gallery features shape: {gallery_features.shape}")

    if query_features.shape[1] != gallery_features.shape[1]:
        raise ValueError('Feature dimensions between query and gallery do not match.')

    if query_features.shape[0] == 0 or gallery_features.shape[0] == 0:
        raise ValueError('Empty query or gallery feature matrices. Check the feature files.')

    query_data = build_metadata_list(query_metadata)
    gallery_data = build_metadata_list(gallery_metadata)

    if len(query_data) != query_features.shape[0]:
        raise ValueError('Mismatch between query metadata entries and feature rows.')
    if len(gallery_data) != gallery_features.shape[0]:
        raise ValueError('Mismatch between gallery metadata entries and feature rows.')

    if args.verbose:
        print('Computing distance matrix...')

    # Get GPU device if requested
    is_gpu, device = get_device(args.use_gpu)
    if is_gpu and args.verbose:
        print(f"Using GPU for distance computation")
    elif args.use_gpu and not is_gpu and args.verbose:
        print(f"GPU requested but not available, using CPU")

    distance_matrix = compute_distance_matrix_torchreid(
        query_features,
        gallery_features,
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

    print_results(results, dataset=f'iLIDS-VID (torchreid ResNet50, metric={args.metric})')

    if args.save_results:
        save_dir = os.path.dirname(args.save_results)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        save_results(results, args.save_results)
        if args.verbose:
            print(f"Saved results to {args.save_results}")


if __name__ == '__main__':
    main()

