"""Evaluate Liu et al. 2012 features on the iLIDS-VID dataset using the correct protocol."""

import argparse
import os
import sys
from typing import Dict, List

sys.path.append('/app')

import numpy as np

from liu2012.utils import load_features
from liu2012.evaluation import evaluate_liu2012_ilids_vid
from eval.cmc_map import print_results, save_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute CMC scores on iLIDS-VID using Liu et al. 2012 features",
    )
    parser.add_argument(
        '--features',
        type=str,
        default='data/features/liu2012_ilidsvid_all.npz',
        help='Path to the .npz file containing all features (will be split into gallery/probes per trial).',
    )
    parser.add_argument(
        '--n-persons',
        type=int,
        default=50,
        help='Number of persons to select per trial (default: 50 as in Liu et al. 2012).',
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=10,
        help='Number of trials to average over (default: 10 as in Liu et al. 2012).',
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
    return parser.parse_args()


def ensure_file(path: str, description: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{description} not found: {path}")


def main() -> None:
    args = parse_args()

    ensure_file(args.features, 'Features file')

    if args.verbose:
        print(f"Loading features from {args.features}")
    features, metadata = load_features(args.features)
    if metadata is None:
        raise ValueError('Features file does not contain metadata. Re-run extraction script with metadata enabled.')

    if args.verbose:
        print(f"Features shape: {features.shape}")
        print(f"Number of images: {features.shape[0]}")

    if features.shape[1] != 2784:
        print(f"Warning: Expected feature dimension 2784, got {features.shape[1]}")

    if features.shape[0] == 0:
        raise ValueError('Empty feature matrix. Check the feature file.')

    if args.verbose:
        print(f'\nUsing Liu et al. 2012 protocol:')
        print(f'  - {args.n_persons} persons per trial')
        print(f'  - {args.n_trials} trials')
        print(f'  - Distance metric: {args.metric} (L1/cityblock)')
        print(f'  - Max rank: {args.max_rank}')

    # Evaluate using Liu et al. 2012 protocol
    results = evaluate_liu2012_ilids_vid(
        features,
        metadata,
        n_persons=args.n_persons,
        n_trials=args.n_trials,
        max_rank=args.max_rank,
        metric=args.metric,
        verbose=args.verbose
    )

    print_results(results, dataset=f'iLIDS-VID (Liu et al. 2012 protocol, {args.n_trials} trials, metric={args.metric})')

    if args.save_results:
        save_dir = os.path.dirname(args.save_results)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        save_results(results, args.save_results)
        if args.verbose:
            print(f"Saved results to {args.save_results}")


if __name__ == '__main__':
    main()

