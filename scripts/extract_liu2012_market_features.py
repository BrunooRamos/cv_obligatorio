"""Extract Liu et al. 2012 features for Market-1501 splits and save them to disk."""

import argparse
import os
import sys
from typing import Dict, List, Tuple

sys.path.append('/app')

import cv2  # type: ignore
import numpy as np

from eval.loaders import load_market1501
from liu2012.extractor import extract_liu2012_batch
from liu2012.utils import save_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Liu et al. 2012 features for Market-1501 splits",
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='datasets/Market-1501-v15.09.15',
        help='Path to the root directory of the Market-1501 dataset.',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/features',
        help='Directory where the feature files (.npz) will be saved.',
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'query', 'test'],
        help='Dataset splits to process.',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Number of images to process at once (for progress display).',
    )
    parser.add_argument(
        '--resize-height',
        type=int,
        default=128,
        help='Resize height applied before feature extraction (default: 128 for 6 stripes).',
    )
    parser.add_argument(
        '--resize-width',
        type=int,
        default=64,
        help='Resize width applied before feature extraction (default: 64).',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing feature files.',
    )
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        help='Use GPU for feature extraction if available (requires CuPy).',
    )


def ensure_output_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_images(batch: List[Dict], resize: Tuple[int, int]) -> Tuple[List[np.ndarray], List[Dict]]:
    images: List[np.ndarray] = []
    valid_entries: List[Dict] = []

    for entry in batch:
        img = cv2.imread(entry['path'])
        if img is None:
            print(f"Warning: Could not load image {entry['path']}, skipping")
            continue

        if resize is not None:
            img = cv2.resize(img, resize)

        # Convert from BGR (OpenCV default) to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        images.append(img)
        valid_entries.append(entry)

    return images, valid_entries


def extract_market1501_split(
    dataset_path: str,
    split: str,
    output_dir: str,
    batch_size: int,
    resize: Tuple[int, int],
    overwrite: bool,
    verbose: bool = False,
    use_gpu: bool = False,
) -> None:
    filename = f'liu2012_market1501_{split}.npz'
    output_path = os.path.join(output_dir, filename)

    if os.path.exists(output_path) and not overwrite:
        print(f"[Skip] {split}: File already exists at {output_path}")
        return

    print(f"\n--- Processing Market-1501 split: {split} ---")

    data = load_market1501(
        dataset_path,
        split=split,
        return_images=False,
    )

    if len(data) == 0:
        print(f"WARNING: No images found for split '{split}'. Nothing to do.")
        return

    print(f"Found {len(data)} images in split '{split}'.")

    resize_size = (resize[0], resize[1])

    features_batches: List[np.ndarray] = []
    processed_entries: List[Dict] = []

    for start in range(0, len(data), batch_size):
        end = min(start + batch_size, len(data))
        batch = data[start:end]

        images, valid_entries = load_images(batch, resize_size)

        if not images:
            continue

        # Extract features using Liu et al. 2012 method
        batch_features = extract_liu2012_batch(
            images,
            n_stripes=6,
            n_bins=16,
            verbose=verbose,
            use_gpu=use_gpu
        )

        features_batches.append(batch_features.astype(np.float32))
        processed_entries.extend(valid_entries)

        if verbose:
            print(f"Processed {len(processed_entries)}/{len(data)} images", end='\r')

    print()  # newline after progress

    if not processed_entries:
        print(f"ERROR: No valid images were processed for split '{split}'.")
        return

    features = np.vstack(features_batches)

    person_ids = [int(entry['person_id']) for entry in processed_entries]
    camera_ids = [int(entry['camera_id']) for entry in processed_entries]
    filenames = [entry['filename'] for entry in processed_entries]
    paths = [entry['path'] for entry in processed_entries]

    metadata = {
        'dataset': 'market1501',
        'method': 'liu2012',
        'split': split,
        'num_images': len(processed_entries),
        'person_ids': person_ids,
        'camera_ids': camera_ids,
        'filenames': filenames,
        'paths': paths,
        'feature_dim': 2784,
        'n_stripes': 6,
        'n_bins': 16,
        'resize': {
            'width': resize[0],
            'height': resize[1],
        },
    }

    ensure_output_dir(output_dir)
    save_features(output_path, features, metadata)

    print(f"Saved features to {output_path}")
    print(f"Feature shape: {features.shape}")


def main() -> None:
    args = parse_args()

    ensure_output_dir(args.output_dir)

    resize = (args.resize_width, args.resize_height)

    for split in args.splits:
        extract_market1501_split(
            dataset_path=args.dataset_path,
            split=split,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            resize=resize,
            overwrite=args.overwrite,
            verbose=args.verbose,
            use_gpu=args.use_gpu,
        )


if __name__ == '__main__':
    main()

