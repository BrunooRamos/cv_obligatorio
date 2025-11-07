"""Extract BILP features for Market-1501 splits and save them to disk."""

import argparse
import os
import sys
from typing import Dict, List, Tuple

sys.path.append('/app')

import cv2  # type: ignore
import numpy as np

from eval.loaders import load_market1501
from bilp.utils import (
    extract_bilp_batch,
    load_calibrated_color_ranges,
    save_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract BILP color+texture descriptors for Market-1501 splits",
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
        help='Number of images to process at once.',
    )
    parser.add_argument(
        '--n-stripes',
        type=int,
        default=6,
        help='Number of horizontal stripes for feature extraction.',
    )
    parser.add_argument(
        '--normalize-method',
        choices=['l1', 'l2', 'power', 'l1_power'],
        default='l1',
        help='Normalization method applied per stripe.',
    )
    parser.add_argument(
        '--resize-width',
        type=int,
        default=128,
        help='Resize width applied before feature extraction.',
    )
    parser.add_argument(
        '--resize-height',
        type=int,
        default=64,
        help='Resize height applied before feature extraction.',
    )
    parser.add_argument(
        '--calibration-file',
        type=str,
        default='app/data/color_ranges_market.json',
        help='Path to the calibrated color range JSON file.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing feature files.',
    )
    return parser.parse_args()


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

        # Convert from BGR (OpenCV default) to RGB expected by BILP utilities
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        images.append(img)
        valid_entries.append(entry)

    return images, valid_entries


def extract_market1501_split(
    dataset_path: str,
    split: str,
    output_dir: str,
    batch_size: int,
    n_stripes: int,
    normalize_method: str,
    resize: Tuple[int, int],
    calibration_file: str,
    overwrite: bool,
) -> None:
    filename = f'market1501_{split}.npz'
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

    u_range, v_range = load_calibrated_color_ranges(calibration_file)
    color_params = {
        'n_bins_uv': 16,
        'n_bins_lum': 16,
        'u_range': u_range,
        'v_range': v_range,
    }
    texture_params = {
        'n_scales': 5,
        'n_orientations': 8,
    }

    color_features_batches: List[np.ndarray] = []
    texture_features_batches: List[np.ndarray] = []
    processed_entries: List[Dict] = []

    for start in range(0, len(data), batch_size):
        end = min(start + batch_size, len(data))
        batch = data[start:end]

        images, valid_entries = load_images(batch, resize_size)

        if not images:
            continue

        color_batch, texture_batch = extract_bilp_batch(
            images,
            n_stripes=n_stripes,
            color_params=color_params,
            texture_params=texture_params,
            normalize=True,
            normalize_method=normalize_method,
            verbose=False,
        )

        color_features_batches.append(color_batch.astype(np.float32))
        texture_features_batches.append(texture_batch.astype(np.float32))
        processed_entries.extend(valid_entries)

        print(f"Processed {len(processed_entries)}/{len(data)} images", end='\r')

    print()  # newline after progress

    if not processed_entries:
        print(f"ERROR: No valid images were processed for split '{split}'.")
        return

    color_features = np.vstack(color_features_batches)
    texture_features = np.vstack(texture_features_batches)

    person_ids = [int(entry['person_id']) for entry in processed_entries]
    camera_ids = [int(entry['camera_id']) for entry in processed_entries]
    filenames = [entry['filename'] for entry in processed_entries]
    paths = [entry['path'] for entry in processed_entries]

    metadata = {
        'dataset': 'market1501',
        'split': split,
        'num_images': len(processed_entries),
        'person_ids': person_ids,
        'camera_ids': camera_ids,
        'filenames': filenames,
        'paths': paths,
        'n_stripes': n_stripes,
        'normalize_method': normalize_method,
        'resize': {
            'width': resize[0],
            'height': resize[1],
        },
        'color_params': {
            'n_bins_uv': color_params['n_bins_uv'],
            'n_bins_lum': color_params['n_bins_lum'],
            'u_range': list(color_params['u_range']),
            'v_range': list(color_params['v_range']),
        },
        'texture_params': {
            'n_scales': texture_params['n_scales'],
            'n_orientations': texture_params['n_orientations'],
        },
        'calibration_file': os.path.abspath(calibration_file),
    }

    ensure_output_dir(output_dir)
    save_features(output_path, color_features, texture_features, metadata)

    print(f"Saved features to {output_path}")
    print(
        f"Color shape: {color_features.shape}, Texture shape: {texture_features.shape}"
    )


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
            n_stripes=args.n_stripes,
            normalize_method=args.normalize_method,
            resize=resize,
            calibration_file=args.calibration_file,
            overwrite=args.overwrite,
        )


if __name__ == '__main__':
    main()

