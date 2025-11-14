"""Extract BILP features for iLIDS-VID sequences and save them to disk."""

import argparse
import os
import sys
from typing import Dict, List, Tuple

sys.path.append('/app')

import numpy as np

from eval.loaders import load_ilids_vid
from bilp.utils import (
    extract_bilp_batch,
    load_calibrated_color_ranges,
    save_features,
    normalize_l1,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract BILP descriptors for iLIDS-VID sequences (cam1 as query, cam2 as gallery)",
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='datasets/iLIDS-VID',
        help='Path to the root directory of the iLIDS-VID dataset.',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/features',
        help='Directory where the feature files (.npz) will be saved.',
    )
    parser.add_argument(
        '--query-filename',
        type=str,
        default='ilidsvid_query.npz',
        help='Filename for the query (cam1) features file.',
    )
    parser.add_argument(
        '--gallery-filename',
        type=str,
        default='ilidsvid_gallery.npz',
        help='Filename for the gallery (cam2) features file.',
    )
    parser.add_argument(
        '--num-frames',
        type=int,
        default=10,
        help='Number of frames to sample per sequence.',
    )
    parser.add_argument(
        '--sampling-strategy',
        choices=['uniform', 'random', 'all'],
        default='uniform',
        help='Frame sampling strategy within each sequence.',
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
        help='Normalization method applied per stripe before pooling.',
    )
    parser.add_argument(
        '--normalize-final',
        action='store_true',
        default=False,
        help='Apply L1 normalization after averaging frames (default: False).',
    )
    parser.add_argument(
        '--calibration-file',
        type=str,
        default='data/color_ranges_market.json',
        help='Path to the calibrated color range JSON file.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing feature files.',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print progress information.',
    )
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        help='Use GPU for feature extraction if available (requires CuPy).',
    )
    parser.add_argument(
        '--use-hog',
        action='store_true',
        help='Extract HOG features in addition to color and texture.',
    )
    return parser.parse_args()


def ensure_output_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def aggregate_sequence_features(
    frames: List[np.ndarray],
    n_stripes: int,
    color_params: Dict,
    texture_params: Dict,
    hog_params: Dict,
    normalize_method: str,
    use_gpu: bool = False,
    normalize_final: bool = True,
    use_hog: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # SIMPLIFIED NORMALIZATION: Do NOT normalize per stripe
    # Only normalize once at the end after averaging frames
    color_batch, texture_batch, hog_batch = extract_bilp_batch(
        frames,
        n_stripes=n_stripes,
        color_params=color_params,
        texture_params=texture_params,
        hog_params=hog_params,
        normalize=False,  # CHANGED: No per-stripe normalization
        normalize_method=normalize_method,
        verbose=False,
        use_gpu=use_gpu,
        use_hog=use_hog,
    )

    # Average across frames
    color_mean = np.mean(color_batch, axis=0)
    texture_mean = np.mean(texture_batch, axis=0)
    hog_mean = np.mean(hog_batch, axis=0) if use_hog else np.array([], dtype=np.float32)

    # Single L2 normalization at the end (more gentle than L1)
    if normalize_final:
        # L2 normalization instead of L1 (preserves more variance)
        color_norm = np.linalg.norm(color_mean) + 1e-12
        color_mean = (color_mean / color_norm).astype(np.float32)

        texture_norm = np.linalg.norm(texture_mean) + 1e-12
        texture_mean = (texture_mean / texture_norm).astype(np.float32)

        if use_hog and len(hog_mean) > 0:
            hog_norm = np.linalg.norm(hog_mean) + 1e-12
            hog_mean = (hog_mean / hog_norm).astype(np.float32)
    else:
        color_mean = color_mean.astype(np.float32)
        texture_mean = texture_mean.astype(np.float32)
        if use_hog:
            hog_mean = hog_mean.astype(np.float32)

    return color_mean, texture_mean, hog_mean


def process_sequences(
    sequences: List[Dict],
    camera_id: int,
    n_stripes: int,
    color_params: Dict,
    texture_params: Dict,
    hog_params: Dict,
    normalize_method: str,
    normalize_final: bool,
    verbose: bool,
    use_gpu: bool = False,
    use_hog: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, List]]:
    color_features: List[np.ndarray] = []
    texture_features: List[np.ndarray] = []
    hog_features: List[np.ndarray] = []
    metadata: Dict[str, List] = {
        'dataset': 'iLIDS-VID',
        'camera_id': camera_id,
        'person_ids': [],
        'camera_ids': [],
        'num_frames': [],
        'total_frames': [],
        'filenames': [],
        'paths': [],
    }

    if verbose:
        print(f"Processing {len(sequences)} sequences for camera {camera_id}...")

    for idx, sequence in enumerate(sequences):
        frames = sequence.get('frames', [])
        if not frames:
            if verbose:
                print(f"Skipping sequence without frames: person {sequence['person_id']} cam {sequence['camera_id']}")
            continue

        color_vec, texture_vec, hog_vec = aggregate_sequence_features(
            frames,
            n_stripes=n_stripes,
            color_params=color_params,
            texture_params=texture_params,
            hog_params=hog_params,
            normalize_method=normalize_method,
            use_gpu=use_gpu,
            normalize_final=normalize_final,
            use_hog=use_hog,
        )

        color_features.append(color_vec)
        texture_features.append(texture_vec)
        if use_hog:
            hog_features.append(hog_vec)

        metadata['person_ids'].append(int(sequence['person_id']))
        metadata['camera_ids'].append(int(sequence['camera_id']))
        metadata['num_frames'].append(int(sequence['num_frames']))
        metadata['total_frames'].append(int(sequence['total_frames']))

        # Store representative filename/path (first frame) for reference
        first_frame = sequence['frame_paths'][0]
        metadata['filenames'].append(os.path.basename(first_frame))
        metadata['paths'].append(first_frame)

        if verbose and (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(sequences)} sequences for camera {camera_id}")

    if not color_features:
        raise RuntimeError(f"No valid sequences processed for camera {camera_id}.")

    color_matrix = np.vstack(color_features)
    texture_matrix = np.vstack(texture_features)
    hog_matrix = np.vstack(hog_features) if use_hog and hog_features else np.array([])

    return color_matrix, texture_matrix, hog_matrix, metadata


def main() -> None:
    args = parse_args()

    ensure_output_dir(args.output_dir)

    if args.verbose:
        print("Loading iLIDS-VID sequences...")

    sequences = load_ilids_vid(
        dataset_path=args.dataset_path,
        num_frames=args.num_frames,
        sampling_strategy=args.sampling_strategy,
        return_images=True,
        verbose=args.verbose,
    )

    if args.verbose:
        print(f"Loaded {len(sequences)} sequences")

    cam1_sequences = [seq for seq in sequences if seq['camera_id'] == 1]
    cam2_sequences = [seq for seq in sequences if seq['camera_id'] == 2]

    if args.verbose:
        print(f"Camera 1 sequences: {len(cam1_sequences)} (query)")
        print(f"Camera 2 sequences: {len(cam2_sequences)} (gallery)")

    if not cam1_sequences or not cam2_sequences:
        raise RuntimeError('Could not find both camera splits in the dataset. Check dataset structure.')

    u_range, v_range = load_calibrated_color_ranges(args.calibration_file)
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
    hog_params = {
        'orientations': 8,
        'pixels_per_cell': (8, 8),
        'cells_per_block': (2, 2),
    }

    color_query, texture_query, hog_query, meta_query = process_sequences(
        cam1_sequences,
        camera_id=1,
        n_stripes=args.n_stripes,
        color_params=color_params,
        texture_params=texture_params,
        hog_params=hog_params,
        normalize_method=args.normalize_method,
        normalize_final=args.normalize_final,
        verbose=args.verbose,
        use_gpu=args.use_gpu,
        use_hog=args.use_hog,
    )

    color_gallery, texture_gallery, hog_gallery, meta_gallery = process_sequences(
        cam2_sequences,
        camera_id=2,
        n_stripes=args.n_stripes,
        color_params=color_params,
        texture_params=texture_params,
        hog_params=hog_params,
        normalize_method=args.normalize_method,
        normalize_final=args.normalize_final,
        verbose=args.verbose,
        use_gpu=args.use_gpu,
        use_hog=args.use_hog,
    )

    meta_query.update({
        'n_stripes': args.n_stripes,
        'normalize_method': args.normalize_method,
        'normalize_final': args.normalize_final,
        'sampling_strategy': args.sampling_strategy,
        'num_sampled_frames': args.num_frames,
        'calibration_file': os.path.abspath(args.calibration_file),
        'use_hog': args.use_hog,
    })

    meta_gallery.update({
        'n_stripes': args.n_stripes,
        'normalize_method': args.normalize_method,
        'normalize_final': args.normalize_final,
        'sampling_strategy': args.sampling_strategy,
        'num_sampled_frames': args.num_frames,
        'calibration_file': os.path.abspath(args.calibration_file),
        'use_hog': args.use_hog,
    })

    query_path = os.path.join(args.output_dir, args.query_filename)
    gallery_path = os.path.join(args.output_dir, args.gallery_filename)

    if not args.overwrite:
        if os.path.exists(query_path):
            raise FileExistsError(f"Query features file already exists: {query_path} (use --overwrite to replace)")
        if os.path.exists(gallery_path):
            raise FileExistsError(f"Gallery features file already exists: {gallery_path} (use --overwrite to replace)")

    if args.verbose:
        print("Saving features to disk...")

    hog_query_to_save = hog_query if args.use_hog and len(hog_query) > 0 else None
    hog_gallery_to_save = hog_gallery if args.use_hog and len(hog_gallery) > 0 else None

    save_features(query_path, color_query, texture_query, hog_query_to_save, meta_query)
    save_features(gallery_path, color_gallery, texture_gallery, hog_gallery_to_save, meta_gallery)

    if args.use_hog:
        print(f"Saved query features to {query_path} (shape: color={color_query.shape}, texture={texture_query.shape}, hog={hog_query.shape})")
        print(f"Saved gallery features to {gallery_path} (shape: color={color_gallery.shape}, texture={texture_gallery.shape}, hog={hog_gallery.shape})")
    else:
        print(f"Saved query features to {query_path} (shape: color={color_query.shape}, texture={texture_query.shape})")
        print(f"Saved gallery features to {gallery_path} (shape: color={color_gallery.shape}, texture={texture_gallery.shape})")


if __name__ == '__main__':
    main()

