"""Extract BILP features for iLIDS-VID sequences and save them to disk."""

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

# Add project root to path (works both in Docker and locally)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

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
        choices=['uniform', 'random', 'consecutive', 'all'],
        default='consecutive',
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
        '--pooling-strategy',
        choices=['first_frame', 'mean', 'max', 'median'],
        default='mean',
        help='Pooling strategy: first_frame (use only first frame, no pooling), mean (average), max, or median (default: mean).',
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
        '--n-bins-uv',
        type=int,
        default=8,
        help='Number of bins for UV histogram (default: 8, was 16).',
    )
    parser.add_argument(
        '--n-bins-lum',
        type=int,
        default=8,
        help='Number of bins for luminance histogram (default: 8, was 16).',
    )
    parser.add_argument(
        '--max-sequences',
        type=int,
        default=None,
        help='Maximum number of sequences to process per camera (for testing). If None, processes all sequences.',
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
    normalize_method: str,
    use_gpu: bool = False,
    normalize_final: bool = True,
    pooling_strategy: str = 'first_frame',  # 'first_frame', 'mean', 'max', 'median'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate features from multiple frames.
    
    Args:
        pooling_strategy: 'first_frame' (use only first frame), 'mean', 'max', 'median'
    """
    if pooling_strategy == 'first_frame':
        # Use only the first frame (no pooling)
        frames_to_use = [frames[0]] if frames else []
    else:
        # Use all frames for pooling
        frames_to_use = frames
    
    if not frames_to_use:
        raise ValueError("No frames provided")
    
    color_batch, texture_batch = extract_bilp_batch(
        frames_to_use,
        n_stripes=n_stripes,
        color_params=color_params,
        texture_params=texture_params,
        normalize=True,
        normalize_method='l2',  # Use L2 normalization (less aggressive)
        normalize_per_stripe=False,  # Normalize globally instead of per stripe
        verbose=False,
        use_gpu=use_gpu,
    )

    if pooling_strategy == 'first_frame':
        # Single frame, no pooling needed
        color_features = color_batch[0]
        texture_features = texture_batch[0]
    elif pooling_strategy == 'mean':
        # Average pooling (original behavior)
        color_features = np.mean(color_batch, axis=0)
        texture_features = np.mean(texture_batch, axis=0)
    elif pooling_strategy == 'max':
        # Max pooling
        color_features = np.max(color_batch, axis=0)
        texture_features = np.max(texture_batch, axis=0)
    elif pooling_strategy == 'median':
        # Median pooling
        color_features = np.median(color_batch, axis=0)
        texture_features = np.median(texture_batch, axis=0)
    else:
        raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")

    # Optional final L1 normalization
    if normalize_final:
        color_features = normalize_l1(color_features).astype(np.float32)
        texture_features = normalize_l1(texture_features).astype(np.float32)
    else:
        color_features = color_features.astype(np.float32)
        texture_features = texture_features.astype(np.float32)

    return color_features, texture_features


def process_sequences(
    sequences: List[Dict],
    camera_id: int,
    n_stripes: int,
    color_params: Dict,
    texture_params: Dict,
    normalize_method: str,
    normalize_final: bool,
    pooling_strategy: str,
    verbose: bool,
    use_gpu: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, List]]:
    color_features: List[np.ndarray] = []
    texture_features: List[np.ndarray] = []
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

    start_time = time.time()
    last_print_time = start_time

    for idx, sequence in enumerate(sequences):
        frame_start_time = time.time()
        
        frames = sequence.get('frames', [])
        if not frames:
            if verbose:
                print(f"Skipping sequence without frames: person {sequence['person_id']} cam {sequence['camera_id']}")
            continue

        color_vec, texture_vec = aggregate_sequence_features(
            frames,
            n_stripes=n_stripes,
            color_params=color_params,
            texture_params=texture_params,
            normalize_method=normalize_method,
            use_gpu=use_gpu,
            normalize_final=normalize_final,
            pooling_strategy=pooling_strategy,
        )

        color_features.append(color_vec)
        texture_features.append(texture_vec)

        metadata['person_ids'].append(int(sequence['person_id']))
        metadata['camera_ids'].append(int(sequence['camera_id']))
        metadata['num_frames'].append(int(sequence['num_frames']))
        metadata['total_frames'].append(int(sequence['total_frames']))

        # Store representative filename/path (first frame) for reference
        first_frame = sequence['frame_paths'][0]
        metadata['filenames'].append(os.path.basename(first_frame))
        metadata['paths'].append(first_frame)

        # Calculate and display ETA
        if verbose:
            current_time = time.time()
            elapsed_time = current_time - start_time
            sequences_processed = idx + 1
            
            # Calculate ETA only if we've processed at least 1 sequence
            if sequences_processed > 0:
                avg_time_per_sequence = elapsed_time / sequences_processed
                remaining_sequences = len(sequences) - sequences_processed
                eta_seconds = avg_time_per_sequence * remaining_sequences
                
                # Format ETA
                if eta_seconds < 60:
                    eta_str = f"{eta_seconds:.0f}s"
                elif eta_seconds < 3600:
                    eta_minutes = eta_seconds / 60
                    eta_str = f"{eta_minutes:.1f}min"
                else:
                    eta_hours = eta_seconds / 3600
                    eta_minutes = (eta_seconds % 3600) / 60
                    eta_str = f"{eta_hours:.0f}h {eta_minutes:.0f}min"
                
                # Print progress every 10 sequences or every 30 seconds
                time_since_last_print = current_time - last_print_time
                if (idx + 1) % 10 == 0 or time_since_last_print >= 30:
                    print(f"Processed {sequences_processed}/{len(sequences)} sequences for camera {camera_id} | ETA: {eta_str}")
                    last_print_time = current_time

    # Print final summary
    if verbose:
        total_time = time.time() - start_time
        if total_time < 60:
            time_str = f"{total_time:.1f}s"
        elif total_time < 3600:
            time_str = f"{total_time/60:.1f}min"
        else:
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            time_str = f"{hours}h {minutes}min"
        print(f"Completed processing {len(color_features)} sequences for camera {camera_id} in {time_str}")

    if not color_features:
        raise RuntimeError(f"No valid sequences processed for camera {camera_id}.")

    color_matrix = np.vstack(color_features)
    texture_matrix = np.vstack(texture_features)

    return color_matrix, texture_matrix, metadata


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

    # Limit to subset if requested
    if args.max_sequences is not None:
        cam1_sequences = cam1_sequences[:args.max_sequences]
        cam2_sequences = cam2_sequences[:args.max_sequences]
        if args.verbose:
            print(f"Limiting to {args.max_sequences} sequences per camera (subset mode)")

    if args.verbose:
        print(f"Camera 1 sequences: {len(cam1_sequences)} (query)")
        print(f"Camera 2 sequences: {len(cam2_sequences)} (gallery)")

    if not cam1_sequences or not cam2_sequences:
        raise RuntimeError('Could not find both camera splits in the dataset. Check dataset structure.')

    u_range, v_range = load_calibrated_color_ranges(args.calibration_file)
    color_params = {
        'n_bins_uv': args.n_bins_uv,
        'n_bins_lum': args.n_bins_lum,
        'u_range': u_range,
        'v_range': v_range,
    }
    texture_params = {
        'n_scales': 5,
        'n_orientations': 8,
    }

    color_query, texture_query, meta_query = process_sequences(
        cam1_sequences,
        camera_id=1,
        n_stripes=args.n_stripes,
        color_params=color_params,
        texture_params=texture_params,
        normalize_method=args.normalize_method,
        normalize_final=args.normalize_final,
        pooling_strategy=args.pooling_strategy,
        verbose=args.verbose,
        use_gpu=args.use_gpu,
    )

    color_gallery, texture_gallery, meta_gallery = process_sequences(
        cam2_sequences,
        camera_id=2,
        n_stripes=args.n_stripes,
        color_params=color_params,
        texture_params=texture_params,
        normalize_method=args.normalize_method,
        normalize_final=args.normalize_final,
        pooling_strategy=args.pooling_strategy,
        verbose=args.verbose,
        use_gpu=args.use_gpu,
    )

    meta_query.update({
        'n_stripes': args.n_stripes,
        'n_bins_uv': args.n_bins_uv,
        'n_bins_lum': args.n_bins_lum,
        'normalize_method': args.normalize_method,
        'normalize_final': args.normalize_final,
        'sampling_strategy': args.sampling_strategy,
        'num_sampled_frames': args.num_frames,
        'pooling_strategy': args.pooling_strategy,
        'calibration_file': os.path.abspath(args.calibration_file),
    })

    meta_gallery.update({
        'n_stripes': args.n_stripes,
        'n_bins_uv': args.n_bins_uv,
        'n_bins_lum': args.n_bins_lum,
        'normalize_method': args.normalize_method,
        'normalize_final': args.normalize_final,
        'sampling_strategy': args.sampling_strategy,
        'num_sampled_frames': args.num_frames,
        'pooling_strategy': args.pooling_strategy,
        'calibration_file': os.path.abspath(args.calibration_file),
    })

    # Adjust filenames if using subset
    if args.max_sequences is not None:
        # Add subset suffix to filenames
        base_query_name = os.path.splitext(args.query_filename)[0]
        base_gallery_name = os.path.splitext(args.gallery_filename)[0]
        ext = os.path.splitext(args.query_filename)[1]
        query_filename = f"{base_query_name}_subset{args.max_sequences}{ext}"
        gallery_filename = f"{base_gallery_name}_subset{args.max_sequences}{ext}"
        if args.verbose:
            print(f"Using subset filenames: {query_filename}, {gallery_filename}")
    else:
        query_filename = args.query_filename
        gallery_filename = args.gallery_filename

    query_path = os.path.join(args.output_dir, query_filename)
    gallery_path = os.path.join(args.output_dir, gallery_filename)

    if not args.overwrite:
        if os.path.exists(query_path):
            raise FileExistsError(f"Query features file already exists: {query_path} (use --overwrite to replace)")
        if os.path.exists(gallery_path):
            raise FileExistsError(f"Gallery features file already exists: {gallery_path} (use --overwrite to replace)")

    if args.verbose:
        print("Saving features to disk...")

    save_features(query_path, color_query, texture_query, meta_query)
    save_features(gallery_path, color_gallery, texture_gallery, meta_gallery)

    print(f"Saved query features to {query_path} (shape: {color_query.shape}, {texture_query.shape})")
    print(f"Saved gallery features to {gallery_path} (shape: {color_gallery.shape}, {texture_gallery.shape})")
    
    if args.max_sequences is not None:
        print(f"\n⚠️  NOTE: These are subset features ({args.max_sequences} sequences per camera).")
        print(f"   To evaluate, use:")
        print(f"   --query-features {query_path}")
        print(f"   --gallery-features {gallery_path}")


if __name__ == '__main__':
    main()

