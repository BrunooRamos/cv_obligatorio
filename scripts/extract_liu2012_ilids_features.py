"""Extract Liu et al. 2012 features for iLIDS-VID sequences and save them to disk."""

import argparse
import os
import sys
from typing import Dict, List, Tuple, Optional

sys.path.append('/app')

import cv2  # type: ignore
import numpy as np

from eval.loaders import load_ilids_vid
from liu2012.extractor import extract_liu2012_features
from liu2012.utils import save_features

# Import GPU utilities
try:
    from bilp.gpu_utils import get_device
except ImportError:
    def get_device(use_gpu: bool = False):
        return False, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Liu et al. 2012 features for iLIDS-VID sequences (cam1 as query, cam2 as gallery)",
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
        '--output-filename',
        type=str,
        default='liu2012_ilidsvid_all.npz',
        help='Filename for the combined features file (all sequences).',
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
        '--pooling-strategy',
        choices=['first_frame', 'mean', 'max', 'median'],
        default='first_frame',
        help='Pooling strategy: first_frame (use only first frame), mean (average), max, or median (default: first_frame).',
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


def aggregate_sequence_features(
    frames: List[np.ndarray],
    pooling_strategy: str = 'first_frame',
    device: Optional = None,
) -> np.ndarray:
    """
    Aggregate features from multiple frames.
    
    Args:
        frames: List of RGB images (each H×W×3, uint8)
        pooling_strategy: 'first_frame' (use only first frame), 'mean', 'max', 'median'
        device: GPU device (CuPy) or None for CPU
    
    Returns:
        Feature vector (2784,)
    """
    if not frames:
        raise ValueError("No frames provided")
    
    if pooling_strategy == 'first_frame':
        # Use only the first frame (no pooling)
        frames_to_use = [frames[0]]
    else:
        # Use all frames for pooling
        frames_to_use = frames
    
    # Extract features from each frame
    frame_features = []
    for frame in frames_to_use:
        feat = extract_liu2012_features(frame, device=device)
        frame_features.append(feat)
    
    frame_features = np.array(frame_features)  # (n_frames, 2784)
    
    if pooling_strategy == 'first_frame':
        # Single frame, no pooling needed
        features = frame_features[0]
    elif pooling_strategy == 'mean':
        # Average pooling
        features = np.mean(frame_features, axis=0)
    elif pooling_strategy == 'max':
        # Max pooling
        features = np.max(frame_features, axis=0)
    elif pooling_strategy == 'median':
        # Median pooling
        features = np.median(frame_features, axis=0)
    else:
        raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")
    
    return features.astype(np.float32)


def process_sequences(
    sequences: List[Dict],
    pooling_strategy: str,
    resize: Tuple[int, int],
    verbose: bool,
    camera_id: int = None,
    device: Optional = None,
) -> Tuple[np.ndarray, Dict[str, List]]:
    features: List[np.ndarray] = []
    metadata: Dict[str, List] = {
        'dataset': 'iLIDS-VID',
        'method': 'liu2012',
        'person_ids': [],
        'camera_ids': [],
        'num_frames': [],
        'total_frames': [],
        'filenames': [],
        'paths': [],
    }

    if verbose:
        if camera_id is not None:
            print(f"Processing {len(sequences)} sequences for camera {camera_id}...")
        else:
            print(f"Processing {len(sequences)} sequences (all cameras)...")

    for idx, sequence in enumerate(sequences):
        frames = sequence.get('frames', [])
        if not frames:
            if verbose:
                print(f"Skipping sequence without frames: person {sequence['person_id']} cam {sequence['camera_id']}")
            continue
        
        # Resize frames if needed
        resized_frames = []
        for frame in frames:
            if resize is not None:
                frame = cv2.resize(frame, resize)
            resized_frames.append(frame)

        feat = aggregate_sequence_features(
            resized_frames,
            pooling_strategy=pooling_strategy,
            device=device,
        )

        features.append(feat)

        metadata['person_ids'].append(int(sequence['person_id']))
        metadata['camera_ids'].append(int(sequence['camera_id']))
        metadata['num_frames'].append(int(sequence['num_frames']))
        metadata['total_frames'].append(int(sequence['total_frames']))

        # Store representative filename/path (first frame) for reference
        first_frame = sequence['frame_paths'][0]
        metadata['filenames'].append(os.path.basename(first_frame))
        metadata['paths'].append(first_frame)

        if verbose and (idx + 1) % 10 == 0:
            if camera_id is not None:
                print(f"Processed {idx + 1}/{len(sequences)} sequences for camera {camera_id}")
            else:
                print(f"Processed {idx + 1}/{len(sequences)} sequences")

    if not features:
        if camera_id is not None:
            raise RuntimeError(f"No valid sequences processed for camera {camera_id}.")
        else:
            raise RuntimeError("No valid sequences processed.")

    feature_matrix = np.vstack(features)

    return feature_matrix, metadata


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

    resize = (args.resize_width, args.resize_height)

    # Get GPU device if requested
    is_gpu, device = get_device(args.use_gpu)
    if is_gpu and args.verbose:
        print(f"Using GPU for feature extraction")
    elif args.use_gpu and not is_gpu and args.verbose:
        print(f"GPU requested but not available, using CPU")

    # Process all sequences together (protocol will split them per trial)
    all_features, all_metadata = process_sequences(
        sequences,
        camera_id=None,  # Process all cameras together
        pooling_strategy=args.pooling_strategy,
        resize=resize,
        verbose=args.verbose,
        device=device,
    )

    all_metadata.update({
        'feature_dim': 2784,
        'n_stripes': 6,
        'n_bins': 16,
        'sampling_strategy': args.sampling_strategy,
        'num_sampled_frames': args.num_frames,
        'pooling_strategy': args.pooling_strategy,
        'resize': {
            'width': resize[0],
            'height': resize[1],
        },
    })

    output_path = os.path.join(args.output_dir, args.output_filename)

    if not args.overwrite:
        if os.path.exists(output_path):
            raise FileExistsError(f"Features file already exists: {output_path} (use --overwrite to replace)")

    if args.verbose:
        print("Saving features to disk...")

    save_features(output_path, all_features, all_metadata)

    print(f"Saved features to {output_path} (shape: {all_features.shape})")
    print(f"Total sequences: {len(sequences)}")
    print(f"Note: Use eval_liu2012_ilids.py with this file for correct protocol evaluation")


if __name__ == '__main__':
    main()

