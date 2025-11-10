"""
Data loaders for Market-1501 and iLIDS-VID datasets
"""

import os
import glob
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path


# ============================================================================
# Market-1501 Loader
# ============================================================================

def parse_market1501_name(filename: str) -> Dict[str, any]:
    """
    Parse Market-1501 image filename.

    Format: XXXX_cYsZ_NNNNNN_FF.jpg
    - XXXX: Person ID (e.g., 0002)
    - cY: Camera ID (e.g., c1, c2, ..., c6)
    - sZ: Sequence ID
    - NNNNNN: Frame number
    - FF: Bounding box ID

    Args:
        filename: Image filename (e.g., "0002_c1s1_000451_03.jpg")

    Returns:
        Dictionary with parsed fields
    """
    # Remove extension
    name = os.path.splitext(filename)[0]
    parts = name.split('_')

    if len(parts) < 4:
        raise ValueError(f"Invalid Market-1501 filename format: {filename}")

    person_id = int(parts[0])

    # Parse camera and sequence (e.g., "c1s1")
    cam_seq = parts[1]
    camera_id = int(cam_seq[1])  # Extract number after 'c'

    frame_num = int(parts[2])
    bbox_id = int(parts[3])

    return {
        'person_id': person_id,
        'camera_id': camera_id,
        'frame_num': frame_num,
        'bbox_id': bbox_id,
        'filename': filename
    }


def load_market1501(
    dataset_path: str,
    split: str = 'train',
    resize: Optional[Tuple[int, int]] = (128, 64),
    return_images: bool = False
) -> List[Dict]:
    """
    Load Market-1501 dataset.

    Args:
        dataset_path: Path to Market-1501-v15.09.15 directory
        split: 'train', 'test', or 'query'
        resize: Target size (width, height) or None to keep original
        return_images: If True, load actual images; otherwise just metadata

    Returns:
        List of dictionaries containing image info and optionally the image array
    """
    # Map split to directory name
    split_dirs = {
        'train': 'bounding_box_train',
        'test': 'bounding_box_test',
        'query': 'query'
    }

    if split not in split_dirs:
        raise ValueError(f"Invalid split '{split}'. Choose from: {list(split_dirs.keys())}")

    split_dir = os.path.join(dataset_path, split_dirs[split])

    if not os.path.exists(split_dir):
        # Check if dataset_path exists
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Dataset directory not found: {dataset_path}\n"
                f"Please verify the dataset path is correct."
            )
        
        # List available directories
        available_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        raise FileNotFoundError(
            f"Split directory not found: {split_dir}\n"
            f"Expected directory: {split_dirs[split]}\n"
            f"Dataset path: {dataset_path}\n"
            f"Available directories: {available_dirs}"
        )

    # Get all jpg files
    image_files = sorted(glob.glob(os.path.join(split_dir, '*.jpg')))

    data = []

    for img_path in image_files:
        filename = os.path.basename(img_path)

        # Skip junk images (person_id == -1 or 0000)
        if filename.startswith('-1') or filename.startswith('0000'):
            continue

        try:
            info = parse_market1501_name(filename)
            info['path'] = img_path
            info['split'] = split

            if return_images:
                # Load and resize image
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Could not load image {img_path}")
                    continue

                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if resize is not None:
                    img = cv2.resize(img, resize)

                info['image'] = img

            data.append(info)

        except Exception as e:
            print(f"Warning: Error parsing {filename}: {e}")
            continue

    print(f"Loaded {len(data)} images from Market-1501 {split} set")
    return data


def get_market1501_stats(data: List[Dict]) -> Dict:
    """
    Get statistics from Market-1501 data.

    Args:
        data: List of data dictionaries from load_market1501

    Returns:
        Dictionary with dataset statistics
    """
    person_ids = set(d['person_id'] for d in data)
    camera_ids = set(d['camera_id'] for d in data)

    return {
        'num_images': len(data),
        'num_identities': len(person_ids),
        'num_cameras': len(camera_ids),
        'person_ids': sorted(person_ids),
        'camera_ids': sorted(camera_ids)
    }


# ============================================================================
# iLIDS-VID Loader
# ============================================================================

def parse_ilids_name(filename: str) -> Dict[str, any]:
    """
    Parse iLIDS-VID frame filename.

    Format: cam{1,2}_person{XXX}_{frame}.png
    - cam: Camera ID (1 or 2)
    - person: Person ID (e.g., 001, 002, ...)
    - frame: Frame number

    Args:
        filename: Frame filename (e.g., "cam1_person001_00317.png")

    Returns:
        Dictionary with parsed fields
    """
    # Remove extension
    name = os.path.splitext(filename)[0]
    parts = name.split('_')

    if len(parts) < 3:
        raise ValueError(f"Invalid iLIDS-VID filename format: {filename}")

    camera_id = int(parts[0].replace('cam', ''))
    person_id = int(parts[1].replace('person', ''))
    frame_num = int(parts[2])

    return {
        'camera_id': camera_id,
        'person_id': person_id,
        'frame_num': frame_num,
        'filename': filename
    }


def sample_frames_from_sequence(
    frame_paths: List[str],
    num_frames: int = 10,
    strategy: str = 'uniform'
) -> List[str]:
    """
    Sample frames from a video sequence.

    Args:
        frame_paths: List of paths to all frames in sequence (sorted)
        num_frames: Number of frames to sample
        strategy: Sampling strategy ('uniform', 'random', or 'all')

    Returns:
        List of sampled frame paths
    """
    if strategy == 'all' or len(frame_paths) <= num_frames:
        return frame_paths

    if strategy == 'uniform':
        # Sample uniformly across the sequence
        indices = np.linspace(0, len(frame_paths) - 1, num_frames, dtype=int)
        return [frame_paths[i] for i in indices]

    elif strategy == 'random':
        # Random sampling
        indices = np.random.choice(len(frame_paths), num_frames, replace=False)
        indices = sorted(indices)
        return [frame_paths[i] for i in indices]

    else:
        raise ValueError(f"Invalid strategy '{strategy}'. Choose from: uniform, random, all")


def load_ilids_vid(
    dataset_path: str,
    num_frames: int = 10,
    sampling_strategy: str = 'uniform',
    resize: Optional[Tuple[int, int]] = (128, 64),
    return_images: bool = False,
    verbose: bool = False
) -> List[Dict]:
    """
    Load iLIDS-VID dataset sequences.

    Args:
        dataset_path: Path to iLIDS-VID directory
        num_frames: Number of frames to sample per sequence
        sampling_strategy: 'uniform', 'random', or 'all'
        resize: Target size (width, height) or None to keep original
        return_images: If True, load actual images; otherwise just metadata
        verbose: If True, print progress information

    Returns:
        List of dictionaries, one per sequence (person + camera combination)
    """
    # Try different possible structures
    possible_paths = [
        os.path.join(dataset_path, 'i-LIDS-VID', 'sequences'),  # Standard structure
        os.path.join(dataset_path, 'sequences'),  # Direct sequences folder
        os.path.join(dataset_path, 'iLIDS-VID', 'sequences'),  # Alternative naming
    ]
    
    sequences_path = None
    for path in possible_paths:
        if os.path.exists(path):
            sequences_path = path
            break
    
    if sequences_path is None:
        raise FileNotFoundError(
            f"Sequences directory not found. Tried:\n" +
            "\n".join([f"  - {p}" for p in possible_paths]) +
            f"\n\nDataset path provided: {dataset_path}"
        )

    data = []

    # Iterate over cameras
    for camera in ['cam1', 'cam2']:
        camera_path = os.path.join(sequences_path, camera)

        if not os.path.exists(camera_path):
            continue

        # Iterate over person directories
        person_dirs = sorted(glob.glob(os.path.join(camera_path, 'person*')))

        for person_dir in person_dirs:
            person_name = os.path.basename(person_dir)
            person_id = int(person_name.replace('person', ''))

            # Get all frames for this sequence
            frame_paths = sorted(glob.glob(os.path.join(person_dir, '*.png')))

            if len(frame_paths) == 0:
                continue

            # Sample frames
            sampled_paths = sample_frames_from_sequence(
                frame_paths,
                num_frames,
                sampling_strategy
            )

            sequence_info = {
                'person_id': person_id,
                'camera_id': int(camera.replace('cam', '')),
                'num_frames': len(sampled_paths),
                'total_frames': len(frame_paths),
                'frame_paths': sampled_paths
            }

            if return_images:
                # Load sampled frames
                frames = []
                for frame_path in sampled_paths:
                    img = cv2.imread(frame_path)
                    if img is None:
                        if verbose:
                            print(f"Warning: Could not load frame {frame_path}")
                        continue

                    # Convert BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    if resize is not None:
                        img = cv2.resize(img, resize)

                    frames.append(img)

                sequence_info['frames'] = frames

            data.append(sequence_info)

    if verbose:
        print(f"Loaded {len(data)} sequences from iLIDS-VID ({num_frames} frames each)")
    return data


def get_ilids_stats(data: List[Dict]) -> Dict:
    """
    Get statistics from iLIDS-VID data.

    Args:
        data: List of sequence dictionaries from load_ilids_vid

    Returns:
        Dictionary with dataset statistics
    """
    person_ids = set(d['person_id'] for d in data)
    camera_ids = set(d['camera_id'] for d in data)

    # Count sequences per person
    from collections import defaultdict
    sequences_per_person = defaultdict(int)
    for d in data:
        sequences_per_person[d['person_id']] += 1

    return {
        'num_sequences': len(data),
        'num_identities': len(person_ids),
        'num_cameras': len(camera_ids),
        'person_ids': sorted(person_ids),
        'camera_ids': sorted(camera_ids),
        'avg_sequences_per_person': np.mean(list(sequences_per_person.values()))
    }


# ============================================================================
# Utility Functions
# ============================================================================

def create_train_val_split(
    data: List[Dict],
    val_ratio: float = 0.2,
    random_seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split data into train and validation sets by person ID.

    Args:
        data: List of data dictionaries
        val_ratio: Fraction of identities for validation
        random_seed: Random seed for reproducibility

    Returns:
        train_data, val_data
    """
    np.random.seed(random_seed)

    # Get unique person IDs
    person_ids = sorted(set(d['person_id'] for d in data))

    # Shuffle and split
    np.random.shuffle(person_ids)
    split_idx = int(len(person_ids) * (1 - val_ratio))
    train_ids = set(person_ids[:split_idx])
    val_ids = set(person_ids[split_idx:])

    # Split data
    train_data = [d for d in data if d['person_id'] in train_ids]
    val_data = [d for d in data if d['person_id'] in val_ids]

    return train_data, val_data
