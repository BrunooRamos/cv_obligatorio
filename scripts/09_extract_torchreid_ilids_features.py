"""Extract torchreid ResNet50 features for iLIDS-VID sequences and save them to disk."""

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

sys.path.append('/app')

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchreid import models

from eval.loaders import load_ilids_vid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract torchreid ResNet50 features for iLIDS-VID sequences",
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
        default='ilidsvid_query_torchreid.npz',
        help='Filename for the query (cam1) features file.',
    )
    parser.add_argument(
        '--gallery-filename',
        type=str,
        default='ilidsvid_gallery_torchreid.npz',
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
        '--batch-size',
        type=int,
        default=32,
        help='Number of frames to process at once.',
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='resnet50',
        help='torchreid model name (default: resnet50).',
    )
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='Path to model weights file. If None, uses pre-trained weights.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing feature files.',
    )
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        help='Use GPU for feature extraction if available.',
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging to show progress during processing.',
    )
    return parser.parse_args()


def ensure_output_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def build_model(model_name: str, weights_path: str = None, use_gpu: bool = False, verbose: bool = False) -> torch.nn.Module:
    """Build and load torchreid model."""
    # Build model
    model = models.build_model(
        name=model_name,
        num_classes=100,  # Dummy, we only need features
        pretrained=True if weights_path is None else False,
    )
    
    # Load custom weights if provided
    if weights_path is not None and os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        if verbose:
            print(f"Loaded weights from {weights_path}")
    
    # Move to GPU if available
    if use_gpu and torch.cuda.is_available():
        model = model.cuda()
        model.eval()
        if verbose:
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        model.eval()
        if verbose:
            print("Using CPU")
    
    return model


def get_preprocess_transform() -> transforms.Compose:
    """Get ImageNet preprocessing transform for torchreid."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 128)),  # Standard person Re-ID size
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def extract_features_batch(
    model: torch.nn.Module,
    images: List[np.ndarray],
    transform: transforms.Compose,
    use_gpu: bool = False
) -> np.ndarray:
    """Extract features for a batch of images."""
    # Preprocess images
    batch_tensors = []
    for img in images:
        tensor = transform(img)
        batch_tensors.append(tensor)
    
    batch_tensor = torch.stack(batch_tensors)
    
    # Move to GPU if available
    if use_gpu and torch.cuda.is_available():
        batch_tensor = batch_tensor.cuda()
    
    # Extract features
    with torch.no_grad():
        features = model(batch_tensor)
        # Normalize features (L2 normalization is standard for Re-ID)
        features = F.normalize(features, p=2, dim=1)
    
    # Move back to CPU and convert to numpy
    features_np = features.cpu().numpy().astype(np.float32)
    
    return features_np


def aggregate_sequence_features(
    frames: List[np.ndarray],
    model: torch.nn.Module,
    transform: transforms.Compose,
    batch_size: int,
    use_gpu: bool = False,
) -> np.ndarray:
    """Extract and average features across frames in a sequence."""
    if not frames:
        raise ValueError("Empty frame list")
    
    # Extract features for all frames
    all_features = []
    
    for start in range(0, len(frames), batch_size):
        end = min(start + batch_size, len(frames))
        frame_batch = frames[start:end]
        
        features_batch = extract_features_batch(
            model,
            frame_batch,
            transform,
            use_gpu=use_gpu
        )
        
        all_features.append(features_batch)
    
    # Stack all features
    features_matrix = np.vstack(all_features)
    
    # Average across frames (temporal pooling)
    sequence_feature = np.mean(features_matrix, axis=0)
    
    # L2 normalize the averaged feature
    norm = np.linalg.norm(sequence_feature)
    if norm > 0:
        sequence_feature = sequence_feature / norm
    
    return sequence_feature.astype(np.float32)


def process_sequences(
    sequences: List[Dict],
    camera_id: int,
    model: torch.nn.Module,
    transform: transforms.Compose,
    batch_size: int,
    verbose: bool,
    use_gpu: bool = False,
) -> Tuple[np.ndarray, Dict[str, List]]:
    features: List[np.ndarray] = []
    metadata: Dict[str, List] = {
        'dataset': 'iLIDS-VID',
        'method': 'torchreid_resnet50',
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

        sequence_feature = aggregate_sequence_features(
            frames,
            model,
            transform,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )

        features.append(sequence_feature)

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

    if not features:
        raise RuntimeError(f"No valid sequences processed for camera {camera_id}.")

    features_matrix = np.vstack(features)
    
    # Add feature dimension to metadata
    metadata['feature_dim'] = features_matrix.shape[1]

    return features_matrix, metadata


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

    # Build model
    if args.verbose:
        print(f"Building model: {args.model_name}")
    
    model = build_model(
        model_name=args.model_name,
        weights_path=args.weights,
        use_gpu=args.use_gpu,
        verbose=args.verbose
    )
    
    # Get preprocessing transform
    transform = get_preprocess_transform()

    # Process query (camera 1)
    query_path = os.path.join(args.output_dir, args.query_filename)
    if os.path.exists(query_path) and not args.overwrite:
        print(f"[Skip] Query features already exist at {query_path}", flush=True)
    else:
        query_features, meta_query = process_sequences(
            cam1_sequences,
            camera_id=1,
            model=model,
            transform=transform,
            batch_size=args.batch_size,
            verbose=args.verbose,
            use_gpu=args.use_gpu,
        )
        
        save_dict = {
            'features': query_features,
            'metadata': meta_query
        }
        
        ensure_output_dir(args.output_dir)
        np.savez_compressed(query_path, **save_dict)
        print(f"Saved query features to {query_path} (shape: {query_features.shape})", flush=True)

    # Process gallery (camera 2)
    gallery_path = os.path.join(args.output_dir, args.gallery_filename)
    if os.path.exists(gallery_path) and not args.overwrite:
        print(f"[Skip] Gallery features already exist at {gallery_path}", flush=True)
    else:
        gallery_features, meta_gallery = process_sequences(
            cam2_sequences,
            camera_id=2,
            model=model,
            transform=transform,
            batch_size=args.batch_size,
            verbose=args.verbose,
            use_gpu=args.use_gpu,
        )
        
        save_dict = {
            'features': gallery_features,
            'metadata': meta_gallery
        }
        
        ensure_output_dir(args.output_dir)
        np.savez_compressed(gallery_path, **save_dict)
        print(f"Saved gallery features to {gallery_path} (shape: {gallery_features.shape})", flush=True)


if __name__ == '__main__':
    main()

