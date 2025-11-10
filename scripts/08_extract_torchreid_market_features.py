"""Extract torchreid ResNet50 features for Market-1501 splits and save them to disk."""

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

from eval.loaders import load_market1501


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract torchreid ResNet50 features for Market-1501 splits",
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
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='Number of images processed before logging progress (only if --verbose is set).',
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
        num_classes=1,  # Dummy, we only need features
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


def load_images(batch: List[Dict], verbose: bool = False) -> Tuple[List[np.ndarray], List[Dict]]:
    """Load images from batch entries."""
    images: List[np.ndarray] = []
    valid_entries: List[Dict] = []

    for entry in batch:
        img = cv2.imread(entry['path'])
        if img is None:
            if verbose:
                print(f"Warning: Could not load image {entry['path']}, skipping")
            continue

        # Convert from BGR (OpenCV default) to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        valid_entries.append(entry)

    return images, valid_entries


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


def extract_market1501_split(
    dataset_path: str,
    split: str,
    output_dir: str,
    model: torch.nn.Module,
    transform: transforms.Compose,
    batch_size: int,
    overwrite: bool,
    use_gpu: bool = False,
    verbose: bool = False,
    log_interval: int = 100,
) -> None:
    filename = f'market1501_{split}_torchreid.npz'
    output_path = os.path.join(output_dir, filename)

    if os.path.exists(output_path) and not overwrite:
        print(f"[Skip] {split}: File already exists at {output_path}", flush=True)
        return

    print(f"\n--- Processing Market-1501 split: {split} ---", flush=True)

    data = load_market1501(
        dataset_path,
        split=split,
        return_images=False,
    )

    if len(data) == 0:
        print(f"WARNING: No images found for split '{split}'. Nothing to do.", flush=True)
        return

    print(f"Found {len(data)} images in split '{split}'.", flush=True)

    features_batches: List[np.ndarray] = []
    processed_entries: List[Dict] = []

    start_time = time.time()
    last_log_time = start_time
    last_log_count = 0

    if verbose:
        print(f"Starting processing of {len(data)} images...", flush=True)
        print(f"Logging progress every {log_interval} images", flush=True)

    for start in range(0, len(data), batch_size):
        end = min(start + batch_size, len(data))
        batch = data[start:end]

        images, valid_entries = load_images(batch, verbose=verbose)

        if not images:
            continue

        features_batch = extract_features_batch(
            model,
            images,
            transform,
            use_gpu=use_gpu
        )

        features_batches.append(features_batch)
        processed_entries.extend(valid_entries)

        num_processed = len(processed_entries)
        
        # Logging periódico si verbose está activado
        if verbose and num_processed % log_interval == 0:
            current_time = time.time()
            elapsed = current_time - start_time
            batch_elapsed = current_time - last_log_time
            images_since_last_log = num_processed - last_log_count
            rate = images_since_last_log / batch_elapsed if batch_elapsed > 0.1 else num_processed / elapsed
            percentage = (num_processed / len(data)) * 100
            remaining = len(data) - num_processed
            eta = remaining / rate if rate > 0 else 0
            
            print(f"[{split}] Processed {num_processed}/{len(data)} images ({percentage:.1f}%) | "
                  f"Rate: {rate:.2f} img/s | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", flush=True)
            last_log_time = current_time
            last_log_count = num_processed
        elif not verbose:
            print(f"Processed {num_processed}/{len(data)} images", end='\r', flush=True)

    if not verbose:
        print()  # newline after progress
    else:
        total_time = time.time() - start_time
        print(f"[{split}] Completed! Processed {len(processed_entries)}/{len(data)} images in {total_time:.1f}s "
              f"({len(processed_entries)/total_time:.1f} img/s average)", flush=True)

    if not processed_entries:
        print(f"ERROR: No valid images were processed for split '{split}'.", flush=True)
        return

    features = np.vstack(features_batches)

    person_ids = [int(entry['person_id']) for entry in processed_entries]
    camera_ids = [int(entry['camera_id']) for entry in processed_entries]
    filenames = [entry['filename'] for entry in processed_entries]
    paths = [entry['path'] for entry in processed_entries]

    metadata = {
        'dataset': 'market1501',
        'split': split,
        'method': 'torchreid_resnet50',
        'num_images': len(processed_entries),
        'feature_dim': features.shape[1],
        'person_ids': person_ids,
        'camera_ids': camera_ids,
        'filenames': filenames,
        'paths': paths,
    }

    ensure_output_dir(output_dir)
    
    # Save in compatible format: features as single array
    # For evaluation, we'll treat this as "color" features with alpha=1.0
    save_dict = {
        'features': features,
        'metadata': metadata
    }
    
    np.savez_compressed(output_path, **save_dict)

    print(f"Saved features to {output_path}", flush=True)
    print(f"Features shape: {features.shape}", flush=True)


def main() -> None:
    args = parse_args()

    ensure_output_dir(args.output_dir)

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

    for split in args.splits:
        extract_market1501_split(
            dataset_path=args.dataset_path,
            split=split,
            output_dir=args.output_dir,
            model=model,
            transform=transform,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
            use_gpu=args.use_gpu,
            verbose=args.verbose,
            log_interval=args.log_interval,
        )


if __name__ == '__main__':
    main()

