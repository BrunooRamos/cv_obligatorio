"""
Calibrate color ranges (u, v) specifically for iLIDS-VID dataset.

This script:
1. Loads a sample of iLIDS-VID sequences
2. Converts RGB to log-chromaticity (u, v)
3. Computes percentile-based ranges to avoid outliers
4. Saves calibrated ranges to JSON file
"""

import argparse
import os
import sys
import json
from typing import Tuple

sys.path.append('/app')

import numpy as np

from eval.loaders import load_ilids_vid
from bilp.color import rgb_to_log_chromaticity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate color ranges for iLIDS-VID dataset",
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='datasets/iLIDS-VID',
        help='Path to the root directory of the iLIDS-VID dataset.',
    )
    parser.add_argument(
        '--num-sequences',
        type=int,
        default=200,
        help='Number of sequences to sample for calibration (default: 200)',
    )
    parser.add_argument(
        '--num-frames',
        type=int,
        default=5,
        help='Number of frames per sequence to sample (default: 5)',
    )
    parser.add_argument(
        '--percentile-low',
        type=float,
        default=1.0,
        help='Lower percentile for range computation (default: 1.0)',
    )
    parser.add_argument(
        '--percentile-high',
        type=float,
        default=99.0,
        help='Upper percentile for range computation (default: 99.0)',
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='data/color_ranges_ilids.json',
        help='Output JSON file for calibrated ranges',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print progress information',
    )
    return parser.parse_args()


def calibrate_color_ranges(
    sequences: list,
    num_sequences: int,
    percentile_low: float,
    percentile_high: float,
    verbose: bool = False
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Calibrate u and v ranges from a sample of sequences.

    Args:
        sequences: List of sequence dictionaries from load_ilids_vid
        num_sequences: Number of sequences to use for calibration
        percentile_low: Lower percentile for range computation
        percentile_high: Upper percentile for range computation
        verbose: Print progress information

    Returns:
        (u_range, v_range): Tuples of (min, max) for u and v channels
    """
    u_vals = []
    v_vals = []

    # Sample sequences
    num_to_sample = min(num_sequences, len(sequences))
    sampled_sequences = sequences[:num_to_sample]

    if verbose:
        print(f"Processing {num_to_sample} sequences for calibration...")

    for idx, seq in enumerate(sampled_sequences):
        frames = seq.get('frames', [])
        if not frames:
            continue

        for frame in frames:
            # Convert to log-chromaticity
            u, v = rgb_to_log_chromaticity(frame)

            # Collect values
            u_vals.append(u.flatten())
            v_vals.append(v.flatten())

        if verbose and (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{num_to_sample} sequences")

    # Concatenate all values
    u_vals = np.concatenate(u_vals)
    v_vals = np.concatenate(v_vals)

    if verbose:
        print(f"\nCollected {len(u_vals):,} pixel values")
        print(f"  U channel: min={u_vals.min():.4f}, max={u_vals.max():.4f}")
        print(f"  V channel: min={v_vals.min():.4f}, max={v_vals.max():.4f}")

    # Compute percentile-based ranges to avoid outliers
    u_min, u_max = np.percentile(u_vals, [percentile_low, percentile_high])
    v_min, v_max = np.percentile(v_vals, [percentile_low, percentile_high])

    u_range = (float(u_min), float(u_max))
    v_range = (float(v_min), float(v_max))

    if verbose:
        print(f"\nCalibrated ranges (percentiles {percentile_low}-{percentile_high}):")
        print(f"  U range: [{u_range[0]:.4f}, {u_range[1]:.4f}]")
        print(f"  V range: [{v_range[0]:.4f}, {v_range[1]:.4f}]")

        # Coverage statistics
        u_in_range = np.sum((u_vals >= u_min) & (u_vals <= u_max))
        v_in_range = np.sum((v_vals >= v_min) & (v_vals <= v_max))
        coverage_u = 100.0 * u_in_range / len(u_vals)
        coverage_v = 100.0 * v_in_range / len(v_vals)

        print(f"\nCoverage:")
        print(f"  U: {coverage_u:.2f}% of pixels within range")
        print(f"  V: {coverage_v:.2f}% of pixels within range")

    return u_range, v_range


def main():
    args = parse_args()

    print("="*60)
    print("Color Calibration Script for iLIDS-VID")
    print("="*60)

    # Load sequences
    if args.verbose:
        print(f"\nLoading iLIDS-VID sequences from: {args.dataset_path}")

    sequences = load_ilids_vid(
        dataset_path=args.dataset_path,
        num_frames=args.num_frames,
        sampling_strategy='random',
        return_images=True,
        verbose=args.verbose,
    )

    if args.verbose:
        print(f"Loaded {len(sequences)} sequences total")

    # Calibrate ranges
    u_range, v_range = calibrate_color_ranges(
        sequences,
        num_sequences=args.num_sequences,
        percentile_low=args.percentile_low,
        percentile_high=args.percentile_high,
        verbose=args.verbose,
    )

    # Prepare output
    ranges = {
        "u_range": list(u_range),
        "v_range": list(v_range),
        "dataset": "iLIDS-VID",
        "num_sequences": min(args.num_sequences, len(sequences)),
        "num_frames_per_sequence": args.num_frames,
        "percentile_low": args.percentile_low,
        "percentile_high": args.percentile_high,
    }

    # Save to file
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_file, 'w') as f:
        json.dump(ranges, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Calibration complete!")
    print(f"Saved calibrated ranges to: {args.output_file}")
    print(f"{'='*60}")

    print("\nTo use these ranges in feature extraction, run:")
    print(f"  --calibration-file {args.output_file}")


if __name__ == '__main__':
    main()
