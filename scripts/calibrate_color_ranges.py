"""
Script to calibrate (u,v) color ranges from Market-1501 training set
"""

import sys
sys.path.append('/app')

import numpy as np
from eval.loaders import load_market1501
from bilp.color import rgb_to_log_chromaticity, calibrate_color_ranges
import json


def calibrate_from_train_set(
    dataset_path: str,
    sample_size: int = 1000,
    percentile_low: float = 1,
    percentile_high: float = 99,
    save_path: str = 'data/color_ranges.json'
):
    """
    Calibrate (u,v) color ranges from Market-1501 training set.

    Args:
        dataset_path: Path to Market-1501 dataset
        sample_size: Number of images to sample (None for all)
        percentile_low: Lower percentile for range
        percentile_high: Upper percentile for range
        save_path: Path to save calibrated ranges

    Returns:
        Dictionary with calibrated ranges
    """
    print("=" * 60)
    print("CALIBRATING COLOR RANGES")
    print("=" * 60)

    # Load training set
    print(f"\nLoading Market-1501 train set...")
    train_data = load_market1501(
        dataset_path,
        split='train',
        return_images=True
    )

    print(f"Loaded {len(train_data)} training images")

    # Sample if needed
    if sample_size is not None and sample_size < len(train_data):
        print(f"Sampling {sample_size} images for calibration...")
        indices = np.random.choice(len(train_data), sample_size, replace=False)
        train_data = [train_data[i] for i in indices]

    # Extract images
    images = [d['image'] for d in train_data]

    print(f"\nComputing (u,v) ranges from {len(images)} images...")
    print(f"Using percentiles: {percentile_low}% - {percentile_high}%")

    # Calibrate ranges
    u_range, v_range = calibrate_color_ranges(
        images,
        percentile_low=percentile_low,
        percentile_high=percentile_high
    )

    print(f"\nCalibrated ranges:")
    print(f"  u_range: [{u_range[0]:.6f}, {u_range[1]:.6f}]")
    print(f"  v_range: [{v_range[0]:.6f}, {v_range[1]:.6f}]")

    # Compute statistics
    print(f"\nComputing statistics across all images...")

    all_u = []
    all_v = []

    for i, image in enumerate(images):
        if (i + 1) % 100 == 0:
            print(f"  Processing {i + 1}/{len(images)}...")

        u, v = rgb_to_log_chromaticity(image)
        all_u.append(u.ravel())
        all_v.append(v.ravel())

    all_u = np.concatenate(all_u)
    all_v = np.concatenate(all_v)

    print(f"\nStatistics:")
    print(f"  Total pixels: {len(all_u):,}")
    print(f"\n  u channel:")
    print(f"    Mean: {all_u.mean():.6f}")
    print(f"    Std:  {all_u.std():.6f}")
    print(f"    Min:  {all_u.min():.6f}")
    print(f"    Max:  {all_u.max():.6f}")
    print(f"    P01:  {np.percentile(all_u, 1):.6f}")
    print(f"    P50:  {np.percentile(all_u, 50):.6f}")
    print(f"    P99:  {np.percentile(all_u, 99):.6f}")

    print(f"\n  v channel:")
    print(f"    Mean: {all_v.mean():.6f}")
    print(f"    Std:  {all_v.std():.6f}")
    print(f"    Min:  {all_v.min():.6f}")
    print(f"    Max:  {all_v.max():.6f}")
    print(f"    P01:  {np.percentile(all_v, 1):.6f}")
    print(f"    P50:  {np.percentile(all_v, 50):.6f}")
    print(f"    P99:  {np.percentile(all_v, 99):.6f}")

    # Save calibrated ranges
    calibration_data = {
        'u_range': [float(u_range[0]), float(u_range[1])],
        'v_range': [float(v_range[0]), float(v_range[1])],
        'percentile_low': percentile_low,
        'percentile_high': percentile_high,
        'num_images': len(images),
        'statistics': {
            'u': {
                'mean': float(all_u.mean()),
                'std': float(all_u.std()),
                'min': float(all_u.min()),
                'max': float(all_u.max())
            },
            'v': {
                'mean': float(all_v.mean()),
                'std': float(all_v.std()),
                'min': float(all_v.min()),
                'max': float(all_v.max())
            }
        }
    }

    print(f"\nSaving calibrated ranges to {save_path}...")
    with open(save_path, 'w') as f:
        json.dump(calibration_data, f, indent=2)

    print("Calibration complete!")
    print("=" * 60)

    return calibration_data


def load_calibrated_ranges(filepath: str = 'data/color_ranges.json'):
    """
    Load calibrated color ranges from file.

    Args:
        filepath: Path to calibration file

    Returns:
        Dictionary with ranges
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    return data


def test_calibration(
    dataset_path: str,
    calibration_path: str = 'data/color_ranges.json'
):
    """
    Test calibrated ranges on sample images.

    Args:
        dataset_path: Path to Market-1501 dataset
        calibration_path: Path to calibration file
    """
    print("\n" + "=" * 60)
    print("TESTING CALIBRATED RANGES")
    print("=" * 60)

    # Load calibrated ranges
    print(f"\nLoading calibrated ranges from {calibration_path}...")
    calib_data = load_calibrated_ranges(calibration_path)

    u_range = tuple(calib_data['u_range'])
    v_range = tuple(calib_data['v_range'])

    print(f"  u_range: {u_range}")
    print(f"  v_range: {v_range}")

    # Load test images
    print(f"\nLoading test images...")
    test_data = load_market1501(dataset_path, split='test', return_images=True)[:100]

    print(f"Loaded {len(test_data)} test images")

    # Check how many pixels fall within calibrated ranges
    within_range_counts = []

    for i, d in enumerate(test_data):
        u, v = rgb_to_log_chromaticity(d['image'])

        u_within = np.sum((u >= u_range[0]) & (u <= u_range[1]))
        v_within = np.sum((v >= v_range[0]) & (v <= v_range[1]))
        both_within = np.sum((u >= u_range[0]) & (u <= u_range[1]) &
                             (v >= v_range[0]) & (v <= v_range[1]))

        total_pixels = u.size
        pct_both = 100 * both_within / total_pixels

        within_range_counts.append(pct_both)

    within_range_counts = np.array(within_range_counts)

    print(f"\nPercentage of pixels within calibrated ranges:")
    print(f"  Mean:   {within_range_counts.mean():.2f}%")
    print(f"  Median: {np.median(within_range_counts):.2f}%")
    print(f"  Min:    {within_range_counts.min():.2f}%")
    print(f"  Max:    {within_range_counts.max():.2f}%")

    if within_range_counts.mean() > 95:
        print("\nCalibration looks good! Most pixels within range.")
    elif within_range_counts.mean() > 90:
        print("\nCalibration is acceptable.")
    else:
        print("\nWarning: Many pixels fall outside calibrated range.")
        print("Consider adjusting percentiles or using wider range.")

    print("=" * 60)


def main():
    """Main calibration pipeline"""
    dataset_path = "datasets/Market-1501-v15.09.15"
    save_path = "data/color_ranges_market.json"

    # Calibrate
    calibration_data = calibrate_from_train_set(
        dataset_path,
        sample_size=1000,  # Use 1000 images for speed
        percentile_low=1,
        percentile_high=99,
        save_path=save_path
    )

    # Test
    test_calibration(dataset_path, save_path)


if __name__ == '__main__':
    main()
