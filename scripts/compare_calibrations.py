"""
Compare two color calibration files and analyze expected impact on features.

This script compares calibration ranges and estimates the impact on histogram binning.
"""

import argparse
import json
import sys

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two color calibration files"
    )
    parser.add_argument(
        '--calib1',
        type=str,
        default='data/color_ranges_market.json',
        help='First calibration file (default: Market-1501)'
    )
    parser.add_argument(
        '--calib2',
        type=str,
        default='data/color_ranges_ilids.json',
        help='Second calibration file (default: iLIDS-VID)'
    )
    parser.add_argument(
        '--n-bins',
        type=int,
        default=16,
        help='Number of bins per dimension (default: 16)'
    )
    return parser.parse_args()


def load_calibration(filepath):
    """Load calibration file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def analyze_range_difference(range1, range2, name, n_bins=16):
    """Analyze difference between two ranges."""
    min1, max1 = range1
    min2, max2 = range2

    span1 = max1 - min1
    span2 = max2 - min2

    bin_width1 = span1 / n_bins
    bin_width2 = span2 / n_bins

    print(f"\n{name} Channel Analysis:")
    print(f"{'='*60}")
    print(f"  Calibration 1: [{min1:.4f}, {max1:.4f}], span={span1:.4f}")
    print(f"  Calibration 2: [{min2:.4f}, {max2:.4f}], span={span2:.4f}")
    print(f"\n  Difference:")
    print(f"    Min shift: {min2 - min1:+.4f}")
    print(f"    Max shift: {max2 - max1:+.4f}")
    print(f"    Span change: {span2 - span1:+.4f} ({100*(span2-span1)/span1:+.1f}%)")

    print(f"\n  Bin widths (n_bins={n_bins}):")
    print(f"    Calibration 1: {bin_width1:.4f}")
    print(f"    Calibration 2: {bin_width2:.4f}")
    print(f"    Width change: {bin_width2 - bin_width1:+.4f} ({100*(bin_width2-bin_width1)/bin_width1:+.1f}%)")

    # Calculate overlap
    overlap_min = max(min1, min2)
    overlap_max = min(max1, max2)
    overlap = max(0, overlap_max - overlap_min)

    overlap_pct1 = 100 * overlap / span1
    overlap_pct2 = 100 * overlap / span2

    print(f"\n  Overlap:")
    print(f"    Absolute: [{overlap_min:.4f}, {overlap_max:.4f}] = {overlap:.4f}")
    print(f"    % of Calib1 range: {overlap_pct1:.1f}%")
    print(f"    % of Calib2 range: {overlap_pct2:.1f}%")

    # Estimate bin mismatch
    # How many bins from calib2 data would fall outside calib1 range?
    outside_low = max(0, min1 - min2) / bin_width2
    outside_high = max(0, max2 - max1) / bin_width2

    print(f"\n  Estimated bin mismatch (if using Calib1 for Calib2 data):")
    print(f"    Bins below range: ~{outside_low:.1f} bins worth of data")
    print(f"    Bins above range: ~{outside_high:.1f} bins worth of data")
    print(f"    Total out-of-range: ~{outside_low + outside_high:.1f} / {n_bins} bins")

    return {
        'span_change_pct': 100*(span2-span1)/span1,
        'overlap_pct': overlap_pct2,
        'bins_out_of_range': outside_low + outside_high
    }


def main():
    args = parse_args()

    print("="*60)
    print("Color Calibration Comparison")
    print("="*60)

    # Load calibrations
    print(f"\nLoading calibrations...")
    print(f"  Calibration 1: {args.calib1}")
    calib1 = load_calibration(args.calib1)
    print(f"  Calibration 2: {args.calib2}")
    calib2 = load_calibration(args.calib2)

    # Extract ranges
    u_range1 = tuple(calib1['u_range'])
    v_range1 = tuple(calib1['v_range'])
    u_range2 = tuple(calib2['u_range'])
    v_range2 = tuple(calib2['v_range'])

    # Analyze differences
    u_stats = analyze_range_difference(u_range1, u_range2, "U", args.n_bins)
    v_stats = analyze_range_difference(v_range1, v_range2, "V", args.n_bins)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    print(f"\nRange changes:")
    print(f"  U span: {u_stats['span_change_pct']:+.1f}%")
    print(f"  V span: {v_stats['span_change_pct']:+.1f}%")

    print(f"\nOverlap with Calibration 2 range:")
    print(f"  U: {u_stats['overlap_pct']:.1f}%")
    print(f"  V: {v_stats['overlap_pct']:.1f}%")

    print(f"\nEstimated data loss (using Calib1 for Calib2 data):")
    print(f"  U: ~{u_stats['bins_out_of_range']:.1f} / {args.n_bins} bins out of range")
    print(f"  V: ~{v_stats['bins_out_of_range']:.1f} / {args.n_bins} bins out of range")

    total_bins = args.n_bins * args.n_bins
    u_coverage = (args.n_bins - u_stats['bins_out_of_range']) / args.n_bins
    v_coverage = (args.n_bins - v_stats['bins_out_of_range']) / args.n_bins
    effective_coverage = u_coverage * v_coverage * 100

    print(f"\n  Effective 2D histogram coverage: {effective_coverage:.1f}%")
    print(f"  Lost bins (approx): {total_bins * (1 - effective_coverage/100):.0f} / {total_bins}")

    # Impact assessment
    print(f"\n{'='*60}")
    print("EXPECTED IMPACT OF USING CORRECT CALIBRATION")
    print(f"{'='*60}")

    if effective_coverage < 70:
        impact = "CRITICAL"
        color = "🔴"
    elif effective_coverage < 85:
        impact = "HIGH"
        color = "🟠"
    elif effective_coverage < 95:
        impact = "MODERATE"
        color = "🟡"
    else:
        impact = "LOW"
        color = "🟢"

    print(f"\n{color} Impact Level: {impact}")
    print(f"\nWith only {effective_coverage:.1f}% effective coverage using wrong calibration:")
    print(f"  - Histograms are severely sparse and poorly distributed")
    print(f"  - Most color information is compressed into few bins")
    print(f"  - Features have very low variance across identities")
    print(f"  - Distance metrics cannot discriminate effectively")

    print(f"\nExpected improvements with correct calibration:")
    print(f"  ✓ Histograms will be better populated")
    print(f"  ✓ Color information spread across more bins")
    print(f"  ✓ Feature variance should increase significantly")
    print(f"  ✓ Better discrimination between identities")
    print(f"  ✓ Improved Re-ID metrics (Rank-1, mAP)")

    if u_stats['bins_out_of_range'] > 3 or v_stats['bins_out_of_range'] > 3:
        print(f"\n⚠️  WARNING: Significant out-of-range data detected!")
        print(f"  Consider also reducing bins (e.g., 8x8 instead of 16x16)")
        print(f"  to ensure better coverage with coarser quantization.")


if __name__ == '__main__':
    main()
