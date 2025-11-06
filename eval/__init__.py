"""
Evaluation module for Person Re-Identification
"""

from .loaders import (
    load_market1501,
    load_ilids_vid,
    parse_market1501_name,
    parse_ilids_name,
    sample_frames_from_sequence,
    get_market1501_stats,
    get_ilids_stats
)

from .cmc_map import (
    compute_cmc,
    compute_map,
    evaluate_reid,
    evaluate_market1501,
    evaluate_ilids_vid,
    print_results,
    save_results,
    load_results
)

__all__ = [
    # Loaders
    'load_market1501',
    'load_ilids_vid',
    'parse_market1501_name',
    'parse_ilids_name',
    'sample_frames_from_sequence',
    'get_market1501_stats',
    'get_ilids_stats',
    # Metrics
    'compute_cmc',
    'compute_map',
    'evaluate_reid',
    'evaluate_market1501',
    'evaluate_ilids_vid',
    'print_results',
    'save_results',
    'load_results'
]
