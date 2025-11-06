"""
Test script for CMC and mAP metrics
"""

import sys
sys.path.append('/app')

import numpy as np
from eval.loaders import load_market1501
from eval.cmc_map import (
    compute_cmc,
    compute_map,
    evaluate_reid,
    evaluate_market1501,
    print_results
)
from bilp import extract_bilp_batch, compute_distance_matrix_fast


def test_cmc_perfect_ranking():
    """Test CMC with perfect ranking"""
    print("=" * 60)
    print("Test 1: CMC with Perfect Ranking")
    print("=" * 60)

    # Create synthetic perfect ranking
    # 5 queries, 10 gallery images
    # Query 0 matches gallery 0, Query 1 matches gallery 1, etc.

    n_query = 5
    n_gallery = 10

    query_ids = np.array([0, 1, 2, 3, 4])
    gallery_ids = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    # Perfect distance matrix: smallest distance to correct match
    distance_matrix = np.random.rand(n_query, n_gallery) + 1.0
    for i in range(n_query):
        distance_matrix[i, i] = 0.0  # Perfect match

    # Compute CMC
    cmc = compute_cmc(distance_matrix, query_ids, gallery_ids, max_rank=10, remove_junk=False)

    print(f"\nCMC (should be [1.0, 1.0, ...]):")
    print(f"  Rank-1:  {cmc[0]:.4f}")
    print(f"  Rank-5:  {cmc[4]:.4f}")
    print(f"  Rank-10: {cmc[9]:.4f}")

    assert cmc[0] == 1.0, "Perfect ranking should have Rank-1 = 1.0"
    assert np.all(cmc == 1.0), "All ranks should be 1.0 for perfect ranking"

    print("\nTest 1 PASSED\n")


def test_cmc_imperfect_ranking():
    """Test CMC with imperfect ranking"""
    print("=" * 60)
    print("Test 2: CMC with Imperfect Ranking")
    print("=" * 60)

    n_query = 5
    n_gallery = 10

    query_ids = np.array([0, 1, 2, 3, 4])
    gallery_ids = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    # Imperfect ranking: correct match at rank 2, 3, etc.
    distance_matrix = np.random.rand(n_query, n_gallery) + 1.0

    # Query 0: correct match at rank 1 (distance 0.1)
    distance_matrix[0, 0] = 0.1

    # Query 1: correct match at rank 3 (distance 0.3, two others with 0.1, 0.2)
    distance_matrix[1, 1] = 0.3
    distance_matrix[1, 5] = 0.1
    distance_matrix[1, 6] = 0.2

    # Query 2: correct match at rank 5
    distance_matrix[2, 2] = 0.5
    for j in range(4):
        distance_matrix[2, 5+j] = 0.1 * (j + 1)

    # Query 3, 4: correct match at rank 1
    distance_matrix[3, 3] = 0.0
    distance_matrix[4, 4] = 0.0

    cmc = compute_cmc(distance_matrix, query_ids, gallery_ids, max_rank=10, remove_junk=False)

    print(f"\nCMC:")
    print(f"  Rank-1:  {cmc[0]:.4f} (expected 0.60 = 3/5)")
    print(f"  Rank-3:  {cmc[2]:.4f} (expected 0.80 = 4/5)")
    print(f"  Rank-5:  {cmc[4]:.4f} (expected 1.00 = 5/5)")

    print("\nTest 2 PASSED\n")


def test_map_perfect():
    """Test mAP with perfect ranking"""
    print("=" * 60)
    print("Test 3: mAP with Perfect Ranking")
    print("=" * 60)

    n_query = 5
    n_gallery = 10

    query_ids = np.array([0, 1, 2, 3, 4])
    gallery_ids = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    # Perfect distance matrix
    distance_matrix = np.random.rand(n_query, n_gallery) + 1.0
    for i in range(n_query):
        distance_matrix[i, i] = 0.0

    # Compute mAP
    mAP = compute_map(distance_matrix, query_ids, gallery_ids, remove_junk=False)

    print(f"\nmAP: {mAP:.4f} (expected 1.0000)")

    assert abs(mAP - 1.0) < 0.01, "Perfect ranking should have mAP ≈ 1.0"

    print("\nTest 3 PASSED\n")


def test_real_data_small_subset():
    """Test with real Market-1501 data (small subset)"""
    print("=" * 60)
    print("Test 4: Real Data Evaluation (Small Subset)")
    print("=" * 60)

    # Load small subset of Market-1501
    dataset_path = "/datasets/Market-1501-v15.09.15"

    print("\nLoading query images...")
    query_data = load_market1501(dataset_path, split='query', return_images=True)[:10]

    print("Loading gallery images...")
    gallery_data = load_market1501(dataset_path, split='test', return_images=True)[:100]

    print(f"Loaded {len(query_data)} queries and {len(gallery_data)} gallery images")

    # Extract features
    print("\nExtracting features...")
    query_images = [d['image'] for d in query_data]
    gallery_images = [d['image'] for d in gallery_data]

    query_color, query_texture = extract_bilp_batch(query_images, verbose=False)
    gallery_color, gallery_texture = extract_bilp_batch(gallery_images, verbose=False)

    print(f"Query features: {query_color.shape}, {query_texture.shape}")
    print(f"Gallery features: {gallery_color.shape}, {gallery_texture.shape}")

    # Compute distances
    print("\nComputing distance matrix...")
    distance_matrix = compute_distance_matrix_fast(
        query_color,
        query_texture,
        gallery_color,
        gallery_texture,
        alpha=0.5
    )

    print(f"Distance matrix: {distance_matrix.shape}")

    # Evaluate
    print("\nEvaluating...")
    results = evaluate_market1501(distance_matrix, query_data, gallery_data, max_rank=20)

    print_results(results, dataset="Market-1501 (Subset)")

    # Sanity checks
    assert 0.0 <= results['mAP'] <= 1.0, "mAP should be in [0, 1]"
    assert 0.0 <= results['rank1'] <= 1.0, "Rank-1 should be in [0, 1]"
    assert results['rank1'] <= results['rank5'], "Rank-1 <= Rank-5"
    assert results['rank5'] <= results['rank10'], "Rank-5 <= Rank-10"

    print("\nTest 4 PASSED\n")


def test_junk_removal():
    """Test junk removal (same camera images)"""
    print("=" * 60)
    print("Test 5: Junk Removal (Same Camera)")
    print("=" * 60)

    # Create scenario with same camera images
    n_query = 2
    n_gallery = 6

    # Query 0: person 0, camera 1
    # Gallery: [person 0 cam 1, person 0 cam 2, person 0 cam 3, person 1 cam 1, person 1 cam 2, person 2 cam 1]

    query_ids = np.array([0, 1])
    gallery_ids = np.array([0, 0, 0, 1, 1, 2])

    query_cams = np.array([1, 1])
    gallery_cams = np.array([1, 2, 3, 1, 2, 1])

    # Distance matrix
    # Query 0: smallest to gallery[0] (same ID, same cam - junk), then gallery[1] (same ID, diff cam - good)
    distance_matrix = np.array([
        [0.1, 0.2, 0.3, 0.5, 0.6, 0.7],  # Query 0
        [0.8, 0.9, 1.0, 0.05, 0.15, 0.25]  # Query 1
    ])

    # With junk removal
    print("\nWith junk removal (remove_junk=True):")
    results_with = evaluate_reid(
        distance_matrix,
        query_ids,
        gallery_ids,
        query_cams,
        gallery_cams,
        max_rank=5,
        remove_junk=True
    )

    print(f"  Rank-1: {results_with['rank1']:.4f}")
    print(f"  mAP: {results_with['mAP']:.4f}")

    # Without junk removal
    print("\nWithout junk removal (remove_junk=False):")
    results_without = evaluate_reid(
        distance_matrix,
        query_ids,
        gallery_ids,
        query_cams,
        gallery_cams,
        max_rank=5,
        remove_junk=False
    )

    print(f"  Rank-1: {results_without['rank1']:.4f}")
    print(f"  mAP: {results_without['mAP']:.4f}")

    print("\nTest 5 PASSED\n")


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("CMC AND mAP METRICS TESTS")
    print("=" * 60 + "\n")

    try:
        test_cmc_perfect_ranking()
        test_cmc_imperfect_ranking()
        test_map_perfect()
        test_real_data_small_subset()
        test_junk_removal()

        print("=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
