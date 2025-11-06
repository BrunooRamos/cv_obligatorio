"""
Test script for BILP feature extraction pipeline
"""

import sys
sys.path.append('/app')

import numpy as np
from eval.loaders import load_market1501
from bilp import (
    extract_bilp_descriptor,
    extract_bilp_batch,
    compute_gating_weight,
    bilp_distance,
    compute_distance_matrix_fast
)


def test_single_image_extraction():
    """Test feature extraction on a single image"""
    print("=" * 60)
    print("Test 1: Single Image Feature Extraction")
    print("=" * 60)

    # Load a single image from Market-1501
    dataset_path = "/datasets/Market-1501-v15.09.15"
    data = load_market1501(dataset_path, split='train', return_images=True)

    if len(data) == 0:
        print("ERROR: No images loaded")
        return

    # Extract BILP descriptor
    image = data[0]['image']
    print(f"\nImage shape: {image.shape}")
    print(f"Person ID: {data[0]['person_id']}, Camera: {data[0]['camera_id']}")

    descriptor = extract_bilp_descriptor(image, n_stripes=6, normalize=True)

    print(f"\nExtracted features:")
    print(f"  Color features: {descriptor['color'].shape} = {len(descriptor['color'])} dims")
    print(f"  Texture features: {descriptor['texture'].shape} = {len(descriptor['texture'])} dims")
    print(f"  Total: {len(descriptor['color']) + len(descriptor['texture'])} dims")

    # Expected: 6 stripes × (272 color + 42 texture) = 1884 features
    expected_color = 6 * 272  # 1632
    expected_texture = 6 * 42  # 252
    expected_total = expected_color + expected_texture  # 1884

    print(f"\nExpected dimensions:")
    print(f"  Color: {expected_color}")
    print(f"  Texture: {expected_texture}")
    print(f"  Total: {expected_total}")

    assert len(descriptor['color']) == expected_color, "Color dimension mismatch"
    assert len(descriptor['texture']) == expected_texture, "Texture dimension mismatch"

    print("\nTest 1 PASSED")


def test_batch_extraction():
    """Test batch feature extraction"""
    print("\n" + "=" * 60)
    print("Test 2: Batch Feature Extraction")
    print("=" * 60)

    # Load 10 images
    dataset_path = "/datasets/Market-1501-v15.09.15"
    data = load_market1501(dataset_path, split='train', return_images=True)[:10]

    images = [d['image'] for d in data]

    print(f"\nExtracting features from {len(images)} images...")

    color_features, texture_features = extract_bilp_batch(
        images,
        n_stripes=6,
        normalize=True,
        verbose=False
    )

    print(f"Color features: {color_features.shape}")
    print(f"Texture features: {texture_features.shape}")

    assert color_features.shape == (10, 1632), "Color batch shape mismatch"
    assert texture_features.shape == (10, 252), "Texture batch shape mismatch"

    print("\nTest 2 PASSED")


def test_gating():
    """Test gating weight computation"""
    print("\n" + "=" * 60)
    print("Test 3: Gating Weight Computation")
    print("=" * 60)

    # Load a single image
    dataset_path = "/datasets/Market-1501-v15.09.15"
    data = load_market1501(dataset_path, split='train', return_images=True)[:5]

    images = [d['image'] for d in data]
    color_features, texture_features = extract_bilp_batch(images)

    # Compute gating weights
    params = {'a1': 2.0, 'a2': 1.0, 'b': 0.0}

    print(f"\nGating parameters: {params}")
    print(f"\nComputing gating weights for {len(images)} images...")

    for i in range(len(images)):
        alpha = compute_gating_weight(
            color_features[i],
            texture_features[i],
            params
        )
        print(f"  Image {i+1} (ID {data[i]['person_id']}): alpha = {alpha:.3f}")

        assert 0.0 <= alpha <= 1.0, "Alpha out of range"

    print("\nTest 3 PASSED")


def test_distance_computation():
    """Test distance computation between images"""
    print("\n" + "=" * 60)
    print("Test 4: Distance Computation")
    print("=" * 60)

    # Load images
    dataset_path = "/datasets/Market-1501-v15.09.15"
    data = load_market1501(dataset_path, split='train', return_images=True)[:20]

    images = [d['image'] for d in data]
    person_ids = np.array([d['person_id'] for d in data])

    # Extract features
    color_features, texture_features = extract_bilp_batch(images)

    # Split into query and gallery
    query_color = color_features[:5]
    query_texture = texture_features[:5]
    gallery_color = color_features[5:]
    gallery_texture = texture_features[5:]

    query_ids = person_ids[:5]
    gallery_ids = person_ids[5:]

    print(f"\nQuery set: {len(query_color)} images")
    print(f"Gallery set: {len(gallery_color)} images")

    # Compute distance matrix
    print("\nComputing distance matrix (alpha=0.5)...")
    dist_matrix = compute_distance_matrix_fast(
        query_color,
        query_texture,
        gallery_color,
        gallery_texture,
        alpha=0.5,
        metric='cityblock'
    )

    print(f"Distance matrix shape: {dist_matrix.shape}")
    print(f"\nDistance matrix (first 5 queries × first 5 gallery):")
    print(dist_matrix[:5, :5])

    # Check that distance to same ID is smaller
    print(f"\nQuery IDs: {query_ids}")
    print(f"Gallery IDs: {gallery_ids}")

    for i in range(len(query_ids)):
        same_id_mask = (gallery_ids == query_ids[i])
        if np.any(same_id_mask):
            same_id_dist = dist_matrix[i, same_id_mask].mean()
            diff_id_dist = dist_matrix[i, ~same_id_mask].mean()
            print(f"Query {i} (ID {query_ids[i]}): "
                  f"same_id_dist={same_id_dist:.3f}, "
                  f"diff_id_dist={diff_id_dist:.3f}")

    print("\nTest 4 PASSED")


def test_feature_statistics():
    """Test feature statistics"""
    print("\n" + "=" * 60)
    print("Test 5: Feature Statistics")
    print("=" * 60)

    # Load images
    dataset_path = "/datasets/Market-1501-v15.09.15"
    data = load_market1501(dataset_path, split='train', return_images=True)[:100]

    images = [d['image'] for d in data]

    # Extract features
    print(f"\nExtracting features from {len(images)} images...")
    color_features, texture_features = extract_bilp_batch(images, verbose=True)

    print(f"\nColor features statistics:")
    print(f"  Shape: {color_features.shape}")
    print(f"  Mean: {color_features.mean():.6f}")
    print(f"  Std: {color_features.std():.6f}")
    print(f"  Min: {color_features.min():.6f}")
    print(f"  Max: {color_features.max():.6f}")

    print(f"\nTexture features statistics:")
    print(f"  Shape: {texture_features.shape}")
    print(f"  Mean: {texture_features.mean():.6f}")
    print(f"  Std: {texture_features.std():.6f}")
    print(f"  Min: {texture_features.min():.6f}")
    print(f"  Max: {texture_features.max():.6f}")

    print("\nTest 5 PASSED")


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("BILP PIPELINE TESTS")
    print("=" * 60 + "\n")

    try:
        test_single_image_extraction()
        test_batch_extraction()
        test_gating()
        test_distance_computation()
        test_feature_statistics()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
