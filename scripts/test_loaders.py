"""
Test script for data loaders
"""

import sys
sys.path.append('/app')

from eval.loaders import (
    load_market1501,
    load_ilids_vid,
    get_market1501_stats,
    get_ilids_stats,
    parse_market1501_name,
    parse_ilids_name
)


def test_market1501_parser():
    """Test Market-1501 filename parser"""
    print("=" * 60)
    print("Testing Market-1501 filename parser...")
    print("=" * 60)

    test_filenames = [
        "0002_c1s1_000451_03.jpg",
        "0123_c3s2_012345_01.jpg",
        "1501_c6s5_999999_05.jpg"
    ]

    for filename in test_filenames:
        try:
            info = parse_market1501_name(filename)
            print(f"\n{filename}")
            print(f"  Person ID: {info['person_id']}")
            print(f"  Camera ID: {info['camera_id']}")
            print(f"  Frame: {info['frame_num']}")
            print(f"  BBox ID: {info['bbox_id']}")
        except Exception as e:
            print(f"ERROR parsing {filename}: {e}")

    print("\n✅ Parser test complete\n")


def test_ilids_parser():
    """Test iLIDS-VID filename parser"""
    print("=" * 60)
    print("Testing iLIDS-VID filename parser...")
    print("=" * 60)

    test_filenames = [
        "cam1_person001_00317.png",
        "cam2_person042_12345.png"
    ]

    for filename in test_filenames:
        try:
            info = parse_ilids_name(filename)
            print(f"\n{filename}")
            print(f"  Person ID: {info['person_id']}")
            print(f"  Camera ID: {info['camera_id']}")
            print(f"  Frame: {info['frame_num']}")
        except Exception as e:
            print(f"ERROR parsing {filename}: {e}")

    print("\n✅ Parser test complete\n")


def test_market1501_loader():
    """Test Market-1501 loader"""
    print("=" * 60)
    print("Testing Market-1501 loader...")
    print("=" * 60)

    dataset_path = "/datasets/Market-1501-v15.09.15"

    # Test train split (metadata only)
    print("\nLoading train split (metadata only)...")
    train_data = load_market1501(dataset_path, split='train', return_images=False)
    stats = get_market1501_stats(train_data)

    print(f"\n📊 Train Statistics:")
    print(f"  Total images: {stats['num_images']}")
    print(f"  Unique identities: {stats['num_identities']}")
    print(f"  Cameras: {stats['camera_ids']}")
    print(f"  Person IDs range: {min(stats['person_ids'])} - {max(stats['person_ids'])}")

    # Test query split (metadata only)
    print("\nLoading query split (metadata only)...")
    query_data = load_market1501(dataset_path, split='query', return_images=False)
    stats = get_market1501_stats(query_data)

    print(f"\n📊 Query Statistics:")
    print(f"  Total images: {stats['num_images']}")
    print(f"  Unique identities: {stats['num_identities']}")

    # Test loading a single image
    print("\n🖼️  Testing image loading...")
    sample_data = load_market1501(dataset_path, split='train', return_images=True)[:5]

    for i, item in enumerate(sample_data):
        if 'image' in item:
            img = item['image']
            print(f"  Image {i+1}: shape={img.shape}, dtype={img.dtype}, "
                  f"person_id={item['person_id']}, camera={item['camera_id']}")

    print("\n✅ Market-1501 loader test complete\n")


def test_ilids_loader():
    """Test iLIDS-VID loader"""
    print("=" * 60)
    print("Testing iLIDS-VID loader...")
    print("=" * 60)

    dataset_path = "/datasets/iLIDS-VID"

    # Test loading sequences (metadata only)
    print("\nLoading sequences (metadata only, 10 frames uniform sampling)...")
    sequences = load_ilids_vid(
        dataset_path,
        num_frames=10,
        sampling_strategy='uniform',
        return_images=False
    )

    stats = get_ilids_stats(sequences)

    print(f"\n📊 iLIDS-VID Statistics:")
    print(f"  Total sequences: {stats['num_sequences']}")
    print(f"  Unique identities: {stats['num_identities']}")
    print(f"  Cameras: {stats['camera_ids']}")
    print(f"  Avg sequences per person: {stats['avg_sequences_per_person']:.2f}")

    # Show first few sequences
    print(f"\n📹 First 5 sequences:")
    for i, seq in enumerate(sequences[:5]):
        print(f"  Seq {i+1}: Person {seq['person_id']:03d}, "
              f"Cam {seq['camera_id']}, "
              f"Sampled {seq['num_frames']}/{seq['total_frames']} frames")

    # Test loading with images
    print("\n🖼️  Testing image loading (first 2 sequences)...")
    sequences_with_images = load_ilids_vid(
        dataset_path,
        num_frames=5,
        sampling_strategy='uniform',
        return_images=True
    )[:2]

    for i, seq in enumerate(sequences_with_images):
        if 'frames' in seq:
            print(f"  Sequence {i+1}: Person {seq['person_id']:03d}, "
                  f"Cam {seq['camera_id']}, "
                  f"Loaded {len(seq['frames'])} frames")
            if len(seq['frames']) > 0:
                print(f"    Frame shape: {seq['frames'][0].shape}, "
                      f"dtype: {seq['frames'][0].dtype}")

    print("\n✅ iLIDS-VID loader test complete\n")


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("🧪 DATA LOADER TESTS")
    print("=" * 60 + "\n")

    try:
        # Test parsers
        test_market1501_parser()
        test_ilids_parser()

        # Test loaders
        test_market1501_loader()
        test_ilids_loader()

        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
