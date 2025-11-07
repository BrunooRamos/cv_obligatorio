"""
Quick test to verify everything works
"""

import sys
sys.path.append('/app')

import numpy as np
from eval.loaders import load_market1501
from bilp import extract_bilp_batch, compute_distance_matrix_fast

print("Loading data...")
query_data = load_market1501('/datasets/Market-1501-v15.09.15', split='query', return_images=True)[:10]
gallery_data = load_market1501('/datasets/Market-1501-v15.09.15', split='test', return_images=True)[:50]

print(f"Loaded {len(query_data)} queries and {len(gallery_data)} gallery images")

print("Extracting features...")
query_images = [d['image'] for d in query_data]
gallery_images = [d['image'] for d in gallery_data]

query_color, query_texture = extract_bilp_batch(query_images, verbose=True)
print(f"Query features: {query_color.shape}, {query_texture.shape}")

gallery_color, gallery_texture = extract_bilp_batch(gallery_images, verbose=True)
print(f"Gallery features: {gallery_color.shape}, {gallery_texture.shape}")

print("Computing distances...")
distance_matrix = compute_distance_matrix_fast(
    query_color, query_texture,
    gallery_color, gallery_texture,
    alpha=0.5
)

print(f"Distance matrix: {distance_matrix.shape}")
print(f"Distance range: [{distance_matrix.min():.4f}, {distance_matrix.max():.4f}]")
print("\nSuccess!")
