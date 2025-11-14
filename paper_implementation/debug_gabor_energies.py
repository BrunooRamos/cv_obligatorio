#!/usr/bin/env python3
"""Script para depurar las energías Gabor y ver por qué T no tiene variación."""
import numpy as np
import pickle
from gating import compute_gabor_energies_from_responses, compute_texture_complexity

# Cargar features
with open('features_ilidsvid_gating.pkl', 'rb') as f:
    tracklets = pickle.load(f)

# Tomar algunos ejemplos
sample_ids = list(tracklets.keys())[:5]

print("="*70)
print("ANÁLISIS DE ENERGÍAS GABOR")
print("="*70)

for i, person_id in enumerate(sample_ids):
    print(f"\nPersona {i+1} (ID: {person_id}):")
    cameras = tracklets[person_id]
    cam_names = list(cameras.keys())
    
    if len(cam_names) > 0:
        cam = cam_names[0]
        frames = cameras[cam]
        
        if len(frames) > 0:
            # Tomar primer frame
            frame = frames[0]
            gabor_responses = frame['gabor_responses']
            
            print(f"  Cámara: {cam}, Frames disponibles: {len(frames)}")
            print(f"  Número de stripes: {len(gabor_responses)}")
            print(f"  Filtros Gabor por stripe: {len(gabor_responses[0])}")
            
            # Calcular energías para cada stripe
            stripe_energies = []
            for stripe_idx, stripe_gabor in enumerate(gabor_responses):
                energies = compute_gabor_energies_from_responses(stripe_gabor)
                stripe_energies.append(energies)
                print(f"\n  Stripe {stripe_idx}:")
                print(f"    Energías Gabor: {energies}")
                print(f"    Suma total: {np.sum(energies):.2f}")
                complexity = compute_texture_complexity(energies)
                print(f"    Complejidad: {complexity:.6f}")
            
            # Agregar energías
            aggregated = np.zeros(len(stripe_energies[0]))
            for energies in stripe_energies:
                aggregated += energies
            
            print(f"\n  Energías agregadas (suma de todos los stripes):")
            print(f"    {aggregated}")
            print(f"    Suma total: {np.sum(aggregated):.2f}")
            complexity_agg = compute_texture_complexity(aggregated)
            print(f"    Complejidad agregada: {complexity_agg:.6f}")
            
            # Verificar si todas las complejidades son iguales
            complexities = [compute_texture_complexity(e) for e in stripe_energies]
            print(f"\n  Complejidades por stripe: {complexities}")
            print(f"    Media: {np.mean(complexities):.6f}")
            print(f"    Desviación estándar: {np.std(complexities):.6f}")
            
            if i < 2:  # Solo mostrar detalles de los primeros 2
                break

print("\n" + "="*70)
print("VERIFICANDO SI TODAS LAS IMÁGENES TIENEN LA MISMA COMPLEJIDAD")
print("="*70)

all_complexities = []
for person_id, cameras in list(tracklets.items())[:20]:
    for cam_name, frames in cameras.items():
        if len(frames) > 0:
            frame = frames[0]
            gabor_responses = frame['gabor_responses']
            
            # Agregar energías
            aggregated = np.zeros(len(gabor_responses[0]))
            for stripe_gabor in gabor_responses:
                energies = compute_gabor_energies_from_responses(stripe_gabor)
                aggregated += energies
            
            complexity = compute_texture_complexity(aggregated)
            all_complexities.append(complexity)

all_complexities = np.array(all_complexities)
print(f"\nComplejidades de {len(all_complexities)} imágenes:")
print(f"  Media: {np.mean(all_complexities):.6f}")
print(f"  Desviación estándar: {np.std(all_complexities):.6f}")
print(f"  Mínimo: {np.min(all_complexities):.6f}")
print(f"  Máximo: {np.max(all_complexities):.6f}")
print(f"  Valores únicos: {len(np.unique(all_complexities))}")

if len(np.unique(all_complexities)) == 1:
    print("\n⚠ TODAS LAS COMPLEJIDADES SON IGUALES!")
    print("Esto sugiere que todas las imágenes tienen la misma distribución de energía Gabor.")
    print("Posibles causas:")
    print("  1. Las imágenes son muy similares")
    print("  2. El cálculo de energías Gabor tiene un problema")
    print("  3. Las respuestas Gabor están siendo normalizadas de manera que pierden variación")

