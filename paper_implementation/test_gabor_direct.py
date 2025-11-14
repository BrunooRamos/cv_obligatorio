#!/usr/bin/env python3
"""Test para verificar si las respuestas Gabor son diferentes entre imágenes."""
import numpy as np
from PIL import Image
import os
from extractor import extract_liu2012_features_with_gating_info
from gating import compute_gabor_energies_from_responses, compute_texture_complexity

# Buscar algunas imágenes diferentes
sequences_dir = 'iLIDS-VID/i-LIDS-VID/sequences'
cam_dirs = [d for d in os.listdir(sequences_dir) if os.path.isdir(os.path.join(sequences_dir, d)) and 'cam' in d.lower()]

if len(cam_dirs) == 0:
    print("No se encontraron directorios de cámaras")
    exit(1)

print("="*70)
print("TEST DIRECTO DE RESPUESTAS GABOR")
print("="*70)

# Tomar algunas imágenes diferentes
images_tested = []
for cam_dir in cam_dirs[:1]:  # Solo primera cámara
    cam_path = os.path.join(sequences_dir, cam_dir)
    person_dirs = [d for d in os.listdir(cam_path) if os.path.isdir(os.path.join(cam_path, d))][:3]
    
    for person_dir in person_dirs:
        person_path = os.path.join(cam_path, person_dir)
        frame_files = sorted([f for f in os.listdir(person_path) 
                             if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        
        if len(frame_files) > 0:
            frame_path = os.path.join(person_path, frame_files[0])
            img = Image.open(frame_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_array = np.array(img, dtype=np.uint8)
            
            # Extraer features
            feature_dict = extract_liu2012_features_with_gating_info(img_array)
            gabor_responses = feature_dict['gabor_responses']
            
            # Calcular complejidad
            aggregated = np.zeros(len(gabor_responses[0]))
            for stripe_gabor in gabor_responses:
                energies = compute_gabor_energies_from_responses(stripe_gabor)
                aggregated += energies
            
            complexity = compute_texture_complexity(aggregated)
            
            images_tested.append({
                'path': frame_path,
                'complexity': complexity,
                'energies': aggregated.copy()
            })
            
            print(f"\nImagen: {frame_path}")
            print(f"  Complejidad: {complexity:.6f}")
            print(f"  Energías Gabor (primeros 4): {aggregated[:4]}")
            
            if len(images_tested) >= 3:
                break
    
    if len(images_tested) >= 3:
        break

# Comparar
print("\n" + "="*70)
print("COMPARACIÓN")
print("="*70)

if len(images_tested) >= 2:
    complexities = [img['complexity'] for img in images_tested]
    print(f"\nComplejidades: {complexities}")
    print(f"¿Son todas iguales? {len(set(complexities)) == 1}")
    
    # Comparar energías
    energies_list = [img['energies'] for img in images_tested]
    print(f"\n¿Todas las energías son iguales?")
    for i in range(len(energies_list)):
        for j in range(i+1, len(energies_list)):
            are_equal = np.allclose(energies_list[i], energies_list[j], rtol=1e-5)
            print(f"  Imagen {i+1} vs Imagen {j+1}: {are_equal}")
            if not are_equal:
                diff = np.abs(energies_list[i] - energies_list[j])
                print(f"    Diferencia máxima: {np.max(diff):.6f}")
                print(f"    Diferencia promedio: {np.mean(diff):.6f}")

