#!/usr/bin/env python3
"""
Script para extraer features de Market-1501 COMPLETO con información de gating.

Este script extrae features con información de gating de TODO el dataset Market-1501.
Puede tomar bastante tiempo dependiendo del tamaño del dataset.

Ejecutar: python3 extract_market1501_gating_complete.py
"""
import numpy as np
from PIL import Image
import os
from pathlib import Path
import time
import sys
import pickle
import re
import argparse
from extractor import extract_liu2012_features_with_gating_info


def load_image_rgb(image_path):
    """Carga una imagen y la convierte a RGB uint8."""
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return np.array(img, dtype=np.uint8)


def extract_id_from_filename(filename):
    """
    Extrae el ID de persona del nombre de archivo de Market-1501.
    
    Formato: 0001_c1s1_000151_00.jpg o -1_c1s1_000151_00.jpg (distractor)
    - 0001: ID de persona (o -1 para distractors)
    - c1s1: cámara y secuencia
    - 000151: frame
    - 00: bounding box
    
    Args:
        filename: nombre del archivo
    
    Returns:
        person_id: ID de la persona (int) o None si es distractor
    """
    # Formato: 0001_c1s1_000151_00.jpg o -1_c1s1_000151_00.jpg
    match = re.match(r'^(-?\d+)_', filename)
    if match:
        person_id = int(match.group(1))
        # Ignorar distractors (ID negativo)
        if person_id < 0:
            return None
        return person_id
    return None


def extract_features_market1501_gating_complete(
    query_dir=None,
    gallery_dir=None,
    verbose=True
):
    """
    Extrae features con información de gating de TODO Market-1501.
    
    Args:
        query_dir: directorio con imágenes query (opcional)
        gallery_dir: directorio con imágenes gallery
        verbose: si True, muestra progreso detallado
    
    Returns:
        features_by_id: dict {id: [feature_dicts...]} donde cada feature_dict tiene:
                       'color_features', 'texture_features', 'gabor_responses', 'features'
        total_images: número total de imágenes procesadas
    """
    print("="*70)
    print("EXTRACCIÓN DE FEATURES CON GATING - MARKET-1501 (COMPLETO)")
    print("="*70)
    
    if gallery_dir:
        print(f"Gallery dir: {gallery_dir}")
    if query_dir:
        print(f"Query dir: {query_dir}")
    print()
    
    features_by_id = {}
    total_images = 0
    
    # Procesar query
    if query_dir and os.path.exists(query_dir):
        print(f"Procesando query...")
        query_files = sorted([f for f in os.listdir(query_dir) 
                             if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        
        query_start_time = time.time()
        for idx, img_file in enumerate(query_files, 1):
            img_path = os.path.join(query_dir, img_file)
            person_id = extract_id_from_filename(img_file)
            
            if person_id is None:
                continue
            
            try:
                img_rgb = load_image_rgb(img_path)
                feature_dict = extract_liu2012_features_with_gating_info(img_rgb)
                
                if person_id not in features_by_id:
                    features_by_id[person_id] = []
                features_by_id[person_id].append(feature_dict)
                total_images += 1
                
                if verbose and idx % 100 == 0:
                    elapsed = time.time() - query_start_time
                    fps = idx / elapsed if elapsed > 0 else 0
                    eta_seconds = (len(query_files) - idx) / fps if fps > 0 else 0
                    eta_minutes = eta_seconds / 60
                    sys.stdout.write(f'\r  Query: {idx}/{len(query_files)} imágenes | '
                                   f'{fps:.1f} img/s | ETA: {eta_minutes:.1f}m')
                    sys.stdout.flush()
            except Exception as e:
                if verbose:
                    print(f"\n⚠ Error procesando {img_file}: {e}")
                continue
        
        if verbose:
            print()  # Nueva línea
        
        query_time = time.time() - query_start_time
        if verbose:
            print(f"✓ Query procesado: {len(query_files)} archivos, "
                  f"{total_images} imágenes válidas en {query_time:.2f}s ({query_time/60:.2f}m)")
    
    # Procesar gallery
    if gallery_dir and os.path.exists(gallery_dir):
        print(f"\nProcesando gallery...")
        gallery_files = sorted([f for f in os.listdir(gallery_dir) 
                               if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        
        gallery_start_time = time.time()
        for idx, img_file in enumerate(gallery_files, 1):
            img_path = os.path.join(gallery_dir, img_file)
            person_id = extract_id_from_filename(img_file)
            
            if person_id is None:
                continue
            
            try:
                img_rgb = load_image_rgb(img_path)
                feature_dict = extract_liu2012_features_with_gating_info(img_rgb)
                
                if person_id not in features_by_id:
                    features_by_id[person_id] = []
                features_by_id[person_id].append(feature_dict)
                total_images += 1
                
                if verbose and idx % 500 == 0:
                    elapsed = time.time() - gallery_start_time
                    fps = idx / elapsed if elapsed > 0 else 0
                    eta_seconds = (len(gallery_files) - idx) / fps if fps > 0 else 0
                    eta_minutes = eta_seconds / 60
                    sys.stdout.write(f'\r  Gallery: {idx}/{len(gallery_files)} imágenes | '
                                   f'{fps:.1f} img/s | ETA: {eta_minutes:.1f}m')
                    sys.stdout.flush()
            except Exception as e:
                if verbose:
                    print(f"\n⚠ Error procesando {img_file}: {e}")
                continue
        
        if verbose:
            print()  # Nueva línea
        
        gallery_time = time.time() - gallery_start_time
        if verbose:
            print(f"✓ Gallery procesado: {len(gallery_files)} archivos, "
                  f"{total_images} imágenes válidas en {gallery_time:.2f}s ({gallery_time/60:.2f}m)")
    
    print(f"\n✓ Total: {len(features_by_id)} IDs únicos, {total_images} imágenes")
    
    return features_by_id, total_images


def save_features_with_gating(features_by_id, output_path):
    """Guarda features con información de gating en un archivo pickle."""
    print(f"\nGuardando features en: {output_path}")
    save_start = time.time()
    with open(output_path, 'wb') as f:
        pickle.dump(features_by_id, f)
    save_time = time.time() - save_start
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ Features guardadas exitosamente ({file_size_mb:.1f} MB en {save_time:.2f}s)")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Extraer features con gating de Market-1501 completo')
    parser.add_argument('--gallery-dir', type=str, 
                       default='Market-1501-v15.09.15/bounding_box_test',
                       help='Directorio con imágenes gallery (default: bounding_box_test)')
    parser.add_argument('--query-dir', type=str, 
                       default='Market-1501-v15.09.15/query',
                       help='Directorio con imágenes query (default: query)')
    parser.add_argument('--output', type=str, default='features_market1501_gating_complete.pkl',
                       help='Archivo de salida (default: features_market1501_gating_complete.pkl)')
    parser.add_argument('--no-verbose', action='store_true',
                       help='Desactivar modo verbose')
    
    args = parser.parse_args()
    
    verbose = not args.no_verbose
    
    print("\n" + "="*70)
    print("EXTRACCIÓN DE FEATURES CON GATING - MARKET-1501 (COMPLETO)")
    print("="*70)
    print(f"Configuración:")
    print(f"  Gallery: {args.gallery_dir}")
    if args.query_dir:
        print(f"  Query: {args.query_dir}")
    print(f"  Archivo de salida: {args.output}")
    print("="*70 + "\n")
    
    # Verificar que existe al menos uno de los directorios
    if not os.path.exists(args.gallery_dir) and not (args.query_dir and os.path.exists(args.query_dir)):
        print(f"❌ Error: No se encontraron los directorios")
        if args.gallery_dir:
            print(f"   Gallery: {args.gallery_dir}")
        if args.query_dir:
            print(f"   Query: {args.query_dir}")
        return
    
    try:
        # Extraer features
        start_time = time.time()
        features_by_id, total_images = extract_features_market1501_gating_complete(
            query_dir=args.query_dir,
            gallery_dir=args.gallery_dir,
            verbose=verbose
        )
        extraction_time = time.time() - start_time
        
        if verbose:
            fps_avg = total_images / extraction_time if extraction_time > 0 else 0
            print(f"\n⏱ Tiempo de extracción total: {extraction_time:.2f} segundos ({extraction_time/60:.2f} minutos)")
            print(f"   Velocidad promedio: {fps_avg:.2f} imágenes/segundo\n")
        else:
            print(f"⏱ Tiempo de extracción: {extraction_time:.2f} segundos ({extraction_time/60:.2f} minutos)\n")
        
        # Guardar features
        save_features_with_gating(features_by_id, args.output)
        
        print("\n✅ Extracción completada exitosamente!")
        print(f"\nFeatures guardadas en: {args.output}")
        print(f"Puedes usar este archivo para evaluar con gating:")
        print(f"  python3 eval_market1501_with_gating.py --skip-extraction --features-file {args.output} --a1 0.2 --a2 0.8 --b -2.0")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Interrumpido por el usuario")
        print("Las features procesadas hasta ahora se han perdido.")
        print("Puedes reanudar ejecutando el script nuevamente.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

