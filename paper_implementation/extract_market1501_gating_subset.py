#!/usr/bin/env python3
"""
Script para extraer features de Market-1501 con información de gating (subset).

Este script extrae features con información de gating de un subset de Market-1501
para poder analizar las estadísticas de T y C sin procesar todo el dataset.

Ejecutar: python3 extract_market1501_gating_subset.py --max-ids 100
"""
import numpy as np
from PIL import Image
import os
from pathlib import Path
import time
import sys
import pickle
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
    Extrae el ID de persona desde el nombre del archivo de Market-1501.
    
    Formato: "0001_c1_001.jpg" -> ID = 1
    """
    # Remover extensión
    name = os.path.splitext(filename)[0]
    
    # Dividir por guiones bajos
    parts = name.split('_')
    
    if len(parts) >= 1:
        try:
            # Primera parte es el ID (ej: "0001" -> 1)
            id_str = parts[0].lstrip('0')  # Remover ceros a la izquierda
            if id_str == '':
                id_str = '0'
            person_id = int(id_str)
            return person_id
        except ValueError:
            pass
    
    return None


def extract_features_market1501_gating_subset(
    query_dir=None,
    gallery_dir=None,
    max_ids=None,
    max_images_per_id=None,
    verbose=True
):
    """
    Extrae features con información de gating de un subset de Market-1501.
    
    Args:
        query_dir: directorio con imágenes query (opcional)
        gallery_dir: directorio con imágenes gallery
        max_ids: máximo número de IDs únicos a procesar (None = todos)
        max_images_per_id: máximo número de imágenes por ID (None = todas)
        verbose: si True, muestra progreso detallado
    
    Returns:
        features_by_id: dict {id: [feature_dicts...]} donde cada feature_dict tiene:
                       'color_features', 'texture_features', 'gabor_responses', 'features'
        total_images: número total de imágenes procesadas
    """
    print("="*70)
    print("EXTRACCIÓN DE FEATURES CON GATING - MARKET-1501 (SUBSET)")
    print("="*70)
    
    if gallery_dir:
        print(f"Gallery dir: {gallery_dir}")
    if query_dir:
        print(f"Query dir: {query_dir}")
    if max_ids:
        print(f"Máximo IDs a procesar: {max_ids}")
    if max_images_per_id:
        print(f"Máximo imágenes por ID: {max_images_per_id}")
    print()
    
    features_by_id = {}
    total_images = 0
    processed_ids = set()
    
    # Procesar gallery
    if gallery_dir and os.path.exists(gallery_dir):
        print(f"Procesando gallery...")
        gallery_files = sorted([f for f in os.listdir(gallery_dir) 
                               if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        
        gallery_start_time = time.time()
        for idx, img_file in enumerate(gallery_files, 1):
            # Verificar límite de IDs
            if max_ids and len(processed_ids) >= max_ids:
                if verbose:
                    print(f"\n  Límite de {max_ids} IDs alcanzado. Deteniendo extracción.")
                break
            
            img_path = os.path.join(gallery_dir, img_file)
            person_id = extract_id_from_filename(img_file)
            
            if person_id is None:
                continue
            
            # Verificar límite de imágenes por ID
            if max_images_per_id and person_id in features_by_id:
                if len(features_by_id[person_id]) >= max_images_per_id:
                    continue
            
            try:
                img_rgb = load_image_rgb(img_path)
                feature_dict = extract_liu2012_features_with_gating_info(img_rgb)
                
                if person_id not in features_by_id:
                    features_by_id[person_id] = []
                    processed_ids.add(person_id)
                
                features_by_id[person_id].append(feature_dict)
                total_images += 1
                
                if verbose:
                    elapsed = time.time() - gallery_start_time
                    if idx > 0:
                        avg_time = elapsed / idx
                        remaining = len(gallery_files) - idx
                        eta = avg_time * remaining / 60
                        percent = (idx / len(gallery_files)) * 100
                        sys.stdout.write(f'\r  Gallery: {idx}/{len(gallery_files)} ({percent:.1f}%) | '
                                       f'IDs: {len(processed_ids)} | Imágenes: {total_images} | ETA: {eta:.1f}m')
                        sys.stdout.flush()
                    else:
                        sys.stdout.write(f'\r  Gallery: {idx}/{len(gallery_files)}')
                        sys.stdout.flush()
            except Exception as e:
                if verbose:
                    print(f"\n    ⚠ Error en {img_file}: {e}")
                continue
        
        if verbose:
            print()  # Nueva línea al completar
        gallery_elapsed = time.time() - gallery_start_time
        print(f"  ✓ Gallery: {total_images} imágenes procesadas de {len(processed_ids)} IDs ({gallery_elapsed:.1f}s)")
    
    # Procesar query (opcional)
    if query_dir and os.path.exists(query_dir):
        print(f"\nProcesando query...")
        query_files = sorted([f for f in os.listdir(query_dir) 
                             if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        
        query_start_time = time.time()
        query_images = 0
        
        for idx, img_file in enumerate(query_files, 1):
            # Verificar límite de IDs
            if max_ids and len(processed_ids) >= max_ids:
                if verbose:
                    print(f"\n  Límite de {max_ids} IDs alcanzado. Deteniendo extracción.")
                break
            
            img_path = os.path.join(query_dir, img_file)
            person_id = extract_id_from_filename(img_file)
            
            if person_id is None:
                continue
            
            # Verificar límite de imágenes por ID
            if max_images_per_id and person_id in features_by_id:
                if len(features_by_id[person_id]) >= max_images_per_id:
                    continue
            
            try:
                img_rgb = load_image_rgb(img_path)
                feature_dict = extract_liu2012_features_with_gating_info(img_rgb)
                
                if person_id not in features_by_id:
                    features_by_id[person_id] = []
                    processed_ids.add(person_id)
                
                features_by_id[person_id].append(feature_dict)
                total_images += 1
                query_images += 1
                
                if verbose:
                    elapsed = time.time() - query_start_time
                    if idx > 0:
                        avg_time = elapsed / idx
                        remaining = len(query_files) - idx
                        eta = avg_time * remaining / 60
                        percent = (idx / len(query_files)) * 100
                        sys.stdout.write(f'\r  Query: {idx}/{len(query_files)} ({percent:.1f}%) | '
                                       f'IDs: {len(processed_ids)} | Imágenes: {total_images} | ETA: {eta:.1f}m')
                        sys.stdout.flush()
                    else:
                        sys.stdout.write(f'\r  Query: {idx}/{len(query_files)}')
                        sys.stdout.flush()
            except Exception as e:
                if verbose:
                    print(f"\n    ⚠ Error en {img_file}: {e}")
                continue
        
        if verbose:
            print()  # Nueva línea al completar
        query_elapsed = time.time() - query_start_time
        print(f"  ✓ Query: {query_images} imágenes procesadas ({query_elapsed:.1f}s)")
    
    print("\n" + "-"*70)
    print(f"RESUMEN:")
    print(f"  IDs únicos: {len(features_by_id)}")
    print(f"  Imágenes totales: {total_images}")
    print("="*70 + "\n")
    
    return features_by_id, total_images


def save_features_with_gating(features_by_id, output_path):
    """Guarda las features con información de gating en disco."""
    print(f"\nGuardando features con gating en: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(features_by_id, f)
    print(f"✓ Features guardadas exitosamente")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Extraer features con gating de Market-1501 (subset)')
    parser.add_argument('--gallery-dir', type=str, 
                       default='Market-1501-v15.09.15/bounding_box_train',
                       help='Directorio con imágenes gallery (default: bounding_box_train)')
    parser.add_argument('--query-dir', type=str, default=None,
                       help='Directorio con imágenes query (opcional)')
    parser.add_argument('--output', type=str, default='features_market1501_gating_subset.pkl',
                       help='Archivo de salida (default: features_market1501_gating_subset.pkl)')
    parser.add_argument('--max-ids', type=int, default=100,
                       help='Máximo número de IDs únicos a procesar (default: 100)')
    parser.add_argument('--max-images-per-id', type=int, default=None,
                       help='Máximo número de imágenes por ID (default: todas)')
    parser.add_argument('--no-verbose', action='store_true',
                       help='Desactivar modo verbose')
    
    args = parser.parse_args()
    
    verbose = not args.no_verbose
    
    print("\n" + "="*70)
    print("EXTRACCIÓN DE FEATURES CON GATING - MARKET-1501 (SUBSET)")
    print("="*70)
    print(f"Configuración:")
    print(f"  Gallery: {args.gallery_dir}")
    if args.query_dir:
        print(f"  Query: {args.query_dir}")
    print(f"  Máximo IDs: {args.max_ids}")
    if args.max_images_per_id:
        print(f"  Máximo imágenes por ID: {args.max_images_per_id}")
    print(f"  Archivo de salida: {args.output}")
    print("="*70 + "\n")
    
    # Verificar que existe el directorio gallery
    if not os.path.exists(args.gallery_dir):
        print(f"❌ Error: No se encontró el directorio gallery: {args.gallery_dir}")
        return
    
    try:
        # Extraer features
        start_time = time.time()
        features_by_id, total_images = extract_features_market1501_gating_subset(
            query_dir=args.query_dir,
            gallery_dir=args.gallery_dir,
            max_ids=args.max_ids,
            max_images_per_id=args.max_images_per_id,
            verbose=verbose
        )
        extraction_time = time.time() - start_time
        
        if verbose:
            fps_avg = total_images / extraction_time if extraction_time > 0 else 0
            print(f"\n⏱ Tiempo de extracción: {extraction_time:.2f} segundos ({extraction_time/60:.2f} minutos)")
            print(f"   Velocidad promedio: {fps_avg:.2f} imágenes/segundo\n")
        else:
            print(f"⏱ Tiempo de extracción: {extraction_time:.2f} segundos ({extraction_time/60:.2f} minutos)\n")
        
        # Guardar features
        save_features_with_gating(features_by_id, args.output)
        
        print("✅ Extracción completada exitosamente!")
        print(f"\nFeatures guardadas en: {args.output}")
        print(f"Puedes usar este archivo para analizar estadísticas de gating:")
        print(f"  python3 analyze_gating_stats.py --features-file {args.output}")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

