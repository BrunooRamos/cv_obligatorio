#!/usr/bin/env python3
"""
Script completo para evaluar Liu 2012 en Market-1501.

Este script:
1. Extrae features de imágenes query y gallery en Market-1501
2. Evalúa CMC con protocolo single-shot
3. Guarda las features para reutilización
4. Muestra los resultados

Ejecutar: python3 eval_market1501_complete.py
"""
import numpy as np
from PIL import Image
import os
from pathlib import Path
import time
import sys
import pickle
import re
from extractor import extract_liu2012_features
from eval_stills import evaluate_cmc_stills
from constants import DIMS_TOTAL


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


def extract_features_market1501(query_dir, gallery_dir, verbose=True):
    """
    Extrae features de imágenes query y gallery en Market-1501.
    
    Args:
        query_dir: directorio con imágenes query
        gallery_dir: directorio con imágenes gallery
        verbose: si True, muestra progreso detallado
    
    Returns:
        features_by_id: dict {id: [features...]}
        total_images: número total de imágenes procesadas
    """
    print("="*70)
    print("PASO 1: EXTRACCIÓN DE FEATURES - MARKET-1501")
    print("="*70)
    print(f"Query dir: {query_dir}")
    print(f"Gallery dir: {gallery_dir}\n")
    
    features_by_id = {}
    total_images = 0
    
    # Procesar query
    if os.path.exists(query_dir):
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
                features = extract_liu2012_features(img_rgb)
                
                if person_id not in features_by_id:
                    features_by_id[person_id] = []
                features_by_id[person_id].append(features)
                total_images += 1
                
                if verbose:
                    elapsed = time.time() - query_start_time
                    if idx > 0:
                        avg_time = elapsed / idx
                        remaining = len(query_files) - idx
                        eta = avg_time * remaining / 60
                        percent = (idx / len(query_files)) * 100
                        sys.stdout.write(f'\r  Query: {idx}/{len(query_files)} ({percent:.1f}%) | ETA: {eta:.1f}m')
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
        print(f"  ✓ Query: {len(query_files)} imágenes procesadas ({query_elapsed:.1f}s)")
    else:
        print(f"⚠ No se encontró directorio query: {query_dir}")
    
    # Procesar gallery
    if os.path.exists(gallery_dir):
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
                features = extract_liu2012_features(img_rgb)
                
                if person_id not in features_by_id:
                    features_by_id[person_id] = []
                features_by_id[person_id].append(features)
                total_images += 1
                
                if verbose:
                    elapsed = time.time() - gallery_start_time
                    if idx > 0:
                        avg_time = elapsed / idx
                        remaining = len(gallery_files) - idx
                        eta = avg_time * remaining / 60
                        percent = (idx / len(gallery_files)) * 100
                        sys.stdout.write(f'\r  Gallery: {idx}/{len(gallery_files)} ({percent:.1f}%) | ETA: {eta:.1f}m')
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
        print(f"  ✓ Gallery: {len(gallery_files)} imágenes procesadas ({gallery_elapsed:.1f}s)")
    else:
        print(f"⚠ No se encontró directorio gallery: {gallery_dir}")
    
    print("\n" + "-"*70)
    print(f"RESUMEN:")
    print(f"  IDs únicos: {len(features_by_id)}")
    print(f"  Imágenes totales: {total_images}")
    print("="*70 + "\n")
    
    return features_by_id, total_images


def save_features(features_by_id, output_path):
    """
    Guarda las features extraídas en disco.
    
    Args:
        features_by_id: dict {id: [features...]}
        output_path: ruta donde guardar el archivo
    """
    print(f"\nGuardando features en: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(features_by_id, f)
    print(f"✓ Features guardadas exitosamente")


def load_features(input_path):
    """
    Carga features previamente guardadas desde disco.
    
    Args:
        input_path: ruta del archivo con features guardadas
    
    Returns:
        features_by_id: dict {id: [features...]}
    """
    print(f"Cargando features desde: {input_path}")
    with open(input_path, 'rb') as f:
        features_by_id = pickle.load(f)
    print(f"✓ Features cargadas exitosamente")
    return features_by_id


def evaluate_market1501_complete(features_by_id, trials=10, verbose=True):
    """
    Evalúa CMC en Market-1501 y muestra resultados.
    
    Args:
        features_by_id: dict {id: [features...]}
        trials: número de trials
        verbose: si True, muestra progreso detallado
    """
    print("="*70)
    print("PASO 2: EVALUACIÓN CMC - MARKET-1501")
    print("="*70)
    print(f"Trials: {trials}")
    print(f"IDs disponibles: {len(features_by_id)}")
    
    # Contar IDs con múltiples imágenes
    ids_with_multiple = sum(1 for feats in features_by_id.values() if len(feats) > 1)
    print(f"IDs con múltiples imágenes: {ids_with_multiple}\n")
    
    if len(features_by_id) == 0:
        raise ValueError("No hay features disponibles para evaluación")
    
    if verbose:
        print("Ejecutando evaluación CMC...")
        print(f"Procesando {trials} trials...")
    else:
        print("Ejecutando evaluación CMC...")
    
    start_time = time.time()
    
    # Usar todos los IDs disponibles (o limitar a 750 como en el paper)
    p = min(750, len(features_by_id))
    
    results = evaluate_cmc_stills(
        features_by_id,
        trials=trials,
        p=p,
        ranks=(1, 5, 10, 20),
        verbose=verbose
    )
    
    elapsed_time = time.time() - start_time
    
    if verbose:
        print(f"✓ Evaluación completada en {elapsed_time:.2f} segundos")
    
    print("\n" + "="*70)
    print("RESULTADOS CMC - MARKET-1501")
    print("="*70)
    print(f"\nConfiguración:")
    print(f"  Trials: {trials}")
    print(f"  IDs por trial: {p}")
    print(f"  Tiempo de evaluación: {elapsed_time:.2f} segundos")
    print(f"\nResultados:")
    print("-"*70)
    for rank, accuracy in sorted(results.items()):
        print(f"  Rank-{rank:2d}: {accuracy:.4f} ({accuracy*100:6.2f}%)")
    print("="*70 + "\n")
    
    return results


def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluación completa Liu 2012 en Market-1501')
    parser.add_argument('--no-verbose', action='store_true', 
                       help='Desactivar modo verbose (menos output)')
    parser.add_argument('--trials', type=int, default=10,
                       help='Número de trials para evaluación CMC (default: 10)')
    parser.add_argument('--query-dir', type=str, 
                       default='Market-1501-v15.09.15/query',
                       help='Directorio con imágenes query')
    parser.add_argument('--gallery-dir', type=str, 
                       default='Market-1501-v15.09.15/bounding_box_test',
                       help='Directorio con imágenes gallery')
    parser.add_argument('--features-file', type=str, default='features_market1501.pkl',
                       help='Archivo para guardar/cargar features (default: features_market1501.pkl)')
    parser.add_argument('--skip-extraction', action='store_true',
                       help='Saltar extracción y cargar features desde archivo')
    parser.add_argument('--skip-saving', action='store_true',
                       help='No guardar features después de extraerlas')
    
    args = parser.parse_args()
    
    verbose = not args.no_verbose
    
    print("\n" + "="*70)
    print("EVALUACIÓN COMPLETA - LIU 2012 EN MARKET-1501")
    print("="*70)
    if verbose:
        print("Modo: VERBOSE (progreso detallado)")
    else:
        print("Modo: SILENCIOSO")
    print("="*70 + "\n")
    
    # Configuración
    query_dir = args.query_dir
    gallery_dir = args.gallery_dir
    trials = args.trials
    features_file = args.features_file
    skip_extraction = args.skip_extraction
    skip_saving = args.skip_saving
    
    if verbose:
        print(f"Configuración:")
        print(f"  Query dir: {query_dir}")
        print(f"  Gallery dir: {gallery_dir}")
        print(f"  Archivo de features: {features_file}")
        print()
    
    # Verificar que existen los directorios
    if not skip_extraction:
        if not os.path.exists(query_dir):
            print(f"❌ Error: No se encontró el directorio query: {query_dir}")
            return
        if not os.path.exists(gallery_dir):
            print(f"❌ Error: No se encontró el directorio gallery: {gallery_dir}")
            return
    
    try:
        # PASO 1: Extraer o cargar features
        if skip_extraction:
            # Cargar features desde archivo
            if not os.path.exists(features_file):
                print(f"❌ Error: No se encontró el archivo de features: {features_file}")
                print("   Ejecuta sin --skip-extraction para extraer features primero")
                return
            features_by_id = load_features(features_file)
            total_images = sum(len(feats) for feats in features_by_id.values())
        else:
            # Extraer features
            start_time = time.time()
            features_by_id, total_images = extract_features_market1501(
                query_dir,
                gallery_dir,
                verbose=verbose
            )
            extraction_time = time.time() - start_time
            
            if verbose:
                fps_avg = total_images / extraction_time if extraction_time > 0 else 0
                print(f"\n⏱ Tiempo de extracción: {extraction_time:.2f} segundos ({extraction_time/60:.2f} minutos)")
                print(f"   Velocidad promedio: {fps_avg:.2f} imágenes/segundo\n")
            else:
                print(f"⏱ Tiempo de extracción: {extraction_time:.2f} segundos ({extraction_time/60:.2f} minutos)\n")
            
            # Guardar features si no se especifica skip_saving
            if not skip_saving:
                save_features(features_by_id, features_file)
        
        # PASO 2: Evaluar CMC
        results = evaluate_market1501_complete(features_by_id, trials=trials, verbose=verbose)
        
        print("✅ Evaluación completada exitosamente!")
        print("\nResultados finales:")
        for rank, accuracy in sorted(results.items()):
            print(f"  Rank-{rank}: {accuracy*100:.2f}%")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

