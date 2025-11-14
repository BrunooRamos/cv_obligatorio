#!/usr/bin/env python3
"""
Script completo para evaluar Liu 2012 en iLIDS-VID.

Este script:
1. Extrae features de todos los tracklets en iLIDS-VID
2. Evalúa CMC con pooling por tracklet y cross-camera
3. Muestra los resultados

Ejecutar: python3 eval_ilidsvid_complete.py
"""
import numpy as np
from PIL import Image
import os
from pathlib import Path
import time
import sys
import pickle
from extractor import extract_liu2012_features
from eval_ilidsvid import evaluate_cmc_ilidsvid
from constants import DIMS_TOTAL


def load_image_rgb(image_path):
    """Carga una imagen y la convierte a RGB uint8."""
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return np.array(img, dtype=np.uint8)


def extract_features_ilidsvid(sequences_dir, verbose=True, max_frames_per_tracklet=10):
    """
    Extrae features de todos los tracklets en iLIDS-VID.
    
    Args:
        sequences_dir: directorio con subdirectorios cam1/ y cam2/
        verbose: si True, muestra progreso detallado
        max_frames_per_tracklet: máximo número de frames a procesar por tracklet (default: 10)
    
    Returns:
        tracklets: dict {id: {cam: [features...]}}
    """
    print("="*70)
    print("PASO 1: EXTRACCIÓN DE FEATURES - iLIDS-VID")
    print("="*70)
    print(f"Directorio: {sequences_dir}\n")
    
    tracklets = {}
    
    # Buscar directorios de cámaras
    if not os.path.exists(sequences_dir):
        raise FileNotFoundError(f"No se encontró el directorio: {sequences_dir}")
    
    cam_dirs = [d for d in os.listdir(sequences_dir) 
                if os.path.isdir(os.path.join(sequences_dir, d)) and 'cam' in d.lower()]
    
    if len(cam_dirs) == 0:
        raise ValueError(f"No se encontraron directorios de cámaras en: {sequences_dir}")
    
    print(f"Encontradas {len(cam_dirs)} cámaras: {cam_dirs}\n")
    
    total_frames = 0
    total_tracklets = 0
    
    # Contar total de IDs para progreso
    total_ids = 0
    for cam_dir in sorted(cam_dirs):
        cam_path = os.path.join(sequences_dir, cam_dir)
        person_dirs = [d for d in os.listdir(cam_path) 
                      if os.path.isdir(os.path.join(cam_path, d))]
        total_ids += len(person_dirs)
    
    if verbose:
        print(f"Total de IDs a procesar: {total_ids}\n")
    
    processed_ids = 0
    
    for cam_idx, cam_dir in enumerate(sorted(cam_dirs), 1):
        cam_path = os.path.join(sequences_dir, cam_dir)
        cam_name = cam_dir  # ej: "cam1"
        
        if verbose:
            print(f"[{cam_idx}/{len(cam_dirs)}] Procesando {cam_name}...")
        else:
            print(f"Procesando {cam_name}...")
        
        # Buscar subdirectorios de personas (IDs)
        person_dirs = [d for d in os.listdir(cam_path) 
                      if os.path.isdir(os.path.join(cam_path, d))]
        
        if verbose:
            print(f"  Encontrados {len(person_dirs)} IDs")
        else:
            print(f"  Encontrados {len(person_dirs)} IDs")
        
        frames_this_cam = 0
        tracklets_this_cam = 0
        cam_start_time = time.time()
        
        for idx, person_dir in enumerate(sorted(person_dirs), 1):
            person_path = os.path.join(cam_path, person_dir)
            
            # Extraer ID (ej: "001" -> 1)
            try:
                person_id = int(person_dir)
            except:
                # Si no es numérico, usar hash
                person_id = hash(person_dir) % 10000
            
            if person_id not in tracklets:
                tracklets[person_id] = {}
            
            # Cargar frames del tracklet (limitado a max_frames_per_tracklet)
            frame_files = sorted([f for f in os.listdir(person_path) 
                                 if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            
            if len(frame_files) == 0:
                continue
            
            # Limitar a max_frames_per_tracklet frames
            if len(frame_files) > max_frames_per_tracklet:
                # Tomar frames distribuidos uniformemente
                step = len(frame_files) / max_frames_per_tracklet
                indices = [int(i * step) for i in range(max_frames_per_tracklet)]
                frame_files = [frame_files[i] for i in indices]
            
            tracklet_features = []
            
            for frame_file in frame_files:
                frame_path = os.path.join(person_path, frame_file)
                try:
                    img_rgb = load_image_rgb(frame_path)
                    features = extract_liu2012_features(img_rgb)
                    tracklet_features.append(features)
                    frames_this_cam += 1
                except Exception as e:
                    if verbose:
                        print(f"\n    ⚠ Error en {frame_file}: {e}")
                    continue
            
            if len(tracklet_features) > 0:
                tracklets[person_id][cam_name] = tracklet_features
                tracklets_this_cam += 1
            
            processed_ids += 1
            
            # Mostrar progreso de cada ID
            if verbose:
                elapsed = time.time() - cam_start_time
                if idx > 0:
                    avg_time_per_id = elapsed / idx
                    remaining_ids = len(person_dirs) - idx
                    eta_seconds = avg_time_per_id * remaining_ids
                    eta_minutes = eta_seconds / 60
                    
                    # Calcular porcentaje
                    percent = (idx / len(person_dirs)) * 100
                    
                    sys.stdout.write(f'\r  {cam_name}: {idx}/{len(person_dirs)} IDs ({percent:.1f}%) | '
                                   f'ETA: {eta_minutes:.1f}m | {frames_this_cam} frames')
                    sys.stdout.flush()
                else:
                    sys.stdout.write(f'\r  {cam_name}: {idx}/{len(person_dirs)} IDs')
                    sys.stdout.flush()
                
                if idx == len(person_dirs):
                    print()  # Nueva línea al completar
        
        total_frames += frames_this_cam
        total_tracklets += tracklets_this_cam
        cam_elapsed = time.time() - cam_start_time
        
        if verbose:
            print(f"  ✓ {cam_name}: {tracklets_this_cam} tracklets, {frames_this_cam} frames")
        else:
            print(f"  ✓ {cam_name}: {tracklets_this_cam} tracklets, {frames_this_cam} frames procesados")
    
    print("\n" + "-"*70)
    print(f"RESUMEN:")
    print(f"  IDs únicos: {len(tracklets)}")
    print(f"  Tracklets totales: {total_tracklets}")
    print(f"  Frames totales: {total_frames}")
    
    # Contar IDs con ambas cámaras
    ids_both_cams = sum(1 for id, cams in tracklets.items() if len(cams) >= 2)
    print(f"  IDs con ambas cámaras: {ids_both_cams}")
    print("="*70 + "\n")
    
    return tracklets, total_frames


def save_features(tracklets, output_path):
    """
    Guarda las features extraídas en disco.
    
    Args:
        tracklets: dict {id: {cam: [features...]}}
        output_path: ruta donde guardar el archivo
    """
    print(f"\nGuardando features en: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(tracklets, f)
    print(f"✓ Features guardadas exitosamente")


def load_features(input_path):
    """
    Carga features previamente guardadas desde disco.
    
    Args:
        input_path: ruta del archivo con features guardadas
    
    Returns:
        tracklets: dict {id: {cam: [features...]}}
    """
    print(f"Cargando features desde: {input_path}")
    with open(input_path, 'rb') as f:
        tracklets = pickle.load(f)
    print(f"✓ Features cargadas exitosamente")
    return tracklets


def evaluate_ilidsvid_complete(tracklets, trials=10, pool_method='mean', verbose=True):
    """
    Evalúa CMC en iLIDS-VID y muestra resultados.
    
    Args:
        tracklets: dict {id: {cam: [features...]}}
        trials: número de trials
        pool_method: 'mean' o 'median'
        verbose: si True, muestra progreso detallado
    """
    print("="*70)
    print("PASO 2: EVALUACIÓN CMC - iLIDS-VID")
    print("="*70)
    print(f"Trials: {trials}")
    print(f"Pooling: {pool_method}")
    print(f"IDs disponibles: {len(tracklets)}")
    
    # Contar IDs con ambas cámaras
    ids_both_cams = sum(1 for id, cams in tracklets.items() if len(cams) >= 2)
    print(f"IDs con ambas cámaras: {ids_both_cams}\n")
    
    if ids_both_cams == 0:
        raise ValueError("No hay IDs con ambas cámaras disponibles para evaluación cross-camera")
    
    if verbose:
        print("Ejecutando evaluación CMC...")
        print(f"Procesando {trials} trials...")
    else:
        print("Ejecutando evaluación CMC...")
    
    start_time = time.time()
    
    results = evaluate_cmc_ilidsvid(
        tracklets, 
        trials=trials, 
        ranks=(1, 5, 10, 20), 
        pool_method=pool_method,
        verbose=verbose
    )
    
    elapsed_time = time.time() - start_time
    
    if verbose:
        print(f"✓ Evaluación completada en {elapsed_time:.2f} segundos")
    
    print("\n" + "="*70)
    print("RESULTADOS CMC - iLIDS-VID")
    print("="*70)
    print(f"\nConfiguración:")
    print(f"  Trials: {trials}")
    print(f"  Pooling: {pool_method}")
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
    
    parser = argparse.ArgumentParser(description='Evaluación completa Liu 2012 en iLIDS-VID')
    parser.add_argument('--no-verbose', action='store_true', 
                       help='Desactivar modo verbose (menos output)')
    parser.add_argument('--trials', type=int, default=10,
                       help='Número de trials para evaluación CMC (default: 10)')
    parser.add_argument('--pool-method', choices=['mean', 'median'], default='mean',
                       help='Método de pooling para tracklets (default: mean)')
    parser.add_argument('--sequences-dir', type=str, 
                       default='iLIDS-VID/i-LIDS-VID/sequences',
                       help='Directorio con secuencias de iLIDS-VID')
    parser.add_argument('--max-frames', type=int, default=10,
                       help='Máximo número de frames por tracklet (default: 10)')
    parser.add_argument('--features-file', type=str, default='features_ilidsvid.pkl',
                       help='Archivo para guardar/cargar features (default: features_ilidsvid.pkl)')
    parser.add_argument('--skip-extraction', action='store_true',
                       help='Saltar extracción y cargar features desde archivo')
    parser.add_argument('--skip-saving', action='store_true',
                       help='No guardar features después de extraerlas')
    
    args = parser.parse_args()
    
    verbose = not args.no_verbose
    
    print("\n" + "="*70)
    print("EVALUACIÓN COMPLETA - LIU 2012 EN iLIDS-VID")
    print("="*70)
    if verbose:
        print("Modo: VERBOSE (progreso detallado)")
    else:
        print("Modo: SILENCIOSO")
    print("="*70 + "\n")
    
    # Configuración
    sequences_dir = args.sequences_dir
    trials = args.trials
    pool_method = args.pool_method
    max_frames = args.max_frames
    features_file = args.features_file
    skip_extraction = args.skip_extraction
    skip_saving = args.skip_saving
    
    if verbose:
        print(f"Configuración:")
        print(f"  Máximo frames por tracklet: {max_frames}")
        print(f"  Archivo de features: {features_file}")
        print()
    
    # Verificar que existe el directorio
    if not os.path.exists(sequences_dir):
        print(f"❌ Error: No se encontró el directorio {sequences_dir}")
        print(f"\nEstructura esperada:")
        print(f"  {sequences_dir}/")
        print(f"    cam1/")
        print(f"      person_id_1/")
        print(f"        frame1.png, frame2.png, ...")
        print(f"      person_id_2/")
        print(f"        ...")
        print(f"    cam2/")
        print(f"      ...")
        return
    
    try:
        # PASO 1: Extraer o cargar features
        if skip_extraction:
            # Cargar features desde archivo
            if not os.path.exists(features_file):
                print(f"❌ Error: No se encontró el archivo de features: {features_file}")
                print("   Ejecuta sin --skip-extraction para extraer features primero")
                return
            tracklets = load_features(features_file)
            total_frames = sum(sum(len(frames) for frames in cams.values()) 
                             for cams in tracklets.values())
        else:
            # Extraer features
            start_time = time.time()
            tracklets, total_frames = extract_features_ilidsvid(
                sequences_dir, 
                verbose=verbose, 
                max_frames_per_tracklet=max_frames
            )
            extraction_time = time.time() - start_time
            
            if verbose:
                fps_avg = total_frames / extraction_time if extraction_time > 0 else 0
                print(f"\n⏱ Tiempo de extracción: {extraction_time:.2f} segundos ({extraction_time/60:.2f} minutos)")
                print(f"   Velocidad promedio: {fps_avg:.2f} fps\n")
            else:
                print(f"⏱ Tiempo de extracción: {extraction_time:.2f} segundos ({extraction_time/60:.2f} minutos)\n")
            
            # Guardar features si no se especifica skip_saving
            if not skip_saving:
                save_features(tracklets, features_file)
        
        # PASO 2: Evaluar CMC
        results = evaluate_ilidsvid_complete(tracklets, trials=trials, pool_method=pool_method, verbose=verbose)
        
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

