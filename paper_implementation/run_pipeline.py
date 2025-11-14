"""
Script guía paso a paso para ejecutar el pipeline completo Liu 2012.

ORDEN DE EJECUCIÓN:
1. Primero: Extraer features de todas las imágenes/tracklets
2. Segundo: Evaluar CMC con las features extraídas
"""
import numpy as np
from PIL import Image
import os
from pathlib import Path
from extractor import extract_liu2012_features
from eval_stills import evaluate_cmc_stills
from eval_ilidsvid import evaluate_cmc_ilidsvid
from constants import DIMS_TOTAL


def load_image_rgb(image_path):
    """
    Carga una imagen y la convierte a RGB uint8.
    
    Args:
        image_path: ruta a la imagen
    
    Returns:
        img: array (H, W, 3) uint8 RGB
    """
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return np.array(img, dtype=np.uint8)


# ============================================================================
# PASO 1: EXTRAER FEATURES
# ============================================================================

def extract_features_stills(image_dir, id_from_filename=True):
    """
    Extrae features de todas las imágenes en un directorio.
    
    Args:
        image_dir: directorio con imágenes
        id_from_filename: si True, extrae ID del nombre del archivo
    
    Returns:
        features_by_id: dict {id: [features...]}
    """
    print(f"Extrayendo features de imágenes en: {image_dir}")
    features_by_id = {}
    
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        
        try:
            # Cargar imagen
            img_rgb = load_image_rgb(img_path)
            
            # Extraer features
            features = extract_liu2012_features(img_rgb)
            
            # Extraer ID del nombre del archivo (ej: "001_c1_001.jpg" -> id=1)
            if id_from_filename:
                # Intentar extraer ID (asumiendo formato común)
                parts = img_file.split('_')
                if len(parts) > 0:
                    try:
                        id = int(parts[0])
                    except:
                        id = hash(img_file) % 10000  # Fallback
                else:
                    id = hash(img_file) % 10000
            else:
                id = hash(img_file) % 10000
            
            if id not in features_by_id:
                features_by_id[id] = []
            features_by_id[id].append(features)
            
        except Exception as e:
            print(f"Error procesando {img_file}: {e}")
            continue
    
    print(f"✓ Extraídas features de {len(features_by_id)} IDs")
    return features_by_id


def extract_features_ilidsvid(sequences_dir):
    """
    Extrae features de tracklets en iLIDS-VID.
    
    Args:
        sequences_dir: directorio con subdirectorios cam1/ y cam2/
    
    Returns:
        tracklets: dict {id: {cam: [features...]}}
    """
    print(f"Extrayendo features de tracklets en: {sequences_dir}")
    tracklets = {}
    
    # Buscar directorios de cámaras
    cam_dirs = [d for d in os.listdir(sequences_dir) if os.path.isdir(os.path.join(sequences_dir, d)) and 'cam' in d.lower()]
    
    for cam_dir in sorted(cam_dirs):
        cam_path = os.path.join(sequences_dir, cam_dir)
        cam_name = cam_dir  # ej: "cam1"
        
        print(f"  Procesando {cam_name}...")
        
        # Buscar subdirectorios de personas (IDs)
        person_dirs = [d for d in os.listdir(cam_path) if os.path.isdir(os.path.join(cam_path, d))]
        
        for person_dir in sorted(person_dirs):
            person_path = os.path.join(cam_path, person_dir)
            
            # Extraer ID (ej: "001" -> 1)
            try:
                person_id = int(person_dir)
            except:
                person_id = hash(person_dir) % 10000
            
            if person_id not in tracklets:
                tracklets[person_id] = {}
            
            # Cargar todos los frames del tracklet
            frame_files = sorted([f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            
            tracklet_features = []
            for frame_file in frame_files:
                frame_path = os.path.join(person_path, frame_file)
                try:
                    img_rgb = load_image_rgb(frame_path)
                    features = extract_liu2012_features(img_rgb)
                    tracklet_features.append(features)
                except Exception as e:
                    print(f"    Error en {frame_file}: {e}")
                    continue
            
            if len(tracklet_features) > 0:
                tracklets[person_id][cam_name] = tracklet_features
        
        print(f"  ✓ {cam_name} procesado")
    
    print(f"✓ Extraídas features de {len(tracklets)} IDs")
    return tracklets


# ============================================================================
# PASO 2: EVALUAR CMC
# ============================================================================

def evaluate_stills_pipeline(features_by_id, p=50, trials=10):
    """
    Evalúa CMC en dataset de stills.
    
    Args:
        features_by_id: dict {id: [features...]}
        p: número de IDs a usar (50 para i-LIDS MCTS, 316 para VIPeR)
        trials: número de trials
    """
    print("\n" + "="*60)
    print("EVALUACIÓN CMC - STILLS")
    print("="*60)
    
    results = evaluate_cmc_stills(features_by_id, p=p, trials=trials, ranks=(1, 5, 10, 20))
    
    print(f"\nResultados CMC (p={p}, {trials} trials):")
    print("-" * 60)
    for rank, accuracy in results.items():
        print(f"  Rank-{rank:2d}: {accuracy:.4f} ({accuracy*100:6.2f}%)")
    print("="*60 + "\n")
    
    return results


def evaluate_ilidsvid_pipeline(tracklets, trials=10, pool_method='mean'):
    """
    Evalúa CMC en iLIDS-VID.
    
    Args:
        tracklets: dict {id: {cam: [features...]}}
        trials: número de trials
        pool_method: 'mean' o 'median'
    """
    print("\n" + "="*60)
    print("EVALUACIÓN CMC - iLIDS-VID")
    print("="*60)
    
    results = evaluate_cmc_ilidsvid(tracklets, trials=trials, ranks=(1, 5, 10, 20), pool_method=pool_method)
    
    print(f"\nResultados CMC (pooling={pool_method}, {trials} trials):")
    print("-" * 60)
    for rank, accuracy in results.items():
        print(f"  Rank-{rank:2d}: {accuracy:.4f} ({accuracy*100:6.2f}%)")
    print("="*60 + "\n")
    
    return results


# ============================================================================
# EJEMPLO DE USO COMPLETO
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("PIPELINE COMPLETO - LIU 2012")
    print("="*60)
    print("\nORDEN DE EJECUCIÓN:")
    print("1. PASO 1: Extraer features de imágenes/tracklets")
    print("2. PASO 2: Evaluar CMC con las features extraídas")
    print("\n" + "="*60 + "\n")
    
    # ========================================================================
    # OPCIÓN A: Evaluar en iLIDS-VID (video)
    # ========================================================================
    print("OPCIÓN A: iLIDS-VID (video)")
    print("-" * 60)
    
    ilidsvid_sequences = "iLIDS-VID/i-LIDS-VID/sequences"
    if os.path.exists(ilidsvid_sequences):
        print(f"\n✓ Encontrado: {ilidsvid_sequences}")
        print("\nPara ejecutar:")
        print("  tracklets = extract_features_ilidsvid('iLIDS-VID/i-LIDS-VID/sequences')")
        print("  results = evaluate_ilidsvid_pipeline(tracklets, trials=10)")
    else:
        print(f"\n✗ No encontrado: {ilidsvid_sequences}")
    
    # ========================================================================
    # OPCIÓN B: Evaluar en stills (si tienes i-LIDS MCTS o VIPeR)
    # ========================================================================
    print("\n" + "-" * 60)
    print("OPCIÓN B: Stills (i-LIDS MCTS / VIPeR)")
    print("-" * 60)
    print("\nSi tienes imágenes de i-LIDS MCTS o VIPeR:")
    print("  features_by_id = extract_features_stills('ruta/a/imagenes')")
    print("  results = evaluate_stills_pipeline(features_by_id, p=50)  # p=50 para MCTS, p=316 para VIPeR")
    
    # ========================================================================
    # EJEMPLO CON DATOS SIMULADOS (para probar)
    # ========================================================================
    print("\n" + "="*60)
    print("EJEMPLO RÁPIDO CON DATOS SIMULADOS")
    print("="*60)
    
    # Simular features para probar
    print("\n1. Simulando features de stills...")
    np.random.seed(42)
    features_by_id = {}
    for id in range(100):
        n_images = np.random.randint(2, 6)
        features_by_id[id] = [
            np.random.rand(DIMS_TOTAL).astype(np.float32) for _ in range(n_images)
        ]
    
    print("2. Evaluando CMC...")
    results = evaluate_stills_pipeline(features_by_id, p=50, trials=5)
    
    print("\n✓ Pipeline completado exitosamente!")
    print("\nPara usar con tus datos reales:")
    print("  1. Ejecuta extract_features_stills() o extract_features_ilidsvid()")
    print("  2. Ejecuta evaluate_stills_pipeline() o evaluate_ilidsvid_pipeline()")



