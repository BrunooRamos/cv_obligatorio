"""
Ejemplo de uso de la implementación Liu 2012.

Este script muestra cómo:
1. Extraer features de una imagen
2. Evaluar CMC en datasets de stills
3. Evaluar CMC en iLIDS-VID con pooling por tracklet
"""
import numpy as np
from PIL import Image
from extractor import extract_liu2012_features
from eval_stills import evaluate_cmc_stills
from eval_ilidsvid import evaluate_cmc_ilidsvid
from constants import DIMS_TOTAL


def example_extract_features():
    """Ejemplo de extracción de features de una imagen."""
    print("=" * 60)
    print("Ejemplo 1: Extracción de features")
    print("=" * 60)
    
    # Crear una imagen de prueba (RGB uint8)
    img = np.random.randint(0, 255, size=(256, 128, 3), dtype=np.uint8)
    
    # Extraer features
    features = extract_liu2012_features(img)
    
    print(f"Imagen de entrada: {img.shape}, dtype={img.dtype}")
    print(f"Features extraídos: shape={features.shape}, dtype={features.dtype}")
    print(f"Dimensión esperada: {DIMS_TOTAL}")
    print(f"✓ Features válidos: {features.shape[0] == DIMS_TOTAL}")
    print(f"✓ Sin NaN: {not np.any(np.isnan(features))}")
    print(f"✓ Sin Inf: {not np.any(np.isinf(features))}")
    print()


def example_eval_stills():
    """Ejemplo de evaluación CMC en stills (simulado)."""
    print("=" * 60)
    print("Ejemplo 2: Evaluación CMC en stills")
    print("=" * 60)
    
    # Simular features para 100 IDs, cada uno con 2-5 imágenes
    features_by_id = {}
    np.random.seed(42)
    
    for id in range(100):
        n_images = np.random.randint(2, 6)
        features_by_id[id] = [
            np.random.rand(DIMS_TOTAL).astype(np.float32) for _ in range(n_images)
        ]
    
    # Evaluar CMC
    results = evaluate_cmc_stills(features_by_id, p=50, trials=5, ranks=(1, 5, 10, 20))
    
    print("Resultados CMC (5 trials, p=50):")
    for rank, accuracy in results.items():
        print(f"  Rank-{rank}: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print()


def example_eval_ilidsvid():
    """Ejemplo de evaluación CMC en iLIDS-VID (simulado)."""
    print("=" * 60)
    print("Ejemplo 3: Evaluación CMC en iLIDS-VID")
    print("=" * 60)
    
    # Simular tracklets para 50 IDs, cada uno con cámaras A y B
    tracklets = {}
    np.random.seed(42)
    
    for id in range(50):
        # Cada cámara tiene un tracklet con 10-30 frames
        n_frames_a = np.random.randint(10, 31)
        n_frames_b = np.random.randint(10, 31)
        
        tracklets[id] = {
            'A': [np.random.rand(DIMS_TOTAL).astype(np.float32) for _ in range(n_frames_a)],
            'B': [np.random.rand(DIMS_TOTAL).astype(np.float32) for _ in range(n_frames_b)]
        }
    
    # Evaluar CMC
    results = evaluate_cmc_ilidsvid(tracklets, trials=5, ranks=(1, 5, 10, 20), pool_method='mean')
    
    print("Resultados CMC iLIDS-VID (5 trials, pooling=mean):")
    for rank, accuracy in results.items():
        print(f"  Rank-{rank}: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Ejemplos de uso - Implementación Liu 2012")
    print("=" * 60 + "\n")
    
    try:
        example_extract_features()
        example_eval_stills()
        example_eval_ilidsvid()
        
        print("=" * 60)
        print("✓ Todos los ejemplos se ejecutaron correctamente")
        print("=" * 60)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()



