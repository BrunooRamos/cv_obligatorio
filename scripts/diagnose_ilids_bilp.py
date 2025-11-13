"""Script de diagnóstico exhaustivo para iLIDS-VID con BILP."""

import sys
sys.path.append('/app')

import numpy as np
from bilp.utils import load_features
from bilp.distance import compute_distance_matrix_fast

def diagnose_ilids_vid_bilp():
    """Diagnostica problemas exhaustivos en features BILP de iLIDS-VID."""
    
    query_file = 'data/features/ilidsvid_query.npz'
    gallery_file = 'data/features/ilidsvid_gallery.npz'
    
    print("="*70)
    print("DIAGNÓSTICO EXHAUSTIVO iLIDS-VID - BILP")
    print("="*70)
    
    # Cargar datos
    print("\n1. CARGANDO FEATURES...")
    query_color, query_texture, query_metadata = load_features(query_file)
    gallery_color, gallery_texture, gallery_metadata = load_features(gallery_file)
    
    query_ids = np.array(query_metadata['person_ids'])
    gallery_ids = np.array(gallery_metadata['person_ids'])
    query_cams = np.array(query_metadata['camera_ids'])
    gallery_cams = np.array(gallery_metadata['camera_ids'])
    
    print(f"   Query: {query_color.shape}, {query_texture.shape}")
    print(f"   Gallery: {gallery_color.shape}, {gallery_texture.shape}")
    
    # Verificar estructura básica
    print("\n2. VERIFICACIÓN DE ESTRUCTURA:")
    print(f"   Query sequences: {len(query_ids)}")
    print(f"   Gallery sequences: {len(gallery_ids)}")
    print(f"   Query IDs únicos: {len(np.unique(query_ids))}")
    print(f"   Gallery IDs únicos: {len(np.unique(gallery_ids))}")
    print(f"   Query IDs range: [{query_ids.min()}, {query_ids.max()}]")
    print(f"   Gallery IDs range: [{gallery_ids.min()}, {gallery_ids.max()}]")
    
    # VERIFICACIÓN CRÍTICA: Orden de IDs
    print("\n3. VERIFICACIÓN CRÍTICA: ORDEN DE PERSON IDs")
    print(f"   Query IDs (primeros 20): {query_ids[:20].tolist()}")
    print(f"   Gallery IDs (primeros 20): {gallery_ids[:20].tolist()}")
    
    if np.array_equal(query_ids, gallery_ids):
        print("   ✓ Los IDs están en el mismo orden")
        order_ok = True
    else:
        print("   ✗ PROBLEMA CRÍTICO: Los IDs NO están en el mismo orden!")
        print("   Esto causaría matching incorrecto entre query y gallery.")
        order_ok = False
        
        # Mostrar diferencias
        diff_indices = np.where(query_ids != gallery_ids)[0]
        print(f"   Primeras diferencias en índices: {diff_indices[:10].tolist()}")
        print(f"   Query IDs en diferencias: {query_ids[diff_indices[:10]].tolist()}")
        print(f"   Gallery IDs en diferencias: {gallery_ids[diff_indices[:10]].tolist()}")
    
    # Verificar correspondencia 1:1
    print("\n4. VERIFICACIÓN DE CORRESPONDENCIA 1:1:")
    unique_query_ids = np.unique(query_ids)
    unique_gallery_ids = np.unique(gallery_ids)
    
    if np.array_equal(np.sort(unique_query_ids), np.sort(unique_gallery_ids)):
        print("   ✓ Todos los IDs de query tienen match en gallery")
        print(f"   IDs únicos: {len(unique_query_ids)}")
    else:
        missing_in_gallery = set(unique_query_ids) - set(unique_gallery_ids)
        missing_in_query = set(unique_gallery_ids) - set(unique_query_ids)
        if missing_in_gallery:
            print(f"   ✗ IDs en query pero no en gallery: {sorted(missing_in_gallery)}")
        if missing_in_query:
            print(f"   ✗ IDs en gallery pero no en query: {sorted(missing_in_query)}")
    
    # Verificar duplicados
    print("\n5. VERIFICACIÓN DE DUPLICADOS:")
    from collections import Counter
    query_counts = Counter(query_ids)
    gallery_counts = Counter(gallery_ids)
    
    query_duplicates = {id: count for id, count in query_counts.items() if count > 1}
    gallery_duplicates = {id: count for id, count in gallery_counts.items() if count > 1}
    
    if not query_duplicates and not gallery_duplicates:
        print("   ✓ No hay duplicados (cada ID aparece exactamente una vez)")
    else:
        if query_duplicates:
            print(f"   ✗ Duplicados en query: {dict(list(query_duplicates.items())[:5])}")
        if gallery_duplicates:
            print(f"   ✗ Duplicados en gallery: {dict(list(gallery_duplicates.items())[:5])}")
    
    # Verificar cámaras
    print("\n6. VERIFICACIÓN DE CÁMARAS:")
    print(f"   Query cameras únicos: {np.unique(query_cams)}")
    print(f"   Gallery cameras únicos: {np.unique(gallery_cams)}")
    if np.all(query_cams == 1) and np.all(gallery_cams == 2):
        print("   ✓ Query es cam1, Gallery es cam2 (correcto)")
    else:
        print("   ✗ PROBLEMA: Cámaras no coinciden con lo esperado")
    
    # Análisis de features
    print("\n7. ANÁLISIS DE FEATURES:")
    print(f"   Query color - Media: {query_color.mean():.6f}, Std: {query_color.std():.6f}")
    print(f"   Query color - Min: {query_color.min():.6f}, Max: {query_color.max():.6f}")
    print(f"   Query texture - Media: {query_texture.mean():.6f}, Std: {query_texture.std():.6f}")
    print(f"   Query texture - Min: {query_texture.min():.6f}, Max: {query_texture.max():.6f}")
    
    # Verificar si hay features completamente cero
    query_color_zero = np.sum(np.all(query_color == 0, axis=1))
    query_texture_zero = np.sum(np.all(query_texture == 0, axis=1))
    print(f"\n   Features completamente cero:")
    print(f"   Query color: {query_color_zero}/{len(query_color)}")
    print(f"   Query texture: {query_texture_zero}/{len(query_texture)}")
    
    # Calcular distancias para diagnóstico
    print("\n8. ANÁLISIS DE DISTANCIAS:")
    
    if order_ok:
        # Calcular distancias entre query[i] y gallery[i] (mismo ID, debería ser bajo)
        print("   Calculando distancias mismo ID (query[i] <-> gallery[i])...")
        same_id_distances_color = []
        same_id_distances_texture = []
        same_id_distances_combined = []
        
        for i in range(len(query_ids)):
            # Distancia color
            dist_color = np.sum(np.abs(query_color[i] - gallery_color[i]))
            same_id_distances_color.append(dist_color)
            
            # Distancia textura
            dist_texture = np.sum(np.abs(query_texture[i] - gallery_texture[i]))
            same_id_distances_texture.append(dist_texture)
            
            # Distancia combinada (alpha=0.5)
            dist_combined = 0.5 * dist_texture + 0.5 * dist_color
            same_id_distances_combined.append(dist_combined)
        
        same_id_distances_color = np.array(same_id_distances_color)
        same_id_distances_texture = np.array(same_id_distances_texture)
        same_id_distances_combined = np.array(same_id_distances_combined)
        
        print(f"\n   Distancias mismo ID (query[i] <-> gallery[i]):")
        print(f"   Color:")
        print(f"     Media: {same_id_distances_color.mean():.4f}")
        print(f"     Mediana: {np.median(same_id_distances_color):.4f}")
        print(f"     Min: {same_id_distances_color.min():.4f}, Max: {same_id_distances_color.max():.4f}")
        print(f"   Texture:")
        print(f"     Media: {same_id_distances_texture.mean():.4f}")
        print(f"     Mediana: {np.median(same_id_distances_texture):.4f}")
        print(f"     Min: {same_id_distances_texture.min():.4f}, Max: {same_id_distances_texture.max():.4f}")
        print(f"   Combinada (alpha=0.5):")
        print(f"     Media: {same_id_distances_combined.mean():.4f}")
        print(f"     Mediana: {np.median(same_id_distances_combined):.4f}")
        print(f"     Min: {same_id_distances_combined.min():.4f}, Max: {same_id_distances_combined.max():.4f}")
        
        # Calcular distancias entre diferentes IDs (deberían ser más altas)
        print("\n   Calculando distancias diferente ID (sample)...")
        different_id_distances = []
        sample_size = min(100, len(query_ids))
        
        for i in range(sample_size):
            q_id = query_ids[i]
            # Encontrar índices con diferente ID
            diff_indices = np.where(gallery_ids != q_id)[0]
            if len(diff_indices) > 0:
                # Calcular distancias a 10 diferentes IDs aleatorios
                np.random.seed(42)
                sample_indices = np.random.choice(diff_indices, min(10, len(diff_indices)), replace=False)
                
                for j in sample_indices:
                    dist_color = np.sum(np.abs(query_color[i] - gallery_color[j]))
                    dist_texture = np.sum(np.abs(query_texture[i] - gallery_texture[j]))
                    dist_combined = 0.5 * dist_texture + 0.5 * dist_color
                    different_id_distances.append(dist_combined)
        
        if different_id_distances:
            different_id_distances = np.array(different_id_distances)
            print(f"   Distancias diferente ID:")
            print(f"     Media: {different_id_distances.mean():.4f}")
            print(f"     Mediana: {np.median(different_id_distances):.4f}")
            print(f"     Min: {different_id_distances.min():.4f}, Max: {different_id_distances.max():.4f}")
            
            # Verificar separabilidad
            print(f"\n   SEPARABILIDAD:")
            print(f"     Media mismo ID: {same_id_distances_combined.mean():.4f}")
            print(f"     Media diferente ID: {different_id_distances.mean():.4f}")
            
            if same_id_distances_combined.mean() < different_id_distances.mean():
                separation_ratio = different_id_distances.mean() / same_id_distances_combined.mean()
                print(f"     Ratio (diferente/mismo): {separation_ratio:.2f}x")
                if separation_ratio > 1.2:
                    print(f"     ✓ Separación adecuada (>1.2x)")
                else:
                    print(f"     ✗ Separación insuficiente (<1.2x) - features no discriminativas")
            else:
                print(f"     ✗ PROBLEMA CRÍTICO: Mismo ID más lejano que diferente ID!")
                print(f"     Esto indica que las features no son discriminativas.")
    
    # Verificar metadata adicional
    print("\n9. METADATA ADICIONAL:")
    if 'num_frames' in query_metadata:
        query_num_frames = np.array(query_metadata['num_frames'])
        print(f"   Frames por secuencia (query):")
        print(f"     Media: {query_num_frames.mean():.1f}")
        print(f"     Min: {query_num_frames.min()}, Max: {query_num_frames.max()}")
    
    print("\n" + "="*70)
    print("FIN DEL DIAGNÓSTICO")
    print("="*70)
    
    # Resumen de problemas encontrados
    print("\nRESUMEN DE PROBLEMAS ENCONTRADOS:")
    problems = []
    
    if not order_ok:
        problems.append("CRÍTICO: Person IDs no están en el mismo orden entre query y gallery")
    
    if query_duplicates or gallery_duplicates:
        problems.append("CRÍTICO: Hay person IDs duplicados")
    
    if not (np.all(query_cams == 1) and np.all(gallery_cams == 2)):
        problems.append("CRÍTICO: Cámaras no coinciden (query debe ser cam1, gallery cam2)")
    
    if query_color_zero > 0 or query_texture_zero > 0:
        problems.append(f"ADVERTENCIA: {query_color_zero + query_texture_zero} secuencias con features completamente cero")
    
    if order_ok and len(same_id_distances_combined) > 0:
        if same_id_distances_combined.mean() >= different_id_distances.mean():
            problems.append("CRÍTICO: Features no discriminativas (mismo ID más lejano que diferente ID)")
        elif (different_id_distances.mean() / same_id_distances_combined.mean()) < 1.2:
            problems.append("ADVERTENCIA: Separación insuficiente entre mismo ID y diferente ID")
    
    if not problems:
        print("   ✓ No se encontraron problemas críticos")
    else:
        for i, problem in enumerate(problems, 1):
            print(f"   {i}. {problem}")


if __name__ == '__main__':
    diagnose_ilids_vid_bilp()

