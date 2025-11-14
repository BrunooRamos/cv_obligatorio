"""
Módulo para evaluación CMC en iLIDS-VID (video con pooling por tracklet).
Protocolo cross-camera con pooling mean/median por tracklet.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from constants import RANKS, TRIALS, N_COLOR_DIMS_PER_STRIPE, N_TEXTURE_DIMS_PER_STRIPE, N_STRIPES


def pool_tracklet_features_separate(
    color_features_list: List[np.ndarray],
    texture_features_list: List[np.ndarray],
    gabor_responses_list: List[List[List[np.ndarray]]],
    method: str = 'mean'
) -> Tuple[np.ndarray, np.ndarray, List[List[np.ndarray]]]:
    """
    Agrega features de color y textura de múltiples frames de un tracklet.
    
    Args:
        color_features_list: lista de arrays (768,) - features de color por frame
        texture_features_list: lista de arrays (2016,) - features de textura por frame
        gabor_responses_list: lista de listas de listas - respuestas Gabor por frame
        method: 'mean' o 'median'
    
    Returns:
        pooled_color: array (768,) con features de color agregadas
        pooled_texture: array (2016,) con features de textura agregadas
        pooled_gabor: lista de listas con respuestas Gabor agregadas
    """
    if len(color_features_list) == 0:
        raise ValueError("Lista de features vacía")
    
    # Pooling de features de color
    color_array = np.array(color_features_list)  # (N_frames, 768)
    if method == 'mean':
        pooled_color = np.mean(color_array, axis=0)
    elif method == 'median':
        pooled_color = np.median(color_array, axis=0)
    else:
        raise ValueError(f"Método de pooling desconocido: {method}")
    
    # Pooling de features de textura
    texture_array = np.array(texture_features_list)  # (N_frames, 2016)
    if method == 'mean':
        pooled_texture = np.mean(texture_array, axis=0)
    elif method == 'median':
        pooled_texture = np.median(texture_array, axis=0)
    
    # Pooling de respuestas Gabor (promediar energía por filtro)
    # gabor_responses_list es lista de listas: [frame][stripe][filter]
    n_frames = len(gabor_responses_list)
    n_stripes = len(gabor_responses_list[0]) if n_frames > 0 else 0
    n_filters = len(gabor_responses_list[0][0]) if n_frames > 0 and n_stripes > 0 else 0
    
    pooled_gabor = []
    for stripe_idx in range(n_stripes):
        stripe_gabor = []
        for filter_idx in range(n_filters):
            # Recopilar respuestas de este filtro en este stripe de todos los frames
            filter_responses = [gabor_responses_list[f][stripe_idx][filter_idx] 
                              for f in range(n_frames)]
            # Promediar respuestas 2D
            if method == 'mean':
                pooled_filter = np.mean([r for r in filter_responses], axis=0)
            else:  # median
                pooled_filter = np.median([r for r in filter_responses], axis=0)
            stripe_gabor.append(pooled_filter)
        pooled_gabor.append(stripe_gabor)
    
    # Normalización L1 por stripe
    pooled_color_normalized = np.zeros_like(pooled_color)
    pooled_texture_normalized = np.zeros_like(pooled_texture)
    
    for stripe_idx in range(N_STRIPES):
        # Normalizar color
        start_color = stripe_idx * N_COLOR_DIMS_PER_STRIPE
        end_color = (stripe_idx + 1) * N_COLOR_DIMS_PER_STRIPE
        stripe_color = pooled_color[start_color:end_color]
        total_color = np.sum(np.abs(stripe_color))
        if total_color > 0:
            stripe_color = stripe_color / total_color
        pooled_color_normalized[start_color:end_color] = stripe_color
        
        # Normalizar textura
        start_texture = stripe_idx * N_TEXTURE_DIMS_PER_STRIPE
        end_texture = (stripe_idx + 1) * N_TEXTURE_DIMS_PER_STRIPE
        stripe_texture = pooled_texture[start_texture:end_texture]
        total_texture = np.sum(np.abs(stripe_texture))
        if total_texture > 0:
            stripe_texture = stripe_texture / total_texture
        pooled_texture_normalized[start_texture:end_texture] = stripe_texture
    
    return (pooled_color_normalized.astype(np.float32),
            pooled_texture_normalized.astype(np.float32),
            pooled_gabor)


def pool_tracklet_features(features_list: List[np.ndarray], method: str = 'mean') -> np.ndarray:
    """
    Agrega features de múltiples frames de un tracklet.
    
    Args:
        features_list: lista de arrays (2784,) - features por frame
        method: 'mean' o 'median'
    
    Returns:
        pooled_features: array (2784,) con features agregadas
    """
    if len(features_list) == 0:
        raise ValueError("Lista de features vacía")
    
    features_array = np.array(features_list)  # (N_frames, 2784)
    
    if method == 'mean':
        pooled = np.mean(features_array, axis=0)
    elif method == 'median':
        pooled = np.median(features_array, axis=0)
    else:
        raise ValueError(f"Método de pooling desconocido: {method}")
    
    # Re-normalizar por bloques (opcional pero recomendado)
    # Cada stripe tiene 464 dims (128 color + 336 textura)
    # Normalizar color (128) y textura (336) por separado por stripe
    from constants import DIMS_PER_STRIPE, N_STRIPES
    
    pooled_normalized = np.zeros_like(pooled)
    for stripe_idx in range(N_STRIPES):
        start_idx = stripe_idx * DIMS_PER_STRIPE
        end_idx = (stripe_idx + 1) * DIMS_PER_STRIPE
        
        stripe_feats = pooled[start_idx:end_idx]
        
        # Normalización L1 del stripe
        total = np.sum(np.abs(stripe_feats))
        if total > 0:
            stripe_feats = stripe_feats / total
        
        pooled_normalized[start_idx:end_idx] = stripe_feats
    
    return pooled_normalized.astype(np.float32)


def pairwise_l1_distance(features1: np.ndarray, features2: np.ndarray) -> np.ndarray:
    """
    Calcula distancias L1 (cityblock) entre dos conjuntos de features.
    
    Args:
        features1: array (N1, D) de features
        features2: array (N2, D) de features
    
    Returns:
        distances: array (N1, N2) con distancias L1
    """
    diff = features1[:, np.newaxis, :] - features2[np.newaxis, :, :]
    distances = np.sum(np.abs(diff), axis=2)
    return distances


def pairwise_l1_distance_separate(
    color_features1: np.ndarray,
    texture_features1: np.ndarray,
    color_features2: np.ndarray,
    texture_features2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula distancias L1 separadas para color y textura.
    
    Args:
        color_features1: array (N1, D_color) de features de color
        texture_features1: array (N1, D_texture) de features de textura
        color_features2: array (N2, D_color) de features de color
        texture_features2: array (N2, D_texture) de features de textura
    
    Returns:
        color_distances: array (N1, N2) con distancias de color
        texture_distances: array (N1, N2) con distancias de textura
    """
    # Distancias de color
    color_diff = color_features1[:, np.newaxis, :] - color_features2[np.newaxis, :, :]
    color_distances = np.sum(np.abs(color_diff), axis=2)
    
    # Distancias de textura
    texture_diff = texture_features1[:, np.newaxis, :] - texture_features2[np.newaxis, :, :]
    texture_distances = np.sum(np.abs(texture_diff), axis=2)
    
    return color_distances, texture_distances


def compute_gating_metrics(
    color_features: np.ndarray,
    texture_features: np.ndarray,
    gabor_responses: List[List[np.ndarray]],
    texture_metric: str = 'entropy'
) -> Tuple[float, float]:
    """
    Calcula complejidad de textura y entropía cromática para gating.
    
    Args:
        color_features: array (768,) con features de color
        texture_features: array (2016,) con features de textura
        gabor_responses: lista de listas - respuestas Gabor por stripe y filtro
        texture_metric: métrica a usar para textura ('entropy', 'variance', 'gabor_energy')
    
    Returns:
        texture_complexity: Complejidad de textura T
        color_entropy: Entropía cromática C
    """
    from gating import (
        compute_gabor_energies_from_responses,
        compute_texture_complexity,
        compute_texture_entropy,
        compute_texture_variance,
        compute_color_entropy
    )
    
    # Calcular complejidad de textura usando la métrica seleccionada
    if texture_metric == 'entropy':
        # Usar entropía de los histogramas de textura (más discriminativa)
        texture_complexity = compute_texture_entropy(texture_features)
    elif texture_metric == 'variance':
        # Usar varianza entre stripes (mide variación espacial)
        texture_complexity = compute_texture_variance(texture_features)
    elif texture_metric == 'gabor_energy':
        # Usar energía Gabor (métrica original, pero tiene problemas)
        all_gabor_energies_aggregated = []
        for stripe_gabor in gabor_responses:
            gabor_energies = compute_gabor_energies_from_responses(stripe_gabor)
            all_gabor_energies_aggregated.append(gabor_energies)
        
        n_filters = len(all_gabor_energies_aggregated[0])
        aggregated_energies = np.zeros(n_filters, dtype=np.float32)
        for stripe_energies in all_gabor_energies_aggregated:
            aggregated_energies += stripe_energies
        
        texture_complexity = compute_texture_complexity(aggregated_energies)
    else:
        raise ValueError(f"Métrica de textura desconocida: {texture_metric}")
    
    # Calcular entropía cromática
    color_entropy = compute_color_entropy(color_features)
    
    return texture_complexity, color_entropy


def compute_ranks(distances: np.ndarray, probe_ids: np.ndarray, gallery_ids: np.ndarray) -> np.ndarray:
    """
    Calcula el rank de la identidad correcta para cada probe.
    
    Args:
        distances: array (N_probe, N_gallery) con distancias
        probe_ids: array (N_probe,) con IDs de probes
        gallery_ids: array (N_gallery,) con IDs de gallery
    
    Returns:
        ranks: array (N_probe,) con rank de cada probe (1-indexed)
    """
    ranks = []
    for i, probe_id in enumerate(probe_ids):
        sorted_indices = np.argsort(distances[i, :])
        correct_positions = np.where(gallery_ids[sorted_indices] == probe_id)[0]
        
        if len(correct_positions) > 0:
            rank = correct_positions[0] + 1
        else:
            rank = len(gallery_ids) + 1
        
        ranks.append(rank)
    
    return np.array(ranks)


def compute_cmc_curve(ranks: np.ndarray, max_rank: int) -> np.ndarray:
    """
    Calcula la curva CMC a partir de los ranks.
    
    Args:
        ranks: array (N_probe,) con ranks
        max_rank: rank máximo a calcular
    
    Returns:
        cmc: array (max_rank,) donde cmc[r-1] = % probes con rank <= r
    """
    cmc = np.zeros(max_rank, dtype=np.float32)
    n_probes = len(ranks)
    
    for r in range(1, max_rank + 1):
        cmc[r - 1] = np.sum(ranks <= r) / n_probes
    
    return cmc


def sample_ids_with_both_cameras(tracklets: Dict, p: int, random_state: Optional[int] = None) -> List:
    """
    Samplea p IDs que tengan ambas cámaras disponibles.
    
    Args:
        tracklets: dict {id: {cam: [features...]}}
        p: número de IDs a samplear
        random_state: seed para reproducibilidad
    
    Returns:
        sampled_ids: lista de p IDs
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Filtrar IDs con ambas cámaras
    valid_ids = []
    for id, cameras in tracklets.items():
        if len(cameras) >= 2:
            # Verificar que ambas cámaras tengan al menos un frame
            cam_names = list(cameras.keys())
            if len(cam_names) >= 2:
                has_frames = all(len(cameras[cam]) > 0 for cam in cam_names[:2])
                if has_frames:
                    valid_ids.append(id)
    
    if len(valid_ids) < p:
        p = min(p, len(valid_ids))
        if p == 0:
            raise ValueError("No hay IDs con ambas cámaras disponibles")
    
    sampled_ids = np.random.choice(valid_ids, size=p, replace=False).tolist()
    return sampled_ids


def build_cross_camera_gallery_and_probes(
    tracklets: Dict,
    ids: List,
    random_state: Optional[int] = None,
    pool_method: str = 'mean'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Construye gallery y probes cross-camera con pooling por tracklet.
    
    Args:
        tracklets: dict {id: {cam: [features...]}}
        ids: lista de IDs a usar
        random_state: seed para reproducibilidad
        pool_method: 'mean' o 'median' para pooling
    
    Returns:
        gallery_features: array (N_gallery, D)
        gallery_ids: array (N_gallery,)
        probe_features: array (N_probe, D)
        probe_ids: array (N_probe,)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    gallery_features = []
    gallery_ids = []
    probe_features = []
    probe_ids = []
    
    for id in ids:
        cameras = tracklets[id]
        cam_names = list(cameras.keys())
        
        if len(cam_names) < 2:
            continue
        
        # Usar primeras dos cámaras (típicamente A y B)
        cam_a = cam_names[0]
        cam_b = cam_names[1]
        
        # Gallery: cámara A
        if cam_a in cameras and len(cameras[cam_a]) > 0:
            # Si hay múltiples tracklets, elegir uno aleatorio o promediar todos
            # Opción: promediar todos los tracklets de la cámara
            gallery_feat = pool_tracklet_features(cameras[cam_a], method=pool_method)
            gallery_features.append(gallery_feat)
            gallery_ids.append(id)
        
        # Probe: cámara B
        if cam_b in cameras and len(cameras[cam_b]) > 0:
            # cameras[cam_b] es una lista de features (frames del tracklet)
            # Pooling de todos los frames del tracklet
            probe_feat = pool_tracklet_features(cameras[cam_b], method=pool_method)
            probe_features.append(probe_feat)
            probe_ids.append(id)
    
    if len(gallery_features) == 0 or len(probe_features) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    gallery_features = np.array(gallery_features)
    gallery_ids = np.array(gallery_ids)
    probe_features = np.array(probe_features)
    probe_ids = np.array(probe_ids)
    
    return gallery_features, gallery_ids, probe_features, probe_ids


def evaluate_cmc_ilidsvid(
    tracklets: Dict[int, Dict[str, List[np.ndarray]]],
    trials: int = TRIALS,
    ranks: Tuple[int, ...] = RANKS,
    pool_method: str = 'mean',
    p: Optional[int] = None,
    verbose: bool = False
) -> Dict[int, float]:
    """
    Evalúa CMC en iLIDS-VID con pooling por tracklet y cross-camera.
    
    Args:
        tracklets: dict {id: {cam: [features...]}} donde cada feature es array (2784,)
        trials: número de trials (default: 10)
        ranks: tupla de ranks a reportar (default: (1, 5, 10, 20))
        pool_method: 'mean' o 'median' para pooling (default: 'mean')
        p: número de IDs a usar por trial. Si None, usa min(50, #IDs_disponibles)
    
    Returns:
        results: dict {rank: accuracy} con accuracy promedio sobre trials
    """
    max_rank = max(ranks)
    cmc_accumulator = np.zeros(max_rank, dtype=np.float32)
    
    # Determinar p si no se especifica
    if p is None:
        # Contar IDs con ambas cámaras
        valid_ids = []
        for id, cameras in tracklets.items():
            if len(cameras) >= 2:
                cam_names = list(cameras.keys())
                if len(cam_names) >= 2:
                    has_frames = all(len(cameras[cam]) > 0 for cam in cam_names[:2])
                    if has_frames:
                        valid_ids.append(id)
        p = min(50, len(valid_ids))
    
    import sys
    import time
    eval_start_time = time.time()
    
    for trial in range(trials):
        # Seed fijo por trial para reproducibilidad
        trial_seed = trial * 42
        
        if verbose:
            elapsed = time.time() - eval_start_time
            if trial > 0:
                avg_time_per_trial = elapsed / trial
                remaining_trials = trials - trial
                eta_seconds = avg_time_per_trial * remaining_trials
                eta_minutes = eta_seconds / 60
                percent = (trial / trials) * 100
                sys.stdout.write(f'\r  Trial {trial+1}/{trials} ({percent:.1f}%) | ETA: {eta_minutes:.1f}m')
                sys.stdout.flush()
            else:
                sys.stdout.write(f'\r  Trial {trial+1}/{trials}')
                sys.stdout.flush()
        
        # Samplear IDs con ambas cámaras
        sampled_ids = sample_ids_with_both_cameras(tracklets, p, random_state=trial_seed)
        
        # Construir gallery y probes cross-camera
        gallery_feats, gallery_ids, probe_feats, probe_ids = build_cross_camera_gallery_and_probes(
            tracklets, sampled_ids, random_state=trial_seed + 1, pool_method=pool_method
        )
        
        if len(probe_feats) == 0:
            if verbose:
                print(f"\nWarning: Trial {trial} no tiene probes. Saltando.")
            continue
        
        # Calcular distancias
        distances = pairwise_l1_distance(probe_feats, gallery_feats)
        
        # Calcular ranks
        ranks_this_trial = compute_ranks(distances, probe_ids, gallery_ids)
        
        # Calcular CMC
        cmc_this_trial = compute_cmc_curve(ranks_this_trial, max_rank)
        cmc_accumulator += cmc_this_trial
    
    if verbose:
        print()  # Nueva línea al finalizar
    
    # Promediar sobre trials
    cmc_accumulator /= trials
    
    # Construir resultados
    results = {r: float(cmc_accumulator[r - 1]) for r in ranks}
    
    return results


def build_cross_camera_gallery_and_probes_with_gating(
    tracklets: Dict,
    ids: List,
    random_state: Optional[int] = None,
    pool_method: str = 'mean'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List, List]:
    """
    Construye gallery y probes cross-camera con información de gating.
    
    Args:
        tracklets: dict {id: {cam: [feature_dicts...]}} donde cada feature_dict tiene:
                   'color_features', 'texture_features', 'gabor_responses'
        ids: lista de IDs a usar
        random_state: seed para reproducibilidad
        pool_method: 'mean' o 'median' para pooling
    
    Returns:
        gallery_color: array (N_gallery, 768)
        gallery_texture: array (N_gallery, 2016)
        gallery_ids: array (N_gallery,)
        probe_color: array (N_probe, 768)
        probe_texture: array (N_probe, 2016)
        probe_ids: array (N_probe,)
        probe_gabor: lista de listas de listas - respuestas Gabor por probe
        gallery_gabor: lista de listas de listas - respuestas Gabor por gallery
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    gallery_color_features = []
    gallery_texture_features = []
    gallery_ids_list = []
    gallery_gabor_list = []
    
    probe_color_features = []
    probe_texture_features = []
    probe_ids_list = []
    probe_gabor_list = []
    
    for id in ids:
        cameras = tracklets[id]
        cam_names = list(cameras.keys())
        
        if len(cam_names) < 2:
            continue
        
        # Usar primeras dos cámaras
        cam_a = cam_names[0]
        cam_b = cam_names[1]
        
        # Gallery: cámara A
        if cam_a in cameras and len(cameras[cam_a]) > 0:
            # Extraer listas de features separadas
            color_list = [f['color_features'] for f in cameras[cam_a]]
            texture_list = [f['texture_features'] for f in cameras[cam_a]]
            gabor_list = [f['gabor_responses'] for f in cameras[cam_a]]
            
            # Pooling
            gallery_color, gallery_texture, gallery_gabor = pool_tracklet_features_separate(
                color_list, texture_list, gabor_list, method=pool_method
            )
            
            gallery_color_features.append(gallery_color)
            gallery_texture_features.append(gallery_texture)
            gallery_ids_list.append(id)
            gallery_gabor_list.append(gallery_gabor)
        
        # Probe: cámara B
        if cam_b in cameras and len(cameras[cam_b]) > 0:
            color_list = [f['color_features'] for f in cameras[cam_b]]
            texture_list = [f['texture_features'] for f in cameras[cam_b]]
            gabor_list = [f['gabor_responses'] for f in cameras[cam_b]]
            
            probe_color, probe_texture, probe_gabor = pool_tracklet_features_separate(
                color_list, texture_list, gabor_list, method=pool_method
            )
            
            probe_color_features.append(probe_color)
            probe_texture_features.append(probe_texture)
            probe_ids_list.append(id)
            probe_gabor_list.append(probe_gabor)
    
    if len(gallery_color_features) == 0 or len(probe_color_features) == 0:
        return (np.array([]), np.array([]), np.array([]),
                np.array([]), np.array([]), np.array([]), [], [])
    
    gallery_color = np.array(gallery_color_features)
    gallery_texture = np.array(gallery_texture_features)
    gallery_ids = np.array(gallery_ids_list)
    
    probe_color = np.array(probe_color_features)
    probe_texture = np.array(probe_texture_features)
    probe_ids = np.array(probe_ids_list)
    
    return (gallery_color, gallery_texture, gallery_ids,
            probe_color, probe_texture, probe_ids,
            probe_gabor_list, gallery_gabor_list)


def evaluate_cmc_ilidsvid_with_gating(
    tracklets: Dict[int, Dict[str, List[dict]]],
    trials: int = TRIALS,
    ranks: Tuple[int, ...] = RANKS,
    pool_method: str = 'mean',
    p: Optional[int] = None,
    a1: float = 1.0,
    a2: float = 1.0,
    b: float = 0.0,
    verbose: bool = False
) -> Dict[int, float]:
    """
    Evalúa CMC en iLIDS-VID con gating adaptativo entre color y textura.
    
    Args:
        tracklets: dict {id: {cam: [feature_dicts...]}} donde cada feature_dict tiene:
                   'color_features', 'texture_features', 'gabor_responses'
        trials: número de trials (default: 10)
        ranks: tupla de ranks a reportar (default: (1, 5, 10, 20))
        pool_method: 'mean' o 'median' para pooling (default: 'mean')
        p: número de IDs a usar por trial. Si None, usa min(50, #IDs_disponibles)
        a1: peso de complejidad de textura para gating
        a2: peso de entropía cromática para gating
        b: sesgo para gating
        verbose: si True, muestra progreso detallado
    
    Returns:
        results: dict {rank: accuracy} con accuracy promedio sobre trials
    """
    from gating import (
        compute_statistics_for_normalization,
        compute_gating_weight,
        combine_distances_with_gating
    )
    
    max_rank = max(ranks)
    cmc_accumulator = np.zeros(max_rank, dtype=np.float32)
    
    # Determinar p si no se especifica
    if p is None:
        valid_ids = []
        for id, cameras in tracklets.items():
            if len(cameras) >= 2:
                cam_names = list(cameras.keys())
                if len(cam_names) >= 2:
                    has_frames = all(len(cameras[cam]) > 0 for cam in cam_names[:2])
                    if has_frames:
                        valid_ids.append(id)
        p = min(50, len(valid_ids))
    
    import sys
    import time
    eval_start_time = time.time()
    
    for trial in range(trials):
        trial_seed = trial * 42
        
        if verbose:
            elapsed = time.time() - eval_start_time
            if trial > 0:
                avg_time_per_trial = elapsed / trial
                remaining_trials = trials - trial
                eta_seconds = avg_time_per_trial * remaining_trials
                eta_minutes = eta_seconds / 60
                percent = (trial / trials) * 100
                sys.stdout.write(f'\r  Trial {trial+1}/{trials} ({percent:.1f}%) | ETA: {eta_minutes:.1f}m')
                sys.stdout.flush()
            else:
                sys.stdout.write(f'\r  Trial {trial+1}/{trials}')
                sys.stdout.flush()
        
        # Samplear IDs
        sampled_ids = sample_ids_with_both_cameras(tracklets, p, random_state=trial_seed)
        
        # Construir gallery y probes con información de gating
        (gallery_color, gallery_texture, gallery_ids,
         probe_color, probe_texture, probe_ids,
         probe_gabor, gallery_gabor) = build_cross_camera_gallery_and_probes_with_gating(
            tracklets, sampled_ids, random_state=trial_seed + 1, pool_method=pool_method
        )
        
        if len(probe_color) == 0:
            if verbose:
                print(f"\nWarning: Trial {trial} no tiene probes. Saltando.")
            continue
        
        # Calcular métricas de gating para todos los probes
        probe_texture_complexities = []
        probe_color_entropies = []
        
        for i in range(len(probe_color)):
            T, C = compute_gating_metrics(
                probe_color[i], 
                probe_texture[i], 
                probe_gabor[i],
                texture_metric='entropy'  # Usar entropía como métrica de textura
            )
            probe_texture_complexities.append(T)
            probe_color_entropies.append(C)
        
        probe_texture_complexities = np.array(probe_texture_complexities)
        probe_color_entropies = np.array(probe_color_entropies)
        
        # Calcular estadísticas para normalización (usar solo probes en este trial)
        T_mean, T_std, C_mean, C_std = compute_statistics_for_normalization(
            probe_texture_complexities, probe_color_entropies
        )
        
        # Calcular alphas para cada probe
        alphas = np.array([
            compute_gating_weight(
                T, C, T_mean, T_std, C_mean, C_std, a1, a2, b
            )
            for T, C in zip(probe_texture_complexities, probe_color_entropies)
        ])
        
        # Calcular distancias separadas
        color_distances, texture_distances = pairwise_l1_distance_separate(
            probe_color, probe_texture, gallery_color, gallery_texture
        )
        
        # Combinar distancias con gating
        combined_distances = combine_distances_with_gating(
            color_distances, texture_distances, alphas
        )
        
        # Calcular ranks
        ranks_this_trial = compute_ranks(combined_distances, probe_ids, gallery_ids)
        
        # Calcular CMC
        cmc_this_trial = compute_cmc_curve(ranks_this_trial, max_rank)
        cmc_accumulator += cmc_this_trial
    
    if verbose:
        print()  # Nueva línea al finalizar
    
    # Promediar sobre trials
    cmc_accumulator /= trials
    
    # Construir resultados
    results = {r: float(cmc_accumulator[r - 1]) for r in ranks}
    
    return results

