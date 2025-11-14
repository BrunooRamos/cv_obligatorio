"""
Módulo para evaluación CMC en datasets de stills (i-LIDS MCTS / VIPeR).
Protocolo single-shot con L1 distance.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from constants import RANKS, TRIALS, P_ILIDS_MCTS, P_VIPER


def pairwise_l1_distance(features1: np.ndarray, features2: np.ndarray, chunk_size: int = 1000) -> np.ndarray:
    """
    Calcula distancias L1 (cityblock) entre dos conjuntos de features.
    Optimizado para memoria: procesa en chunks si hay muchos probes.
    
    Args:
        features1: array (N1, D) de features (probes)
        features2: array (N2, D) de features (gallery)
        chunk_size: tamaño del chunk para procesamiento (default: 1000)
    
    Returns:
        distances: array (N1, N2) con distancias L1
    """
    n1, d = features1.shape
    n2 = features2.shape[0]
    
    # Si hay muchos probes, procesar en chunks para ahorrar memoria
    if n1 > chunk_size:
        distances = np.zeros((n1, n2), dtype=np.float32)
        for i in range(0, n1, chunk_size):
            end_i = min(i + chunk_size, n1)
            chunk = features1[i:end_i]  # (chunk_size, D)
            # Calcular distancias del chunk contra toda la gallery
            diff = chunk[:, np.newaxis, :] - features2[np.newaxis, :, :]  # (chunk_size, N2, D)
            distances[i:end_i] = np.sum(np.abs(diff), axis=2).astype(np.float32)
        return distances
    else:
        # Si hay pocos probes, usar método directo
        diff = features1[:, np.newaxis, :] - features2[np.newaxis, :, :]
        distances = np.sum(np.abs(diff), axis=2).astype(np.float32)
        return distances


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
        # Ordenar distancias (menor a mayor)
        sorted_indices = np.argsort(distances[i, :])
        
        # Encontrar posición de la identidad correcta
        correct_positions = np.where(gallery_ids[sorted_indices] == probe_id)[0]
        
        if len(correct_positions) > 0:
            # Rank = posición + 1 (1-indexed)
            rank = correct_positions[0] + 1
        else:
            # Si no hay match, asignar rank muy alto
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


def sample_ids(features_by_id: Dict, p: int, random_state: Optional[int] = None) -> List:
    """
    Samplea p IDs que tengan al menos 1 imagen.
    
    Args:
        features_by_id: dict {id: [features...]}
        p: número de IDs a samplear
        random_state: seed para reproducibilidad
    
    Returns:
        sampled_ids: lista de p IDs
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Filtrar IDs con al menos 1 imagen
    valid_ids = [id for id, feats in features_by_id.items() if len(feats) > 0]
    
    if len(valid_ids) < p:
        raise ValueError(f"No hay suficientes IDs con imágenes. Requeridos: {p}, Disponibles: {len(valid_ids)}")
    
    sampled_ids = np.random.choice(valid_ids, size=p, replace=False).tolist()
    return sampled_ids


def build_single_shot_gallery_and_probes(
    features_by_id: Dict,
    ids: List,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Construye gallery (single-shot) y probes para un trial.
    
    Args:
        features_by_id: dict {id: [features...]}
        ids: lista de IDs a usar
        random_state: seed para reproducibilidad
    
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
        feats = features_by_id[id]
        
        if len(feats) == 0:
            continue
        
        # Gallery: 1 imagen aleatoria
        gallery_idx = np.random.randint(0, len(feats))
        gallery_features.append(feats[gallery_idx])
        gallery_ids.append(id)
        
        # Probe: todas las demás imágenes
        for i, feat in enumerate(feats):
            if i != gallery_idx:
                probe_features.append(feat)
                probe_ids.append(id)
    
    gallery_features = np.array(gallery_features)
    gallery_ids = np.array(gallery_ids)
    probe_features = np.array(probe_features)
    probe_ids = np.array(probe_ids)
    
    return gallery_features, gallery_ids, probe_features, probe_ids


def evaluate_cmc_stills(
    features: Dict[int, List[np.ndarray]],
    trials: int = TRIALS,
    p: Optional[int] = None,
    ranks: Tuple[int, ...] = RANKS,
    verbose: bool = False
) -> Dict[int, float]:
    """
    Evalúa CMC en dataset de stills (i-LIDS MCTS / VIPeR).
    
    Args:
        features: dict {id: [features...]} donde cada feature es array (2784,)
        trials: número de trials (default: 10)
        p: número de IDs a usar por trial. Si None, usa P_ILIDS_MCTS=50
        ranks: tupla de ranks a reportar (default: (1, 5, 10, 20))
    
    Returns:
        results: dict {rank: accuracy} con accuracy promedio sobre trials
    """
    if p is None:
        p = P_ILIDS_MCTS
    
    max_rank = max(ranks)
    cmc_accumulator = np.zeros(max_rank, dtype=np.float32)
    
    import sys
    import time
    eval_start_time = time.time()
    
    for trial in range(trials):
        # Seed fijo por trial para reproducibilidad
        trial_seed = trial * 42  # Seed base
        
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
        sampled_ids = sample_ids(features, p, random_state=trial_seed)
        
        # Construir gallery y probes
        gallery_feats, gallery_ids, probe_feats, probe_ids = build_single_shot_gallery_and_probes(
            features, sampled_ids, random_state=trial_seed + 1
        )
        
        if len(probe_feats) == 0:
            print(f"Warning: Trial {trial} no tiene probes. Saltando.")
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


def build_single_shot_gallery_and_probes_with_gating(
    features_by_id: Dict,
    ids: List,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List, List]:
    """
    Construye gallery y probes con información de gating para Market-1501.
    
    Args:
        features_by_id: dict {id: [feature_dicts...]} donde cada feature_dict tiene:
                       'color_features', 'texture_features', 'gabor_responses', 'features'
        ids: lista de IDs a usar
        random_state: seed para reproducibilidad
    
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
        feat_dicts = features_by_id[id]
        
        if len(feat_dicts) == 0:
            continue
        
        # Gallery: 1 imagen aleatoria
        gallery_idx = np.random.randint(0, len(feat_dicts))
        gallery_feat = feat_dicts[gallery_idx]
        
        gallery_color_features.append(gallery_feat['color_features'])
        gallery_texture_features.append(gallery_feat['texture_features'])
        gallery_ids_list.append(id)
        gallery_gabor_list.append(gallery_feat['gabor_responses'])
        
        # Probe: todas las demás imágenes
        for i, feat_dict in enumerate(feat_dicts):
            if i != gallery_idx:
                probe_color_features.append(feat_dict['color_features'])
                probe_texture_features.append(feat_dict['texture_features'])
                probe_ids_list.append(id)
                probe_gabor_list.append(feat_dict['gabor_responses'])
    
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


def evaluate_cmc_market1501_with_gating(
    features_by_id: Dict[int, List[dict]],
    trials: int = TRIALS,
    ranks: Tuple[int, ...] = RANKS,
    p: Optional[int] = None,
    a1: float = 1.0,
    a2: float = 1.0,
    b: float = 0.0,
    verbose: bool = False
) -> Dict[int, float]:
    """
    Evalúa CMC en Market-1501 con gating adaptativo entre color y textura.
    
    Args:
        features_by_id: dict {id: [feature_dicts...]} donde cada feature_dict tiene:
                       'color_features', 'texture_features', 'gabor_responses'
        trials: número de trials (default: 10)
        ranks: tupla de ranks a reportar (default: (1, 5, 10, 20))
        p: número de IDs a usar por trial. Si None, usa min(100, #IDs_disponibles)
        a1: peso de complejidad de textura para gating
        a2: peso de entropía cromática para gating
        b: sesgo para gating
        verbose: si True, muestra progreso detallado
    
    Returns:
        results: dict {rank: accuracy} con accuracy promedio sobre trials
    """
    from eval_ilidsvid import (
        pairwise_l1_distance_separate,
        compute_gating_metrics,
        compute_ranks,
        compute_cmc_curve
    )
    from gating import (
        compute_statistics_for_normalization,
        compute_gating_weight,
        combine_distances_with_gating
    )
    
    max_rank = max(ranks)
    cmc_accumulator = np.zeros(max_rank, dtype=np.float32)
    
    # Determinar p si no se especifica
    if p is None:
        p = min(100, len(features_by_id))
    
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
        np.random.seed(trial_seed)
        all_ids = list(features_by_id.keys())
        sampled_ids = np.random.choice(all_ids, size=min(p, len(all_ids)), replace=False).tolist()
        
        # Construir gallery y probes con información de gating
        (gallery_color, gallery_texture, gallery_ids,
         probe_color, probe_texture, probe_ids,
         probe_gabor, gallery_gabor) = build_single_shot_gallery_and_probes_with_gating(
            features_by_id, sampled_ids, random_state=trial_seed + 1
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
                texture_metric='entropy'
            )
            probe_texture_complexities.append(T)
            probe_color_entropies.append(C)
        
        probe_texture_complexities = np.array(probe_texture_complexities)
        probe_color_entropies = np.array(probe_color_entropies)
        
        # Calcular estadísticas para normalización
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



