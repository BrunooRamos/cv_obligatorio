"""
Módulo para mecanismo de gating adaptativo entre features de color y textura.

El gating combina distancias de color y textura con un peso adaptativo:
    d_total = α * d_textura + (1 - α) * d_color

donde α se calcula usando:
    - Complejidad de textura (T): proporción de energía en altas frecuencias Gabor
    - Entropía cromática (C): entropía del histograma de color
    - logit = a1 * T_norm - a2 * C_norm + b
    - α = sigmoid(logit)
"""
import numpy as np
from scipy.stats import entropy
from typing import Tuple, Dict, List, Optional
from constants import N_GABOR, N_COLOR_DIMS_PER_STRIPE, N_STRIPES


def compute_texture_complexity(
    gabor_energies: np.ndarray,
    high_freq_threshold: float = 0.5
) -> float:
    """
    Calcula la complejidad de textura como proporción de energía en altas frecuencias.
    
    Args:
        gabor_energies: Respuestas de filtros Gabor (n_filters,) - energías por filtro
                       Ordenados por frecuencia creciente: [freq_baja, ..., freq_alta]
        high_freq_threshold: Proporción de filtros a considerar como alta frecuencia (default: 0.5)
    
    Returns:
        Proporción de energía en altas frecuencias [0, 1]
    """
    if len(gabor_energies) == 0:
        return 0.0
    
    total_energy = np.sum(gabor_energies) + 1e-10
    
    # Los filtros Gabor están ordenados por frecuencia creciente (baja -> alta)
    # GABOR_FREQUENCIES = [0.10, 0.14, 0.20, 0.28] (baja a alta)
    # Entonces las altas frecuencias están al FINAL del array
    n_filters = len(gabor_energies)
    n_high_freq = int(n_filters * high_freq_threshold)
    
    # Tomar última mitad como altas frecuencias
    high_freq_energy = np.sum(gabor_energies[-n_high_freq:])
    
    return float(high_freq_energy / total_energy)


def compute_texture_entropy(texture_features: np.ndarray) -> float:
    """
    Calcula la entropía de las features de textura como medida de complejidad/discriminatividad.
    
    Una textura con mayor entropía es más variada y potencialmente más discriminativa.
    
    Args:
        texture_features: Vector de features de textura (2016 dims = 6 stripes × 336)
                         Histogramas normalizados L1
    
    Returns:
        Entropía promedio de los histogramas de textura
    """
    from constants import N_TEXTURE_DIMS_PER_STRIPE, N_STRIPES
    
    n_stripes = len(texture_features) // N_TEXTURE_DIMS_PER_STRIPE
    bins_per_filter = 16  # Cada filtro tiene un histograma de 16 bins
    
    entropies = []
    
    for stripe_idx in range(n_stripes):
        start_idx = stripe_idx * N_TEXTURE_DIMS_PER_STRIPE
        end_idx = (stripe_idx + 1) * N_TEXTURE_DIMS_PER_STRIPE
        stripe_texture = texture_features[start_idx:end_idx]
        
        # Calcular entropía de cada histograma de filtro
        n_filters = len(stripe_texture) // bins_per_filter
        for filter_idx in range(n_filters):
            hist_start = filter_idx * bins_per_filter
            hist_end = (filter_idx + 1) * bins_per_filter
            hist = stripe_texture[hist_start:hist_end]
            
            # Asegurar que el histograma esté normalizado y sin ceros
            hist = hist + 1e-10
            hist = hist / np.sum(hist)
            
            # Calcular entropía
            hist_entropy = entropy(hist, base=2)
            entropies.append(hist_entropy)
    
    return float(np.mean(entropies))


def compute_texture_variance(texture_features: np.ndarray) -> float:
    """
    Calcula la varianza de las features de textura como medida de complejidad.
    
    Una textura con mayor varianza entre stripes es más variada espacialmente.
    
    Args:
        texture_features: Vector de features de textura (2016 dims = 6 stripes × 336)
    
    Returns:
        Varianza promedio entre stripes
    """
    from constants import N_TEXTURE_DIMS_PER_STRIPE, N_STRIPES
    
    n_stripes = len(texture_features) // N_TEXTURE_DIMS_PER_STRIPE
    
    stripe_features = []
    for stripe_idx in range(n_stripes):
        start_idx = stripe_idx * N_TEXTURE_DIMS_PER_STRIPE
        end_idx = (stripe_idx + 1) * N_TEXTURE_DIMS_PER_STRIPE
        stripe_features.append(texture_features[start_idx:end_idx])
    
    # Calcular varianza entre stripes
    stripe_features_array = np.array(stripe_features)  # (n_stripes, 336)
    variance_per_dim = np.var(stripe_features_array, axis=0)  # Varianza por dimensión
    mean_variance = np.mean(variance_per_dim)
    
    return float(mean_variance)


def compute_color_entropy(color_features: np.ndarray) -> float:
    """
    Calcula la entropía cromática desde las features de color.
    
    Args:
        color_features: Vector de features de color (128 dims por stripe o concatenado)
                       Histogramas normalizados L1
    
    Returns:
        Entropía cromática promedio sobre todos los canales
    """
    # Si es un vector concatenado de múltiples stripes, procesar por stripe
    if len(color_features.shape) == 1:
        n_stripes = len(color_features) // N_COLOR_DIMS_PER_STRIPE
        if n_stripes > 1:
            # Procesar cada stripe por separado y promediar
            entropies = []
            for i in range(n_stripes):
                start_idx = i * N_COLOR_DIMS_PER_STRIPE
                end_idx = (i + 1) * N_COLOR_DIMS_PER_STRIPE
                stripe_color = color_features[start_idx:end_idx]
                stripe_entropy = compute_color_entropy_single_stripe(stripe_color)
                entropies.append(stripe_entropy)
            return float(np.mean(entropies))
        else:
            return compute_color_entropy_single_stripe(color_features)
    else:
        raise ValueError(f"color_features debe ser 1D, recibido shape: {color_features.shape}")


def compute_color_entropy_single_stripe(color_features: np.ndarray) -> float:
    """
    Calcula entropía cromática para un solo stripe.
    
    Args:
        color_features: Vector de 128 elementos (8 canales × 16 bins)
    
    Returns:
        Entropía promedio sobre los 8 canales
    """
    n_channels = 8
    bins_per_channel = 16
    
    if len(color_features) != n_channels * bins_per_channel:
        raise ValueError(f"color_features debe tener {n_channels * bins_per_channel} elementos")
    
    entropies = []
    for ch_idx in range(n_channels):
        start_idx = ch_idx * bins_per_channel
        end_idx = (ch_idx + 1) * bins_per_channel
        hist = color_features[start_idx:end_idx]
        
        # Asegurar que el histograma esté normalizado y sin ceros para entropy
        hist = hist + 1e-10  # Evitar log(0)
        hist = hist / np.sum(hist)
        
        # Calcular entropía del histograma
        ch_entropy = entropy(hist, base=2)  # Entropía en bits
        entropies.append(ch_entropy)
    
    return float(np.mean(entropies))


def compute_gabor_energies_from_responses(gabor_responses: List[np.ndarray]) -> np.ndarray:
    """
    Calcula las energías de los filtros Gabor desde las respuestas 2D.
    
    Args:
        gabor_responses: Lista de arrays 2D con respuestas de filtros Gabor
    
    Returns:
        energies: Array 1D con energía por filtro (suma de valores absolutos)
    """
    energies = []
    for resp in gabor_responses:
        # Energía = suma de valores absolutos de la respuesta
        energy = np.sum(np.abs(resp))
        energies.append(energy)
    
    return np.array(energies, dtype=np.float32)


def z_score_normalize(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    """
    Normaliza valores usando z-score.
    
    Args:
        values: Array de valores a normalizar
        mean: Media para normalización
        std: Desviación estándar para normalización
    
    Returns:
        Valores normalizados
    """
    if std < 1e-10:
        return np.zeros_like(values)
    return (values - mean) / std


def compute_gating_weight(
    texture_complexity: float,
    color_entropy: float,
    T_mean: float,
    T_std: float,
    C_mean: float,
    C_std: float,
    a1: float,
    a2: float,
    b: float
) -> float:
    """
    Calcula el peso de gating α usando la fórmula sigmoidal.
    
    Args:
        texture_complexity: Complejidad de textura T
        color_entropy: Entropía cromática C
        T_mean: Media de T para normalización
        T_std: Desviación estándar de T para normalización
        C_mean: Media de C para normalización
        C_std: Desviación estándar de C para normalización
        a1: Peso de complejidad de textura
        a2: Peso de entropía cromática
        b: Sesgo
    
    Returns:
        α: Peso de gating entre 0 y 1
    """
    # Normalización z-score
    T_norm = (texture_complexity - T_mean) / (T_std + 1e-10)
    C_norm = (color_entropy - C_mean) / (C_std + 1e-10)
    
    # Calcular logit
    logit = a1 * T_norm - a2 * C_norm + b
    
    # Sigmoid
    alpha = 1.0 / (1.0 + np.exp(-logit))
    
    return float(np.clip(alpha, 0.0, 1.0))


def compute_statistics_for_normalization(
    texture_complexities: np.ndarray,
    color_entropies: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Calcula estadísticas (media y desviación estándar) para normalización z-score.
    
    Args:
        texture_complexities: Array de complejidades de textura
        color_entropies: Array de entropías cromáticas
    
    Returns:
        T_mean, T_std, C_mean, C_std
    """
    T_mean = float(np.mean(texture_complexities))
    T_std = float(np.std(texture_complexities))
    C_mean = float(np.mean(color_entropies))
    C_std = float(np.std(color_entropies))
    
    return T_mean, T_std, C_mean, C_std


def combine_distances_with_gating(
    color_distances: np.ndarray,
    texture_distances: np.ndarray,
    alphas: np.ndarray
) -> np.ndarray:
    """
    Combina distancias de color y textura usando pesos de gating adaptativos.
    
    Args:
        color_distances: Array (N1, N2) con distancias de color
        texture_distances: Array (N1, N2) con distancias de textura
        alphas: Array (N1,) con pesos de gating para cada probe
    
    Returns:
        combined_distances: Array (N1, N2) con distancias combinadas
    """
    if color_distances.shape != texture_distances.shape:
        raise ValueError(f"Shapes deben coincidir: color {color_distances.shape} vs texture {texture_distances.shape}")
    
    if len(alphas.shape) == 1:
        # Expandir alphas para broadcasting: (N1,) -> (N1, 1)
        alphas = alphas[:, np.newaxis]
    
    # d_total = α * d_textura + (1 - α) * d_color
    combined = alphas * texture_distances + (1 - alphas) * color_distances
    
    return combined


def grid_search_gating_parameters(
    texture_complexities: np.ndarray,
    color_entropies: np.ndarray,
    color_distances: np.ndarray,
    texture_distances: np.ndarray,
    probe_ids: np.ndarray,
    gallery_ids: np.ndarray,
    a1_range: List[float],
    a2_range: List[float],
    b_range: List[float],
    metric: str = 'rank1'
) -> Tuple[Dict, float]:
    """
    Realiza grid search para optimizar parámetros a1, a2, b del gating.
    
    Args:
        texture_complexities: Array (N_probe,) con complejidades de textura
        color_entropies: Array (N_probe,) con entropías cromáticas
        color_distances: Array (N_probe, N_gallery) con distancias de color
        texture_distances: Array (N_probe, N_gallery) con distancias de textura
        probe_ids: Array (N_probe,) con IDs de probes
        gallery_ids: Array (N_gallery,) con IDs de gallery
        a1_range: Lista de valores a probar para a1
        a2_range: Lista de valores a probar para a2
        b_range: Lista de valores a probar para b
        metric: Métrica a optimizar ('rank1', 'rank5', 'rank10', 'rank20')
    
    Returns:
        best_params: Dict con mejores parámetros {'a1': float, 'a2': float, 'b': float}
        best_score: Mejor score obtenido
    """
    from eval_ilidsvid import compute_ranks, compute_cmc_curve
    
    # Calcular estadísticas para normalización
    T_mean, T_std, C_mean, C_std = compute_statistics_for_normalization(
        texture_complexities, color_entropies
    )
    
    best_score = -1.0
    best_params = None
    
    rank_map = {'rank1': 1, 'rank5': 5, 'rank10': 10, 'rank20': 20}
    target_rank = rank_map.get(metric, 1)
    
    total_combinations = len(a1_range) * len(a2_range) * len(b_range)
    print(f"Grid search: {total_combinations} combinaciones a evaluar...")
    
    for a1 in a1_range:
        for a2 in a2_range:
            for b in b_range:
                # Calcular alphas para todos los probes
                alphas = np.array([
                    compute_gating_weight(
                        T, C, T_mean, T_std, C_mean, C_std, a1, a2, b
                    )
                    for T, C in zip(texture_complexities, color_entropies)
                ])
                
                # Combinar distancias
                combined_distances = combine_distances_with_gating(
                    color_distances, texture_distances, alphas
                )
                
                # Calcular ranks
                ranks = compute_ranks(combined_distances, probe_ids, gallery_ids)
                
                # Calcular CMC
                cmc = compute_cmc_curve(ranks, max_rank=target_rank)
                
                # Score es la accuracy en el rank objetivo
                score = float(cmc[target_rank - 1])
                
                if score > best_score:
                    best_score = score
                    best_params = {'a1': a1, 'a2': a2, 'b': b}
    
    return best_params, best_score

