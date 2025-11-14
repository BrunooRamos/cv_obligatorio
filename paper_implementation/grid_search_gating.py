#!/usr/bin/env python3
"""
Script para hacer grid search de parámetros de gating (a1, a2, b).

Este script:
1. Carga features con información de gating
2. Realiza grid search sobre rangos de parámetros
3. Reporta los mejores parámetros y resultados
"""
import numpy as np
import os
import pickle
import argparse
from eval_ilidsvid import (
    build_cross_camera_gallery_and_probes_with_gating,
    sample_ids_with_both_cameras,
    pairwise_l1_distance_separate,
    compute_gating_metrics,
    compute_ranks,
    compute_cmc_curve
)
from eval_stills import (
    build_single_shot_gallery_and_probes_with_gating
)
from gating import (
    compute_statistics_for_normalization,
    compute_gating_weight,
    combine_distances_with_gating,
    grid_search_gating_parameters
)


def load_features_with_gating(input_path):
    """Carga features con información de gating."""
    print(f"Cargando features desde: {input_path}")
    with open(input_path, 'rb') as f:
        tracklets = pickle.load(f)
    print(f"✓ Features cargadas exitosamente")
    return tracklets


def grid_search_single_trial(
    data,
    trial_seed=42,
    p=50,
    pool_method='mean',
    a1_range=[0.5, 1.0, 1.5, 2.0],
    a2_range=[0.5, 1.0, 1.5, 2.0],
    b_range=[-1.0, -0.5, 0.0, 0.5, 1.0],
    metric='rank1',
    is_market1501=False
):
    """
    Realiza grid search en un solo trial.
    
    Args:
        data: tracklets (iLIDS-VID) o features_by_id (Market-1501)
        is_market1501: si True, trata data como Market-1501, sino como iLIDS-VID
    
    Returns:
        best_params: dict con mejores parámetros
        best_score: mejor score
        all_results: lista de dicts con todos los resultados
    """
    import numpy as np
    
    if is_market1501:
        # Market-1501: samplear IDs directamente
        np.random.seed(trial_seed)
        all_ids = list(data.keys())
        sampled_ids = np.random.choice(all_ids, size=min(p, len(all_ids)), replace=False).tolist()
        
        # Construir gallery y probes
        (gallery_color, gallery_texture, gallery_ids,
         probe_color, probe_texture, probe_ids,
         probe_gabor, gallery_gabor) = build_single_shot_gallery_and_probes_with_gating(
            data, sampled_ids, random_state=trial_seed + 1
        )
    else:
        # iLIDS-VID: samplear IDs con ambas cámaras
        sampled_ids = sample_ids_with_both_cameras(data, p, random_state=trial_seed)
        
        # Construir gallery y probes
        (gallery_color, gallery_texture, gallery_ids,
         probe_color, probe_texture, probe_ids,
         probe_gabor, gallery_gabor) = build_cross_camera_gallery_and_probes_with_gating(
            data, sampled_ids, random_state=trial_seed + 1, pool_method=pool_method
        )
    
    if len(probe_color) == 0:
        print(f"Warning: No hay probes disponibles")
        return None, -1.0, []
    
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
    
    # Calcular estadísticas para normalización
    T_mean, T_std, C_mean, C_std = compute_statistics_for_normalization(
        probe_texture_complexities, probe_color_entropies
    )
    
    # Calcular distancias separadas
    color_distances, texture_distances = pairwise_l1_distance_separate(
        probe_color, probe_texture, gallery_color, gallery_texture
    )
    
    # Grid search
    best_score = -1.0
    best_params = None
    all_results = []
    
    rank_map = {'rank1': 1, 'rank5': 5, 'rank10': 10, 'rank20': 20}
    target_rank = rank_map.get(metric, 1)
    
    total_combinations = len(a1_range) * len(a2_range) * len(b_range)
    print(f"Evaluando {total_combinations} combinaciones de parámetros...")
    
    for a1 in a1_range:
        for a2 in a2_range:
            for b in b_range:
                # Calcular alphas para todos los probes
                alphas = np.array([
                    compute_gating_weight(
                        T, C, T_mean, T_std, C_mean, C_std, a1, a2, b
                    )
                    for T, C in zip(probe_texture_complexities, probe_color_entropies)
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
                
                result = {
                    'a1': a1,
                    'a2': a2,
                    'b': b,
                    'score': score,
                    'alpha_mean': float(np.mean(alphas)),
                    'alpha_std': float(np.std(alphas))
                }
                all_results.append(result)
                
                if score > best_score:
                    best_score = score
                    best_params = {'a1': a1, 'a2': a2, 'b': b}
    
    return best_params, best_score, all_results


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Grid search de parámetros de gating')
    parser.add_argument('--features-file', type=str, default='features_ilidsvid_gating.pkl',
                       help='Archivo con features con información de gating')
    parser.add_argument('--trials', type=int, default=1,
                       help='Número de trials para grid search (default: 1)')
    parser.add_argument('--p', type=int, default=50,
                       help='Número de IDs a usar por trial (default: 50)')
    parser.add_argument('--pool-method', choices=['mean', 'median'], default='mean',
                       help='Método de pooling (default: mean)')
    parser.add_argument('--metric', choices=['rank1', 'rank5', 'rank10', 'rank20'], default='rank1',
                       help='Métrica a optimizar (default: rank1)')
    parser.add_argument('--a1-range', type=float, nargs='+', default=[0.5, 1.0, 1.5, 2.0],
                       help='Rango de valores para a1 (default: 0.5 1.0 1.5 2.0)')
    parser.add_argument('--a2-range', type=float, nargs='+', default=[0.5, 1.0, 1.5, 2.0],
                       help='Rango de valores para a2 (default: 0.5 1.0 1.5 2.0)')
    parser.add_argument('--b-range', type=float, nargs='+', default=[-1.0, -0.5, 0.0, 0.5, 1.0],
                       help='Rango de valores para b (default: -1.0 -0.5 0.0 0.5 1.0)')
    parser.add_argument('--output', type=str, default=None,
                       help='Archivo para guardar resultados (opcional)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("GRID SEARCH DE PARÁMETROS DE GATING")
    print("="*70)
    print(f"Archivo de features: {args.features_file}")
    print(f"Trials: {args.trials}")
    print(f"IDs por trial: {args.p}")
    print(f"Métrica: {args.metric}")
    print(f"Rangos:")
    print(f"  a1: {args.a1_range}")
    print(f"  a2: {args.a2_range}")
    print(f"  b: {args.b_range}")
    print("="*70 + "\n")
    
    # Verificar que existe el archivo
    if not os.path.exists(args.features_file):
        print(f"❌ Error: No se encontró el archivo: {args.features_file}")
        print("   Ejecuta primero eval_ilidsvid_with_gating.py para extraer features")
        return
    
    # Cargar features
    data = load_features_with_gating(args.features_file)
    
    # Detectar tipo de dataset
    first_key = list(data.keys())[0]
    first_value = data[first_key]
    is_market1501 = isinstance(first_value, list)  # Market-1501 tiene lista, iLIDS-VID tiene dict
    
    if is_market1501:
        print("Detectado: Market-1501")
    else:
        print("Detectado: iLIDS-VID")
    
    # Realizar grid search en múltiples trials
    all_best_params = []
    all_best_scores = []
    all_results_combined = []
    
    for trial in range(args.trials):
        print(f"\nTrial {trial + 1}/{args.trials}")
        print("-"*70)
        
        trial_seed = trial * 42
        best_params, best_score, all_results = grid_search_single_trial(
            data,
            trial_seed=trial_seed,
            p=args.p,
            pool_method=args.pool_method,
            a1_range=args.a1_range,
            a2_range=args.a2_range,
            b_range=args.b_range,
            metric=args.metric,
            is_market1501=is_market1501
        )
        
        if best_params is not None:
            all_best_params.append(best_params)
            all_best_scores.append(best_score)
            all_results_combined.extend(all_results)
            
            print(f"Mejores parámetros: a1={best_params['a1']}, a2={best_params['a2']}, b={best_params['b']}")
            print(f"Mejor score ({args.metric}): {best_score:.4f} ({best_score*100:.2f}%)")
    
    if len(all_best_params) == 0:
        print("\n❌ No se encontraron resultados válidos")
        return
    
    # Promediar resultados sobre trials
    avg_best_score = np.mean(all_best_scores)
    
    # Encontrar parámetros más frecuentes
    from collections import Counter
    param_strings = [f"{p['a1']}_{p['a2']}_{p['b']}" for p in all_best_params]
    most_common = Counter(param_strings).most_common(1)[0]
    most_common_params = all_best_params[param_strings.index(most_common[0])]
    
    print("\n" + "="*70)
    print("RESULTADOS DEL GRID SEARCH")
    print("="*70)
    print(f"\nPromedio de mejor score ({args.metric}): {avg_best_score:.4f} ({avg_best_score*100:.2f}%)")
    print(f"\nParámetros más frecuentes:")
    print(f"  a1 = {most_common_params['a1']}")
    print(f"  a2 = {most_common_params['a2']}")
    print(f"  b = {most_common_params['b']}")
    print(f"  (apareció en {most_common[1]}/{args.trials} trials)")
    
    # Mostrar top 5 combinaciones
    print(f"\nTop 5 combinaciones de parámetros:")
    print("-"*70)
    
    # Agrupar por combinación de parámetros y promediar scores
    from collections import defaultdict
    grouped_results = defaultdict(list)
    for result in all_results_combined:
        key = (result['a1'], result['a2'], result['b'])
        grouped_results[key].append(result['score'])
    
    averaged_results = [
        {'a1': k[0], 'a2': k[1], 'b': k[2], 'score': np.mean(v), 'count': len(v)}
        for k, v in grouped_results.items()
    ]
    averaged_results.sort(key=lambda x: x['score'], reverse=True)
    
    for i, result in enumerate(averaged_results[:5], 1):
        print(f"{i}. a1={result['a1']:.2f}, a2={result['a2']:.2f}, b={result['b']:.2f} "
              f"-> score={result['score']:.4f} ({result['score']*100:.2f}%) "
              f"[{result['count']} evaluaciones]")
    
    print("="*70)
    
    # Guardar resultados si se especifica
    if args.output:
        output_data = {
            'best_params': most_common_params,
            'avg_best_score': avg_best_score,
            'all_best_params': all_best_params,
            'all_best_scores': all_best_scores,
            'top_results': averaged_results[:10],
            'config': {
                'a1_range': args.a1_range,
                'a2_range': args.a2_range,
                'b_range': args.b_range,
                'metric': args.metric,
                'trials': args.trials,
                'p': args.p
            }
        }
        with open(args.output, 'wb') as f:
            pickle.dump(output_data, f)
        print(f"\n✓ Resultados guardados en: {args.output}")


if __name__ == "__main__":
    main()

