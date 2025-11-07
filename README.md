# BILP: Brightness-Invariant Local Patterns for Person Re-Identification

Implementación de BILP (Brightness-Invariant Local Patterns) para Re-Identificación de Personas usando los datasets **Market-1501** e **iLIDS-VID**.

## Descripción del Proyecto

Este proyecto implementa un método de re-identificación de personas basado en características de bajo nivel (handcrafted features) que son invariantes a cambios de iluminación. La técnica combina:

- **Color**: Log-chromaticity para invariancia a brillo
- **Textura**: Filtros Gabor y análisis espectral (FFT)
- **Gating adaptativo**: Fusión dinámica basada en contenido de la imagen

## Datasets

### Market-1501 (Single-Query)
- **751 IDs** en train, **750 IDs** en test
- Imágenes estáticas: **128×64 px**
- Métricas: **CMC (Rank-1/5/10) + mAP**

### iLIDS-VID (Multi-Shot)
- **Video sequences**, 2 cámaras por ID
- Protocolo multi-shot: pooling de frames
- Métrica: **CMC (Rank-1/5/10)**

---

## Roadmap del Proyecto

### Setup y Estructura
**Completado:**
- [x] Entorno Docker con dependencias (numpy, opencv, scikit-image, scipy, scikit-learn, matplotlib)
- [x] Estructura de directorios:
  ```
  /data/{market1501, ilids-vid}/
  /bilp/{color.py, texture.py, gating.py, distance.py, utils.py}
  /eval/{loaders.py, cmc_map.py, splits.py, plots.py}
  /scripts/{01_prepare_data.py, 02_extract_market_features.py,
            02_extract_ilids_features.py, 03_eval_market.py,
            04_eval_ilidsvid.py, 05_ablation.py}
  ```
- [x] Módulos BILP base creados:
  - `bilp/color.py`: Log-chromaticity (u,v) + luminancia
  - `bilp/texture.py`: Gabor filters + FFT features

### Data Loaders
**Completado:**
- [x] Implementación de data loaders:
  - `eval/loaders.py`: Loaders completos para Market-1501 e iLIDS-VID
  - `eval/__init__.py`: Módulo de evaluación
- [x] Funcionalidades implementadas:
  - Parser de nombres Market-1501: `parse_market1501_name()`
  - Parser de nombres iLIDS-VID: `parse_ilids_name()`
  - Loader Market-1501: `load_market1501()` con splits train/test/query
  - Loader iLIDS-VID: `load_ilids_vid()` con sampling de frames
  - Estrategias de sampling: uniform, random, all
  - Función de resize automático a 128×64
  - Estadísticas de datasets: `get_market1501_stats()`, `get_ilids_stats()`
- [x] Testing completo:
  - Script de prueba: `scripts/test_loaders.py`
  - Market-1501: 12,936 train + 3,362 queries cargadas correctamente
  - iLIDS-VID: 600 secuencias (300 IDs x 2 cámaras) cargadas correctamente

### Módulos BILP Completos
**Completado:**
- [x] Implementación completa de módulos BILP:
  - `bilp/utils.py`: Normalización y extracción batch
  - `bilp/distance.py`: Cálculo de distancias y ranking
  - `bilp/gating.py`: Fusión adaptativa color/textura
- [x] Funcionalidades implementadas:
  - **utils.py**: normalize_l1, power_normalize, extract_bilp_descriptor, extract_bilp_batch, save/load_features
  - **distance.py**: bilp_distance, compute_distance_matrix_fast, rank_gallery, compute_ap
  - **gating.py**: compute_gating_weight, optimize_gating_params, per-stripe gating
- [x] Testing completo:
  - Script de prueba: `scripts/test_bilp.py`
  - Extracción de features: 1,884 dims (1,632 color + 252 texture)
  - Batch processing funcionando
  - Gating weights calculados correctamente
  - Distance matrix computada

### Métricas de Evaluación
**Completado:**
- [x] Implementación completa de métricas:
  - `eval/cmc_map.py`: CMC y mAP para re-identificación
- [x] Funcionalidades implementadas:
  - **CMC**: compute_cmc() con soporte para junk removal
  - **mAP**: compute_map() y compute_single_ap()
  - **Evaluación**: evaluate_market1501(), evaluate_ilids_vid()
  - **Utilidades**: print_results(), save/load_results()
- [x] Testing completo:
  - Script de prueba: `scripts/test_metrics.py`
  - Tests sintéticos: CMC y mAP perfectos verificados
  - Test con datos reales: Rank-1 = 70%, mAP = 63.73% (subset)
  - Junk removal funcionando correctamente

**Próximos pasos:**
- [ ] Calibrar rangos (u,v) de color con train set de Market-1501
- [ ] Script de extracción de features para dataset completo
- [ ] Evaluación end-to-end en Market-1501 completo
- [ ] Evaluación en iLIDS-VID

### Verificación de Datasets
**Completado:**
- [x] **Market-1501**: Dataset completo y verificado
  - 12,936 imágenes train (751 IDs)
  - 19,732 imágenes test (750 IDs)
  - 3,368 queries
  - Split oficial listo para usar
- [x] **iLIDS-VID**: Dataset completo y verificado
  - Secuencias de video por persona
  - 2 cámaras (cam1, cam2)
  - Splits oficiales en formato .mat
  - Frames en formato .png

Ver detalles completos en [DATASET_VERIFICATION.md](DATASET_VERIFICATION.md)

---

## Estructura del Código

### `/eval/` - Módulo de evaluación y data loading

#### `loaders.py` - Implementado
Carga de datasets con funcionalidades completas:

**Market-1501:**
- `load_market1501(dataset_path, split, resize, return_images)`: Carga train/test/query
- `parse_market1501_name(filename)`: Parser de nombres (person_id, camera_id, frame, bbox)
- `get_market1501_stats(data)`: Estadísticas del dataset
- Soporte para resize automático a 128×64
- Filtrado automático de imágenes junk (ID -1, 0000)

**iLIDS-VID:**
- `load_ilids_vid(dataset_path, num_frames, sampling_strategy, resize, return_images)`: Carga secuencias
- `parse_ilids_name(filename)`: Parser de nombres de frames
- `sample_frames_from_sequence(frames, num_frames, strategy)`: Sampling uniforme/random/all
- `get_ilids_stats(data)`: Estadísticas del dataset
- Default: K=10 frames por secuencia (sampling uniforme)

**Utilidades:**
- `create_train_val_split(data, val_ratio, seed)`: Split train/val por person ID

#### `cmc_map.py` - Implementado
Métricas de evaluación completas:

**CMC (Cumulative Matching Characteristic):**
- `compute_cmc()`: Calcula curva CMC hasta rank-k
- Soporte para junk removal (same camera images)
- Retorna array con scores para cada rank

**mAP (mean Average Precision):**
- `compute_map()`: Calcula mAP para Market-1501
- `compute_single_ap()`: AP para una sola query
- Manejo correcto de junk images (same ID + same camera)

**Funciones de evaluación:**
- `evaluate_reid()`: Evaluación completa (CMC + mAP)
- `evaluate_market1501()`: Wrapper específico para Market-1501
- `evaluate_ilids_vid()`: Wrapper para iLIDS-VID (solo CMC)
- `print_results()`: Impresión formateada de resultados
- `save_results() / load_results()`: Persistencia de resultados

#### `splits.py` (Por implementar)
Manejo de splits oficiales:
- Parser de archivos .mat (iLIDS-VID)
- Validación de splits

#### `plots.py` (Por implementar)
Visualización:
- Curvas CMC
- Top-k retrieval examples
- Distribución de distancias

---

### `/bilp/` - Módulo principal BILP

#### `color.py`
Extracción de características de color invariantes a brillo:
- **Log-chromaticity**: `u = log(R/G)`, `v = log(B/G)`
- **Luminancia**: Canal Y para información de intensidad
- **Histogramas por stripe**: 16×16 bins (u,v) + 16 bins (luminancia)
- **Calibración**: Rangos (u,v) calculados con percentiles 1-99% del train set

**Dimensión por stripe**: 272 features (256 + 16)

#### `texture.py`
Extracción de características de textura:
- **Gabor bank**: 5 escalas × 8 orientaciones = 40 filtros
- **FFT features**: Frecuencia pico + entropía espectral
- **Energía por banda**: L2 norm de respuesta filtrada

**Dimensión por stripe**: 42 features (40 Gabor + 2 FFT)

#### `gating.py` - Implementado
Fusión adaptativa color/textura:
- **`compute_gating_weight(color, texture, params)`**: Calcula α = σ(a1·T - a2·C + b)
  - C: entropía cromática (de histograma 2D u,v)
  - T: proporción de energía en altas frecuencias (Gabor)
- **`compute_gating_weights_batch()`**: Cálculo batch de pesos
- **`optimize_gating_params()`**: Grid-search de parámetros (a1, a2, b)
- **`compute_per_stripe_gating()`**: Gating independiente por stripe

#### `distance.py` - Implementado
Cálculo de distancias BILP:
- **`bilp_distance()`**: d = α·d_tex + (1-α)·d_col
- **`compute_distance_matrix_fast()`**: Matriz de distancias optimizada con scipy
- **`l1_distance(), l2_distance(), chi_square_distance()`**: Métricas de distancia
- **`rank_gallery()`**: Ranking de galería para una query
- **`compute_ap()`**: Average Precision para Market-1501

Soporta múltiples métricas: L1 (cityblock), L2 (euclidean), Chi-square, Bhattacharyya

#### `utils.py` - Implementado
Utilidades completas:
- **Normalización**: `normalize_l1()`, `power_normalize()`, `normalize_features()`
- **`extract_bilp_descriptor(image)`**: Extracción end-to-end para una imagen
- **`extract_bilp_batch(images)`**: Procesamiento batch optimizado
- **`save_features() / load_features()`**: Guardado/carga de features en .npz
- **`compute_feature_stats()`**: Estadísticas de features

---

## Uso del Proyecto

### 1. Construcción del entorno Docker

```bash
docker build -t cv-project .
```

### 2. Ejecutar scripts en el contenedor

```bash
# Modo interactivo
docker run -it --rm -v $(pwd):/app cv-project bash

# Ejecutar script específico
docker run --rm -v $(pwd):/app cv-project python scripts/01_prepare_data.py
```

### 3. Ejecución en VM con GPU

Para ejecutar el proyecto en una VM de Google Cloud con GPU usando Docker, consulta la guía completa en **[GPU_SETUP.md](GPU_SETUP.md)**.

La guía incluye:
- Configuración de la VM con GPU
- Instalación de Docker y NVIDIA Container Toolkit
- Preparación de datasets
- Construcción de imagen Docker con soporte GPU (CuPy)
- Ejecución del pipeline completo con aceleración GPU
- Troubleshooting y optimización de costos

**Mejora de rendimiento con GPU:**
- Extracción de features: 5-20x más rápido
- Cálculo de distancias: 10-50x más rápido
- Tiempo total estimado: ~1-1.5 horas (vs ~5-6 horas en CPU)

---

## Dimensiones de Features

### Por imagen (6 stripes):
- **Color**: 6 × 272 = **1,632 features**
- **Textura**: 6 × 42 = **252 features**
- **Total**: **1,884 features**

### Procesamiento por stripe:
Cada imagen se divide en **6 horizontal stripes** para capturar información espacial.

---

## Próximos Pasos Inmediatos

1. **Completar módulo de gating** (`bilp/gating.py`)
2. **Completar módulo de distancia** (`bilp/distance.py`)
3. **Crear utils** (`bilp/utils.py`)
4. **Testear pipeline completo** con imágenes sintéticas
5. **Preparar loaders de datasets** (Market-1501 e iLIDS-VID)

---

## Referencias

- **Market-1501**: Zheng et al. "Scalable Person Re-identification: A Benchmark" (ICCV 2015)
- **iLIDS-VID**: Wang et al. "Person Re-Identification by Video Ranking" (ECCV 2014)
- **BILP**: Basado en técnicas de color invariante y análisis de textura multi-escala

---

## Notas de Desarrollo

### Log de Cambios

**Setup completo**
- Setup inicial del proyecto con Docker
- Creación de Dockerfile y requirements.txt
- Implementación de `bilp/color.py` y `bilp/texture.py`
- Estructura de directorios completa
- Verificación de datasets: Market-1501 e iLIDS-VID correctos y completos
- Market-1501: 12,936 train + 19,732 test + 3,368 queries
- iLIDS-VID: Secuencias de video, 2 cámaras, splits oficiales

**Data Loaders**
- Implementación completa de `eval/loaders.py`
- Parser de nombres para ambos datasets
- Loader Market-1501 con splits train/test/query
- Loader iLIDS-VID con sampling de frames (uniform/random/all)
- Funciones de estadísticas y train/val split
- Testing completo: todos los loaders funcionando correctamente
- Resultados: 12,936 train images + 600 video sequences cargadas

**Módulos BILP Completos**
- Implementación completa de todos los módulos BILP
- `bilp/utils.py`: Normalización L1/power, extracción batch, save/load features
- `bilp/distance.py`: Distancias L1/L2/Chi2/Bhattacharyya, ranking, mAP
- `bilp/gating.py`: Fusión adaptativa con parámetros optimizables
- Testing completo del pipeline BILP
- Extracción verificada: 1,884 features (1,632 color + 252 texture)
- Batch processing funcionando en 100 imágenes
- Distance matrix computada correctamente

**Métricas de Evaluación**
- Implementación de `eval/cmc_map.py` con CMC y mAP
- CMC (Cumulative Matching Characteristic) completo
- mAP (mean Average Precision) para Market-1501
- Soporte para junk removal (same camera images)
- Funciones de evaluación: evaluate_market1501(), evaluate_ilids_vid()
- Testing completo con datos sintéticos y reales
- Resultados en subset (10 queries, 100 gallery): Rank-1=70%, mAP=63.73%

**Calibración de Color**
- Implementación de calibración automática de rangos (u,v)
- Script `calibrate_color_ranges.py` para calibrar desde train set
- Rangos calibrados: u=[-0.663, 1.069], v=[-0.656, 0.480]
- Función `load_calibrated_color_ranges()` para cargar rangos
- Integración automática en BILP descriptor extraction
- Testing: 96.53% de píxeles dentro de rangos calibrados
- Persistencia en JSON: `data/color_ranges.json`

## Evaluación End-to-End (Subset)

### Script de Evaluación
Creado scripts/eval_market_subset.py para pipeline completo de evaluación:
- Carga queries y gallery (con muestreo opcional)
- Extrae features BILP usando rangos de color calibrados
- Computa matriz de distancias con alpha configurable
- Evalúa métricas CMC y mAP
- Guarda features y resultados

### Resultados Iniciales (Gallery=100)

Configuración:
- Queries: 3,362 (todas)
- Gallery: 100 imágenes (muestra aleatoria)
- Alpha (gating): 0.5
- Tiempo de extracción de features: 1788.80s (0.517s por imagen)

Resultados:
- mAP: 24.91%
- Rank-1: 1.55%
- Rank-5: 3.57%
- Rank-10: 4.88%
- Rank-20: 6.07%

Análisis:
- mAP de 24.91% es alentador para features handcrafted
- Rank-1 bajo (1.55%) es esperado con gallery muy pequeña (100 imágenes)
- El sistema maneja correctamente el junk removal (imágenes de misma cámara)
- La extracción de features es el cuello de botella (99.8% del tiempo total)
- El cómputo de distancias es muy rápido (0.27s para 336,200 distancias)

Archivos guardados:
- data/features_subset.npz: Features BILP extraídas
- data/results_subset.npz: Resultados de evaluación

Próximos pasos:
- Evaluar con gallery más grande (5,000-10,000 imágenes) para resultados más realistas
- Optimizar parámetros de gating (a1, a2, b) si es necesario
- Considerar extraer features para el dataset completo para evaluaciones posteriores más rápidas


## Tareas Pendientes

### Corto Plazo (Evaluación y Optimización)
1. Ejecutar evaluación con tamaños de gallery más grandes:
   - 5,000 imágenes: Evaluación de rendimiento más realista
   - 10,000 imágenes: Mejor significancia estadística
   - Test set completo (19,732 imágenes): Resultados finales de benchmark

2. Optimizar parámetros de gating:
   - Actual: alpha=0.5 (peso fijo igual)
   - TODO: Grid search para parámetros óptimos a1, a2, b
   - Implementar gating adaptativo por stripe usando funciones de bilp/gating.py

3. Estudios de ablación:
   - Features solo color (alpha=0.0)
   - Features solo textura (alpha=1.0)
   - Comparar con diferentes valores de alpha (0.3, 0.5, 0.7)

### Mediano Plazo (Dataset Completo)
4. Extraer y guardar features para dataset completo:
   - Train set: 12,936 imágenes
   - Test/gallery: 19,732 imágenes
   - Query: 3,362 imágenes
   - Total: ~33k imágenes (~4.8 horas a 0.517s/imagen)
   - Beneficio: Ejecutar múltiples evaluaciones sin re-extraer features

5. Evaluar en dataset iLIDS-VID:
   - Implementar manejo de secuencias de video (muestreo de K frames)
   - Protocolo de evaluación multi-shot
   - Comparar con resultados de Market-1501

### Largo Plazo (Features Avanzadas)
6. Optimizaciones de rendimiento:
   - Paralelizar extracción de features (multiprocessing)
   - Optimizar cómputo de filtros Gabor
   - Considerar aceleración GPU para matriz de distancias

7. Comparaciones con baselines:
   - Implementar baselines simples (histograma HSV, LBP)
   - Comparar BILP vs baselines
   - Documentar trade-offs de rendimiento

8. Herramientas de visualización:
   - Visualización de resultados de queries
   - Visualización de features
   - Heatmaps de matriz de distancias
   - Gráficos de curvas CMC

### Documentación
9. Reporte final:
   - Tabla completa de resultados
   - Análisis de rendimiento
   - Limitaciones y trabajo futuro
   - Instrucciones de uso

### Estado Actual
- COMPLETO
  - Configuración de entorno Docker
  - Implementación BILP (color, textura, gating, distancia)
  - Data loaders (Market-1501, iLIDS-VID)
  - Métricas de evaluación (CMC, mAP)
  - Calibración de color desde training set
  - Pipeline de evaluación end-to-end

- Próximo paso inmediato: Ejecutar evaluación con gallery más grande (5,000-10,000 imágenes)

---

## Actualizaciones recientes

- `scripts/02_extract_market_features.py`: nuevo nombre y parámetros claros para extraer y guardar features de Market-1501 por split (`train`, `query`, `test`). Produce `.npz` con color, textura y metadata.
- `scripts/02_extract_ilids_features.py`: extracción para iLIDS-VID. Muestras K frames por secuencia (cam1 → query, cam2 → gallery), promedia features y guarda `.npz` separados.
- `scripts/03_eval_market.py`: carga features precomputadas de Market-1501, calcula matriz de distancias (color+textura) y reporta mAP/CMC completos. Permite guardar resultados.
- `scripts/04_eval_ilidsvid.py`: evaluación CMC para iLIDS-VID usando los `.npz` generados. Configurable (`alpha`, métrica, rank máximo).
