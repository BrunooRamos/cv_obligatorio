# Evaluación Comparativa: BILP vs torchreid ResNet50

Este documento describe cómo ejecutar y comparar los métodos BILP (Brightness-Invariant Local Patterns) y torchreid ResNet50 en los datasets Market-1501 e iLIDS-VID.

## Resumen

Este proyecto implementa dos enfoques para Person Re-Identification:

1. **BILP**: Método handcrafted que combina características de color y textura usando gating adaptativo
2. **torchreid ResNet50**: Baseline de deep learning usando ResNet50 pre-entrenado

Ambos métodos se evalúan usando las mismas métricas estándar:
- **mAP** (mean Average Precision)
- **CMC** (Cumulative Matching Characteristics) en ranks 1, 5, 10, 20, etc.

## Requisitos Previos

1. Docker con soporte GPU configurado (ver `GPU_SETUP.md`)
2. Datasets descargados:
   - Market-1501-v15.09.15
   - iLIDS-VID
3. Imagen Docker construida: `cv-bilp-gpu`

## Estructura de Archivos

### Scripts de Extracción de Features

- **BILP**:
  - `scripts/02_extract_market_features.py` - Extrae features BILP para Market-1501
  - `scripts/02_extract_ilids_features.py` - Extrae features BILP para iLIDS-VID

- **torchreid**:
  - `scripts/08_extract_torchreid_market_features.py` - Extrae features ResNet50 para Market-1501
  - `scripts/09_extract_torchreid_ilids_features.py` - Extrae features ResNet50 para iLIDS-VID

### Scripts de Evaluación

- **BILP**:
  - `scripts/03_eval_market.py` - Evalúa BILP en Market-1501
  - `scripts/04_eval_ilidsvid.py` - Evalúa BILP en iLIDS-VID

- **torchreid**:
  - `scripts/10_eval_torchreid_market.py` - Evalúa torchreid en Market-1501
  - `scripts/11_eval_torchreid_ilidsvid.py` - Evalúa torchreid en iLIDS-VID

## Ejecución Paso a Paso

### 1. Construir Imagen Docker (si aún no está construida)

```bash
docker build -f Dockerfile.gpu -t cv-bilp-gpu .
```

### 2. Extraer Features BILP

#### Market-1501

```bash
docker run --gpus all --rm \
  -v $(pwd):/app \
  -v /home/obligatorio/dataset:/datasets \
  -v $(pwd)/data:/app/data \
  cv-bilp-gpu python scripts/02_extract_market_features.py \
  --dataset-path /datasets/Market-1501-v15.09.15 \
  --output-dir data/features \
  --splits train query test \
  --batch-size 128 \
  --use-gpu \
  --overwrite \
  --verbose
```

#### iLIDS-VID

```bash
docker run --gpus all --rm \
  -v $(pwd):/app \
  -v /home/obligatorio/dataset:/datasets \
  -v $(pwd)/data:/app/data \
  cv-bilp-gpu python scripts/02_extract_ilids_features.py \
  --dataset-path /datasets/iLIDS-VID \
  --output-dir data/features \
  --num-frames 10 \
  --sampling-strategy uniform \
  --use-gpu \
  --overwrite \
  --verbose
```

### 3. Extraer Features torchreid ResNet50

#### Market-1501

```bash
docker run --gpus all --rm \
  -v $(pwd):/app \
  -v /home/obligatorio/dataset:/datasets \
  -v $(pwd)/data:/app/data \
  cv-bilp-gpu python scripts/08_extract_torchreid_market_features.py \
  --dataset-path /datasets/Market-1501-v15.09.15 \
  --output-dir data/features \
  --splits train query test \
  --batch-size 64 \
  --use-gpu \
  --overwrite \
  --verbose
```

#### iLIDS-VID

```bash
docker run --gpus all --rm \
  -v $(pwd):/app \
  -v /home/obligatorio/dataset:/datasets \
  -v $(pwd)/data:/app/data \
  cv-bilp-gpu python scripts/09_extract_torchreid_ilids_features.py \
  --dataset-path /datasets/iLIDS-VID \
  --output-dir data/features \
  --num-frames 10 \
  --sampling-strategy uniform \
  --batch-size 32 \
  --use-gpu \
  --overwrite \
  --verbose
```

### 4. Evaluar Features BILP

#### Market-1501

```bash
docker run --gpus all --rm \
  -v $(pwd):/app \
  -v $(pwd)/data:/app/data \
  cv-bilp-gpu python scripts/03_eval_market.py \
  --query-features data/features/market1501_query.npz \
  --gallery-features data/features/market1501_test.npz \
  --alpha 0.5 \
  --metric cityblock \
  --max-rank 50 \
  --use-gpu \
  --verbose
```

#### iLIDS-VID

```bash
docker run --gpus all --rm \
  -v $(pwd):/app \
  -v $(pwd)/data:/app/data \
  cv-bilp-gpu python scripts/04_eval_ilidsvid.py \
  --query-features data/features/ilidsvid_query.npz \
  --gallery-features data/features/ilidsvid_gallery.npz \
  --alpha 0.5 \
  --metric cityblock \
  --max-rank 20 \
  --use-gpu \
  --verbose
```

### 5. Evaluar Features torchreid

#### Market-1501

```bash
docker run --gpus all --rm \
  -v $(pwd):/app \
  -v $(pwd)/data:/app/data \
  cv-bilp-gpu python scripts/10_eval_torchreid_market.py \
  --query-features data/features/market1501_query_torchreid.npz \
  --gallery-features data/features/market1501_test_torchreid.npz \
  --metric cosine \
  --max-rank 50 \
  --use-gpu \
  --verbose
```

#### iLIDS-VID

```bash
docker run --gpus all --rm \
  -v $(pwd):/app \
  -v $(pwd)/data:/app/data \
  cv-bilp-gpu python scripts/11_eval_torchreid_ilidsvid.py \
  --query-features data/features/ilidsvid_query_torchreid.npz \
  --gallery-features data/features/ilidsvid_gallery_torchreid.npz \
  --metric cosine \
  --max-rank 20 \
  --use-gpu \
  --verbose
```

## Formato de Features

### BILP

Los archivos `.npz` de BILP contienen:
- `color`: Array numpy de forma `(n_samples, 1632)` - Features de color
- `texture`: Array numpy de forma `(n_samples, 252)` - Features de textura
- `metadata`: Diccionario con información de las imágenes

**Dimensión total**: 1884 features por muestra (1632 color + 252 textura)

### torchreid ResNet50

Los archivos `.npz` de torchreid contienen:
- `features`: Array numpy de forma `(n_samples, 2048)` - Features unificadas
- `metadata`: Diccionario con información de las imágenes

**Dimensión total**: 2048 features por muestra

## Métricas de Evaluación

Ambos métodos usan las mismas métricas estándar de Re-ID:

### Market-1501
- **mAP**: Mean Average Precision (precisión promedio)
- **CMC@k**: Cumulative Matching Characteristics en rank k (k=1, 5, 10, 20, 50)

### iLIDS-VID
- **CMC@k**: Cumulative Matching Characteristics en rank k (k=1, 5, 10, 20)

## Comparación de Métodos

### Diferencias Metodológicas

| Aspecto | BILP | torchreid ResNet50 |
|---------|------|---------------------|
| **Tipo** | Handcrafted features | Deep learning |
| **Entrenamiento** | Sin entrenamiento | Pre-entrenado en ImageNet |
| **Features** | Color (1632) + Textura (252) | Vector único (2048) |
| **Gating** | Adaptativo (alpha variable) | No aplica |
| **Distancia** | Combinación ponderada | Cosine (estándar) |
| **Preprocesamiento** | Resize a 128x64, normalización L1 | Resize a 256x128, normalización ImageNet |

### Ventajas y Desventajas

**BILP**:
- ✅ No requiere entrenamiento
- ✅ Interpretable (color y textura separados)
- ✅ Gating adaptativo para combinar características
- ❌ Rendimiento generalmente inferior a deep learning
- ❌ Más sensible a variaciones de iluminación

**torchreid ResNet50**:
- ✅ Alto rendimiento (baseline establecido)
- ✅ Features aprendidas automáticamente
- ✅ Robusto a variaciones de iluminación
- ❌ Requiere GPU para entrenamiento (aunque usamos pre-entrenado)
- ❌ Menos interpretable

## Interpretación de Resultados

### Ejemplo de Salida

```
============================================================
Evaluation Results - Market-1501 (torchreid ResNet50, metric=cosine)
============================================================
mAP: 0.7234 (72.34%)
CMC Scores:
Rank-1:  0.8456 (84.56%)
Rank-5:  0.9234 (92.34%)
Rank-10: 0.9456 (94.56%)
Rank-20: 0.9634 (96.34%)
```

### Comparación Esperada

En general, se espera que torchreid ResNet50 supere a BILP en ambas métricas:
- **Market-1501**: torchreid típicamente alcanza mAP > 70% y Rank-1 > 80%
- **iLIDS-VID**: torchreid típicamente alcanza Rank-1 > 60%

BILP puede tener rendimiento más bajo pero sigue siendo útil como baseline handcrafted.

## Guardar Resultados

Para guardar resultados en archivo `.npz`:

```bash
# BILP
--save-results data/results/bilp_market1501.npz

# torchreid
--save-results data/results/torchreid_market1501.npz
```

Los archivos guardados contienen:
- `mAP`: Valor de mAP
- `CMC`: Array con scores CMC@k para k=1 hasta max_rank
- `metadata`: Información sobre parámetros de evaluación

## Troubleshooting

### Error: "CUDA out of memory"
- Reduce `--batch-size` en los scripts de extracción
- Usa CPU en lugar de GPU: elimina `--use-gpu`

### Error: "File not found"
- Verifica que los paths de los datasets sean correctos
- Asegúrate de que los archivos `.npz` existan antes de evaluar

### Error: "torchreid model not found"
- Verifica que `torchreid` esté instalado en el contenedor
- Reconstruye la imagen Docker si es necesario

## Notas Adicionales

1. **GPU**: Ambos métodos pueden usar GPU, pero torchreid se beneficia más del uso de GPU durante la extracción de features.

2. **Tiempo de Ejecución**: 
   - BILP: ~2-5 minutos para Market-1501 completo
   - torchreid: ~5-10 minutos para Market-1501 completo (depende de GPU)

3. **Memoria**: torchreid requiere más memoria GPU que BILP debido al tamaño del modelo ResNet50.

4. **Reproducibilidad**: Los resultados de torchreid pueden variar ligeramente debido a operaciones no determinísticas en PyTorch. Para reproducibilidad exacta, configura seeds de aleatoriedad.

## Referencias

- **BILP**: Método handcrafted desarrollado para este proyecto
- **torchreid**: [https://github.com/KaiyangZhou/deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid)
- **Market-1501**: [https://www.aitribune.com/dataset/2018051063](https://www.aitribune.com/dataset/2018051063)
- **iLIDS-VID**: [http://www.eecs.qmul.ac.uk/~xiatian/downloads_qmul_iLIDS-VID_ReID_dataset.html](http://www.eecs.qmul.ac.uk/~xiatian/downloads_qmul_iLIDS-VID_ReID_dataset.html)

