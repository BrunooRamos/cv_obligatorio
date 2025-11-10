# Liu et al. 2012 Feature Extraction

Implementación de la extracción de features del paper "Person Re-identification: What Features Are Important?" (Liu et al., ECCV Workshops 2012).

## Características

- **6 franjas horizontales** (stripes)
- **Features de color**: 8 canales (RGB, HSV, YCbCr) con histogramas de 16 bins → 128 dims/stripe
- **Features de textura**: 21 filtros (8 Gabor + 13 Schmid) sobre luminancia con histogramas de 16 bins → 336 dims/stripe
- **Dimensión total**: 6 × (128 + 336) = **2784 dimensiones**

## Estructura

```
liu2012/
├── __init__.py          # Exporta función principal
├── color.py             # Extracción de features de color
├── texture.py           # Extracción de features de textura (Gabor + Schmid)
├── extractor.py          # Función principal de extracción
└── utils.py             # Utilidades (guardar/cargar features)
```

## Uso

### Extracción de features de una imagen

```python
from liu2012.extractor import extract_liu2012_features
import cv2

# Cargar imagen RGB (uint8)
image = cv2.imread('image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Extraer features
features = extract_liu2012_features(image)
# features.shape = (2784,)
# features.dtype = float32
```

### Extracción de features de un batch

```python
from liu2012.extractor import extract_liu2012_batch

images = [img1, img2, ...]  # Lista de imágenes RGB (uint8)
features = extract_liu2012_batch(images, verbose=True)
# features.shape = (n_images, 2784)
```

## Scripts de extracción

### Market-1501

```bash
python scripts/extract_liu2012_market_features.py \
    --dataset-path datasets/Market-1501-v15.09.15 \
    --output-dir data/features \
    --splits train query test \
    --resize-height 128 \
    --resize-width 64 \
    --use-gpu \
    --verbose
```

**Nota importante**: Las dimensiones de resize son `--resize-height 128 --resize-width 64` (alto × ancho), no al revés. Esto es crucial porque el método divide la imagen en 6 franjas horizontales, por lo que necesitamos suficiente altura (≥128px) para que cada franja tenga información significativa.

**Soporte GPU**: Usa `--use-gpu` para acelerar la extracción de features de textura (convoluciones de filtros Gabor y Schmid). Requiere CuPy instalado.

### iLIDS-VID

```bash
python scripts/extract_liu2012_ilids_features.py \
    --dataset-path datasets/iLIDS-VID \
    --output-dir data/features \
    --num-frames 10 \
    --sampling-strategy uniform \
    --pooling-strategy first_frame \
    --resize-height 128 \
    --resize-width 64 \
    --use-gpu \
    --verbose
```

**Nota importante**: Las dimensiones de resize son `--resize-height 128 --resize-width 64` (alto × ancho), no al revés. Esto es crucial porque el método divide la imagen en 6 franjas horizontales, por lo que necesitamos suficiente altura (≥128px) para que cada franja tenga información significativa.

**Soporte GPU**: Usa `--use-gpu` para acelerar la extracción de features de textura (convoluciones de filtros Gabor y Schmid). Requiere CuPy instalado.

## Scripts de evaluación

### Market-1501

```bash
python scripts/eval_liu2012_market.py \
    --query-features data/features/liu2012_market1501_query.npz \
    --gallery-features data/features/liu2012_market1501_test.npz \
    --metric cityblock \
    --max-rank 50 \
    --verbose
```

### iLIDS-VID

**Protocolo Liu et al. 2012**:
- Seleccionar p=50 personas aleatoriamente
- Para cada trial (10 trials):
  - Gallery: 1 imagen aleatoria por persona (50 imágenes)
  - Probes: todas las demás imágenes de esas 50 personas
  - Calcular CMC
- Promediar CMC sobre los 10 trials
- Usar distancia L1 (cityblock)

```bash
python scripts/eval_liu2012_ilids.py \
    --features data/features/liu2012_ilidsvid_all.npz \
    --n-persons 50 \
    --n-trials 10 \
    --metric cityblock \
    --max-rank 20 \
    --verbose
```

**Nota**: El script de extracción guarda todas las features en un solo archivo (`liu2012_ilidsvid_all.npz`), ya que el protocolo necesita acceso a todas las imágenes para hacer el split aleatorio en cada trial.

## Detalles de implementación

### Features de color

- **Canales**: R, G, B, H, S, V, Cb, Cr (8 canales)
- **Histogramas**: 16 bins por canal, normalizados (suma = 1)
- **Total por stripe**: 8 × 16 = 128 dimensiones

### Features de textura

- **Filtros Gabor**: 8 filtros
  - Frecuencias: [0.10, 0.14, 0.20, 0.28]
  - Orientaciones: [0, π/2]
  - sigma = 0.56 / freq
  
- **Filtros Schmid**: 13 filtros rotacionalmente invariantes
  - Parámetros (σ, τ): (2,1), (4,1), (4,2), (6,1), (6,2), (6,3), (8,1), (8,2), (8,3), (10,1), (10,2), (10,3), (10,4)
  
- **Aplicación**: Sobre canal de luminancia (Y de YCbCr)
- **Histogramas**: 16 bins por respuesta de filtro, normalizados (suma = 1)
- **Total por stripe**: 21 × 16 = 336 dimensiones

### Orden del vector final

Para cada stripe (de arriba hacia abajo):
1. Color (128 dims): [R_hist, G_hist, B_hist, H_hist, S_hist, V_hist, Cb_hist, Cr_hist]
2. Textura (336 dims): [Gabor_1_hist, ..., Gabor_8_hist, Schmid_1_hist, ..., Schmid_13_hist]

Vector completo: stripe 1 → stripe 6 concatenados = **2784 dimensiones**

## Validación

Ejecutar el script de prueba:

```bash
python scripts/test_liu2012.py
```

Este script verifica:
- Dimensión correcta (2784)
- Tipo correcto (float32)
- Normalización de histogramas (suma ≈ 1)
- Ausencia de NaN o Inf

## Referencias

Liu, C., Gong, S., Loy, C. C., & Lin, X. (2012). Person re-identification: What features are important?. In European conference on computer vision (pp. 391-401). Springer, Berlin, Heidelberg.

