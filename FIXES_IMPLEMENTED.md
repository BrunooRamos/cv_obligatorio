# Correcciones Implementadas para Mejorar Performance de BILP en iLIDS-VID

## 🔍 Diagnóstico Realizado

### Problemas Identificados (Test 1)

El análisis diagnóstico reveló **3 problemas críticos**:

1. **Varianza Extremadamente Baja en Features**
   - Color: 88.6% de dimensiones con varianza ~0
   - Texture: 84.1% de dimensiones con varianza ~0
   - Varianza media color: 0.000002
   - Varianza media texture: 0.000000

2. **Ratios Inter/Intra-Persona Invertidos**
   - Ratios observados: 0.34-0.61 (debería ser >1.5)
   - Distancia entre personas DIFERENTES < Distancia entre muestras MISMA persona
   - El sistema está clasificando al revés

3. **Histogramas de Color Colapsados**
   - Histogramas extremadamente sparse (mayoría de bins = 0)
   - Solo 2-3 picos por stripe
   - Todos los IDs tienen patrones casi idénticos
   - Causa: Calibración de Market-1501 incompatible con iLIDS-VID

## ✅ Soluciones Implementadas

### 1. Script de Calibración Específico para iLIDS-VID

**Archivo:** `scripts/calibrate_color_ilids.py`

**Qué hace:**
- Carga muestra de secuencias de iLIDS-VID
- Extrae valores (u, v) de log-chromaticity de frames reales
- Calcula rangos basados en percentiles (1-99%) para evitar outliers
- Guarda rangos calibrados en JSON

**Cómo usar:**

```bash
# Opción 1: Con Docker (recomendado)
docker run --rm -v $(pwd):/app cv-project python scripts/calibrate_color_ilids.py \
    --dataset-path datasets/iLIDS-VID \
    --num-sequences 200 \
    --num-frames 5 \
    --output-file data/color_ranges_ilids.json \
    --verbose

# Opción 2: Directamente (si tienes el entorno configurado)
python scripts/calibrate_color_ilids.py \
    --dataset-path datasets/iLIDS-VID \
    --num-sequences 200 \
    --output-file data/color_ranges_ilids.json \
    --verbose
```

**Parámetros:**
- `--num-sequences`: Número de secuencias para calibración (default: 200)
- `--num-frames`: Frames por secuencia (default: 5)
- `--percentile-low/high`: Percentiles para rangos (default: 1-99)
- `--output-file`: Archivo JSON de salida

**Output esperado:**
```
Calibrated ranges (percentiles 1-99):
  U range: [-0.XXXX, X.XXXX]  # Valores específicos de iLIDS-VID
  V range: [-0.XXXX, X.XXXX]  # Diferentes de Market-1501

Coverage:
  U: ~98% of pixels within range
  V: ~98% of pixels within range
```

### 2. Uso de Rangos Calibrados en Extracción de Features

**Modificar:** `scripts/02_extract_ilids_features.py`

**Cambio necesario:**
```bash
# ANTES (usaba calibración de Market-1501)
python scripts/02_extract_ilids_features.py \
    --calibration-file data/color_ranges_market.json \
    --n-stripes 6 \
    --num-frames 10

# DESPUÉS (usa calibración específica de iLIDS-VID)
python scripts/02_extract_ilids_features.py \
    --calibration-file data/color_ranges_ilids.json \
    --n-stripes 6 \
    --num-frames 10
```

### 3. Recomendaciones Adicionales (Pendientes de Implementar)

#### 3.1. Reducir Número de Bins (8x8 en lugar de 16x16)

**Modificación en extracción:**
```bash
# Agregar parámetros para bins más gruesos
python scripts/02_extract_ilids_features.py \
    --calibration-file data/color_ranges_ilids.json \
    --n-bins-uv 8 \     # En lugar de 16
    --n-bins-lum 8 \    # En lugar de 16
    --num-frames 10
```

**Nota:** Esto requiere modificar `scripts/02_extract_ilids_features.py` para aceptar estos parámetros.

#### 3.2. Simplificar Normalización

**Problema actual:** Múltiples normalizaciones en cascada aplastando diferencias

**En `bilp/utils.py` - función `extract_bilp_batch`:**

Cambiar de:
```python
# Normaliza por stripe → por frame → por secuencia
color_vec = normalize_l1(color_vec)  # Por stripe
texture_vec = normalize_l1(texture_vec)
# ... más normalizaciones ...
```

A:
```python
# UNA SOLA normalización al final
# En extract_bilp_batch, después de promediar frames:
color_mean = np.mean(color_batch, axis=0)
texture_mean = np.mean(texture_batch, axis=0)

# L2 normalization final SOLAMENTE
norm_color = np.linalg.norm(color_mean) + 1e-12
color_final = (color_mean / norm_color).astype(np.float32)

norm_texture = np.linalg.norm(texture_mean) + 1e-12
texture_final = (texture_mean / norm_texture).astype(np.float32)
```

#### 3.3. Revisar Normalización de Gabor (Texture)

**Archivo:** `bilp/texture.py`

**Acción:** Verificar si las respuestas de Gabor tienen varianza ANTES de normalizar:
- Si varianza es buena antes → problema es la normalización
- Si varianza es mala antes → ajustar parámetros de filtros

## 📋 Pipeline Completo de Corrección

### Paso 1: Calibrar Rangos de Color
```bash
docker run --rm -v $(pwd):/app cv-project \
    python scripts/calibrate_color_ilids.py \
    --dataset-path datasets/iLIDS-VID \
    --num-sequences 200 \
    --output-file data/color_ranges_ilids.json \
    --verbose
```

### Paso 2: Re-extraer Features con Nueva Calibración

**IMPORTANTE:** Montar el dataset iLIDS-VID en el contenedor Docker:
- Dataset ubicado en: `../datasets/iLIDS-VID` (relativo al directorio `code/`)
- Montaje: `-v $(pwd)/../datasets/iLIDS-VID:/app/datasets/iLIDS-VID`

**Opción A (recomendada): Una línea**
```bash
docker run --rm -v $(pwd):/app -v $(pwd)/../datasets/iLIDS-VID:/app/datasets/iLIDS-VID cv-project python scripts/02_extract_ilids_features.py --dataset-path datasets/iLIDS-VID --calibration-file data/color_ranges_ilids.json --output-dir data/features --query-filename ilidsvid_query_recalibrated.npz --gallery-filename ilidsvid_gallery_recalibrated.npz --num-frames 10 --n-stripes 6 --overwrite --verbose
```

**Opción B: Multi-línea (sin espacios después de `\`)**
```bash
docker run --rm \
-v $(pwd):/app \
-v $(pwd)/../datasets/iLIDS-VID:/app/datasets/iLIDS-VID \
cv-project \
python scripts/02_extract_ilids_features.py \
--dataset-path datasets/iLIDS-VID \
--calibration-file data/color_ranges_ilids.json \
--output-dir data/features \
--query-filename ilidsvid_query_recalibrated.npz \
--gallery-filename ilidsvid_gallery_recalibrated.npz \
--num-frames 10 \
--n-stripes 6 \
--overwrite \
--verbose
```

### Paso 3: Ejecutar Test Diagnóstico con Nuevas Features
```bash
docker run --rm -v $(pwd):/app cv-project \
    python tests/test_1.py \
    --query-features data/features/ilidsvid_query_recalibrated.npz \
    --gallery-features data/features/ilidsvid_gallery_recalibrated.npz \
    --num-persons 10 \
    --output-dir debug_output_recalibrated
```

### Paso 4: Comparar Resultados

**Métricas a comparar:**

| Métrica | Antes (Market calibration) | Después (iLIDS calibration) | Target |
|---------|---------------------------|----------------------------|--------|
| Varianza media color | 0.000002 | ??? | >0.001 |
| Dims con var ~0 (color) | 88.6% | ??? | <30% |
| Ratio Inter/Intra | 0.34-0.61 | ??? | >1.5 |
| Separación same/diff | 0.0007 | ??? | >0.01 |
| Rank-1 (evaluación) | ~17% | ??? | >30% |

## 🎯 Resultados Esperados

Con las correcciones implementadas, esperamos:

1. **Histogramas de Color más Poblados**
   - Distribución más uniforme de valores
   - Más bins con valores >0
   - Patrones diferenciables entre personas

2. **Mayor Varianza en Features**
   - Varianza media color >0.001
   - <30% de dimensiones con varianza ~0
   - Dimensiones más discriminativas

3. **Ratios Correctos**
   - Inter-persona / Intra-persona >1.5
   - Distancias same-ID < Distancias diff-ID

4. **Mejor Performance**
   - Rank-1 mejorado (target >30%)
   - mAP mejorado
   - Curva CMC con pendiente más pronunciada

## 📁 Archivos Creados/Modificados

### Nuevos Archivos
- `scripts/calibrate_color_ilids.py` - Script de calibración
- `tests/test_1.py` - Script de diagnóstico extendido
- `FIXES_IMPLEMENTED.md` - Este documento

### Archivos a Modificar (Pendiente)
- `scripts/02_extract_ilids_features.py` - Agregar parámetros para bins
- `bilp/utils.py` - Simplificar normalización
- `bilp/texture.py` - Revisar normalización Gabor (si es necesario)

## 🚀 Próximos Pasos

1. ✅ **COMPLETADO:** Crear script de calibración
2. ⏳ **PENDIENTE:** Ejecutar calibración en iLIDS-VID con dataset montado
3. ⏳ **PENDIENTE:** Re-extraer features con nueva calibración
4. ⏳ **PENDIENTE:** Ejecutar test diagnóstico con nuevas features
5. ⏳ **PENDIENTE:** Evaluar mejoras y ajustar si es necesario
6. ⏳ **PENDIENTE:** Implementar reducción de bins si se necesita más mejora
7. ⏳ **PENDIENTE:** Simplificar normalización si persisten problemas

## 📊 Logging de Resultados

Mantener registro de todas las ejecuciones:

```bash
# Ejemplo de log
echo "=== Calibración iLIDS-VID ===" >> results_log.txt
date >> results_log.txt
docker run --rm -v $(pwd):/app cv-project \
    python scripts/calibrate_color_ilids.py --verbose \
    2>&1 | tee -a results_log.txt
```

---

**Última actualización:** $(date)
**Autor:** Claude Code
**Estado:** Calibración implementada, pendiente de ejecución con dataset
