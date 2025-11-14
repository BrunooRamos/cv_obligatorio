# Implementación Liu 2012 - Extracción de Features y Evaluación CMC

Nota: esta carpeta se realizó como un proyecto aparte pero se agrega acá para entregar un único repositorio. Para ejecutar cosas se debe tratar esta carpeta como raíz de proyecto y tener los datasets en la raíz del proyecto también.

## 📋 Orden de Ejecución

### **PASO 1: Extraer Features** (PRIMERO)
Antes de evaluar, necesitas extraer features de todas tus imágenes/tracklets.

### **PASO 2: Evaluar CMC** (SEGUNDO)
Una vez que tengas las features, puedes evaluar el rendimiento con CMC.

---

## 🚀 Inicio Rápido

### ⭐ Opción Recomendada: Script Completo para iLIDS-VID

**Ejecuta directamente el script completo:**

```bash
python3 eval_ilidsvid_complete.py
```

Este script automáticamente:
1. Extrae features de todos los tracklets en iLIDS-VID
2. Evalúa CMC con pooling por tracklet y cross-camera
3. Muestra los resultados

### Opción A: Evaluar en iLIDS-VID (Video) - Manual

```python
from run_pipeline import extract_features_ilidsvid, evaluate_ilidsvid_pipeline

# PASO 1: Extraer features
tracklets = extract_features_ilidsvid('iLIDS-VID/i-LIDS-VID/sequences')

# PASO 2: Evaluar CMC
results = evaluate_ilidsvid_pipeline(tracklets, trials=10, pool_method='mean')
```

### Opción B: Evaluar en Stills (i-LIDS MCTS / VIPeR)

```python
from run_pipeline import extract_features_stills, evaluate_stills_pipeline

# PASO 1: Extraer features
features_by_id = extract_features_stills('ruta/a/tus/imagenes')

# PASO 2: Evaluar CMC
# Para i-LIDS MCTS: p=50
# Para VIPeR: p=316
results = evaluate_stills_pipeline(features_by_id, p=50, trials=10)
```

---

## 📝 Ejecución Paso a Paso

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Ejecutar el script guía

```bash
python3 run_pipeline.py
```

Este script te mostrará:
- Qué datasets tienes disponibles
- Cómo ejecutar cada paso
- Un ejemplo con datos simulados

### 3. Para iLIDS-VID (recomendado, ya que tienes el dataset)

```python
python3 -c "
from run_pipeline import extract_features_ilidsvid, evaluate_ilidsvid_pipeline

# Extraer features (esto puede tardar varios minutos)
print('Extrayendo features...')
tracklets = extract_features_ilidsvid('iLIDS-VID/i-LIDS-VID/sequences')

# Evaluar
print('Evaluando CMC...')
results = evaluate_ilidsvid_pipeline(tracklets, trials=10)
print('Resultados:', results)
"
```

---

## 📁 Estructura del Proyecto

```
paper_implementation/
├── constants.py              # Constantes (dimensiones, parámetros)
├── color.py                  # Extracción de features de color
├── texture.py                # Filtros Gabor y Schmid
├── extractor.py              # Función principal extract_liu2012_features()
├── eval_stills.py            # Evaluación CMC para stills
├── eval_ilidsvid.py          # Evaluación CMC para video
├── eval_ilidsvid_complete.py # ⭐ Script completo para iLIDS-VID (EJECUTA ESTE)
├── run_pipeline.py           # Script guía paso a paso
├── example_usage.py          # Ejemplos de uso
└── requirements.txt          # Dependencias
```

---

## 🔍 Detalles Técnicos

### Features Extraídos
- **Dimensión**: 2784 (6 stripes × 464 dims/stripe)
- **Color**: 8 canales × 16 bins = 128 dims/stripe
- **Textura**: 21 filtros × 16 bins = 336 dims/stripe
  - 8 Gabor (4 freqs × 2 orientaciones)
  - 13 Schmid (pares sigma, tau)

### Evaluación CMC
- **Stills**: Protocolo single-shot, distancia L1, 10 trials
- **Video**: Pooling por tracklet (mean/median), cross-camera

---

## ⚠️ Notas Importantes

1. **Primero extrae features, luego evalúa**: No puedes evaluar sin haber extraído features primero.

2. **Tiempo de ejecución**: La extracción de features puede tardar varios minutos dependiendo del tamaño del dataset.

3. **Memoria**: Asegúrate de tener suficiente RAM. Para datasets grandes, considera procesar por lotes.

4. **Reproducibilidad**: Los seeds están fijos para garantizar resultados reproducibles.

---

## 🐛 Solución de Problemas

### Error: "No hay suficientes IDs"
- Verifica que tus imágenes/tracklets tengan IDs válidos
- Asegúrate de que haya al menos `p` IDs con imágenes

### Error: "Features deben tener 2784 dims"
- Verifica que las imágenes sean RGB uint8
- Revisa que el resize funcione correctamente

### Error de memoria
- Procesa el dataset en lotes más pequeños
- Guarda las features en disco y cárgalas cuando las necesites

---

## 📚 Referencias

- Liu 2012: "Person Re-identification: What Features Are Important?"
- i-LIDS MCTS: p=50, single-shot
- VIPeR: p=316, single-shot
- iLIDS-VID: pooling por tracklet, cross-camera

