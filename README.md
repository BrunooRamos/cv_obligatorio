## bilp – Person Re-ID por color, textura y gating adaptativo

Implementación y extensión de un sistema de **Person Re-ID** basado en el trabajo de Liu et al. (2012), usando descriptores de **color** y **textura** por franjas horizontales y un mecanismo de **gating adaptativo** que pondera ambas distancias según la complejidad de textura y la entropía cromática de cada imagen.

Se evalúan las features propuestas y el mecanismo de gating en los datasets **Market-1501** e **iLIDS-VID**, midiendo desempeño con curvas **CMC** y **mAP**.


### Características principales

- **Extracción de features BILP** (`bilp/`):  
  - Color: histograma de log-cromaticidad \((u = \log(R/G), v = \log(B/G))\) y luminancia.  
  - Textura: banco de filtros de Gabor en varias escalas y orientaciones.
- **Gating adaptativo** (`bilp/gating.py`): combina distancias de color y textura con un peso aprendido \(\alpha\) por imagen.
- **Evaluación** (`eval/`): métricas CMC/mAP específicas para Market-1501 e iLIDS-VID.
- **Scripts de experimentos** (`scripts/`): calibración de rangos de color, extracción de features, evaluación y optimización de parámetros de gating.

### Estructura del repositorio

- **`bilp/`**: implementación del extractor BILP, distancias, gating y utilidades.
- **`eval/`**: loaders de datasets y métricas de evaluación (CMC, mAP).
- **`scripts/`**: scripts de alto nivel para correr los experimentos del proyecto.
- **`paper_implementation/`**: implementación de las features originales de Liu et al. (2012) y pipelines usados como baseline.
- **`data/`**: archivos auxiliares (rangos de color calibrados, parámetros de gating, etc.).

### Instalación (Docker)

- **Opción recomendada**: usar los `Dockerfile` incluidos en el repo (CPU o GPU).
- **CPU**:

```bash
docker build -t bilp-cpu -f Dockerfile .
docker run --rm -it \
  -v /ruta/local/datasets:/app/datasets \
  -v /ruta/local/data:/app/data \
  bilp-cpu /bin/bash
```

- **GPU** (requiere drivers y `nvidia-docker`):

```bash
docker build -t bilp-gpu -f Dockerfile.gpu .
docker run --rm -it --gpus all \
  -v /ruta/local/datasets:/app/datasets \
  -v /ruta/local/data:/app/data \
  bilp-gpu /bin/bash
```

Dentro del contenedor puedes ejecutar directamente los scripts de extracción y evaluación descritos abajo.  
Si prefieres no usar Docker, puedes instalar las dependencias con `pip install -r requirements.txt` (no garantizado para todas las combinaciones de SO/GPU).

### Uso básico

- **Market-1501**:

```bash
python scripts/02_extract_market_features.py --dataset-path <ruta_Market1501>
python scripts/03_eval_market.py
```

- **iLIDS-VID**:

```bash
python scripts/02_extract_ilids_features.py --dataset-path <ruta_iLIDS-VID>
python scripts/04_eval_ilidsvid.py
```

Para experimentar con el **gating adaptativo** (búsqueda de parámetros y evaluación), se pueden usar los scripts de optimización y los flags `--gating-params` en los scripts de evaluación (ver `scripts/05_optimize_gating_params.py` y la ayuda de cada script con `-h`).

### Autores

- **Bruno Ramos**
- **Rodrigo Sotelo**
- **Docentes**: Nicolás Rondán, Elías Masquil

