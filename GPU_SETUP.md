# Guía de Configuración y Ejecución en VM con GPU

Esta guía explica cómo configurar y ejecutar el proyecto BILP en una VM de Google Cloud con GPU usando Docker.

## Requisitos Previos

- VM de Google Cloud con GPU (NVIDIA T4, V100, etc.)
- CUDA instalado en la VM
- Docker instalado
- Acceso SSH a la VM

---

## Paso 1: Configurar la VM

### 1.1 Conectarse a la VM

```bash
gcloud compute ssh nombre-de-tu-vm --zone=us-central1-a
```

### 1.2 Instalar Docker (si no está instalado)

```bash
# Actualizar sistema
sudo apt-get update

# Instalar Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Reiniciar sesión o ejecutar:
newgrp docker
```

### 1.3 Instalar NVIDIA Container Toolkit

```bash
# Configurar repositorio
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Instalar
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Reiniciar Docker
sudo systemctl restart docker
```

### 1.4 Verificar GPU en Docker

```bash
docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
```

Si ves la salida de `nvidia-smi`, la GPU está funcionando correctamente en Docker.

---

## Paso 2: Clonar el Repositorio

### 2.1 Clonar desde GitHub

```bash
git clone https://github.com/tu-usuario/cv_obligatorio.git
cd cv_obligatorio
```

### O subir el código desde tu máquina local:

```bash
# Desde tu máquina local
gcloud compute scp --recurse cv_obligatorio/ nombre-vm:/home/tu-usuario/ --zone=us-central1-a

# En la VM
cd ~/cv_obligatorio
```

---

## Paso 3: Preparar los Datasets

### 3.1 Opción A: Subir datasets desde tu máquina local

```bash
# Desde tu máquina local
gcloud compute scp --recurse /ruta/local/Market-1501-v15.09.15 nombre-vm:/datasets/ --zone=us-central1-a
gcloud compute scp --recurse /ruta/local/iLIDS-VID nombre-vm:/datasets/ --zone=us-central1-a
```

### 3.2 Opción B: Usar Google Cloud Storage

```bash
# Subir datasets a un bucket
gsutil -m cp -r /ruta/local/Market-1501-v15.09.15 gs://tu-bucket/datasets/
gsutil -m cp -r /ruta/local/iLIDS-VID gs://tu-bucket/datasets/

# En la VM, descargar
gsutil -m cp -r gs://tu-bucket/datasets/Market-1501-v15.09.15 /datasets/
gsutil -m cp -r gs://tu-bucket/datasets/iLIDS-VID /datasets/
```

### 3.3 Verificar estructura de datasets

```bash
# Market-1501 debe tener esta estructura:
ls /datasets/Market-1501-v15.09.15/
# Debe mostrar: bounding_box_train/, bounding_box_test/, query/, etc.

# iLIDS-VID debe tener esta estructura:
ls /datasets/iLIDS-VID/
# Debe mostrar: i-LIDS-VID/sequences/cam1/, cam2/, etc.
```

---

## Paso 4: Construir la Imagen Docker

### 4.1 Construir imagen base

```bash
cd ~/cv_obligatorio
docker build -t cv-bilp .
```

### 4.2 Verificar que la imagen se construyó correctamente

```bash
docker images | grep cv-bilp
```

---

## Paso 5: Instalar CuPy en el Contenedor

### 5.1 Crear un Dockerfile con CuPy

Crea un archivo `Dockerfile.gpu` basado en el Dockerfile original pero con CuPy:

```dockerfile
# Dockerfile.gpu
FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt

# Instalar CuPy (ajusta según tu versión de CUDA)
# Para CUDA 11.x:
RUN pip install cupy-cuda11x

# Para CUDA 12.x (descomenta si usas CUDA 12):
# RUN pip install cupy-cuda12x

CMD ["python"]
```

### 5.2 Construir imagen con GPU

```bash
docker build -f Dockerfile.gpu -t cv-bilp-gpu .
```

---

## Paso 6: Ejecutar el Pipeline

### 6.1 Calibrar Rangos de Color

```bash
docker run --gpus all --rm \
  -v $(pwd):/app \
  -v /datasets:/datasets \
  -v $(pwd)/data:/app/data \
  cv-bilp-gpu python scripts/calibrate_color_ranges.py \
  --dataset-path /datasets/Market-1501-v15.09.15 \
  --save-path data/color_ranges_market.json
```

### 6.2 Extraer Features de Market-1501 (con GPU)

```bash
# Extraer features para train, query y test
docker run --gpus all --rm \
  -v $(pwd):/app \
  -v /datasets:/datasets \
  -v $(pwd)/data:/app/data \
  cv-bilp-gpu python scripts/02_extract_market_features.py \
  --dataset-path /datasets/Market-1501-v15.09.15 \
  --output-dir data/features \
  --splits train query test \
  --batch-size 128 \
  --use-gpu \
  --overwrite
```

**Nota:** El flag `--use-gpu` activa la aceleración GPU. Si no hay GPU disponible, automáticamente usa CPU.

### 6.3 Evaluar Market-1501 (con GPU)

```bash
docker run --gpus all --rm \
  -v $(pwd):/app \
  -v $(pwd)/data:/app/data \
  cv-bilp-gpu python scripts/03_eval_market.py \
  --query-features data/features/market1501_query.npz \
  --gallery-features data/features/market1501_test.npz \
  --alpha 0.5 \
  --metric cityblock \
  --use-gpu \
  --save-results data/results/market1501_full.npz \
  --verbose
```

### 6.4 Extraer Features de iLIDS-VID (con GPU)

```bash
docker run --gpus all --rm \
  -v $(pwd):/app \
  -v /datasets:/datasets \
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

### 6.5 Evaluar iLIDS-VID (con GPU)

```bash
docker run --gpus all --rm \
  -v $(pwd):/app \
  -v $(pwd)/data:/app/data \
  cv-bilp-gpu python scripts/04_eval_ilidsvid.py \
  --query-features data/features/ilidsvid_query.npz \
  --gallery-features data/features/ilidsvid_gallery.npz \
  --alpha 0.5 \
  --metric cityblock \
  --use-gpu \
  --save-results data/results/ilidsvid_full.npz \
  --verbose
```

---

## Paso 7: Modo Interactivo (Opcional)

Si prefieres trabajar dentro del contenedor:

```bash
docker run --gpus all -it --rm \
  -v $(pwd):/app \
  -v /datasets:/datasets \
  -v $(pwd)/data:/app/data \
  cv-bilp-gpu bash
```

Dentro del contenedor:

```bash
# Verificar GPU
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount())"

# Ejecutar scripts
python scripts/02_extract_market_features.py --use-gpu --dataset-path /datasets/Market-1501-v15.09.15
```

---

## Paso 8: Guardar Resultados en Cloud Storage

Para no perder resultados al detener la VM:

```bash
# Subir features y resultados a un bucket
gsutil -m cp -r data/features gs://tu-bucket/results/
gsutil -m cp -r data/results gs://tu-bucket/results/
```

---

## Troubleshooting

### Error: "CUDA not available"

**Solución:** Verifica que:
1. La VM tenga GPU asignada
2. NVIDIA Container Toolkit esté instalado
3. Docker tenga acceso a GPU: `docker run --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi`

### Error: "CuPy not installed"

**Solución:** Instala CuPy en el contenedor:
```bash
docker run --gpus all -it --rm \
  -v $(pwd):/app \
  cv-bilp-gpu pip install cupy-cuda11x
```

### Error: "Out of memory"

**Solución:** Reduce el batch size:
```bash
--batch-size 32  # en lugar de 128
```

### Error: "Dataset not found"

**Solución:** Verifica que los datasets estén montados correctamente:
```bash
docker run --gpus all -it --rm \
  -v /datasets:/datasets \
  cv-bilp-gpu ls /datasets/
```

---

## Optimización de Costos

### Detener la VM cuando no la uses:

```bash
# Desde tu máquina local
gcloud compute instances stop nombre-vm --zone=us-central1-a
```

### Iniciar la VM:

```bash
gcloud compute instances start nombre-vm --zone=us-central1-a
```

### Monitorear uso de GPU:

```bash
watch -n 1 nvidia-smi
```

---

## Tiempos Estimados

Con GPU (NVIDIA T4):
- Calibración: ~5 minutos
- Extracción Market-1501 (33k imágenes): ~30-45 minutos
- Evaluación Market-1501: ~2-5 minutos
- Extracción iLIDS-VID: ~10-15 minutos
- Evaluación iLIDS-VID: ~1-2 minutos

**Total estimado:** ~1-1.5 horas (vs ~5-6 horas en CPU)

---

## Script de Ejecución Completa

Puedes crear un script `run_full_pipeline.sh`:

```bash
#!/bin/bash
set -e

echo "=== Calibrating color ranges ==="
docker run --gpus all --rm \
  -v $(pwd):/app -v /datasets:/datasets -v $(pwd)/data:/app/data \
  cv-bilp-gpu python scripts/calibrate_color_ranges.py \
  --dataset-path /datasets/Market-1501-v15.09.15

echo "=== Extracting Market-1501 features ==="
docker run --gpus all --rm \
  -v $(pwd):/app -v /datasets:/datasets -v $(pwd)/data:/app/data \
  cv-bilp-gpu python scripts/02_extract_market_features.py \
  --dataset-path /datasets/Market-1501-v15.09.15 \
  --use-gpu --overwrite

echo "=== Evaluating Market-1501 ==="
docker run --gpus all --rm \
  -v $(pwd):/app -v $(pwd)/data:/app/data \
  cv-bilp-gpu python scripts/03_eval_market.py \
  --query-features data/features/market1501_query.npz \
  --gallery-features data/features/market1501_test.npz \
  --use-gpu --save-results data/results/market1501_full.npz

echo "=== Pipeline complete! ==="
```

Ejecutar:

```bash
chmod +x run_full_pipeline.sh
./run_full_pipeline.sh
```

---

## Referencias

- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [CuPy Installation](https://docs.cupy.dev/en/stable/install.html)
- [Docker GPU Support](https://docs.docker.com/config/containers/resource_constraints/#gpu)

