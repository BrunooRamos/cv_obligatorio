# Guía de Configuración y Ejecución en VM con GPU

Esta guía explica cómo configurar y ejecutar el proyecto BILP en una VM de Google Cloud con GPU usando Docker.

## Especificaciones de la VM Recomendada

**Tipo de máquina:** g2-standard-4
- **CPU:** 4 vCPUs (Intel Cascade Lake)
- **Memoria:** 16 GB RAM
- **GPU:** 1 x NVIDIA L4 (24 GB VRAM)
- **Arquitectura:** x86-64

## Requisitos Previos

- Cuenta de Google Cloud Platform (GCP) activa
- Proyecto de GCP creado
- Permisos para crear instancias de VM con GPU
- Acceso SSH a la VM

---

## Paso 0: Crear la VM en GCP

### 0.1 Crear la instancia desde la consola web

1. Ve a [Google Cloud Console](https://console.cloud.google.com/)
2. Navega a **Compute Engine** > **VM instances**
3. Haz clic en **CREATE INSTANCE**
4. Configura la instancia:
   - **Name:** `bilp-gpu-vm` (o el nombre que prefieras)
   - **Region:** Selecciona una región con disponibilidad de L4 (ej: `us-central1`, `us-east1`, `europe-west4`)
   - **Zone:** Selecciona una zona (ej: `us-central1-a`)
   - **Machine family:** General-purpose
   - **Machine type:** `g2-standard-4` (4 vCPUs, 16 GB RAM)
   - **GPU:** 
     - Tipo: `NVIDIA L4`
     - Cantidad: `1`
   - **Boot disk:** 
     - OS: **Ubuntu 22.04 LTS** o **Ubuntu 20.04 LTS**
     - Tipo: **Balanced persistent disk**
     - Tamaño: **50 GB** (mínimo recomendado)
   - **Firewall:** Marca **Allow HTTP traffic** y **Allow HTTPS traffic** (opcional)
5. Haz clic en **CREATE**

**Nota:** La creación de la VM puede tardar varios minutos. Las GPUs en GCP requieren aprobación de cuota.

### 0.2 Crear la instancia desde gcloud CLI

```bash
# Configurar proyecto
gcloud config set project TU-PROJECT-ID

# Crear VM con GPU L4
gcloud compute instances create bilp-gpu-vm \
  --zone=us-central1-a \
  --machine-type=g2-standard-4 \
  --accelerator=type=nvidia-l4,count=1 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB \
  --boot-disk-type=pd-balanced \
  --maintenance-policy=TERMINATE \
  --restart-on-failure
```

### 0.3 Verificar que la VM tiene GPU

```bash
# Conectarse a la VM
gcloud compute ssh bilp-gpu-vm --zone=us-central1-a

# Verificar GPU (debe mostrar NVIDIA L4)
nvidia-smi
```

Si `nvidia-smi` no funciona, necesitas instalar los drivers de NVIDIA (ver Paso 1.1).

---

## Paso 1: Configurar la VM

### 1.1 Conectarse a la VM

```bash
gcloud compute ssh bilp-gpu-vm --zone=us-central1-a
```

### 1.2 Instalar Drivers de NVIDIA (si no están instalados)

Las imágenes de Ubuntu en GCP generalmente no incluyen los drivers de NVIDIA. Necesitas instalarlos:

```bash
# Actualizar sistema
sudo apt-get update
sudo apt-get upgrade -y

# Instalar drivers de NVIDIA (para Ubuntu 22.04)
# Para Ubuntu 20.04, usa: ubuntu-drivers install nvidia-driver-535
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers install

# O instalar versión específica (recomendado para L4)
sudo apt-get install -y nvidia-driver-535 nvidia-dkms-535

# Reiniciar la VM
sudo reboot
```

**Después del reinicio**, reconéctate y verifica:

```bash
# Verificar drivers instalados
nvidia-smi

# Debe mostrar algo como:
# NVIDIA-SMI 535.xx.xx   Driver Version: 535.xx.xx   CUDA Version: 12.x
```

**Nota:** Si usas una imagen de GCP con CUDA preinstalado (como `cuda-12-0-ubuntu-2204`), puedes saltar este paso.

### 1.3 Instalar CUDA Toolkit (si no está instalado)

Si `nvidia-smi` muestra CUDA Version pero necesitas el toolkit completo:

```bash
# Para CUDA 12.x (recomendado para L4)
wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda_12.3.0_545.23.06_linux.run
sudo sh cuda_12.3.0_545.23.06_linux.run --silent --toolkit

# Agregar CUDA al PATH
echo 'export PATH=/usr/local/cuda-12.3/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verificar instalación
nvcc --version
```

**Nota:** En GCP, generalmente CUDA viene preinstalado. Verifica primero con `nvcc --version`.

### 1.4 Instalar Docker (si no está instalado)

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

### 1.5 Verificar GPU en Docker

```bash
# Verificar que Docker puede acceder a la GPU
# Usa la imagen de CUDA que corresponda a tu versión instalada
docker run --rm --gpus all nvidia/cuda:12.3.0-base-ubuntu22.04 nvidia-smi
```

Si ves la salida de `nvidia-smi` mostrando la GPU L4, Docker está configurado correctamente.

**Troubleshooting:** Si obtienes un error, verifica:
1. NVIDIA Container Toolkit está instalado: `dpkg -l | grep nvidia-container-toolkit`
2. Docker está corriendo: `sudo systemctl status docker`
3. Reinicia Docker: `sudo systemctl restart docker`

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

### 5.1 Construir imagen con GPU

El proyecto ya incluye un `Dockerfile.gpu` configurado para CUDA 12.x (compatible con NVIDIA L4).

**Nota:** Si tu VM usa CUDA 11.x, edita `Dockerfile.gpu` y cambia `cupy-cuda12x` por `cupy-cuda11x`.

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

## Ajustes Necesarios en la VM

### Configuración de Memoria y Swap

Para una VM g2-standard-4 con 16 GB RAM, es recomendable configurar swap para evitar problemas de memoria:

```bash
# Verificar espacio en disco disponible
df -h

# Crear archivo de swap (4 GB recomendado)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Hacer el swap permanente
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Verificar que el swap está activo
free -h
```

### Optimización de Docker

Configura Docker para usar más memoria si es necesario:

```bash
# Editar configuración de Docker
sudo nano /etc/docker/daemon.json
```

Agrega (o modifica) el siguiente contenido:

```json
{
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
```

Reinicia Docker:

```bash
sudo systemctl restart docker
```

### Configuración de GPU

Para optimizar el rendimiento de la GPU L4:

```bash
# Verificar modo de rendimiento de la GPU
sudo nvidia-smi -pm 1  # Habilitar modo de rendimiento persistente

# Verificar que la GPU está en modo de rendimiento máximo
nvidia-smi -q | grep Performance
```

### Variables de Entorno Recomendadas

Agrega estas variables a tu `~/.bashrc`:

```bash
# CUDA paths (ajusta según tu versión de CUDA)
export PATH=/usr/local/cuda-12.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH

# CuPy memory pool (opcional, para mejor gestión de memoria)
export CUPY_ACCELERATORS=cub

# Python optimizaciones
export PYTHONUNBUFFERED=1
```

Recarga la configuración:

```bash
source ~/.bashrc
```

### Verificación Final de la Configuración

Ejecuta este script para verificar que todo está configurado correctamente:

```bash
#!/bin/bash
echo "=== Verificando configuración de la VM ==="
echo ""

echo "1. GPU disponible:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo ""
echo "2. CUDA disponible:"
nvcc --version 2>/dev/null || echo "CUDA toolkit no encontrado (puede estar bien si solo usas drivers)"

echo ""
echo "3. Docker con acceso a GPU:"
docker run --rm --gpus all nvidia/cuda:12.3.0-base-ubuntu22.04 nvidia-smi --query-gpu=name --format=csv,noheader

echo ""
echo "4. Memoria disponible:"
free -h

echo ""
echo "5. Espacio en disco:"
df -h /

echo ""
echo "6. NVIDIA Container Toolkit:"
dpkg -l | grep nvidia-container-toolkit || echo "No instalado"

echo ""
echo "=== Verificación completa ==="
```

Guarda este script como `check_setup.sh`, hazlo ejecutable y ejecútalo:

```bash
chmod +x check_setup.sh
./check_setup.sh
```

---

## Troubleshooting

### Error: "CUDA not available"

**Solución:** Verifica que:
1. La VM tenga GPU asignada: `nvidia-smi` debe mostrar L4
2. Drivers de NVIDIA instalados: `nvidia-smi` funciona
3. NVIDIA Container Toolkit esté instalado: `dpkg -l | grep nvidia-container-toolkit`
4. Docker tenga acceso a GPU: `docker run --rm --gpus all nvidia/cuda:12.3.0-base-ubuntu22.04 nvidia-smi`
5. Reinicia Docker si es necesario: `sudo systemctl restart docker`

### Error: "CuPy not installed" o "CuPy CUDA version mismatch"

**Solución:** 
1. Verifica la versión de CUDA en la VM: `nvcc --version` o `nvidia-smi`
2. Si usas CUDA 12.x, el Dockerfile.gpu ya está configurado correctamente
3. Si usas CUDA 11.x, edita `Dockerfile.gpu` y cambia `cupy-cuda12x` por `cupy-cuda11x`
4. Reconstruye la imagen: `docker build -f Dockerfile.gpu -t cv-bilp-gpu .`

### Error: "Out of memory" en GPU

**Solución:** 
- NVIDIA L4 tiene 24 GB VRAM, suficiente para este proyecto
- Si ocurre, reduce el batch size:
```bash
--batch-size 32  # en lugar de 64 o 128
```

### Error: "VM no tiene GPU asignada"

**Solución:**
1. Verifica en GCP Console que la VM tiene GPU L4 asignada
2. Si no la tiene, detén la VM y edítala para agregar GPU
3. Las GPUs en GCP requieren aprobación de cuota - solicítala si es necesario

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
gcloud compute instances stop bilp-gpu-vm --zone=us-central1-a
```

**Importante:** Detener la VM evita cargos por GPU y CPU, pero el disco persistente sigue generando costos menores.

### Iniciar la VM:

```bash
gcloud compute instances start bilp-gpu-vm --zone=us-central1-a
```

**Nota:** Al iniciar la VM, espera 1-2 minutos para que los servicios se inicialicen antes de conectarte.

### Monitorear uso de GPU:

```bash
# En la VM
watch -n 1 nvidia-smi

# O ver uso de recursos del sistema
htop
```

### Costos estimados (aproximados, varían por región):

- **g2-standard-4 con L4:** ~$1.20-1.50 USD/hora
- **Disco persistente (50 GB):** ~$0.17 USD/mes
- **Tráfico de red:** Varía según uso

**Recomendación:** Detén la VM cuando no la uses para minimizar costos. El pipeline completo toma ~1 hora, así que puedes ejecutarlo y detener la VM inmediatamente después.

---

## Tiempos Estimados

Con GPU **NVIDIA L4** (g2-standard-4):
- Calibración: ~3-5 minutos
- Extracción Market-1501 (33k imágenes): ~25-40 minutos
- Evaluación Market-1501: ~1-3 minutos
- Extracción iLIDS-VID: ~8-12 minutos
- Evaluación iLIDS-VID: ~1-2 minutos

**Total estimado:** ~45 minutos - 1 hora (vs ~5-6 horas en CPU)

**Nota:** NVIDIA L4 es más rápida que T4, especialmente en operaciones de memoria y cómputo de punto flotante.

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

