"""
Explicación: Cómo se manejan los frames y features en iLIDS-VID

ESTRUCTURA DE DATOS:
===================

tracklets = {
    person_id_1: {
        'cam1': [feature_frame1, feature_frame2, ..., feature_frameN],  # Lista de N frames
        'cam2': [feature_frame1, feature_frame2, ..., feature_frameM]  # Lista de M frames
    },
    person_id_2: {
        'cam1': [...],
        'cam2': [...]
    },
    ...
}

PROCESO:
========

1. EXTRACCIÓN (por cada frame individual):
   - Se carga cada frame como imagen RGB uint8
   - Se extrae features Liu 2012 → vector de 2784 dims
   - Se guarda en lista: tracklets[id][cam] = [feat1, feat2, ..., featN]
   
   Ejemplo:
   - Tracklet de persona 1 en cam1 tiene 50 frames
   - Se extraen 50 vectores de 2784 dims cada uno
   - Se almacenan: tracklets[1]['cam1'] = [array(2784), array(2784), ..., array(2784)]  # 50 elementos

2. POOLING (en evaluación):
   - Se toma la lista de features del tracklet
   - Se hace pooling (mean o median) sobre todos los frames
   - Resultado: 1 vector de 2784 dims por tracklet
   
   Ejemplo:
   - tracklets[1]['cam1'] tiene 50 features de 2784 dims
   - pool_tracklet_features() promedia los 50 → 1 vector de 2784 dims
   - Este vector único representa todo el tracklet

3. EVALUACIÓN:
   - Gallery: 1 tracklet por ID (pooling de frames de cam1)
   - Probe: 1 tracklet por ID (pooling de frames de cam2)
   - Se comparan los vectores pooled (no frame-a-frame)

IMPORTANTE:
===========
- NO se compara frame-a-frame (sería muy lento y menos robusto)
- Se compara tracklet-a-tracklet usando el descriptor pooled
- El pooling agrega información temporal y hace el descriptor más estable
"""
print("Ver código fuente para detalles completos")

