"""
Constantes para la extracción de features Liu 2012 y evaluación CMC.
"""
import numpy as np

# Dimensiones de imagen
HEIGHT = 128
WIDTH = 64

# Configuración de stripes y bins
N_STRIPES = 6
N_BINS = 16

# Canales de color
COLOR_CHANNELS = ["R", "G", "B", "H", "S", "V", "Cb", "Cr"]  # 8
N_COLOR_DIMS_PER_STRIPE = 8 * 16  # 128

# Filtros de textura
N_GABOR = 8    # 4 freqs × 2 orient
N_SCHMID = 13  # pares (sigma, tau) listados
N_TEXTURE_DIMS_PER_STRIPE = (8 + 13) * 16  # 336

# Dimensiones totales
DIMS_PER_STRIPE = 128 + 336  # 464
DIMS_TOTAL = 6 * 464         # 2784

# Parámetros de evaluación
RANKS = (1, 5, 10, 20)
TRIALS = 10
P_ILIDS_MCTS = 50
P_VIPER = 316

# Parámetros Gabor
GABOR_FREQUENCIES = [0.10, 0.14, 0.20, 0.28]  # 4 frecuencias
GABOR_ORIENTATIONS = [0.0, np.pi / 2]  # 2 orientaciones
GABOR_SIGMA_FACTOR = 0.56  # sigma = 0.56 / f

# Parámetros Schmid (sigma, tau)
SCHMID_PAIRS = [
    (2, 1), (2, 2),
    (3, 1), (3, 2), (3, 3),
    (4, 1), (4, 2), (4, 3),
    (5, 1), (5, 2), (5, 3),
    (6, 1), (6, 2)
]

