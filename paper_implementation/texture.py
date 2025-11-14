"""
Módulo para extracción de features de textura según Liu 2012.
Incluye filtros Gabor (8) y Schmid (13) aplicados sobre la luminancia Y.
"""
import numpy as np
from scipy import ndimage
from scipy.signal import convolve2d
from constants import (
    N_BINS, GABOR_FREQUENCIES, GABOR_ORIENTATIONS, GABOR_SIGMA_FACTOR,
    SCHMID_PAIRS
)


def get_gabor_kernel_size(sigma):
    """
    Calcula el tamaño del kernel Gabor (impar, mínimo 21).
    
    Args:
        sigma: desviación estándar del filtro
    
    Returns:
        ksize: tamaño del kernel (impar, >= 21)
    """
    ksize = int(np.ceil(6 * sigma))
    if ksize % 2 == 0:
        ksize += 1
    if ksize < 21:
        ksize = 21
    return ksize


def gabor_filter_2d(Y, frequency, orientation, sigma):
    """
    Aplica filtro Gabor 2D y retorna la magnitud de la respuesta.
    
    Args:
        Y: array 2D (luminancia)
        frequency: frecuencia espacial
        orientation: orientación en radianes
        sigma: desviación estándar (sigma = 0.56 / frequency)
    
    Returns:
        magnitude: array 2D con magnitud de la respuesta
    """
    ksize = get_gabor_kernel_size(sigma)
    
    # IMPORTANTE: Limitar el tamaño del kernel al tamaño del stripe
    # Si el kernel es más grande que el stripe, la convolución con padding
    # simétrico produce respuestas idénticas para diferentes imágenes
    H, W = Y.shape
    ksize = min(ksize, min(H, W))
    
    # Asegurar que ksize sea impar
    if ksize % 2 == 0:
        ksize -= 1
    if ksize < 3:
        ksize = 3
    
    center = ksize // 2
    
    # Crear grid
    x = np.arange(ksize) - center
    y = np.arange(ksize) - center
    X, Y_grid = np.meshgrid(x, y)
    
    # Rotar coordenadas según orientación
    X_rot = X * np.cos(orientation) + Y_grid * np.sin(orientation)
    Y_rot = -X * np.sin(orientation) + Y_grid * np.cos(orientation)
    
    # Kernel Gabor (parte real)
    gaussian = np.exp(-(X_rot**2 + Y_rot**2) / (2 * sigma**2))
    sinusoid = np.cos(2 * np.pi * frequency * X_rot)
    kernel = gaussian * sinusoid
    
    # Normalizar kernel
    kernel = kernel / np.sum(np.abs(kernel))
    
    # Convolución
    response = convolve2d(Y, kernel, mode='same', boundary='symm')
    
    # Retornar magnitud (valor absoluto)
    magnitude = np.abs(response)
    
    return magnitude


def schmid_filter_2d(Y, sigma, tau):
    """
    Aplica filtro Schmid (radial, invariante a rotación) y retorna la magnitud.
    
    Args:
        Y: array 2D (luminancia)
        sigma: parámetro sigma del filtro
        tau: parámetro tau del filtro
    
    Returns:
        magnitude: array 2D con magnitud de la respuesta
    """
    # Tamaño del kernel basado en sigma
    ksize = int(np.ceil(6 * sigma))
    if ksize % 2 == 0:
        ksize += 1
    if ksize < 21:
        ksize = 21
    
    # IMPORTANTE: Limitar el tamaño del kernel al tamaño del stripe
    # Si el kernel es más grande que el stripe, la convolución con padding
    # simétrico produce respuestas idénticas para diferentes imágenes
    H, W = Y.shape
    ksize = min(ksize, min(H, W))
    
    # Asegurar que ksize sea impar
    if ksize % 2 == 0:
        ksize -= 1
    if ksize < 3:
        ksize = 3
    
    center = ksize // 2
    
    # Crear grid radial
    x = np.arange(ksize) - center
    y = np.arange(ksize) - center
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    # Kernel Schmid radial
    # F(r) = cos(2*pi*tau*r/sigma) * exp(-r^2/(2*sigma^2))
    kernel = np.cos(2 * np.pi * tau * R / sigma) * np.exp(-R**2 / (2 * sigma**2))
    
    # Normalizar kernel
    kernel = kernel / np.sum(np.abs(kernel))
    
    # Convolución
    response = convolve2d(Y, kernel, mode='same', boundary='symm')
    
    # Retornar magnitud
    magnitude = np.abs(response)
    
    return magnitude


def normalize_to_01(response):
    """
    Normaliza una respuesta de filtro a [0, 1] usando min-max.
    
    Args:
        response: array 2D
    
    Returns:
        normalized: array 2D en [0, 1]
    """
    min_val = np.min(response)
    max_val = np.max(response)
    
    if max_val > min_val:
        return (response - min_val) / (max_val - min_val)
    else:
        return np.zeros_like(response)


def histogram_16_bins(channel):
    """
    Calcula histograma de 16 bins para un canal en [0, 1].
    
    Args:
        channel: array 2D en [0, 1]
    
    Returns:
        hist: array 1D de 16 elementos, normalizado L1
    """
    hist, _ = np.histogram(channel.flatten(), bins=N_BINS, range=(0.0, 1.0), density=False)
    hist = hist.astype(np.float32)
    
    # Normalización L1
    total = np.sum(hist)
    if total > 0:
        hist = hist / total
    else:
        hist = np.zeros(N_BINS, dtype=np.float32)
    
    return hist


def extract_texture_features(stripe_Y):
    """
    Extrae features de textura para una franja usando Gabor y Schmid.
    
    Args:
        stripe_Y: array (H_stripe, W) con luminancia Y
    
    Returns:
        texture_vector: array 1D de 336 elementos (21 filtros × 16 bins)
    """
    responses = []
    
    # 8 filtros Gabor
    for freq in GABOR_FREQUENCIES:
        sigma = GABOR_SIGMA_FACTOR / freq
        for orient in GABOR_ORIENTATIONS:
            resp = gabor_filter_2d(stripe_Y, freq, orient, sigma)
            responses.append(resp)
    
    # 13 filtros Schmid
    for sigma, tau in SCHMID_PAIRS:
        resp = schmid_filter_2d(stripe_Y, sigma, tau)
        responses.append(resp)
    
    # Normalizar cada respuesta a [0, 1] y calcular histogramas
    normalized_responses = [normalize_to_01(r) for r in responses]
    histograms = [histogram_16_bins(r) for r in normalized_responses]
    
    # Concatenar
    texture_vector = np.concatenate(histograms, axis=0)
    
    return texture_vector.astype(np.float32)


def extract_texture_features_with_gabor(stripe_Y):
    """
    Extrae features de textura y también devuelve respuestas Gabor para gating.
    
    Args:
        stripe_Y: array (H_stripe, W) con luminancia Y
    
    Returns:
        texture_vector: array 1D de 336 elementos (21 filtros × 16 bins)
        gabor_responses: lista de arrays 2D con respuestas de filtros Gabor (8 elementos)
    """
    gabor_responses = []
    schmid_responses = []
    
    # 8 filtros Gabor
    for freq in GABOR_FREQUENCIES:
        sigma = GABOR_SIGMA_FACTOR / freq
        for orient in GABOR_ORIENTATIONS:
            resp = gabor_filter_2d(stripe_Y, freq, orient, sigma)
            gabor_responses.append(resp)
    
    # 13 filtros Schmid
    for sigma, tau in SCHMID_PAIRS:
        resp = schmid_filter_2d(stripe_Y, sigma, tau)
        schmid_responses.append(resp)
    
    # Normalizar cada respuesta a [0, 1] y calcular histogramas
    all_responses = gabor_responses + schmid_responses
    normalized_responses = [normalize_to_01(r) for r in all_responses]
    histograms = [histogram_16_bins(r) for r in normalized_responses]
    
    # Concatenar
    texture_vector = np.concatenate(histograms, axis=0)
    
    return texture_vector.astype(np.float32), gabor_responses



