"""
Módulo para extracción de features de color según Liu 2012.
Incluye histogramas de 8 canales (R, G, B, H, S, V, Cb, Cr) con 16 bins cada uno.
"""
import numpy as np
from constants import N_BINS


def rgb_to_hsv(rgb):
    """
    Convierte imagen RGB a HSV.
    
    Args:
        rgb: array (H, W, 3) en [0, 1]
    
    Returns:
        hsv: array (H, W, 3) con H, S, V en [0, 1]
    """
    # Implementación manual de RGB a HSV
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    
    max_val = np.maximum(np.maximum(r, g), b)
    min_val = np.minimum(np.minimum(r, g), b)
    delta = max_val - min_val
    
    # Value
    v = max_val
    
    # Saturation
    s = np.where(max_val > 0, delta / max_val, 0.0)
    
    # Hue
    h = np.zeros_like(r)
    mask = delta > 0
    
    # R es máximo
    mask_r = mask & (max_val == r)
    h[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6.0
    
    # G es máximo
    mask_g = mask & (max_val == g)
    h[mask_g] = ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2.0
    
    # B es máximo
    mask_b = mask & (max_val == b)
    h[mask_b] = ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4.0
    
    h = h / 6.0  # Normalizar a [0, 1]
    
    return np.stack([h, s, v], axis=-1)


def rgb_to_ycbcr(rgb):
    """
    Convierte imagen RGB a YCbCr.
    
    Args:
        rgb: array (H, W, 3) en [0, 1]
    
    Returns:
        ycbcr: array (H, W, 3) con Y, Cb, Cr en [0, 1]
    """
    # Matriz de conversión estándar RGB -> YCbCr
    # Y  =  0.299*R + 0.587*G + 0.114*B
    # Cb = -0.168736*R - 0.331264*G + 0.5*B + 0.5
    # Cr =  0.5*R - 0.418688*G - 0.081312*B + 0.5
    
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    
    # Asegurar que estén en [0, 1]
    y = np.clip(y, 0.0, 1.0)
    cb = np.clip(cb, 0.0, 1.0)
    cr = np.clip(cr, 0.0, 1.0)
    
    return np.stack([y, cb, cr], axis=-1)


def normalize_to_01(channel):
    """
    Normaliza un canal a [0, 1] usando min-max.
    
    Args:
        channel: array 2D
    
    Returns:
        normalized: array 2D en [0, 1]
    """
    min_val = np.min(channel)
    max_val = np.max(channel)
    
    if max_val > min_val:
        return (channel - min_val) / (max_val - min_val)
    else:
        return np.zeros_like(channel)


def histogram_16_bins(channel):
    """
    Calcula histograma de 16 bins para un canal en [0, 1].
    
    Args:
        channel: array 2D en [0, 1]
    
    Returns:
        hist: array 1D de 16 elementos, normalizado L1
    """
    # Bins: [0, 1/16), [1/16, 2/16), ..., [15/16, 1]
    # Usar right=True para incluir el último bin correctamente
    hist, _ = np.histogram(channel.flatten(), bins=N_BINS, range=(0.0, 1.0), density=False)
    hist = hist.astype(np.float32)
    
    # Normalización L1
    total = np.sum(hist)
    if total > 0:
        hist = hist / total
    else:
        hist = np.zeros(N_BINS, dtype=np.float32)
    
    return hist


def extract_color_features(stripe_rgb, stripe_hsv, stripe_ycbcr):
    """
    Extrae features de color para una franja (stripe).
    
    Args:
        stripe_rgb: array (H_stripe, W, 3) RGB en [0, 1]
        stripe_hsv: array (H_stripe, W, 3) HSV en [0, 1]
        stripe_ycbcr: array (H_stripe, W, 3) YCbCr en [0, 1]
    
    Returns:
        color_vector: array 1D de 128 elementos (8 canales × 16 bins)
    """
    # Extraer canales
    r = stripe_rgb[..., 0]
    g = stripe_rgb[..., 1]
    b = stripe_rgb[..., 2]
    
    h = stripe_hsv[..., 0]
    s = stripe_hsv[..., 1]
    v = stripe_hsv[..., 2]
    
    cb = stripe_ycbcr[..., 1]
    cr = stripe_ycbcr[..., 2]
    
    # Asegurar que todos los canales estén en [0, 1] antes de binnear
    channels = [r, g, b, h, s, v, cb, cr]
    normalized_channels = [normalize_to_01(c) for c in channels]
    
    # Calcular histogramas
    histograms = [histogram_16_bins(c) for c in normalized_channels]
    
    # Concatenar
    color_vector = np.concatenate(histograms, axis=0)
    
    return color_vector.astype(np.float32)



