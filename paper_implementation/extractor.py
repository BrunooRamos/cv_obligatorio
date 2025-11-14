"""
Módulo principal para extracción de features Liu 2012.
Función principal: extract_liu2012_features()
"""
import numpy as np
from PIL import Image
from constants import (
    HEIGHT, WIDTH, N_STRIPES, DIMS_TOTAL
)
from color import rgb_to_hsv, rgb_to_ycbcr, extract_color_features
from texture import extract_texture_features, extract_texture_features_with_gabor


def to_float01(img_rgb_uint8):
    """
    Convierte imagen RGB uint8 a float32 en [0, 1].
    
    Args:
        img_rgb_uint8: array (H, W, 3) uint8 en RGB
    
    Returns:
        img_float: array (H, W, 3) float32 en [0, 1]
    """
    img_float = img_rgb_uint8.astype(np.float32) / 255.0
    return np.clip(img_float, 0.0, 1.0)


def resize_image(img, target_height, target_width):
    """
    Redimensiona imagen manteniendo aspecto (alto=128, ancho=64).
    
    Args:
        img: array (H, W, 3) en [0, 1]
        target_height: altura objetivo
        target_width: ancho objetivo
    
    Returns:
        resized: array (target_height, target_width, 3) en [0, 1]
    """
    # Verificar que la imagen tenga al menos 6 píxeles de alto
    if img.shape[0] < 6:
        # Redimensionar proporcionalmente hasta tener al menos 6 píxeles
        scale = 6.0 / img.shape[0]
        new_h = max(6, int(img.shape[0] * scale))
        new_w = max(1, int(img.shape[1] * scale))
        img = np.array(Image.fromarray((img * 255).astype(np.uint8)).resize(
            (new_w, new_h), Image.Resampling.LANCZOS
        )).astype(np.float32) / 255.0
    
    # Redimensionar a tamaño objetivo
    img_uint8 = (img * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_uint8)
    img_resized = img_pil.resize((target_width, target_height), Image.Resampling.LANCZOS)
    img_resized = np.array(img_resized).astype(np.float32) / 255.0
    
    return img_resized


def split_into_stripes(img_rgb, img_hsv, img_ycbcr, img_Y, n_stripes):
    """
    Divide la imagen en n_stripes franjas horizontales iguales.
    
    Args:
        img_rgb: array (H, W, 3) RGB
        img_hsv: array (H, W, 3) HSV
        img_ycbcr: array (H, W, 3) YCbCr
        img_Y: array (H, W) luminancia Y
        n_stripes: número de franjas
    
    Returns:
        stripes: lista de dicts, cada uno con 'rgb', 'hsv', 'ycbcr', 'Y'
    """
    H = img_rgb.shape[0]
    stripe_height = H // n_stripes
    
    stripes = []
    for i in range(n_stripes):
        start_h = i * stripe_height
        if i == n_stripes - 1:
            # Última franja absorbe el sobrante
            end_h = H
        else:
            end_h = (i + 1) * stripe_height
        
        stripe = {
            'rgb': img_rgb[start_h:end_h, :, :],
            'hsv': img_hsv[start_h:end_h, :, :],
            'ycbcr': img_ycbcr[start_h:end_h, :, :],
            'Y': img_Y[start_h:end_h, :]
        }
        stripes.append(stripe)
    
    return stripes


def extract_liu2012_features(img: np.ndarray) -> np.ndarray:
    """
    Extrae features Liu 2012 de una imagen RGB.
    
    Args:
        img: array RGB uint8 de forma (H, W, 3)
    
    Returns:
        features: vector float32 de 2784 dimensiones
    """
    # Validación de entrada
    if img.dtype != np.uint8:
        raise ValueError(f"Imagen debe ser uint8, recibido: {img.dtype}")
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError(f"Imagen debe ser (H, W, 3), recibido: {img.shape}")
    
    # Preprocesamiento
    img_float = to_float01(img)
    
    # Resize a (128, 64)
    img_resized = resize_image(img_float, HEIGHT, WIDTH)
    
    # Conversiones de color
    img_hsv = rgb_to_hsv(img_resized)
    img_ycbcr = rgb_to_ycbcr(img_resized)
    img_Y = img_ycbcr[..., 0]  # Luminancia
    
    # Dividir en 6 franjas
    stripes = split_into_stripes(img_resized, img_hsv, img_ycbcr, img_Y, N_STRIPES)
    
    # Extraer features por franja
    all_features = []
    for stripe in stripes:
        # Color: 128 dims
        color_vec = extract_color_features(stripe['rgb'], stripe['hsv'], stripe['ycbcr'])
        
        # Textura: 336 dims
        texture_vec = extract_texture_features(stripe['Y'])
        
        # Concatenar: 464 dims por franja
        stripe_features = np.concatenate([color_vec, texture_vec], axis=0)
        all_features.append(stripe_features)
    
    # Concatenar todas las franjas: 2784 dims total
    features = np.concatenate(all_features, axis=0)
    
    # Validaciones finales
    assert features.shape == (DIMS_TOTAL,), f"Features deben tener {DIMS_TOTAL} dims, recibido: {features.shape}"
    assert features.dtype == np.float32, f"Features deben ser float32, recibido: {features.dtype}"
    assert not np.any(np.isnan(features)), "Features no pueden contener NaN"
    assert not np.any(np.isinf(features)), "Features no pueden contener Inf"
    
    return features


def extract_liu2012_features_with_gating_info(img: np.ndarray) -> dict:
    """
    Extrae features Liu 2012 de una imagen RGB con información adicional para gating.
    
    Args:
        img: array RGB uint8 de forma (H, W, 3)
    
    Returns:
        dict con:
            'features': vector float32 de 2784 dimensiones (concatenado)
            'color_features': vector float32 de 768 dimensiones (6 stripes × 128)
            'texture_features': vector float32 de 2016 dimensiones (6 stripes × 336)
            'gabor_responses': lista de listas, cada una con 8 respuestas Gabor por stripe
    """
    # Validación de entrada
    if img.dtype != np.uint8:
        raise ValueError(f"Imagen debe ser uint8, recibido: {img.dtype}")
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError(f"Imagen debe ser (H, W, 3), recibido: {img.shape}")
    
    # Preprocesamiento
    img_float = to_float01(img)
    
    # Resize a (128, 64)
    img_resized = resize_image(img_float, HEIGHT, WIDTH)
    
    # Conversiones de color
    img_hsv = rgb_to_hsv(img_resized)
    img_ycbcr = rgb_to_ycbcr(img_resized)
    img_Y = img_ycbcr[..., 0]  # Luminancia
    
    # Dividir en 6 franjas
    stripes = split_into_stripes(img_resized, img_hsv, img_ycbcr, img_Y, N_STRIPES)
    
    # Extraer features por franja
    all_features = []
    all_color_features = []
    all_texture_features = []
    all_gabor_responses = []
    
    for stripe in stripes:
        # Color: 128 dims
        color_vec = extract_color_features(stripe['rgb'], stripe['hsv'], stripe['ycbcr'])
        
        # Textura: 336 dims + respuestas Gabor
        texture_vec, gabor_responses = extract_texture_features_with_gabor(stripe['Y'])
        
        # Concatenar: 464 dims por franja
        stripe_features = np.concatenate([color_vec, texture_vec], axis=0)
        all_features.append(stripe_features)
        
        # Guardar por separado
        all_color_features.append(color_vec)
        all_texture_features.append(texture_vec)
        all_gabor_responses.append(gabor_responses)
    
    # Concatenar todas las franjas
    features = np.concatenate(all_features, axis=0)
    color_features = np.concatenate(all_color_features, axis=0)
    texture_features = np.concatenate(all_texture_features, axis=0)
    
    # Validaciones finales
    assert features.shape == (DIMS_TOTAL,), f"Features deben tener {DIMS_TOTAL} dims, recibido: {features.shape}"
    assert features.dtype == np.float32, f"Features deben ser float32, recibido: {features.dtype}"
    assert not np.any(np.isnan(features)), "Features no pueden contener NaN"
    assert not np.any(np.isinf(features)), "Features no pueden contener Inf"
    
    return {
        'features': features,
        'color_features': color_features,
        'texture_features': texture_features,
        'gabor_responses': all_gabor_responses  # Lista de listas: [stripe][filter]
    }



