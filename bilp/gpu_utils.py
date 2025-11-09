"""
GPU utilities for BILP feature extraction using CuPy
"""

import numpy as np
from typing import Optional, Union, Tuple

# Try to import CuPy
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


def get_device(use_gpu: bool = True) -> Tuple[bool, Optional]:
    """
    Get the appropriate device (CPU or GPU) based on availability.
    
    Args:
        use_gpu: Whether to try to use GPU
        
    Returns:
        (is_gpu, device): Tuple indicating if GPU is used and the device object
    """
    if not use_gpu or not CUPY_AVAILABLE:
        return False, None
    
    try:
        # Check if GPU is available
        cp.cuda.Device(0).use()
        return True, cp
    except:
        return False, None


def to_gpu(array: np.ndarray, device) -> Union[np.ndarray, 'cp.ndarray']:
    """
    Transfer array to GPU if device is available.
    
    Args:
        array: NumPy array
        device: CuPy device or None
        
    Returns:
        Array on GPU (CuPy) or CPU (NumPy)
    """
    if device is not None:
        return cp.asarray(array)
    return array


def to_cpu(array: Union[np.ndarray, 'cp.ndarray']) -> np.ndarray:
    """
    Transfer array from GPU to CPU.
    
    Args:
        array: CuPy or NumPy array
        
    Returns:
        NumPy array on CPU
    """
    if CUPY_AVAILABLE and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)


def get_array_module(device) -> Union:
    """
    Get the appropriate array module (CuPy or NumPy).
    
    Args:
        device: CuPy device or None
        
    Returns:
        cp if GPU, np if CPU
    """
    if device is not None:
        return cp
    return np


def cdist_gpu(
    query: np.ndarray,
    gallery: np.ndarray,
    metric: str = 'cityblock',
    device = None
) -> np.ndarray:
    """
    Compute distance matrix on GPU using CuPy.
    
    Args:
        query: Query features (n_query, n_features)
        gallery: Gallery features (n_gallery, n_features)
        metric: Distance metric ('cityblock', 'euclidean', 'cosine')
        device: CuPy device or None
        
    Returns:
        Distance matrix (n_query, n_gallery) on CPU
    """
    if device is None:
        # Fallback to scipy on CPU
        from scipy.spatial.distance import cdist
        return cdist(query, gallery, metric=metric)
    
    xp = get_array_module(device)
    
    # Transfer to GPU
    query_gpu = to_gpu(query, device)
    gallery_gpu = to_gpu(gallery, device)
    
    if metric == 'cityblock':
        # L1 distance: sum(|x - y|)
        # Expand dimensions for broadcasting: (n_query, 1, n_features) - (1, n_gallery, n_features)
        diff = query_gpu[:, None, :] - gallery_gpu[None, :, :]
        dist = xp.sum(xp.abs(diff), axis=2)
    elif metric == 'euclidean':
        # L2 distance: sqrt(sum((x - y)^2))
        diff = query_gpu[:, None, :] - gallery_gpu[None, :, :]
        dist = xp.sqrt(xp.sum(diff ** 2, axis=2))
    elif metric == 'cosine':
        # Cosine distance: 1 - cosine_similarity
        query_norm = xp.linalg.norm(query_gpu, axis=1, keepdims=True)
        gallery_norm = xp.linalg.norm(gallery_gpu, axis=1, keepdims=True)
        dot_product = query_gpu @ gallery_gpu.T
        dist = 1 - dot_product / (query_norm * gallery_norm.T)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    
    # Transfer back to CPU
    return to_cpu(dist)


def convolve_gpu(
    image: np.ndarray,
    kernel: np.ndarray,
    device = None
) -> np.ndarray:
    """
    Convolve image with kernel on GPU.
    
    Args:
        image: Input image (H, W)
        kernel: Convolution kernel (K, K)
        device: CuPy device or None
        
    Returns:
        Convolved image (H, W) on CPU
    """
    if device is None:
        # Fallback to scipy on CPU
        from scipy import ndimage
        return ndimage.convolve(image, kernel, mode='constant')
    
    xp = get_array_module(device)
    
    # Transfer to GPU
    image_gpu = to_gpu(image, device)
    kernel_gpu = to_gpu(kernel, device)
    
    # CuPy doesn't have direct convolution, use FFT-based convolution
    # Pad kernel to match image size
    h, w = image_gpu.shape
    k_h, k_w = kernel_gpu.shape
    
    # Pad kernel
    pad_h = (h - k_h) // 2
    pad_w = (w - k_w) // 2
    kernel_padded = xp.pad(
        kernel_gpu,
        ((pad_h, h - k_h - pad_h), (pad_w, w - k_w - pad_w)),
        mode='constant'
    )
    
    # FFT-based convolution
    image_fft = xp.fft.fft2(image_gpu)
    kernel_fft = xp.fft.fft2(kernel_padded)
    result_fft = image_fft * kernel_fft
    result = xp.real(xp.fft.ifft2(result_fft))
    
    # Transfer back to CPU
    return to_cpu(result)


def batch_convolve_gpu(
    images: np.ndarray,
    kernels: list,
    device = None
) -> np.ndarray:
    """
    Apply multiple kernels to multiple images in batch on GPU.
    
    Args:
        images: Batch of images (n_images, H, W)
        kernels: List of kernels, each (K, K)
        device: CuPy device or None
        
    Returns:
        Convolved images (n_images, n_kernels, H, W) on CPU
    """
    if device is None:
        # Fallback to CPU
        from scipy import ndimage
        results = []
        for img in images:
            img_results = []
            for kernel in kernels:
                result = ndimage.convolve(img, kernel, mode='constant')
                img_results.append(result)
            results.append(img_results)
        return np.array(results)
    
    xp = get_array_module(device)
    
    # Transfer to GPU
    images_gpu = to_gpu(images, device)
    n_images, h, w = images_gpu.shape
    n_kernels = len(kernels)
    
    results = []
    
    for img in images_gpu:
        img_results = []
        for kernel in kernels:
            kernel_gpu = to_gpu(kernel, device)
            
            # Pad kernel
            k_h, k_w = kernel_gpu.shape
            pad_h = (h - k_h) // 2
            pad_w = (w - k_w) // 2
            kernel_padded = xp.pad(
                kernel_gpu,
                ((pad_h, h - k_h - pad_h), (pad_w, w - k_w - pad_w)),
                mode='constant'
            )
            
            # FFT-based convolution
            img_fft = xp.fft.fft2(img)
            kernel_fft = xp.fft.fft2(kernel_padded)
            result_fft = img_fft * kernel_fft
            result = xp.real(xp.fft.ifft2(result_fft))
            img_results.append(result)
        
        results.append(img_results)
    
    # Transfer back to CPU
    results_array = xp.array(results)
    return to_cpu(results_array)

