"""
GPU utilities for BILP feature extraction using CuPy
"""

import numpy as np
from typing import Optional, Union, Tuple, Any

# Try to import CuPy
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


def get_device(use_gpu: bool = True) -> Tuple[bool, Optional[Any]]:
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


def get_array_module(device) -> Any:
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
    device = None,
    batch_size: int = 50
) -> np.ndarray:
    """
    Compute distance matrix on GPU using CuPy.
    Processes in batches to avoid memory issues with large matrices.
    
    Args:
        query: Query features (n_query, n_features)
        gallery: Gallery features (n_gallery, n_features)
        metric: Distance metric ('cityblock', 'euclidean', 'cosine')
        device: CuPy device or None
        batch_size: Number of queries to process at once (to avoid OOM)
        
    Returns:
        Distance matrix (n_query, n_gallery) on CPU
    """
    if device is None:
        # Fallback to scipy on CPU
        from scipy.spatial.distance import cdist
        return cdist(query, gallery, metric=metric)
    
    xp = get_array_module(device)
    
    n_query, n_features = query.shape
    n_gallery = gallery.shape[0]
    
    # Pre-allocate result matrix
    dist_matrix = np.zeros((n_query, n_gallery), dtype=np.float32)
    
    # For cosine, we can keep gallery on GPU and process in batches
    # For cityblock/euclidean, we need to process both query and gallery in batches
    if metric == 'cosine':
        # Cosine is more memory efficient - transfer gallery once
        gallery_gpu = to_gpu(gallery, device)
        gallery_norm = xp.linalg.norm(gallery_gpu, axis=1, keepdims=True)
        
        for i in range(0, n_query, batch_size):
            end_idx = min(i + batch_size, n_query)
            query_batch = query[i:end_idx]
            query_batch_gpu = to_gpu(query_batch, device)
            
            query_batch_norm = xp.linalg.norm(query_batch_gpu, axis=1, keepdims=True)
            dot_product = query_batch_gpu @ gallery_gpu.T
            dist_batch = 1 - dot_product / (query_batch_norm * gallery_norm.T)
            
            dist_matrix[i:end_idx] = to_cpu(dist_batch)
            del query_batch_gpu, dist_batch
        
        del gallery_gpu, gallery_norm
    else:
        # For cityblock and euclidean, process gallery in batches too
        # to avoid creating huge intermediate arrays
        gallery_batch_size = min(5000, n_gallery)  # Process gallery in chunks
        
        for i in range(0, n_query, batch_size):
            query_end = min(i + batch_size, n_query)
            query_batch = query[i:query_end]
            query_batch_gpu = to_gpu(query_batch, device)
            
            for j in range(0, n_gallery, gallery_batch_size):
                gallery_end = min(j + gallery_batch_size, n_gallery)
                gallery_batch = gallery[j:gallery_end]
                gallery_batch_gpu = to_gpu(gallery_batch, device)
                
                if metric == 'cityblock':
                    # L1 distance: sum(|x - y|)
                    diff = query_batch_gpu[:, None, :] - gallery_batch_gpu[None, :, :]
                    dist_batch = xp.sum(xp.abs(diff), axis=2)
                elif metric == 'euclidean':
                    # L2 distance: sqrt(sum((x - y)^2))
                    diff = query_batch_gpu[:, None, :] - gallery_batch_gpu[None, :, :]
                    dist_batch = xp.sqrt(xp.sum(diff ** 2, axis=2))
                
                dist_matrix[i:query_end, j:gallery_end] = to_cpu(dist_batch)
                del gallery_batch_gpu, dist_batch, diff
            
            del query_batch_gpu
    
    return dist_matrix


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
    
    h, w = image_gpu.shape
    k_h, k_w = kernel_gpu.shape
    
    # If kernel is larger than image, pad image first
    # Do padding in NumPy to avoid CuPy JIT compilation issues with xp.pad()
    if k_h > h or k_w > w:
        # Transfer back to CPU for padding
        image_cpu = to_cpu(image_gpu)
        pad_h_before = max(0, (k_h - h + 1) // 2)
        pad_h_after = max(0, k_h - h - pad_h_before)
        pad_w_before = max(0, (k_w - w + 1) // 2)
        pad_w_after = max(0, k_w - w - pad_w_before)
        
        image_padded_cpu = np.pad(
            image_cpu,
            ((pad_h_before, pad_h_after), (pad_w_before, pad_w_after)),
            mode='constant'
        )
        image_padded = to_gpu(image_padded_cpu, device)
        h_padded, w_padded = image_padded.shape
    else:
        image_padded = image_gpu
        h_padded, w_padded = h, w
    
    # Pad kernel to match (padded) image size for FFT-based convolution
    # Do padding in NumPy to avoid CuPy JIT compilation issues
    kernel_cpu = to_cpu(kernel_gpu)
    pad_h = max(0, (h_padded - k_h) // 2)
    pad_w = max(0, (w_padded - k_w) // 2)
    pad_h_after = max(0, h_padded - k_h - pad_h)
    pad_w_after = max(0, w_padded - k_w - pad_w)
    
    kernel_padded_cpu = np.pad(
        kernel_cpu,
        ((pad_h, pad_h_after), (pad_w, pad_w_after)),
        mode='constant'
    )
    kernel_padded = to_gpu(kernel_padded_cpu, device)
    
    # FFT-based convolution
    image_fft = xp.fft.fft2(image_padded)
    kernel_fft = xp.fft.fft2(kernel_padded)
    result_fft = image_fft * kernel_fft
    result = xp.real(xp.fft.ifft2(result_fft))
    
    # Crop result back to original image size
    if k_h > h or k_w > w:
        # Extract center region matching original image size
        start_h = (result.shape[0] - h) // 2
        start_w = (result.shape[1] - w) // 2
        result = result[start_h:start_h+h, start_w:start_w+w]
    else:
        # Result should already match original size
        result = result[:h, :w]
    
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
            k_h, k_w = kernel_gpu.shape
            
            # Handle case where kernel is larger than image
            # Do padding in NumPy to avoid CuPy JIT compilation issues
            if k_h > h or k_w > w:
                # Transfer back to CPU for padding
                img_cpu = to_cpu(img)
                pad_h_before = max(0, (k_h - h + 1) // 2)
                pad_h_after = max(0, k_h - h - pad_h_before)
                pad_w_before = max(0, (k_w - w + 1) // 2)
                pad_w_after = max(0, k_w - w - pad_w_before)
                
                img_padded_cpu = np.pad(
                    img_cpu,
                    ((pad_h_before, pad_h_after), (pad_w_before, pad_w_after)),
                    mode='constant'
                )
                img_padded = to_gpu(img_padded_cpu, device)
                h_padded, w_padded = img_padded.shape
            else:
                img_padded = img
                h_padded, w_padded = h, w
            
            # Pad kernel to match (padded) image size
            # Do padding in NumPy to avoid CuPy JIT compilation issues
            kernel_cpu = to_cpu(kernel_gpu)
            pad_h = max(0, (h_padded - k_h) // 2)
            pad_w = max(0, (w_padded - k_w) // 2)
            pad_h_after = max(0, h_padded - k_h - pad_h)
            pad_w_after = max(0, w_padded - k_w - pad_w)
            
            kernel_padded_cpu = np.pad(
                kernel_cpu,
                ((pad_h, pad_h_after), (pad_w, pad_w_after)),
                mode='constant'
            )
            kernel_padded = to_gpu(kernel_padded_cpu, device)
            
            # FFT-based convolution
            img_fft = xp.fft.fft2(img_padded)
            kernel_fft = xp.fft.fft2(kernel_padded)
            result_fft = img_fft * kernel_fft
            result = xp.real(xp.fft.ifft2(result_fft))
            
            # Crop result back to original image size
            if k_h > h or k_w > w:
                start_h = (result.shape[0] - h) // 2
                start_w = (result.shape[1] - w) // 2
                result = result[start_h:start_h+h, start_w:start_w+w]
            else:
                result = result[:h, :w]
            
            img_results.append(result)
        
        results.append(img_results)
    
    # Transfer back to CPU
    results_array = xp.array(results)
    return to_cpu(results_array)

