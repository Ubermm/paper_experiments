"""
utils/gpu_utils.py
==================
GPU-accelerated versions of computationally intensive operations.

Only used when:
1. PyTorch with CUDA is available
2. Dataset size > 5000 samples (break-even point)
3. Falls back to CPU if GPU fails

Key optimizations:
- Covariance computation: 1.5x+ speedup for n>=5k
- Kurtosis tensor: 1.5x+ speedup for n>=5k
- Distance computations: Marginal benefit
"""

import numpy as np
from typing import Optional, Union
import warnings

# GPU imports (optional)
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    DEVICE = torch.device('cuda' if GPU_AVAILABLE else 'cpu')
except ImportError:
    torch = None
    GPU_AVAILABLE = False
    DEVICE = None

# Configuration
GPU_THRESHOLD = 5000  # Minimum samples for GPU to be beneficial


def should_use_gpu(X: np.ndarray) -> bool:
    """Check if GPU should be used for this dataset size."""
    return GPU_AVAILABLE and len(X) >= GPU_THRESHOLD


def to_torch(X: np.ndarray, device: Optional[str] = None) -> 'torch.Tensor':
    """Convert numpy array to torch tensor with proper dtype."""
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU not available")

    device = device or DEVICE
    return torch.tensor(X, dtype=torch.float32, device=device)


def compute_covariance_gpu(X: np.ndarray) -> np.ndarray:
    """GPU-accelerated covariance computation."""
    if not should_use_gpu(X):
        # Fallback to CPU
        from .coreset_methods import compute_covariance
        return compute_covariance(X)

    try:
        X_gpu = to_torch(X)

        # Compute mean and center
        mu = X_gpu.mean(dim=0)
        centered = X_gpu - mu

        # Covariance matrix
        cov = (centered.T @ centered) / len(X_gpu)

        return cov.cpu().numpy()

    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        warnings.warn(f"GPU computation failed ({e}), falling back to CPU")
        from .coreset_methods import compute_covariance
        return compute_covariance(X)


def compute_kurtosis_tensor_gpu(X: np.ndarray) -> np.ndarray:
    """GPU-accelerated kurtosis tensor computation."""
    if not should_use_gpu(X):
        # Fallback to CPU
        from .coreset_methods import compute_kurtosis_tensor
        return compute_kurtosis_tensor(X)

    try:
        X_gpu = to_torch(X)
        N = len(X_gpu)

        # Center the data
        mu = X_gpu.mean(dim=0)
        centered = X_gpu - mu

        # Raw 4th moment matrix: E[xi²xj²]
        sq = centered ** 2
        raw_4th = (sq.T @ sq) / N

        # E[xi²]E[xj²] term
        var = sq.mean(dim=0)
        var_outer = torch.outer(var, var)

        # 2E[xixj]² term (covariance squared)
        cov = (centered.T @ centered) / N
        cov_squared = cov ** 2

        # Correction: var_outer + 2*cov_squared
        correction = var_outer + 2 * cov_squared

        # No additional diagonal correction needed - the formula is already correct
        # For diagonal: κ₄[i,i] = E[xi⁴] - E[xi²]² - 2E[xi²]² = E[xi⁴] - 3E[xi²]²
        # This is handled by var_outer + 2*cov_squared since cov[i,i] = var[i]

        kurt = raw_4th - correction

        return kurt.cpu().numpy()

    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        warnings.warn(f"GPU computation failed ({e}), falling back to CPU")
        from .coreset_methods import compute_kurtosis_tensor
        return compute_kurtosis_tensor(X)


def compute_distances_gpu(X: np.ndarray, center: np.ndarray) -> np.ndarray:
    """GPU-accelerated distance computation."""
    if not should_use_gpu(X):
        # CPU fallback
        return np.linalg.norm(X - center, axis=1)

    try:
        X_gpu = to_torch(X)
        center_gpu = to_torch(center.reshape(1, -1))

        dists = torch.norm(X_gpu - center_gpu, dim=1)

        return dists.cpu().numpy()

    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        warnings.warn(f"GPU computation failed ({e}), falling back to CPU")
        return np.linalg.norm(X - center, axis=1)


def mahalanobis_distances_gpu(X: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """GPU-accelerated Mahalanobis distances."""
    if not should_use_gpu(X):
        # CPU fallback - avoid circular import
        try:
            cov_inv = np.linalg.inv(cov + 1e-6 * np.eye(len(cov)))
        except:
            cov_inv = np.eye(len(cov))
        centered = X - mu
        return np.sqrt(np.sum(centered @ cov_inv * centered, axis=1))

    try:
        X_gpu = to_torch(X)
        mu_gpu = to_torch(mu)
        cov_gpu = to_torch(cov)

        # Compute inverse (with regularization)
        eye = torch.eye(len(cov_gpu), device=DEVICE) * 1e-6
        try:
            cov_inv = torch.inverse(cov_gpu + eye)
        except:
            cov_inv = torch.eye(len(cov_gpu), device=DEVICE)

        # Centered data
        centered = X_gpu - mu_gpu

        # Mahalanobis distance: sqrt((x-μ)ᵀ Σ⁻¹ (x-μ))
        mahal_sq = torch.sum(centered @ cov_inv * centered, dim=1)
        mahal = torch.sqrt(torch.clamp(mahal_sq, min=0))  # Avoid numerical issues

        return mahal.cpu().numpy()

    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        warnings.warn(f"GPU computation failed ({e}), falling back to CPU")
        # CPU fallback - avoid circular import
        try:
            cov_inv = np.linalg.inv(cov + 1e-6 * np.eye(len(cov)))
        except:
            cov_inv = np.eye(len(cov))
        centered = X - mu
        return np.sqrt(np.sum(centered @ cov_inv * centered, axis=1))


def print_gpu_info():
    """Print GPU availability and device info."""
    if GPU_AVAILABLE:
        print(f"GPU Acceleration: Available")
        print(f"  Device: {torch.cuda.get_device_name()}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  Threshold: {GPU_THRESHOLD:,} samples")
    else:
        print("GPU Acceleration: Not available")
        if torch is None:
            print("  Reason: PyTorch not installed")
        else:
            print("  Reason: CUDA not available")


# Wrapper functions that automatically choose CPU/GPU
def compute_covariance_auto(X: np.ndarray) -> np.ndarray:
    """Automatically choose CPU or GPU for covariance computation."""
    return compute_covariance_gpu(X)


def compute_kurtosis_tensor_auto(X: np.ndarray) -> np.ndarray:
    """Automatically choose CPU or GPU for kurtosis tensor computation."""
    return compute_kurtosis_tensor_gpu(X)


def mahalanobis_distances_auto(X: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Automatically choose CPU or GPU for Mahalanobis distances."""
    return mahalanobis_distances_gpu(X, mu, cov)


def compute_correlation_matrix_gpu(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    GPU-accelerated correlation matrix computation.
    Useful for ICA source separation quality assessment.
    """
    if not should_use_gpu(A):
        return np.abs(np.corrcoef(A.T, B.T))

    try:
        A_gpu = to_torch(A)
        B_gpu = to_torch(B)

        # Standardize columns
        A_std = (A_gpu - A_gpu.mean(dim=0)) / (A_gpu.std(dim=0) + 1e-10)
        B_std = (B_gpu - B_gpu.mean(dim=0)) / (B_gpu.std(dim=0) + 1e-10)

        # Correlation matrix
        n = len(A_gpu)
        corr = torch.abs((A_std.T @ B_std) / n)

        return corr.cpu().numpy()

    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        warnings.warn(f"GPU computation failed ({e}), falling back to CPU")
        return np.abs(np.corrcoef(A.T, B.T))


def compute_kurtosis_gpu(X: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    GPU-accelerated kurtosis computation.
    """
    if not should_use_gpu(X):
        from scipy.stats import kurtosis
        return kurtosis(X, axis=axis, fisher=True)

    try:
        X_gpu = to_torch(X)

        # Center the data
        if axis == 0:
            mu = X_gpu.mean(dim=0)
            centered = X_gpu - mu
            var = (centered ** 2).mean(dim=0)
            kurt = ((centered ** 4).mean(dim=0) / (var ** 2)) - 3
        else:
            mu = X_gpu.mean(dim=1, keepdim=True)
            centered = X_gpu - mu
            var = (centered ** 2).mean(dim=1, keepdim=True)
            kurt = ((centered ** 4).mean(dim=1) / (var.squeeze() ** 2)) - 3

        return kurt.cpu().numpy()

    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        warnings.warn(f"GPU computation failed ({e}), falling back to CPU")
        from scipy.stats import kurtosis
        return kurtosis(X, axis=axis, fisher=True)