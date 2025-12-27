"""
utils/coreset_methods.py
========================
Shared coreset selection implementations for all experiments.

All methods fixed with proper index handling and cumulant-based kurtosis.
GPU acceleration automatically used for large datasets (n >= 5000).
"""

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from typing import Tuple, Optional, List
import time

# GPU acceleration (auto-fallback to CPU)
try:
    from .gpu_utils import (compute_covariance_auto as compute_covariance_gpu,
                            compute_kurtosis_tensor_auto as compute_kurtosis_tensor_gpu,
                            should_use_gpu, print_gpu_info)
    GPU_UTILS_AVAILABLE = True
except ImportError:
    GPU_UTILS_AVAILABLE = False


# =============================================================================
# EFFICIENT COVARIANCE UPDATE UTILITIES
# =============================================================================

def update_covariance_incremental(X_subset: np.ndarray, new_point: np.ndarray,
                                current_cov: np.ndarray, current_mean: np.ndarray) -> tuple:
    """
    Efficiently update covariance matrix when adding one point using rank-1 updates.
    Returns (new_cov, new_mean)
    """
    n = len(X_subset)
    new_n = n + 1

    # Update mean
    new_mean = (n * current_mean + new_point) / new_n

    # Centered new point
    delta_old = new_point - current_mean
    delta_new = new_point - new_mean

    # Rank-1 update to covariance
    # C_new = (n-1)/(n) * C_old + (1/n) * delta_old * delta_old^T - (1/n) * (delta_new * old_mean_diff^T)
    new_cov = ((n - 1) / new_n) * current_cov + (1 / new_n) * np.outer(delta_old, delta_old)

    # Correction for mean shift
    mean_shift = new_mean - current_mean
    new_cov -= (n / new_n) * np.outer(mean_shift, mean_shift)

    return new_cov, new_mean


def compute_covariance_error_efficient(X_subset: np.ndarray, candidate: np.ndarray,
                                     current_cov: np.ndarray, current_mean: np.ndarray,
                                     target_cov: np.ndarray) -> float:
    """Efficiently compute covariance error when adding candidate point."""
    new_cov, _ = update_covariance_incremental(X_subset, candidate, current_cov, current_mean)
    return np.sum((new_cov - target_cov) ** 2)


# =============================================================================
# MOMENT COMPUTATION UTILITIES
# =============================================================================

def compute_covariance(X: np.ndarray) -> np.ndarray:
    """Compute sample covariance matrix with optional GPU acceleration."""
    if GPU_UTILS_AVAILABLE and should_use_gpu(X):
        return compute_covariance_gpu(X)
    else:
        # CPU fallback
        mu = X.mean(axis=0)
        centered = X - mu
        return (centered.T @ centered) / len(X)


def compute_kurtosis_cumulant(X: np.ndarray) -> np.ndarray:
    """
    Compute 4th-order cumulant (excess kurtosis) per dimension.
    κ₄ = E[(X-μ)⁴] - 3·(E[(X-μ)²])²
    """
    mu = X.mean(axis=0)
    centered = X - mu
    m2 = (centered ** 2).mean(axis=0)
    m4 = (centered ** 4).mean(axis=0)
    return m4 - 3 * (m2 ** 2)


def compute_kurtosis_tensor(X: np.ndarray) -> np.ndarray:
    """
    Compute 4th-order cumulant tensor with optional GPU acceleration.
    For ICA: κ₄[i,j] = E[xᵢ²xⱼ²] - E[xᵢ²]E[xⱼ²] - 2E[xᵢxⱼ]²

    Implementation uses the correct cumulant formula:
    κ₄[i,j] = E[xᵢ²xⱼ²] - E[xᵢ²]E[xⱼ²] - 2(Cov[xᵢ,xⱼ])²
    """
    if GPU_UTILS_AVAILABLE and should_use_gpu(X):
        return compute_kurtosis_tensor_gpu(X)
    else:
        # CPU fallback
        mu = X.mean(axis=0)
        centered = X - mu
        N = len(X)

        # Raw 4th moment matrix: E[xᵢ²xⱼ²]
        sq = centered ** 2
        raw_4th = (sq.T @ sq) / N

        # E[xᵢ²]E[xⱼ²] term
        var = (centered ** 2).mean(axis=0)
        var_outer = np.outer(var, var)

        # 2E[xᵢxⱼ]² term (covariance squared)
        cov = (centered.T @ centered) / N
        cov_squared = cov ** 2

        # For diagonal terms: subtract 3*var²  (excess kurtosis)
        # For off-diagonal terms: subtract var_i*var_j + 2*cov²
        correction = var_outer + 2 * cov_squared

        # No additional diagonal correction needed - the formula is already correct
        # For diagonal: κ₄[i,i] = E[xi⁴] - E[xi²]² - 2E[xi²]² = E[xi⁴] - 3E[xi²]²
        # This is handled by var_outer + 2*cov_squared since cov[i,i] = var[i]

        return raw_4th - correction


def compute_skewness_tensor(X: np.ndarray) -> np.ndarray:
    """Compute 3rd-order moment tensor (diagonal approximation)."""
    mu = X.mean(axis=0)
    centered = X - mu
    N = len(X)
    return ((centered ** 2).T @ centered) / N


# =============================================================================
# CORESET SELECTION METHODS
# =============================================================================

def random_coreset(X: np.ndarray, k: int, seed: int = None) -> np.ndarray:
    """Random sampling baseline."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.choice(len(X), k, replace=False)


def kmeans_pp_coreset(X: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    """K-means++ initialization as coreset."""
    km = KMeans(n_clusters=k, init='k-means++', n_init=1, max_iter=1, random_state=seed)
    km.fit(X)
    
    indices = []
    for c in km.cluster_centers_:
        dists = np.linalg.norm(X - c, axis=1)
        idx = np.argmin(dists)
        while idx in indices:
            dists[idx] = np.inf
            idx = np.argmin(dists)
        indices.append(idx)
    return np.array(indices)


def herding_coreset(X: np.ndarray, k: int) -> np.ndarray:
    """Kernel herding for mean matching."""
    mu = X.mean(axis=0)
    selected = []
    sum_sel = np.zeros(X.shape[1])
    
    for t in range(k):
        if t == 0:
            scores = X @ mu
        else:
            scores = X @ (mu - sum_sel / t)
        
        # Mask already selected
        for idx in selected:
            scores[idx] = -np.inf
        
        best = int(np.argmax(scores))
        selected.append(best)
        sum_sel += X[best]
    
    return np.array(selected)


def greedy_covariance_coreset(X: np.ndarray, k: int) -> np.ndarray:
    """Greedy covariance matching with O(N*k*d^2) complexity using rank-1 updates."""
    N, d = X.shape
    mu = X.mean(axis=0)
    target_cov = compute_covariance(X)

    # Start with point closest to mean
    start_idx = int(np.argmin(np.linalg.norm(X - mu, axis=1)))
    selected = [start_idx]
    remaining = set(range(N)) - {start_idx}

    # Initialize current covariance (single point has zero covariance)
    current_mean = X[start_idx].copy()
    current_cov = np.zeros((d, d))

    for _ in range(k - 1):
        best_idx = None
        best_err = np.inf

        # Current subset for rank-1 updates
        X_current = X[selected]

        for idx in remaining:
            candidate = X[idx]
            err = compute_covariance_error_efficient(X_current, candidate,
                                                   current_cov, current_mean, target_cov)

            if err < best_err:
                best_err = err
                best_idx = idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

            # Update current covariance and mean for next iteration
            current_cov, current_mean = update_covariance_incremental(
                X[selected[:-1]], X[best_idx], current_cov, current_mean)

    return np.array(selected)


def hmp_coreset(X: np.ndarray, k: int, skeleton_ratio: float = 0.5,
                cov_weight: float = 0.3, kurt_weight: float = 0.7) -> np.ndarray:
    """
    Hierarchical Moment-Preserving coreset selection.
    
    Phase 1: Build covariance skeleton (first skeleton_ratio * k points)
    Phase 2: Refine with kurtosis cumulant matching
    
    Parameters:
    -----------
    X : array of shape (N, d)
    k : coreset size
    skeleton_ratio : fraction of k for covariance skeleton
    cov_weight : weight for covariance error in phase 2
    kurt_weight : weight for kurtosis error in phase 2
    """
    N, d = X.shape
    m = max(d + 1, int(k * skeleton_ratio))
    m = min(m, k)
    
    mu = X.mean(axis=0)
    target_cov = compute_covariance(X)
    target_kurt = compute_kurtosis_tensor(X)
    
    # Normalization constants
    cov_norm = np.linalg.norm(target_cov, 'fro') + 1e-10
    kurt_norm = np.linalg.norm(target_kurt, 'fro') + 1e-10
    
    # Start with point closest to mean
    start_idx = int(np.argmin(np.linalg.norm(X - mu, axis=1)))
    selected = [start_idx]
    remaining = set(range(N)) - {start_idx}
    
    # Initialize current covariance tracking for efficient updates
    current_mean = X[start_idx].copy()
    current_cov = np.zeros((d, d))

    # Phase 1: Covariance skeleton (optimized with rank-1 updates)
    for _ in range(m - 1):
        best_idx = None
        best_err = np.inf

        X_current = X[selected]
        for idx in remaining:
            candidate = X[idx]
            err = compute_covariance_error_efficient(X_current, candidate,
                                                   current_cov, current_mean, target_cov)

            if err < best_err:
                best_err = err
                best_idx = idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

            # Update current covariance and mean for next iteration
            current_cov, current_mean = update_covariance_incremental(
                X[selected[:-1]], X[best_idx], current_cov, current_mean)
    
    # Phase 2: Kurtosis refinement
    for _ in range(k - m):
        best_idx = None
        best_err = np.inf
        
        for idx in remaining:
            test_indices = selected + [idx]
            X_test = X[test_indices]
            
            cov_test = compute_covariance(X_test)
            kurt_test = compute_kurtosis_tensor(X_test)
            
            cov_err = np.sum((cov_test - target_cov) ** 2) / (cov_norm ** 2)
            kurt_err = np.sum((kurt_test - target_kurt) ** 2) / (kurt_norm ** 2)
            
            err = cov_weight * cov_err + kurt_weight * kurt_err
            
            if err < best_err:
                best_err = err
                best_idx = idx
        
        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)
    
    return np.array(selected[:k])


def hmp_kurtosis_heavy(X: np.ndarray, k: int, skeleton_ratio: float = 0.3) -> np.ndarray:
    """HMP variant with heavier kurtosis weighting for signal processing."""
    return hmp_coreset(X, k, skeleton_ratio=skeleton_ratio, 
                       cov_weight=0.2, kurt_weight=0.8)


def stratified_coreset(X: np.ndarray, y: np.ndarray, k: int, 
                       seed: int = None) -> np.ndarray:
    """Class-balanced random sampling."""
    if seed is not None:
        np.random.seed(seed)
    
    classes = np.unique(y)
    k_per_class = k // len(classes)
    
    indices = []
    for c in classes:
        class_mask = y == c
        class_indices = np.where(class_mask)[0]
        n_select = min(k_per_class, len(class_indices))
        selected = np.random.choice(class_indices, n_select, replace=False)
        indices.extend(selected)
    
    # Fill remaining slots
    remaining = k - len(indices)
    if remaining > 0:
        all_indices = set(range(len(X))) - set(indices)
        extra = np.random.choice(list(all_indices), remaining, replace=False)
        indices.extend(extra)
    
    return np.array(indices[:k])


def stratified_hmp_coreset(X: np.ndarray, y: np.ndarray, k: int,
                           skeleton_ratio: float = 0.5) -> np.ndarray:
    """Class-balanced HMP: apply HMP within each class."""
    classes = np.unique(y)
    k_per_class = k // len(classes)
    
    indices = []
    for c in classes:
        class_mask = y == c
        X_class = X[class_mask]
        class_indices = np.where(class_mask)[0]
        
        n_select = min(k_per_class, len(X_class))
        if n_select > 0:
            local_indices = hmp_coreset(X_class, n_select, skeleton_ratio)
            indices.extend(class_indices[local_indices])
    
    return np.array(indices[:k])


# =============================================================================
# TIMING WRAPPER
# =============================================================================

def timed_selection(method_fn, X: np.ndarray, k: int, **kwargs) -> Tuple[np.ndarray, float]:
    """Run coreset selection with timing."""
    start = time.time()
    indices = method_fn(X, k, **kwargs)
    elapsed = time.time() - start
    return indices, elapsed


# =============================================================================
# METHOD REGISTRY
# =============================================================================

METHODS = {
    'Random': random_coreset,
    'K-means++': kmeans_pp_coreset,
    'Herding': herding_coreset,
    'Covariance': greedy_covariance_coreset,
    'HMP': hmp_coreset,
    'HMP-Kurt': hmp_kurtosis_heavy,
}

SUPERVISED_METHODS = {
    'Stratified': stratified_coreset,
    'Stratified+HMP': stratified_hmp_coreset,
}


def get_method(name: str):
    """Get method function by name."""
    if name in METHODS:
        return METHODS[name]
    elif name in SUPERVISED_METHODS:
        return SUPERVISED_METHODS[name]
    else:
        raise ValueError(f"Unknown method: {name}")
