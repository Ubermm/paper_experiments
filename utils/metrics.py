"""
utils/metrics.py
================
Shared evaluation metrics for all experiments.
GPU acceleration automatically used for large datasets.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import kurtosis, skew, spearmanr
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Tuple, Dict, List

# GPU acceleration (auto-fallback to CPU)
try:
    from .gpu_utils import mahalanobis_distances_auto as mahalanobis_distances_gpu
    GPU_UTILS_AVAILABLE = True
except ImportError:
    GPU_UTILS_AVAILABLE = False


# =============================================================================
# MOMENT ERROR METRICS
# =============================================================================

def mean_error(X_full: np.ndarray, X_core: np.ndarray) -> float:
    """Relative error in mean estimation."""
    mu_full = X_full.mean(axis=0)
    mu_core = X_core.mean(axis=0)
    return np.linalg.norm(mu_full - mu_core) / (np.linalg.norm(mu_full) + 1e-10)


def covariance_error(X_full: np.ndarray, X_core: np.ndarray, 
                     norm: str = 'fro') -> float:
    """Relative Frobenius/spectral error in covariance estimation."""
    cov_full = np.cov(X_full.T)
    cov_core = np.cov(X_core.T)
    return np.linalg.norm(cov_full - cov_core, norm) / (np.linalg.norm(cov_full, norm) + 1e-10)


def kurtosis_error(X_full: np.ndarray, X_core: np.ndarray) -> float:
    """Relative error in kurtosis (4th cumulant) estimation."""
    # Per-dimension kurtosis
    kurt_full = kurtosis(X_full, axis=0, fisher=True)
    kurt_core = kurtosis(X_core, axis=0, fisher=True)
    return np.mean(np.abs(kurt_full - kurt_core) / (np.abs(kurt_full) + 1e-10))


def kurtosis_tensor_error(X_full: np.ndarray, X_core: np.ndarray) -> float:
    """Error in 4th-order cumulant tensor."""
    def compute_kurt_tensor(X):
        mu = X.mean(axis=0)
        c = X - mu
        N = len(X)

        # Raw 4th moment: E[xᵢ²xⱼ²]
        sq = c ** 2
        raw = (sq.T @ sq) / N

        # E[xᵢ²]E[xⱼ²]
        var = (c ** 2).mean(axis=0)
        var_outer = np.outer(var, var)

        # 2E[xᵢxⱼ]² (covariance squared)
        cov = (c.T @ c) / N
        cov_squared = cov ** 2

        # Correction: var_outer + 2*cov_squared
        correction = var_outer + 2 * cov_squared

        # No additional diagonal correction needed - the formula is already correct
        # For diagonal: κ₄[i,i] = E[xi⁴] - E[xi²]² - 2E[xi²]² = E[xi⁴] - 3E[xi²]²
        # This is handled by var_outer + 2*cov_squared since cov[i,i] = var[i]

        return raw - correction

    kt_full = compute_kurt_tensor(X_full)
    kt_core = compute_kurt_tensor(X_core)
    return np.linalg.norm(kt_full - kt_core, 'fro') / (np.linalg.norm(kt_full, 'fro') + 1e-10)


def skewness_error(X_full: np.ndarray, X_core: np.ndarray) -> float:
    """Relative error in skewness (3rd moment) estimation."""
    skew_full = skew(X_full, axis=0)
    skew_core = skew(X_core, axis=0)
    return np.mean(np.abs(skew_full - skew_core) / (np.abs(skew_full) + 1e-10))


def combined_moment_error(X_full: np.ndarray, X_core: np.ndarray,
                          weights: Tuple[float, float, float] = (0.2, 0.5, 0.3)) -> float:
    """Weighted combination of mean, covariance, and kurtosis errors."""
    w_mean, w_cov, w_kurt = weights
    return (w_mean * mean_error(X_full, X_core) +
            w_cov * covariance_error(X_full, X_core) +
            w_kurt * kurtosis_tensor_error(X_full, X_core))


# =============================================================================
# DISTRIBUTION DISTANCE METRICS
# =============================================================================

def compute_mmd(X: np.ndarray, Y: np.ndarray, sigma: float = None) -> float:
    """
    Maximum Mean Discrepancy with RBF kernel.
    MMD measures distance in RKHS - lower is better.
    """
    # Subsample for efficiency
    max_n = 500
    if len(X) > max_n:
        X = X[np.random.choice(len(X), max_n, replace=False)]
    if len(Y) > max_n:
        Y = Y[np.random.choice(len(Y), max_n, replace=False)]
    
    if sigma is None:
        dists = cdist(X[:100], Y[:100], 'euclidean')
        sigma = np.median(dists) + 1e-6
    
    def rbf(A, B):
        return np.exp(-cdist(A, B, 'sqeuclidean') / (2 * sigma ** 2))
    
    K_xx = rbf(X, X).mean()
    K_yy = rbf(Y, Y).mean()
    K_xy = rbf(X, Y).mean()
    
    return K_xx + K_yy - 2 * K_xy


def compute_fid(X_real: np.ndarray, X_gen: np.ndarray) -> float:
    """
    Fréchet Inception Distance (proxy without InceptionNet).
    FID = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2(Σ₁Σ₂)^½)
    """
    mu1, mu2 = X_real.mean(0), X_gen.mean(0)
    cov1, cov2 = np.cov(X_real.T), np.cov(X_gen.T)
    
    # Ensure positive semi-definiteness
    cov1 = cov1 + 1e-6 * np.eye(len(cov1))
    cov2 = cov2 + 1e-6 * np.eye(len(cov2))
    
    diff = mu1 - mu2
    mean_term = diff @ diff
    
    # Correct matrix sqrt: sqrt(Σ₁ Σ₂) via scipy
    from scipy.linalg import sqrtm
    
    # Compute sqrt(cov1 @ cov2) properly
    cov_sqrt = sqrtm(cov1 @ cov2)
    
    # Take real part (numerical errors can introduce small imaginary components)
    cov_sqrt_trace = np.real(np.trace(cov_sqrt))
    
    cov_term = np.trace(cov1) + np.trace(cov2) - 2 * cov_sqrt_trace
    
    # FID should be non-negative; clamp numerical errors
    return max(0.0, mean_term + cov_term)


# =============================================================================
# MAHALANOBIS DISTANCE METRICS
# =============================================================================

def mahalanobis_distances(X: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Compute Mahalanobis distance for each point with optional GPU acceleration."""
    if GPU_UTILS_AVAILABLE:
        return mahalanobis_distances_gpu(X, mu, cov)
    else:
        # CPU fallback
        try:
            cov_inv = np.linalg.inv(cov + 1e-6 * np.eye(len(cov)))
        except:
            cov_inv = np.eye(len(cov))

        centered = X - mu
        return np.sqrt(np.sum(centered @ cov_inv * centered, axis=1))


def mahalanobis_correlation(X_full: np.ndarray, X_core: np.ndarray) -> float:
    """Correlation between true and coreset-estimated Mahalanobis distances."""
    # True distances (from full data)
    mu_full = X_full.mean(axis=0)
    cov_full = np.cov(X_full.T)
    d_true = mahalanobis_distances(X_full, mu_full, cov_full)
    
    # Estimated distances (from coreset)
    mu_core = X_core.mean(axis=0)
    cov_core = np.cov(X_core.T)
    d_est = mahalanobis_distances(X_full, mu_core, cov_core)
    
    return np.corrcoef(d_true, d_est)[0, 1]


def outlier_detection_precision(X_full: np.ndarray, X_core: np.ndarray,
                                k_outliers: int = 50) -> float:
    """Precision in detecting top-k outliers via Mahalanobis distance."""
    # True outliers
    mu_full = X_full.mean(axis=0)
    cov_full = np.cov(X_full.T)
    d_true = mahalanobis_distances(X_full, mu_full, cov_full)
    true_outliers = set(np.argsort(d_true)[-k_outliers:])
    
    # Detected outliers
    mu_core = X_core.mean(axis=0)
    cov_core = np.cov(X_core.T)
    d_est = mahalanobis_distances(X_full, mu_core, cov_core)
    detected = set(np.argsort(d_est)[-k_outliers:])
    
    return len(true_outliers & detected) / k_outliers


# =============================================================================
# ICA / BSS METRICS
# =============================================================================

def ica_separation_quality(S_true: np.ndarray, S_recovered: np.ndarray) -> Tuple[float, float]:
    """
    Compute ICA source separation quality.
    
    Returns:
    --------
    avg_corr : Average correlation after optimal matching
    sir : Signal-to-Interference Ratio in dB
    """
    n_sources = S_true.shape[1]
    
    # Correlation matrix
    corr_matrix = np.abs(np.corrcoef(S_true.T, S_recovered.T)[:n_sources, n_sources:])
    
    # Hungarian algorithm for optimal assignment
    row_ind, col_ind = linear_sum_assignment(-corr_matrix)
    
    avg_corr = corr_matrix[row_ind, col_ind].mean()
    
    # SIR calculation
    sir = 0
    for i, j in zip(row_ind, col_ind):
        matched = corr_matrix[i, j]
        others = np.delete(corr_matrix[i, :], j)
        if len(others) > 0 and others.max() > 0:
            sir += 20 * np.log10(matched / (others.max() + 1e-10))
    sir /= n_sources
    
    return avg_corr, sir


def negentropy(x: np.ndarray) -> float:
    """Approximate negentropy using log-cosh."""
    return (np.mean(np.log(np.cosh(x))) - 0.3745) ** 2


def negentropy_preservation(X_full: np.ndarray, X_core: np.ndarray) -> float:
    """Relative error in negentropy (non-Gaussianity measure)."""
    neg_full = np.array([negentropy(X_full[:, i]) for i in range(X_full.shape[1])])
    neg_core = np.array([negentropy(X_core[:, i]) for i in range(X_core.shape[1])])
    return np.mean(np.abs(neg_full - neg_core) / (neg_full + 1e-10))


# =============================================================================
# CLASSIFICATION METRICS
# =============================================================================

def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard classification metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
    }


# =============================================================================
# PCA METRICS
# =============================================================================

def pca_subspace_alignment(X_full: np.ndarray, X_core: np.ndarray,
                           n_components: int = 5) -> float:
    """Measure alignment between PCA subspaces."""
    from sklearn.decomposition import PCA
    
    pca_full = PCA(n_components=n_components)
    pca_full.fit(X_full)
    V_full = pca_full.components_
    
    pca_core = PCA(n_components=n_components)
    pca_core.fit(X_core)
    V_core = pca_core.components_
    
    # Average cosine similarity
    alignment = 0
    for i in range(n_components):
        alignment += np.abs(V_core[i] @ V_full[i])
    
    return alignment / n_components


def pca_reconstruction_error(X_full: np.ndarray, X_core: np.ndarray,
                             n_components: int = 5) -> float:
    """Reconstruction error when using PCA from coreset on full data."""
    from sklearn.decomposition import PCA
    
    pca_core = PCA(n_components=n_components)
    pca_core.fit(X_core)
    
    X_recon = pca_core.inverse_transform(pca_core.transform(X_full))
    return np.mean((X_full - X_recon) ** 2)


# =============================================================================
# STATISTICAL SIGNIFICANCE
# =============================================================================

def wilcoxon_test(scores_a: List[float], scores_b: List[float]) -> Tuple[float, float]:
    """Wilcoxon signed-rank test for paired samples."""
    from scipy.stats import wilcoxon
    try:
        stat, pval = wilcoxon(scores_a, scores_b)
        return stat, pval
    except:
        return np.nan, np.nan


def bootstrap_ci(scores: List[float], n_bootstrap: int = 1000,
                 ci: float = 0.95) -> Tuple[float, float, float]:
    """Bootstrap confidence interval."""
    scores = np.array(scores)
    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, len(scores), replace=True)
        means.append(sample.mean())
    
    alpha = (1 - ci) / 2
    lower = np.percentile(means, alpha * 100)
    upper = np.percentile(means, (1 - alpha) * 100)
    
    return scores.mean(), lower, upper
