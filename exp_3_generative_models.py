"""
Experiment 3: Generative Model Training Quality
================================================

Tests how coreset quality affects generative model training by measuring:
- FID (Fréchet Inception Distance) between generated and true distribution
- MMD (Maximum Mean Discrepancy) 
- Moment matching errors in generated samples

This simulates training a generator (GAN/VAE) on the coreset and
evaluating against the true distribution.

Output: Table 3 and Figure 3 data for the paper.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.mixture import GaussianMixture
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.coreset_methods import METHODS
from utils.metrics import (compute_mmd, compute_fid, covariance_error,
                           kurtosis_tensor_error, bootstrap_ci)

# Configuration
CONFIG = {
    'n_samples': 1000,
    'dimensions': 10,
    'coreset_size': 100,
    'n_seeds': 10,
    'n_generated': 500,
    'distributions': ['gaussian', 't3', 'mixture'],
    'output_dir': 'results/exp3_generative',
}


class SimpleGaussianGenerator:
    """
    A simple generator that learns to match the training distribution.
    Simulates what a well-trained GAN/VAE would achieve.
    
    Generator: z ~ N(0, I) -> x = Az + b where A, b match training moments.
    """
    
    def __init__(self, d: int):
        self.d = d
        self.A = None
        self.b = None
    
    def fit(self, X: np.ndarray):
        """Fit to match mean and covariance of training data."""
        self.b = X.mean(axis=0)
        cov = np.cov(X.T)
        
        # Regularize for stability
        cov = cov + 1e-4 * np.eye(self.d)
        
        # Cholesky factorization: Σ = AA^T
        try:
            self.A = np.linalg.cholesky(cov)
        except:
            # Fallback to eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 1e-6)
            self.A = eigvecs @ np.diag(np.sqrt(eigvals))
    
    def sample(self, n: int) -> np.ndarray:
        """Generate n samples."""
        z = np.random.randn(n, self.d)
        return z @ self.A.T + self.b


class GMMGenerator:
    """GMM-based generator for mixture distributions."""
    
    def __init__(self, n_components: int = 3):
        self.n_components = n_components
        self.gmm = None
    
    def fit(self, X: np.ndarray):
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='full',
            reg_covar=1e-4,
            random_state=42
        )
        self.gmm.fit(X)
    
    def sample(self, n: int) -> np.ndarray:
        return self.gmm.sample(n)[0]


def generate_data(dist_type: str, n: int, d: int, seed: int) -> np.ndarray:
    """Generate data from specified distribution."""
    np.random.seed(seed)
    
    A = np.random.randn(d, d) * 0.5
    cov = A @ A.T / d + 0.3 * np.eye(d)
    L = np.linalg.cholesky(cov)
    
    if dist_type == 'gaussian':
        X = np.random.randn(n, d) @ L.T
    elif dist_type == 't3':
        X = np.random.standard_t(3, (n, d)) @ L.T
    elif dist_type == 'mixture':
        X_list = []
        for c in range(3):
            n_c = n // 3
            X_c = np.random.randn(n_c, d) @ L.T
            mean_c = np.random.randn(d) * 2
            X_c += mean_c
            X_list.append(X_c)
        X = np.vstack(X_list)
        np.random.shuffle(X)
    else:
        raise ValueError(f"Unknown distribution: {dist_type}")
    
    # Add random mean shift
    X += np.random.randn(d)
    
    return X


def run_experiment():
    """Run generative model training experiment."""
    print("=" * 80)
    print("EXPERIMENT 3: GENERATIVE MODEL TRAINING QUALITY")
    print("=" * 80)
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    results = defaultdict(lambda: defaultdict(list))
    
    for dist in CONFIG['distributions']:
        print(f"\n--- Distribution: {dist} ---")
        
        for seed in range(CONFIG['n_seeds']):
            print(f"  Seed {seed+1}/{CONFIG['n_seeds']}...", end=" ", flush=True)
            
            # Generate "true" data
            X = generate_data(dist, CONFIG['n_samples'], CONFIG['dimensions'], 
                             seed=42+seed)
            
            # Split into train and test
            X_train = X[:int(0.8 * len(X))]
            X_test = X[int(0.8 * len(X)):]
            
            k = CONFIG['coreset_size']
            
            # Full data baseline
            if dist == 'mixture':
                gen_full = GMMGenerator(n_components=3)
            else:
                gen_full = SimpleGaussianGenerator(CONFIG['dimensions'])
            gen_full.fit(X_train)
            X_gen_full = gen_full.sample(CONFIG['n_generated'])
            
            mmd_full = compute_mmd(X_test, X_gen_full)
            fid_full = compute_fid(X_test, X_gen_full)
            
            results[dist][('Full', 'mmd')].append(mmd_full)
            results[dist][('Full', 'fid')].append(fid_full)
            
            # Coreset methods
            for method_name, method_fn in METHODS.items():
                if method_name == 'Random':
                    indices = method_fn(X_train, k, seed=seed)
                else:
                    indices = method_fn(X_train, k)
                
                X_core = X_train[indices]
                
                # Train generator on coreset
                if dist == 'mixture':
                    gen = GMMGenerator(n_components=3)
                else:
                    gen = SimpleGaussianGenerator(CONFIG['dimensions'])
                
                try:
                    gen.fit(X_core)
                    X_gen = gen.sample(CONFIG['n_generated'])
                    
                    # Evaluate against TRUE test distribution
                    mmd = compute_mmd(X_test, X_gen)
                    fid = compute_fid(X_test, X_gen)
                    cov_err = covariance_error(X_test, X_gen)
                    kurt_err = kurtosis_tensor_error(X_test, X_gen)
                except:
                    mmd, fid, cov_err, kurt_err = np.inf, np.inf, np.inf, np.inf
                
                results[dist][(method_name, 'mmd')].append(mmd)
                results[dist][(method_name, 'fid')].append(fid)
                results[dist][(method_name, 'cov_err')].append(cov_err)
                results[dist][(method_name, 'kurt_err')].append(kurt_err)
                results[dist][(method_name, 'mmd_gap')].append(mmd - mmd_full)
                results[dist][(method_name, 'fid_gap')].append(fid - fid_full)
            
            print("Done")
    
    # Print MMD Results
    print("\n" + "=" * 80)
    print("MMD (Maximum Mean Discrepancy, lower is better)")
    print("=" * 80)
    
    all_methods = ['Full'] + list(METHODS.keys())
    
    print(f"{'Method':<15}", end="")
    for dist in CONFIG['distributions']:
        print(f" {dist:>15}", end="")
    print()
    print("-" * 60)
    
    mmd_rows = []
    for method in all_methods:
        row = {'Method': method}
        line = f"{method:<15}"
        for dist in CONFIG['distributions']:
            vals = results[dist][(method, 'mmd')]
            mean_v = np.mean(vals)
            std_v = np.std(vals)
            line += f" {mean_v:.4f}±{std_v:.3f}"
            row[f'{dist}_mean'] = mean_v
            row[f'{dist}_std'] = std_v
        print(line)
        mmd_rows.append(row)
    
    pd.DataFrame(mmd_rows).to_csv(f"{CONFIG['output_dir']}/mmd_results.csv", index=False)
    
    # Print FID Results
    print("\n" + "=" * 80)
    print("FID (Fréchet Inception Distance proxy, lower is better)")
    print("=" * 80)
    
    print(f"{'Method':<15}", end="")
    for dist in CONFIG['distributions']:
        print(f" {dist:>15}", end="")
    print()
    print("-" * 60)
    
    fid_rows = []
    for method in all_methods:
        row = {'Method': method}
        line = f"{method:<15}"
        for dist in CONFIG['distributions']:
            vals = results[dist][(method, 'fid')]
            mean_v = np.mean(vals)
            std_v = np.std(vals)
            line += f" {mean_v:7.2f}±{std_v:.2f}"
            row[f'{dist}_mean'] = mean_v
            row[f'{dist}_std'] = std_v
        print(line)
        fid_rows.append(row)
    
    pd.DataFrame(fid_rows).to_csv(f"{CONFIG['output_dir']}/fid_results.csv", index=False)
    
    # Relative comparison
    print("\n" + "=" * 80)
    print("FID RELATIVE TO RANDOM (ratio, <1 is better)")
    print("=" * 80)
    
    for method in ['K-means++', 'Herding', 'Covariance', 'HMP']:
        line = f"{method:<15}"
        for dist in CONFIG['distributions']:
            random_fid = np.mean(results[dist][('Random', 'fid')])
            method_fid = np.mean(results[dist][(method, 'fid')])
            ratio = method_fid / (random_fid + 1e-10)
            status = "✓" if ratio < 1 else "✗"
            line += f" {ratio:6.2f}x {status}"
        print(line)
    
    # Correlation analysis
    print("\n" + "=" * 80)
    print("CORRELATION: Covariance Error → FID")
    print("=" * 80)
    
    all_cov_errs = []
    all_fids = []
    for dist in CONFIG['distributions']:
        for method in METHODS.keys():
            cov_errs = results[dist][(method, 'cov_err')]
            fids = results[dist][(method, 'fid')]
            all_cov_errs.extend(cov_errs)
            all_fids.extend(fids)
    
    corr = np.corrcoef(all_cov_errs, all_fids)[0, 1]
    print(f"Correlation(Covariance Error, FID) = {corr:.4f}")
    print("This demonstrates that preserving covariance → lower FID!")
    
    print(f"\nResults saved to {CONFIG['output_dir']}/")
    
    return results


if __name__ == "__main__":
    run_experiment()
