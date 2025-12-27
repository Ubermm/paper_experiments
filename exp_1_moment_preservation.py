"""
Experiment 1: Moment Preservation Quality
==========================================

This experiment measures how well different coreset methods preserve
statistical moments (mean, covariance, kurtosis) across:
- Multiple distributions (Gaussian, t₃, t₂, mixture)
- Multiple coreset sizes (k = 30, 50, 100, 150)
- Multiple seeds for statistical significance

Output: Table 1 and Figure 1 data for the paper.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.coreset_methods import METHODS, compute_covariance, compute_kurtosis_tensor
from utils.metrics import (mean_error, covariance_error, kurtosis_tensor_error,
                           skewness_error, combined_moment_error, bootstrap_ci)

# Configuration
CONFIG = {
    'n_samples': 1000,
    'dimensions': 10,
    'coreset_sizes': [30, 50, 100, 150],  # Match paper Table 1
    'n_seeds': 10,
    'distributions': ['gaussian', 't3', 't2', 'mixture'],
    'output_dir': 'results/exp1_moments',
}


def generate_data(dist_type: str, n: int, d: int, seed: int) -> np.ndarray:
    """Generate data from specified distribution."""
    np.random.seed(seed)
    
    # Random covariance structure
    A = np.random.randn(d, d) * 0.5
    cov = A @ A.T / d + 0.3 * np.eye(d)
    L = np.linalg.cholesky(cov)
    
    if dist_type == 'gaussian':
        X = np.random.randn(n, d) @ L.T
    elif dist_type == 't3':
        X = np.random.standard_t(3, (n, d)) @ L.T
    elif dist_type == 't2':
        X = np.random.standard_t(2, (n, d)) @ L.T
    elif dist_type == 'mixture':
        # 3-component mixture
        X_list = []
        for c in range(3):
            n_c = n // 3
            X_c = np.random.randn(n_c, d) @ L.T
            X_c[:, c % d] += 3.0  # Shift mean
            X_list.append(X_c)
        X = np.vstack(X_list)
        np.random.shuffle(X)
    else:
        raise ValueError(f"Unknown distribution: {dist_type}")
    
    return X


def run_experiment():
    """Run the full moment preservation experiment."""
    print("=" * 80)
    print("EXPERIMENT 1: MOMENT PRESERVATION QUALITY")
    print("=" * 80)
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for dist in CONFIG['distributions']:
        print(f"\n--- Distribution: {dist} ---")
        
        for k in CONFIG['coreset_sizes']:
            print(f"  Coreset size k={k}...", end=" ", flush=True)
            
            for seed in range(CONFIG['n_seeds']):
                X = generate_data(dist, CONFIG['n_samples'], CONFIG['dimensions'], 
                                  seed=42 + seed)
                
                for method_name, method_fn in METHODS.items():
                    # Select coreset
                    if method_name == 'Random':
                        indices = method_fn(X, k, seed=seed)
                    else:
                        indices = method_fn(X, k)
                    
                    X_core = X[indices]
                    
                    # Compute errors
                    err_mean = mean_error(X, X_core)
                    err_cov = covariance_error(X, X_core)
                    err_kurt = kurtosis_tensor_error(X, X_core)
                    err_skew = skewness_error(X, X_core)
                    err_combined = combined_moment_error(X, X_core)
                    
                    # Store results
                    key = f"{dist}_k{k}"
                    results[method_name][key]['mean'].append(err_mean)
                    results[method_name][key]['cov'].append(err_cov)
                    results[method_name][key]['kurt'].append(err_kurt)
                    results[method_name][key]['skew'].append(err_skew)
                    results[method_name][key]['combined'].append(err_combined)
            
            print("Done")
    
    # Create summary tables
    print("\n" + "=" * 80)
    print("COVARIANCE ERROR (Frobenius, lower is better)")
    print("=" * 80)
    
    # Table header
    header = f"{'Method':<15}"
    for dist in CONFIG['distributions']:
        for k in CONFIG['coreset_sizes']:
            header += f" {dist[:4]}_k{k:3d}"
    print(header)
    print("-" * len(header))
    
    # Detailed results dataframe
    rows = []
    for method in METHODS.keys():
        row = {'Method': method}
        line = f"{method:<15}"
        for dist in CONFIG['distributions']:
            for k in CONFIG['coreset_sizes']:
                key = f"{dist}_k{k}"
                vals = results[method][key]['cov']
                mean_val = np.mean(vals)
                std_val = np.std(vals)
                line += f" {mean_val:7.3f}"
                row[f'{dist}_k{k}_cov_mean'] = mean_val
                row[f'{dist}_k{k}_cov_std'] = std_val
        print(line)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(f"{CONFIG['output_dir']}/covariance_errors.csv", index=False)
    
    # Kurtosis error table
    print("\n" + "=" * 80)
    print("KURTOSIS CUMULANT ERROR (Frobenius, lower is better)")
    print("=" * 80)
    
    for method in METHODS.keys():
        line = f"{method:<15}"
        for dist in CONFIG['distributions']:
            for k in CONFIG['coreset_sizes']:
                key = f"{dist}_k{k}"
                vals = results[method][key]['kurt']
                mean_val = np.mean(vals)
                line += f" {mean_val:7.3f}"
        print(line)
    
    # Save detailed results
    detailed_rows = []
    for method in METHODS.keys():
        for dist in CONFIG['distributions']:
            for k in CONFIG['coreset_sizes']:
                key = f"{dist}_k{k}"
                for metric in ['mean', 'cov', 'kurt', 'skew', 'combined']:
                    vals = results[method][key][metric]
                    mean_v, lower, upper = bootstrap_ci(vals)
                    detailed_rows.append({
                        'method': method,
                        'distribution': dist,
                        'k': k,
                        'metric': metric,
                        'mean': mean_v,
                        'std': np.std(vals),
                        'ci_lower': lower,
                        'ci_upper': upper,
                        'n_runs': len(vals),
                    })
    
    df_detailed = pd.DataFrame(detailed_rows)
    df_detailed.to_csv(f"{CONFIG['output_dir']}/detailed_results.csv", index=False)
    
    # Improvement ratios
    print("\n" + "=" * 80)
    print("IMPROVEMENT OVER RANDOM (Covariance Error)")
    print("=" * 80)
    
    for method in ['K-means++', 'Herding', 'Covariance', 'HMP', 'HMP-Kurt']:
        line = f"{method:<15}"
        for dist in CONFIG['distributions']:
            for k in [50, 100]:  # Selected sizes
                key = f"{dist}_k{k}"
                random_err = np.mean(results['Random'][key]['cov'])
                method_err = np.mean(results[method][key]['cov'])
                ratio = random_err / (method_err + 1e-10)
                line += f" {ratio:6.2f}x"
        print(line)
    
    print(f"\nResults saved to {CONFIG['output_dir']}/")
    
    return results


if __name__ == "__main__":
    results = run_experiment()
