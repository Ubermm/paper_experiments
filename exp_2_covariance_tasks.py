"""
Experiment 2: Covariance-Dependent Downstream Tasks
====================================================

Tests coreset quality on tasks that depend on covariance estimation:
- QDA (Quadratic Discriminant Analysis)
- PCA (Principal Component Analysis) subspace recovery
- Mahalanobis distance / Outlier detection
- LDA (Linear Discriminant Analysis)

Output: Table 2 and Figure 2 data for the paper.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.coreset_methods import METHODS, SUPERVISED_METHODS
from utils.metrics import (pca_subspace_alignment, pca_reconstruction_error,
                           mahalanobis_correlation, outlier_detection_precision,
                           classification_metrics, bootstrap_ci, wilcoxon_test)

# Configuration
CONFIG = {
    'n_samples': 1500,
    'dimensions': 10,
    'n_classes': 3,
    'coreset_size': 100,
    'n_seeds': 10,
    'distributions': ['gaussian', 't3', 't2'],
    'output_dir': 'results/exp2_covariance_tasks',
}


def generate_classification_data(dist_type: str, n: int, d: int, n_classes: int, 
                                  seed: int) -> tuple:
    """Generate multi-class data with class-specific covariances."""
    np.random.seed(seed)
    
    X_list, y_list = [], []
    
    for c in range(n_classes):
        n_c = n // n_classes
        
        # Class-specific covariance
        A = np.random.randn(d, d) * (0.3 + 0.2 * c)
        cov_c = A @ A.T / d + 0.3 * np.eye(d)
        L = np.linalg.cholesky(cov_c)
        
        # Generate samples
        if dist_type == 'gaussian':
            X_c = np.random.randn(n_c, d) @ L.T
        elif dist_type == 't3':
            X_c = np.random.standard_t(3, (n_c, d)) @ L.T
        elif dist_type == 't2':
            X_c = np.random.standard_t(2, (n_c, d)) @ L.T
        else:
            raise ValueError(f"Unknown distribution: {dist_type}")
        
        # Class-specific mean
        mean_c = np.zeros(d)
        mean_c[c % d] = 3.0
        X_c += mean_c
        
        X_list.append(X_c)
        y_list.append(np.full(n_c, c))
    
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    # Shuffle
    perm = np.random.permutation(len(X))
    return X[perm], y[perm]


def run_qda_experiment(X_train, y_train, X_test, y_test, k, seed):
    """Run QDA experiment with different coreset methods."""
    results = {}
    
    # Full data baseline
    try:
        qda_full = QuadraticDiscriminantAnalysis(reg_param=0.1)
        qda_full.fit(X_train, y_train)
        acc_full = qda_full.score(X_test, y_test)
    except:
        acc_full = 0.0
    
    results['Full'] = {'accuracy': acc_full}
    
    # Unsupervised methods
    for method_name, method_fn in METHODS.items():
        if method_name == 'Random':
            indices = method_fn(X_train, k, seed=seed)
        else:
            indices = method_fn(X_train, k)
        
        X_core, y_core = X_train[indices], y_train[indices]
        
        try:
            qda = QuadraticDiscriminantAnalysis(reg_param=0.1)
            qda.fit(X_core, y_core)
            acc = qda.score(X_test, y_test)
        except:
            acc = 0.0
        
        results[method_name] = {
            'accuracy': acc,
            'gap': acc_full - acc,
        }
    
    # Supervised methods (class-balanced)
    for method_name, method_fn in SUPERVISED_METHODS.items():
        if 'Stratified' in method_name and 'HMP' not in method_name:
            indices = method_fn(X_train, y_train, k, seed=seed)
        else:
            indices = method_fn(X_train, y_train, k)
        
        X_core, y_core = X_train[indices], y_train[indices]
        
        try:
            qda = QuadraticDiscriminantAnalysis(reg_param=0.1)
            qda.fit(X_core, y_core)
            acc = qda.score(X_test, y_test)
        except:
            acc = 0.0
        
        results[method_name] = {
            'accuracy': acc,
            'gap': acc_full - acc,
        }
    
    return results


def run_pca_experiment(X, k, seed):
    """Run PCA subspace recovery experiment."""
    results = {}
    n_components = min(5, CONFIG['dimensions'])
    
    for method_name, method_fn in METHODS.items():
        if method_name == 'Random':
            indices = method_fn(X, k, seed=seed)
        else:
            indices = method_fn(X, k)
        
        X_core = X[indices]
        
        alignment = pca_subspace_alignment(X, X_core, n_components)
        recon_err = pca_reconstruction_error(X, X_core, n_components)
        
        results[method_name] = {
            'alignment': alignment,
            'reconstruction_error': recon_err,
        }
    
    return results


def run_mahalanobis_experiment(X, k, seed):
    """Run Mahalanobis distance / outlier detection experiment."""
    results = {}
    k_outliers = max(10, len(X) // 20)
    
    for method_name, method_fn in METHODS.items():
        if method_name == 'Random':
            indices = method_fn(X, k, seed=seed)
        else:
            indices = method_fn(X, k)
        
        X_core = X[indices]
        
        corr = mahalanobis_correlation(X, X_core)
        precision = outlier_detection_precision(X, X_core, k_outliers)
        
        results[method_name] = {
            'correlation': corr,
            'outlier_precision': precision,
        }
    
    return results


def run_experiment():
    """Run all covariance-dependent experiments."""
    print("=" * 80)
    print("EXPERIMENT 2: COVARIANCE-DEPENDENT DOWNSTREAM TASKS")
    print("=" * 80)
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Storage for all results
    qda_results = defaultdict(lambda: defaultdict(list))
    pca_results = defaultdict(lambda: defaultdict(list))
    mahal_results = defaultdict(lambda: defaultdict(list))
    
    k = CONFIG['coreset_size']
    
    for dist in CONFIG['distributions']:
        print(f"\n--- Distribution: {dist} ---")
        
        for seed in range(CONFIG['n_seeds']):
            print(f"  Seed {seed+1}/{CONFIG['n_seeds']}...", end=" ", flush=True)
            
            # Generate data
            X, y = generate_classification_data(
                dist, CONFIG['n_samples'], CONFIG['dimensions'],
                CONFIG['n_classes'], seed=42+seed
            )
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, stratify=y, random_state=seed
            )
            
            # QDA
            qda_res = run_qda_experiment(X_train, y_train, X_test, y_test, k, seed)
            for method, metrics in qda_res.items():
                for metric, val in metrics.items():
                    qda_results[dist][(method, metric)].append(val)
            
            # PCA (on full X, no labels)
            pca_res = run_pca_experiment(X, k, seed)
            for method, metrics in pca_res.items():
                for metric, val in metrics.items():
                    pca_results[dist][(method, metric)].append(val)
            
            # Mahalanobis
            mahal_res = run_mahalanobis_experiment(X, k, seed)
            for method, metrics in mahal_res.items():
                for metric, val in metrics.items():
                    mahal_results[dist][(method, metric)].append(val)
            
            print("Done")
    
    # Print QDA Results
    print("\n" + "=" * 80)
    print("QDA CLASSIFICATION ACCURACY")
    print("=" * 80)
    
    all_methods = list(METHODS.keys()) + list(SUPERVISED_METHODS.keys()) + ['Full']
    
    print(f"{'Method':<18}", end="")
    for dist in CONFIG['distributions']:
        print(f" {dist:>12}", end="")
    print()
    print("-" * 60)
    
    qda_rows = []
    for method in all_methods:
        row = {'Method': method}
        line = f"{method:<18}"
        for dist in CONFIG['distributions']:
            vals = qda_results[dist].get((method, 'accuracy'), [0])
            mean_v = np.mean(vals)
            std_v = np.std(vals)
            line += f" {mean_v:.4f}±{std_v:.2f}"
            row[f'{dist}_mean'] = mean_v
            row[f'{dist}_std'] = std_v
        print(line)
        qda_rows.append(row)
    
    pd.DataFrame(qda_rows).to_csv(f"{CONFIG['output_dir']}/qda_results.csv", index=False)
    
    # Print PCA Results
    print("\n" + "=" * 80)
    print("PCA SUBSPACE ALIGNMENT (higher is better)")
    print("=" * 80)
    
    print(f"{'Method':<15}", end="")
    for dist in CONFIG['distributions']:
        print(f" {dist:>12}", end="")
    print()
    print("-" * 55)
    
    pca_rows = []
    for method in METHODS.keys():
        row = {'Method': method}
        line = f"{method:<15}"
        for dist in CONFIG['distributions']:
            vals = pca_results[dist][(method, 'alignment')]
            mean_v = np.mean(vals)
            std_v = np.std(vals)
            line += f" {mean_v:.4f}±{std_v:.2f}"
            row[f'{dist}_mean'] = mean_v
            row[f'{dist}_std'] = std_v
        print(line)
        pca_rows.append(row)
    
    pd.DataFrame(pca_rows).to_csv(f"{CONFIG['output_dir']}/pca_results.csv", index=False)
    
    # Print Mahalanobis Results
    print("\n" + "=" * 80)
    print("MAHALANOBIS OUTLIER DETECTION PRECISION")
    print("=" * 80)
    
    print(f"{'Method':<15}", end="")
    for dist in CONFIG['distributions']:
        print(f" {dist:>12}", end="")
    print()
    print("-" * 55)
    
    mahal_rows = []
    for method in METHODS.keys():
        row = {'Method': method}
        line = f"{method:<15}"
        for dist in CONFIG['distributions']:
            vals = mahal_results[dist][(method, 'outlier_precision')]
            mean_v = np.mean(vals)
            std_v = np.std(vals)
            line += f" {mean_v:.4f}±{std_v:.2f}"
            row[f'{dist}_mean'] = mean_v
            row[f'{dist}_std'] = std_v
        print(line)
        mahal_rows.append(row)
    
    pd.DataFrame(mahal_rows).to_csv(f"{CONFIG['output_dir']}/mahalanobis_results.csv", index=False)
    
    # Statistical significance tests
    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE (Wilcoxon test vs Random)")
    print("=" * 80)
    
    for dist in CONFIG['distributions']:
        print(f"\n{dist}:")
        random_accs = qda_results[dist][('Random', 'accuracy')]
        for method in ['Covariance', 'HMP', 'Stratified+HMP']:
            method_accs = qda_results[dist].get((method, 'accuracy'), random_accs)
            stat, pval = wilcoxon_test(method_accs, random_accs)
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            print(f"  {method} vs Random: p={pval:.4f} {sig}")
    
    print(f"\nResults saved to {CONFIG['output_dir']}/")
    
    return qda_results, pca_results, mahal_results


if __name__ == "__main__":
    run_experiment()
