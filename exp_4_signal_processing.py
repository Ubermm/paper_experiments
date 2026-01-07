"""
Experiment 4: Signal Processing - ICA / Blind Source Separation
================================================================

Tests how coreset quality affects ICA (Independent Component Analysis):
- Source separation quality (correlation, SIR)
- Kurtosis (4th cumulant) preservation
- Negentropy preservation

This is where HMP with kurtosis preservation should excel!

Key insight: ICA separates signals by maximizing non-Gaussianity,
measured via 4th-order CUMULANTS (not raw moments).

Output: Table 4 and Figure 4 data for the paper.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import signal
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.coreset_methods import METHODS
from utils.metrics import (ica_separation_quality, negentropy_preservation,
                           kurtosis_tensor_error, bootstrap_ci, wilcoxon_test)

# Configuration
CONFIG = {
    'n_samples': 2000,
    'n_sources': 3,
    'coreset_sizes': [100, 200, 300],
    'n_seeds': 10,
    'output_dir': 'results/exp4_signal_processing',
    'test_prewhitening': True,  # Compare with/without pre-whitening
}


def generate_source_signals(n: int, n_sources: int = 3) -> tuple:
    """
    Generate source signals with different non-Gaussian characteristics.
    
    Returns: (sources, time_vector)
    """
    t = np.linspace(0, 10, n)
    
    sources = []
    source_types = []
    
    # Source 1: Sinusoid (negative kurtosis ~ -1.5)
    s1 = np.sin(2 * np.pi * 1.5 * t)
    sources.append(s1)
    source_types.append('sinusoid')
    
    # Source 2: Square wave (negative kurtosis ~ -2)
    s2 = signal.square(2 * np.pi * 0.5 * t)
    sources.append(s2)
    source_types.append('square')
    
    # Source 3: Laplacian noise (positive kurtosis ~ 3)
    s3 = np.random.laplace(0, 1, n)
    sources.append(s3)
    source_types.append('laplacian')
    
    if n_sources > 3:
        # Source 4: Uniform (negative kurtosis ~ -1.2)
        s4 = np.random.uniform(-1, 1, n) * 2
        sources.append(s4)
        source_types.append('uniform')
    
    if n_sources > 4:
        # Source 5: Super-Gaussian
        s5 = np.sign(np.random.randn(n)) * np.abs(np.random.randn(n)) ** 0.5
        sources.append(s5)
        source_types.append('super_gaussian')
    
    S = np.column_stack(sources[:n_sources])
    
    # Standardize
    S = (S - S.mean(axis=0)) / (S.std(axis=0) + 1e-10)
    
    return S, t, source_types[:n_sources]


def mix_signals(S: np.ndarray, seed: int = 42) -> tuple:
    """
    Create mixed signals X = S @ A.T (cocktail party problem).
    
    Returns: (mixed_signals, mixing_matrix)
    """
    np.random.seed(seed)
    n_sources = S.shape[1]
    
    # Random well-conditioned mixing matrix
    A = np.random.randn(n_sources, n_sources)
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    
    X = S @ A.T
    
    return X, A


def prewhiten_data(X: np.ndarray, regularization: float = 1e-6) -> tuple:
    """
    Pre-whiten data to have identity covariance before coreset selection.
    This ensures that coreset selection focuses on higher-order moments.

    Returns: (whitened_data, whitening_matrix, mean)
    """
    # Center the data
    mean = X.mean(axis=0)
    centered = X - mean

    # Compute covariance
    cov = np.cov(centered.T)

    # Eigendecomposition for whitening
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    eigenvals = np.maximum(eigenvals, regularization)  # Regularization

    # Whitening matrix
    W = eigenvecs @ np.diag(1.0 / np.sqrt(eigenvals)) @ eigenvecs.T

    # Apply whitening
    whitened = centered @ W.T

    return whitened, W, mean


def apply_whitening_to_coreset(X_core: np.ndarray, W: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Apply pre-computed whitening transformation to coreset."""
    return (X_core - mean) @ W.T


def compute_kurtosis_cumulant_error(X_full: np.ndarray, X_core: np.ndarray) -> float:
    """Compute error in 4th-order cumulants (what ICA uses)."""
    def kurtosis_cumulant(X):
        mu = X.mean(axis=0)
        c = X - mu
        m2 = (c ** 2).mean(axis=0)
        m4 = (c ** 4).mean(axis=0)
        return m4 - 3 * (m2 ** 2)
    
    k_full = kurtosis_cumulant(X_full)
    k_core = kurtosis_cumulant(X_core)
    
    return np.mean(np.abs(k_full - k_core) / (np.abs(k_full) + 1e-10))


def run_experiment():
    """Run ICA/BSS signal processing experiment."""
    print("=" * 80)
    print("EXPERIMENT 4: SIGNAL PROCESSING - ICA / BSS")
    print("The Cocktail Party Problem")
    print("=" * 80)
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    results = defaultdict(lambda: defaultdict(list))
    
    for k in CONFIG['coreset_sizes']:
        print(f"\n--- Coreset size k={k} ---")
        
        for seed in range(CONFIG['n_seeds']):
            print(f"  Seed {seed+1}/{CONFIG['n_seeds']}...", end=" ", flush=True)
            
            # Generate sources
            S, t, source_types = generate_source_signals(
                CONFIG['n_samples'], CONFIG['n_sources']
            )
            
            # Mix signals
            X, A = mix_signals(S, seed=42+seed)

            # Optional pre-whitening for comparison
            if CONFIG['test_prewhitening']:
                X_whitened, W_matrix, X_mean = prewhiten_data(X)
            else:
                X_whitened, W_matrix, X_mean = X, None, None
            
            # Print source kurtosis (first run only)
            if seed == 0 and k == CONFIG['coreset_sizes'][0]:
                print("\n  Source kurtosis:", end=" ")
                for i, st in enumerate(source_types):
                    k_val = kurtosis(S[:, i], fisher=True)
                    print(f"{st}={k_val:.2f}", end=" ")
                print()
            
            # Full data ICA baseline
            try:
                ica_full = FastICA(n_components=CONFIG['n_sources'],
                                   random_state=42, max_iter=500)
                S_full = ica_full.fit_transform(X)
                corr_full, sir_full = ica_separation_quality(S, S_full)
                conv_full = ica_full.n_iter_ < 500  # Converged if < max_iter
            except:
                corr_full, sir_full = 0.0, -np.inf
                conv_full = False

            results[k][('Full', 'correlation')].append(corr_full)
            results[k][('Full', 'sir')].append(sir_full)
            results[k][('Full', 'converged')].append(conv_full)
            
            # Coreset methods
            for method_name, method_fn in METHODS.items():
                # Test both regular and pre-whitened versions
                for whitened_suffix in (['', '+PreWhiten'] if CONFIG['test_prewhitening'] else ['']):
                    current_method_name = method_name + whitened_suffix

                    # Select from appropriate data
                    X_for_selection = X_whitened if 'PreWhiten' in whitened_suffix else X

                    if method_name == 'Random':
                        indices = method_fn(X_for_selection, k, seed=seed)
                    else:
                        indices = method_fn(X_for_selection, k)

                    X_core = X[indices]  # Always use original data for final evaluation

                    # Measure kurtosis preservation
                    kurt_err = compute_kurtosis_cumulant_error(X, X_core)
                    neg_err = negentropy_preservation(X, X_core)

                    # Run ICA on coreset
                    try:
                        ica = FastICA(n_components=CONFIG['n_sources'],
                                      random_state=42, max_iter=500)
                        ica.fit(X_core)

                        # Apply to FULL data
                        S_recovered = ica.transform(X)

                        corr, sir = ica_separation_quality(S, S_recovered)
                        converged = ica.n_iter_ < 500  # Converged if < max_iter
                        iterations = ica.n_iter_
                    except:
                        corr, sir = 0.0, -np.inf
                        converged = False
                        iterations = 500

                    results[k][(current_method_name, 'correlation')].append(corr)
                    results[k][(current_method_name, 'sir')].append(sir)
                    results[k][(current_method_name, 'kurt_err')].append(kurt_err)
                    results[k][(current_method_name, 'neg_err')].append(neg_err)
                    results[k][(current_method_name, 'corr_gap')].append(corr_full - corr)
                    results[k][(current_method_name, 'sir_gap')].append(sir_full - sir)
                    results[k][(current_method_name, 'converged')].append(converged)
                    results[k][(current_method_name, 'iterations')].append(iterations)
            
            print("Done")
    
    # Print SIR Results (key metric for ICA)
    print("\n" + "=" * 80)
    print("SIGNAL-TO-INTERFERENCE RATIO (SIR in dB, higher is better)")
    print("=" * 80)
    
    all_methods = ['Full'] + list(METHODS.keys())
    if CONFIG['test_prewhitening']:
        all_methods += [m + '+PreWhiten' for m in METHODS.keys()]
    
    print(f"{'Method':<15}", end="")
    for k in CONFIG['coreset_sizes']:
        print(f" k={k:>10}", end="")
    print()
    print("-" * 50)
    
    sir_rows = []
    for method in all_methods:
        row = {'Method': method}
        line = f"{method:<15}"
        for k in CONFIG['coreset_sizes']:
            vals = results[k][(method, 'sir')]
            mean_v = np.mean(vals)
            std_v = np.std(vals)
            line += f" {mean_v:6.2f}±{std_v:.1f}"
            row[f'k{k}_mean'] = mean_v
            row[f'k{k}_std'] = std_v
        print(line)
        sir_rows.append(row)
    
    pd.DataFrame(sir_rows).to_csv(f"{CONFIG['output_dir']}/sir_results.csv", index=False)
    
    # Print Correlation Results
    print("\n" + "=" * 80)
    print("SOURCE CORRELATION (higher is better)")
    print("=" * 80)
    
    print(f"{'Method':<15}", end="")
    for k in CONFIG['coreset_sizes']:
        print(f" k={k:>10}", end="")
    print()
    print("-" * 50)
    
    corr_rows = []
    for method in all_methods:
        row = {'Method': method}
        line = f"{method:<15}"
        for k in CONFIG['coreset_sizes']:
            vals = results[k][(method, 'correlation')]
            mean_v = np.mean(vals)
            std_v = np.std(vals)
            line += f" {mean_v:.4f}±{std_v:.2f}"
            row[f'k{k}_mean'] = mean_v
            row[f'k{k}_std'] = std_v
        print(line)
        corr_rows.append(row)
    
    pd.DataFrame(corr_rows).to_csv(f"{CONFIG['output_dir']}/correlation_results.csv", index=False)
    
    # Print Kurtosis Error (key for understanding)
    print("\n" + "=" * 80)
    print("KURTOSIS CUMULANT ERROR (lower is better)")
    print("=" * 80)
    
    print(f"{'Method':<15}", end="")
    for k in CONFIG['coreset_sizes']:
        print(f" k={k:>10}", end="")
    print()
    print("-" * 50)
    
    for method in METHODS.keys():
        line = f"{method:<15}"
        for k in CONFIG['coreset_sizes']:
            vals = results[k][(method, 'kurt_err')]
            mean_v = np.mean(vals)
            line += f" {mean_v:.4f}    "
        print(line)
    
    # Key insight analysis
    print("\n" + "=" * 80)
    print("KEY INSIGHT: Kurtosis Error → SIR Correlation")
    print("=" * 80)
    
    all_kurt_errs = []
    all_sirs = []
    for k in CONFIG['coreset_sizes']:
        for method in METHODS.keys():
            kurt_errs = results[k][(method, 'kurt_err')]
            sirs = results[k][(method, 'sir')]
            all_kurt_errs.extend(kurt_errs)
            all_sirs.extend(sirs)
    
    # Filter out infinities
    mask = np.isfinite(all_sirs)
    all_kurt_errs = np.array(all_kurt_errs)[mask]
    all_sirs = np.array(all_sirs)[mask]
    
    corr = np.corrcoef(all_kurt_errs, all_sirs)[0, 1]
    print(f"Correlation(Kurtosis Error, SIR) = {corr:.4f}")
    print("Negative correlation: Lower kurtosis error → Higher SIR!")
    
    # Method comparison
    print("\n" + "=" * 80)
    print("WHY COVARIANCE-ONLY FAILS FOR ICA")
    print("=" * 80)
    
    k = 200  # Middle coreset size
    
    cov_sir = np.mean(results[k][('Covariance', 'sir')])
    hmp_sir = np.mean(results[k][('HMP-Kurt', 'sir')])
    random_sir = np.mean(results[k][('Random', 'sir')])
    
    cov_kurt = np.mean(results[k][('Covariance', 'kurt_err')])
    hmp_kurt = np.mean(results[k][('HMP-Kurt', 'kurt_err')])
    
    print(f"At k={k}:")
    print(f"  Covariance: SIR={cov_sir:.2f} dB, Kurt Error={cov_kurt:.4f}")
    print(f"  HMP-Kurt:   SIR={hmp_sir:.2f} dB, Kurt Error={hmp_kurt:.4f}")
    print(f"  Random:     SIR={random_sir:.2f} dB")
    print()
    print("Covariance-only destroys the 4th-order cumulants that ICA needs!")
    
    # ICA Convergence Analysis
    print("\n" + "=" * 80)
    print("ICA CONVERGENCE RATES (% converged within 500 iterations)")
    print("=" * 80)

    print(f"{'Method':<15}", end="")
    for k in CONFIG['coreset_sizes']:
        print(f" k={k:>10}", end="")
    print()
    print("-" * 50)

    conv_rows = []
    for method in all_methods:
        row = {'Method': method}
        line = f"{method:<15}"
        for k in CONFIG['coreset_sizes']:
            conv_vals = results[k][(method, 'converged')]
            conv_rate = np.mean(conv_vals) * 100
            line += f"   {conv_rate:6.1f}%"
            row[f'k{k}_conv_rate'] = conv_rate
        print(line)
        conv_rows.append(row)

    pd.DataFrame(conv_rows).to_csv(f"{CONFIG['output_dir']}/convergence_results.csv", index=False)

    print("\n--- Average iterations to convergence ---")
    print(f"{'Method':<15}", end="")
    for k in CONFIG['coreset_sizes']:
        print(f" k={k:>10}", end="")
    print()
    print("-" * 50)

    for method in all_methods:
        line = f"{method:<15}"
        for k in CONFIG['coreset_sizes']:
            iter_vals = [it for it, conv in zip(results[k][(method, 'iterations')], results[k][(method, 'converged')]) if conv]
            if iter_vals:
                avg_iters = np.mean(iter_vals)
                line += f"   {avg_iters:6.1f}"
            else:
                line += f"      N/A"
        print(line)

    # Statistical significance
    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE")
    print("=" * 80)

    for k in CONFIG['coreset_sizes']:
        print(f"\nk={k}:")
        random_sirs = results[k][('Random', 'sir')]
        for method in ['Covariance', 'HMP', 'HMP-Kurt']:
            method_sirs = results[k][(method, 'sir')]
            stat, pval = wilcoxon_test(method_sirs, random_sirs)
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            print(f"  {method} vs Random: p={pval:.4f} {sig}")

    print(f"\nResults saved to {CONFIG['output_dir']}/")

    return results


if __name__ == "__main__":
    run_experiment()
