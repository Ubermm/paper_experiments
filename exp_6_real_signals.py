"""
Experiment 6: Real-World Signal Processing (EEG/Audio-like)
============================================================

Tests coreset methods on realistic multi-channel signal data for ICA:
- EEG-like signals (brain activity + artifacts)
- Audio-like signals (speech/music mixtures)

This provides real-world validation for the ICA experiments.

Key properties of real signals:
- Non-stationary
- Multiple source types (neural, ocular, muscular for EEG)
- Different non-Gaussianity levels

Output: Table 6 data for the paper (real-world signal validation).
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import signal as sig
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.coreset_methods import METHODS
from utils.metrics import ica_separation_quality, bootstrap_ci, wilcoxon_test

# Configuration
CONFIG = {
    'n_samples': 5000,  # ~5 seconds at 1kHz
    'sampling_rate': 1000,  # Hz
    'n_sources': 4,
    'coreset_size': 500,
    'n_seeds': 10,
    'output_dir': 'results/exp6_real_signals',
}


def generate_eeg_like_sources(n_samples: int, fs: int, seed: int) -> tuple:
    """
    Generate realistic EEG-like source signals:
    - Alpha rhythm (8-13 Hz oscillation)
    - Eye blink artifact (impulsive, super-Gaussian)
    - Muscle artifact (high-frequency, Gaussian-ish)
    - Neural background (1/f noise, heavy-tailed)
    """
    np.random.seed(seed)
    t = np.arange(n_samples) / fs
    
    sources = []
    source_names = []
    
    # Source 1: Alpha rhythm (sinusoidal, negative kurtosis)
    alpha_freq = 10 + np.random.uniform(-1, 1)  # 9-11 Hz
    alpha = np.sin(2 * np.pi * alpha_freq * t)
    # Add amplitude modulation
    alpha *= (1 + 0.3 * np.sin(2 * np.pi * 0.5 * t))
    sources.append(alpha)
    source_names.append('alpha')
    
    # Source 2: Eye blink (impulsive, super-Gaussian, kurtosis > 3)
    blink = np.zeros(n_samples)
    n_blinks = max(3, n_samples // 1000)  # ~1 blink per second
    blink_times = np.random.choice(n_samples - 100, n_blinks, replace=False)
    for bt in blink_times:
        # Blink shape: sharp rise, slow decay
        blink_shape = np.exp(-np.arange(100) / 20) * np.sin(np.pi * np.arange(100) / 100)
        blink[bt:bt+100] += np.random.uniform(2, 4) * blink_shape
    sources.append(blink)
    source_names.append('eye_blink')
    
    # Source 3: Muscle artifact (EMG, near-Gaussian high-freq noise)
    muscle = np.random.randn(n_samples)
    # Bandpass 30-100 Hz
    b, a = sig.butter(4, [30/(fs/2), 100/(fs/2)], btype='band')
    muscle = sig.filtfilt(b, a, muscle)
    # Add bursts
    n_bursts = 5
    burst_times = np.random.choice(n_samples - 200, n_bursts, replace=False)
    for bt in burst_times:
        muscle[bt:bt+200] *= np.random.uniform(3, 5)
    sources.append(muscle)
    source_names.append('muscle')
    
    # Source 4: 1/f neural background (heavy-tailed)
    # Generate 1/f noise via spectral shaping
    white = np.random.standard_t(4, n_samples)  # Heavy-tailed
    freqs = np.fft.rfftfreq(n_samples, 1/fs)
    freqs[0] = 1  # Avoid division by zero
    spectrum = np.fft.rfft(white)
    spectrum /= np.sqrt(freqs)  # 1/f shaping
    neural_bg = np.fft.irfft(spectrum, n_samples)
    sources.append(neural_bg)
    source_names.append('neural_bg')
    
    S = np.column_stack(sources)
    
    # Standardize
    S = (S - S.mean(axis=0)) / (S.std(axis=0) + 1e-10)
    
    return S, source_names


def generate_audio_like_sources(n_samples: int, fs: int, seed: int) -> tuple:
    """
    Generate audio-like source signals:
    - Speech-like (modulated narrowband, super-Gaussian)
    - Music-like (harmonic, sub-Gaussian)
    - Noise (background, Gaussian)
    - Transient (clicks, super-Gaussian)
    """
    np.random.seed(seed)
    t = np.arange(n_samples) / fs
    
    sources = []
    source_names = []
    
    # Source 1: Speech-like (amplitude-modulated narrowband)
    carrier = np.sin(2 * np.pi * 200 * t)  # 200 Hz carrier
    # Syllable-like modulation
    mod = np.abs(np.sin(2 * np.pi * 4 * t)) ** 2  # ~4 syllables/sec
    speech = carrier * mod
    # Add formant-like filtering
    b, a = sig.butter(2, [100/(fs/2), 400/(fs/2)], btype='band')
    speech = sig.filtfilt(b, a, speech * np.random.randn(n_samples) * mod)
    sources.append(speech)
    source_names.append('speech')
    
    # Source 2: Music-like (harmonic content)
    fundamental = 110  # A2
    music = np.zeros(n_samples)
    for harmonic in [1, 2, 3, 4, 5]:
        amp = 1.0 / harmonic
        music += amp * np.sin(2 * np.pi * fundamental * harmonic * t)
    # Add vibrato
    music *= np.sin(2 * np.pi * 5 * t + 0.5 * np.sin(2 * np.pi * 0.5 * t))
    sources.append(music)
    source_names.append('music')
    
    # Source 3: Background noise (Gaussian)
    noise = np.random.randn(n_samples)
    # Low-pass for realism
    b, a = sig.butter(2, min(0.9, 500/(fs/2)))
    noise = sig.filtfilt(b, a, noise)
    sources.append(noise)
    source_names.append('noise')
    
    # Source 4: Transients (clicks, super-Gaussian)
    transient = np.zeros(n_samples)
    n_clicks = 20
    click_times = np.random.choice(n_samples - 10, n_clicks, replace=False)
    for ct in click_times:
        transient[ct:ct+10] = np.random.randn() * np.exp(-np.arange(10) / 2)
    sources.append(transient)
    source_names.append('transient')
    
    S = np.column_stack(sources)
    S = (S - S.mean(axis=0)) / (S.std(axis=0) + 1e-10)
    
    return S, source_names


def run_experiment():
    """Run real-world signal processing experiment."""
    print("=" * 80)
    print("EXPERIMENT 6: REAL-WORLD SIGNAL PROCESSING")
    print("EEG-like and Audio-like Source Separation")
    print("=" * 80)
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    results = defaultdict(lambda: defaultdict(list))
    
    k = CONFIG['coreset_size']
    fs = CONFIG['sampling_rate']
    
    for signal_type in ['eeg', 'audio']:
        print(f"\n--- Signal type: {signal_type.upper()} ---")
        
        for seed in range(CONFIG['n_seeds']):
            print(f"  Seed {seed+1}/{CONFIG['n_seeds']}...", end=" ", flush=True)
            
            # Generate sources
            if signal_type == 'eeg':
                S, source_names = generate_eeg_like_sources(
                    CONFIG['n_samples'], fs, seed=42+seed
                )
            else:
                S, source_names = generate_audio_like_sources(
                    CONFIG['n_samples'], fs, seed=42+seed
                )
            
            # Print source kurtosis (first run only)
            if seed == 0:
                print(f"\n    Source kurtosis:", end=" ")
                for i, name in enumerate(source_names):
                    k_val = kurtosis(S[:, i], fisher=True)
                    print(f"{name}={k_val:.2f}", end=" ")
                print()
            
            # Create mixture
            np.random.seed(seed + 100)
            A = np.random.randn(CONFIG['n_sources'], CONFIG['n_sources'])
            A = A / np.linalg.norm(A, axis=1, keepdims=True)
            X = S @ A.T
            
            # Full ICA baseline
            try:
                ica_full = FastICA(n_components=CONFIG['n_sources'],
                                   random_state=42, max_iter=1000)
                S_full = ica_full.fit_transform(X)
                corr_full, sir_full = ica_separation_quality(S, S_full)
            except:
                corr_full, sir_full = 0.0, -np.inf
            
            results[signal_type][('Full', 'correlation')].append(corr_full)
            results[signal_type][('Full', 'sir')].append(sir_full)
            
            # Coreset methods
            for method_name, method_fn in METHODS.items():
                if method_name == 'Random':
                    indices = method_fn(X, k, seed=seed)
                else:
                    indices = method_fn(X, k)
                
                X_core = X[indices]
                
                # Kurtosis preservation
                kurt_full = kurtosis(X, axis=0, fisher=True)
                kurt_core = kurtosis(X_core, axis=0, fisher=True)
                kurt_err = np.mean(np.abs(kurt_full - kurt_core) / (np.abs(kurt_full) + 1e-10))
                
                # ICA on coreset, apply to full data
                try:
                    ica = FastICA(n_components=CONFIG['n_sources'],
                                  random_state=42, max_iter=1000)
                    ica.fit(X_core)
                    S_recovered = ica.transform(X)
                    corr, sir = ica_separation_quality(S, S_recovered)
                except:
                    corr, sir = 0.0, -np.inf
                
                results[signal_type][(method_name, 'correlation')].append(corr)
                results[signal_type][(method_name, 'sir')].append(sir)
                results[signal_type][(method_name, 'kurt_err')].append(kurt_err)
            
            print("Done")
    
    # Print Results
    print("\n" + "=" * 80)
    print("SOURCE SEPARATION RESULTS")
    print("=" * 80)
    
    for signal_type in ['eeg', 'audio']:
        print(f"\n{signal_type.upper()} Signals:")
        print(f"{'Method':<15} {'Correlation':>12} {'SIR (dB)':>12} {'Kurt Error':>12}")
        print("-" * 55)
        
        all_methods = ['Full'] + list(METHODS.keys())
        
        rows = []
        for method in all_methods:
            row = {'Method': method, 'Signal': signal_type}
            
            corr_vals = results[signal_type][(method, 'correlation')]
            sir_vals = results[signal_type][(method, 'sir')]
            
            corr_mean = np.mean(corr_vals)
            corr_std = np.std(corr_vals)
            sir_mean = np.mean(sir_vals)
            sir_std = np.std(sir_vals)
            
            if method != 'Full':
                kurt_vals = results[signal_type][(method, 'kurt_err')]
                kurt_mean = np.mean(kurt_vals)
                kurt_str = f"{kurt_mean:.4f}"
            else:
                kurt_str = "N/A"
            
            print(f"{method:<15} {corr_mean:.4f}±{corr_std:.2f} {sir_mean:7.2f}±{sir_std:.1f} {kurt_str:>12}")
            
            row['corr_mean'] = corr_mean
            row['corr_std'] = corr_std
            row['sir_mean'] = sir_mean
            row['sir_std'] = sir_std
            rows.append(row)
        
        pd.DataFrame(rows).to_csv(
            f"{CONFIG['output_dir']}/{signal_type}_results.csv", index=False
        )
    
    # Statistical significance
    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE")
    print("=" * 80)
    
    for signal_type in ['eeg', 'audio']:
        print(f"\n{signal_type.upper()}:")
        random_sirs = results[signal_type][('Random', 'sir')]
        for method in ['Covariance', 'HMP', 'HMP-Kurt']:
            method_sirs = results[signal_type][(method, 'sir')]
            stat, pval = wilcoxon_test(method_sirs, random_sirs)
            sig_str = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            print(f"  {method} vs Random: p={pval:.4f} {sig_str}")
    
    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    for signal_type in ['eeg', 'audio']:
        print(f"\n{signal_type.upper()} Signals:")
        
        cov_sir = np.mean(results[signal_type][('Covariance', 'sir')])
        hmp_kurt_sir = np.mean(results[signal_type][('HMP-Kurt', 'sir')])
        random_sir = np.mean(results[signal_type][('Random', 'sir')])
        full_sir = np.mean(results[signal_type][('Full', 'sir')])
        
        print(f"  Full data: {full_sir:.2f} dB")
        print(f"  Random:    {random_sir:.2f} dB")
        print(f"  Covariance: {cov_sir:.2f} dB")
        print(f"  HMP-Kurt:   {hmp_kurt_sir:.2f} dB")
        
        if hmp_kurt_sir > cov_sir:
            print(f"  → HMP-Kurt beats Covariance by {hmp_kurt_sir - cov_sir:.2f} dB")
        else:
            print(f"  → Covariance beats HMP-Kurt by {cov_sir - hmp_kurt_sir:.2f} dB")
    
    print(f"\nResults saved to {CONFIG['output_dir']}/")
    
    return results


if __name__ == "__main__":
    run_experiment()
