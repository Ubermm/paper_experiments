# Hierarchical Moment-Preserving Coresets

## Experimental Evaluation Suite

This repository contains all code to reproduce the experiments in:

> **Beyond Coverage: Task-Specific Moment Preservation for Optimal Coreset Selection**

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run synthetic experiments (1-8)
python run_all_experiments.py 1 2 3 4 5 6 7 8

# Run REAL-WORLD experiments (9-10)
python run_all_experiments.py 9 10

# Run specific experiment
python run_all_experiments.py 9   # S&P 500 finance
python run_all_experiments.py 10  # PhysioNet EEG

# Generate figures
python generate_figures.py
```

---

## Installation

```bash
# Basic install (synthetic experiments only)
pip install numpy scipy pandas scikit-learn matplotlib seaborn

# Full install (includes real-world data support)
pip install numpy scipy pandas scikit-learn matplotlib seaborn mne

# Or use requirements.txt
pip install -r requirements.txt
```

### Required Libraries by Experiment

| Experiment | Libraries Needed |
|------------|-----------------|
| 1-8 (Synthetic) | numpy, scipy, pandas, scikit-learn |
| 9 (S&P 500) | pandas (+ manual data download) |
| 10 (EEG) | **mne** (auto-downloads data) |
| Figures | matplotlib, seaborn |

---

## Real-World Data Setup

### S&P 500 Stock Data (Experiment 9)

**Source:** https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks

1. Download `sp-500-stocks.zip` from Kaggle
2. Extract to `data/sp500/` folder
3. Verify: `data/sp500/sp500_stocks.csv` exists

```
paper_experiments/
└── data/
    └── sp500/
        ├── sp500_stocks.csv      # Required (~200MB)
        ├── sp500_companies.csv   # Not required
        └── sp500_index.csv       # Not required
```

**What's in the data:**
- Daily OHLCV data for 500+ S&P 500 stocks
- Date range: 2010-2022+
- We use: Adjusted Close prices → Daily returns

### PhysioNet EEG Data (Experiment 10)

**Source:** https://physionet.org/content/eegmmidb/1.0.0/

- **Automatically downloaded** by MNE-Python on first run
- 109 subjects, 64 channels, motor/imagery tasks
- ~1.9GB total (downloaded incrementally per subject)
- Cached in `~/mne_data/`

```bash
# Install MNE
pip install mne

# Just run - data downloads automatically
python run_all_experiments.py 10
```

**First run note:** May take 5-10 minutes to download data for first few subjects.

---

## Repository Structure

```
paper_experiments/
├── utils/
│   ├── __init__.py
│   ├── coreset_methods.py        # All coreset selection algorithms
│   └── metrics.py                # Evaluation metrics
│
├── Synthetic Experiments:
│   ├── exp_1_moment_preservation.py  # Core moment quality
│   ├── exp_2_covariance_tasks.py     # QDA, PCA, Mahalanobis
│   ├── exp_3_generative_models.py    # FID, MMD for GANs/VAEs
│   ├── exp_4_signal_processing.py    # ICA/BSS (synthetic signals)
│   ├── exp_5_finance_outliers.py     # Finance (synthetic)
│   ├── exp_6_real_signals.py         # EEG/audio (synthetic)
│   ├── exp_7_ablations.py            # Ablation studies
│   └── exp_8_runtime.py              # Computational analysis
│
├── Real-World Experiments:
│   ├── exp_5_finance_real.py     # S&P 500 Stock Data
│   └── exp_6_eeg_real.py         # PhysioNet EEGMMI
│
├── run_all_experiments.py        # Master runner
├── generate_figures.py           # Paper figures
├── paper_draft.tex               # LaTeX paper draft
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Experiments Overview

| # | Name | Data | Key Metric |
|---|------|------|------------|
| 1 | Moment Preservation | Synthetic | Cov/Kurt Error |
| 2 | Covariance Tasks | Synthetic | QDA Accuracy |
| 3 | Generative Models | Synthetic | FID Score |
| 4 | Signal Processing | Synthetic | SIR (dB) |
| 5 | Finance Outliers | Synthetic | VaR Error |
| 6 | Real Signals | Synthetic | Separation quality |
| 7 | Ablations | Synthetic | Various |
| 8 | Runtime | Synthetic | Time (s) |
| **9** | **S&P 500 Finance** | **REAL** | **Outlier Precision, VaR Error** |
| **10** | **PhysioNet EEG** | **REAL** | **ICA Source Correlation** |

---

## Key Methods

### Baseline Methods
- **Random**: Uniform random sampling (unbiased)
- **K-means++**: D² sampling for coverage
- **Herding**: Kernel herding for mean matching

### Our Methods
- **Covariance**: Greedy covariance matching (best for 2nd-order tasks)
- **HMP**: Hierarchical Moment-Preserving (balanced)
- **HMP-Kurt**: Heavy kurtosis weighting (best for ICA)

---

## Expected Results

### Synthetic Data (Quick Test)
```
Method          : Cov Error    Kurt Error
-------------------------------------------
Random          : 0.4266       0.9884
K-means++       : 1.9825       7.6249  ← 10x WORSE!
Covariance      : 0.0388       1.0006  ← BEST for cov
HMP-Kurt        : 0.1735       0.8580  ← Best balanced
```

### Real-World S&P 500
- Covariance method: ~85% outlier precision
- Random: ~70% outlier precision
- **Improvement: +15-20%**

### Real-World EEG
- HMP-Kurt: Best ICA source separation
- Covariance-only: Destroys kurtosis needed for ICA
- **Validates theory on real neural signals**

---

## License

MIT License. See LICENSE file for details.
