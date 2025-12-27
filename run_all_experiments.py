"""
run_all_experiments.py
======================

Master script to run all experiments for the HMP paper.

Usage:
    python run_all_experiments.py [experiment_number]
    
    Examples:
    python run_all_experiments.py        # Run all experiments
    python run_all_experiments.py 1      # Run only experiment 1
    python run_all_experiments.py 1 2 3  # Run experiments 1, 2, 3
"""

import sys
import os
import time
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_experiment(exp_num):
    """Run a single experiment by number."""
    
    experiments = {
        1: ('Moment Preservation', 'exp_1_moment_preservation'),
        2: ('Covariance Tasks', 'exp_2_covariance_tasks'),
        3: ('Generative Models', 'exp_3_generative_models'),
        4: ('Signal Processing (Synthetic)', 'exp_4_signal_processing'),
        5: ('Finance Outliers (Synthetic)', 'exp_5_finance_outliers'),
        6: ('Real Signals (Synthetic)', 'exp_6_real_signals'),
        7: ('Ablations', 'exp_7_ablations'),
        8: ('Runtime', 'exp_8_runtime'),
        # Real-world data experiments
        9: ('S&P 500 Finance (REAL)', 'exp_5_finance_real'),
        10: ('PhysioNet EEG (REAL)', 'exp_6_eeg_real'),
    }
    
    if exp_num not in experiments:
        print(f"Unknown experiment: {exp_num}")
        return False
    
    name, module = experiments[exp_num]
    
    print("\n" + "#" * 80)
    print(f"# EXPERIMENT {exp_num}: {name.upper()}")
    print("#" * 80 + "\n")
    
    start = time.time()
    
    try:
        exec(f"from {module} import run_experiment; run_experiment()")
        elapsed = time.time() - start
        print(f"\n✓ Experiment {exp_num} completed in {elapsed:.1f}s")
        return True
    except Exception as e:
        print(f"\n✗ Experiment {exp_num} failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    
    print("=" * 80)
    print("HIERARCHICAL MOMENT-PRESERVING CORESETS")
    print("Experimental Evaluation Suite")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Parse arguments
    if len(sys.argv) > 1:
        exp_nums = [int(x) for x in sys.argv[1:]]
    else:
        exp_nums = list(range(1, 9))  # All experiments
    
    print(f"\nRunning experiments: {exp_nums}")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Run experiments
    results = {}
    total_start = time.time()
    
    for exp_num in exp_nums:
        success = run_experiment(exp_num)
        results[exp_num] = success
    
    total_time = time.time() - total_start
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    success_count = sum(results.values())
    total_count = len(results)
    
    for exp_num, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} Experiment {exp_num}")
    
    print(f"\n{success_count}/{total_count} experiments completed successfully")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # List result files
    print("\n" + "-" * 40)
    print("Result files:")
    for root, dirs, files in os.walk('results'):
        for f in files:
            print(f"  {os.path.join(root, f)}")


if __name__ == "__main__":
    main()
