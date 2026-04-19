'''
Environment Parity Audit: Single-Sample Integrity Check
Certifies that the new ARM-native environment reproduces Phase 1 bit-parity.
Detailed per-channel reporting for X tensor.

date : 19/04/2026
author : Antigravity (AI Assistant)
'''
import sys
import os
from pathlib import Path

# Force the project root into the path
PROJECT_ROOT = Path(__file__).parents[2].resolve()
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import argparse
from bin.preprocessing.build_dataset import DatasetBuilder
from iriscc.settings import DATASET_EXP5_30Y_DIR

def audit_parity(date_str):
    date = pd.Timestamp(date_str)
    date_formatted = date.strftime('%Y%m%d')
    # AUTHENTIC ARCHIVAL BASELINE
    baseline_root = Path("/scratch/globc/page/idownscale_exp5/datasets/dataset_exp5_30y/")
    baseline_file = baseline_root / f"sample_{date_formatted}.npz"
    
    if not baseline_file.exists():
        print(f"Error: Baseline file {baseline_file} not found.")
        sys.exit(1)
        
    print(f"Loading archival baseline: {baseline_file}")
    baseline = np.load(baseline_file)
    
    # Initialize DatasetBuilder with Experiment 5
    print("Initializing DatasetBuilder for 'exp5'...")
    get_builder = DatasetBuilder(exp='exp5')
    
    print(f"Regenerating tensors for {date_str} in new environment...")
    # Generate tensors using the official production logic
    x_new = get_builder.input_data(date)
    y_new = get_builder.target_data(date)
    
    # Per-channel audit for X
    # In Exp 5, X usually has: [Elevation, TAS Input]
    channels = get_builder.input_vars
    print(f"\n--- Predictor (X) Channel Audit ({date_str}) ---")
    for i, var in enumerate(channels):
        x_new_c = x_new[i]
        x_base_c = baseline['x'][i]
        
        mask = ~np.isnan(x_new_c) & ~np.isnan(x_base_c)
        diff = np.abs(x_new_c[mask] - x_base_c[mask]).max() if mask.any() else 0.0
        mismatch_mask = np.sum(np.isnan(x_new_c) != np.isnan(x_base_c))
        
        print(f"Channel {i} ({var}): Diff={diff:.2e}, MaskMismatches={mismatch_mask}")

    # Audit comparison (Account for NaNs)
    parity_x = np.array_equal(x_new, baseline['x'], equal_nan=True)
    parity_y = np.array_equal(y_new, baseline['y'], equal_nan=True)
    
    print(f"\n--- Final Environment Parity Summary ---")
    print(f"Predictor (X) Identical: {parity_x}")
    print(f"Target (Y) Identical:    {parity_y}")
    
    if parity_x and parity_y:
        print("\n[SUCCESS] ENVIRONMENT PARITY CERTIFIED: 0.00e+00 bit-identical parity achieved.")
    else:
        print("\n[WARNING] PARITY MISMATCH DETECTED.")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default="1980-01-01")
    args = parser.parse_args()
    audit_parity(args.date)
